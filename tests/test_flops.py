"""Measured FLOPs: attention pricing, module attribution and per-phase tracing."""

import pytest
import torch
import torch.nn.functional as F
from transformers import CLIPVisionConfig, LlamaConfig, LlamaForCausalLM, LlavaConfig, LlavaForConditionalGeneration

from llmoe.flops import FlopCounter, theoretical_flops
from llmoe.performance import trace_generate


def _tiny_llava(attn_implementation: str) -> LlavaForConditionalGeneration:
    torch.manual_seed(0)
    config = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=2, image_size=28, patch_size=14),
        text_config=LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, vocab_size=200),
        image_token_id=199,
        vision_feature_layer=-2,
        attn_implementation=attn_implementation,
    )
    return LlavaForConditionalGeneration(config).eval()


def _generate(model: LlavaForConditionalGeneration, new_tokens: int = 3) -> None:
    input_ids = torch.tensor([[1, 5, *[199] * 4, 6, 7]])
    model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                   pixel_values=torch.randn(1, 3, 28, 28), max_new_tokens=new_tokens,
                   min_new_tokens=new_tokens, do_sample=False, pad_token_id=0)


def test_grouped_query_attention_is_priced_per_query_head() -> None:
    query = torch.randn(1, 8, 10, 64)
    key = torch.randn(1, 2, 10, 64)
    with FlopCounter() as counter:
        F.scaled_dot_product_attention(query, key, key, enable_gqa=True)
    # Q @ K^T and P @ V, each 2 * heads * s_q * s_k * d.
    assert counter.total == 2 * 2 * 8 * 10 * 10 * 64
    assert counter.uncounted_ops == []


def test_sdpa_and_eager_attention_count_the_same_work() -> None:
    totals = {}
    for implementation in ("eager", "sdpa"):
        model = _tiny_llava(implementation)
        with torch.inference_mode(), FlopCounter() as counter:
            _generate(model)
        totals[implementation] = (counter.total, counter.by_module())
    assert totals["eager"] == totals["sdpa"]


def test_modules_partition_the_total() -> None:
    model = _tiny_llava("sdpa")
    with torch.inference_mode(), FlopCounter() as counter:
        _generate(model)
    modules = counter.by_module()
    assert set(modules) == {"model.vision_tower", "model.multi_modal_projector", "model.language_model", "lm_head"}
    assert sum(modules.values()) == counter.total


def test_trace_splits_prefill_from_decode_and_restores_forward() -> None:
    model = _tiny_llava("sdpa")
    with torch.inference_mode(), FlopCounter() as counter:
        with trace_generate(model, counter) as trace:
            _generate(model, new_tokens=4)
    assert (trace.prefill_calls, trace.decode_calls, trace.forward_calls) == (1, 3, 4)
    assert trace.prefill_flops + trace.decode_flops == counter.total
    # The image is encoded once, in the prefill.
    assert trace.prefill_flops > counter.by_module()["model.vision_tower"]
    assert trace.wall_seconds > 0.0
    assert trace.wall_seconds >= trace.prefill_seconds + trace.decode_seconds
    assert "forward" not in model.__dict__


def test_trace_sees_the_forwards_under_a_peft_lora_wrapper() -> None:
    # CORE's decoder: PEFT's generate runs the wrapped model, never its own forward.
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    llm = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, vocab_size=200)).eval()
    model = get_peft_model(llm, LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).eval()
    embeds = model.get_input_embeddings()(torch.tensor([[1, 5, 6, 7]]))
    with torch.inference_mode(), FlopCounter() as counter:
        with trace_generate(model, counter) as trace:
            model.generate(inputs_embeds=embeds, attention_mask=torch.ones(1, 4, dtype=torch.long),
                           max_new_tokens=4, min_new_tokens=4, do_sample=False, pad_token_id=0)
    assert (trace.prefill_calls, trace.decode_calls) == (1, 3)
    assert trace.prefill_flops + trace.decode_flops == counter.total
    assert "forward" not in model.get_base_model().__dict__


def test_trace_follows_a_decoder_driven_layer_by_layer() -> None:
    # Varest's [REL] loop (varest.rel.generate_pruned) calls the decoder blocks and the output head directly and never
    # the model's forward; each pass then runs from the first block to the head.
    from varest.rel import generate_pruned

    torch.manual_seed(0)
    llm = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, vocab_size=200)).eval()
    embeds = llm.get_input_embeddings()(torch.tensor([[1, 5, 6, 7, 8, 9]]))
    with torch.inference_mode(), FlopCounter() as counter:
        with trace_generate(llm, counter) as trace:
            ids, _ = generate_pruned(llm.model, llm.lm_head, llm.get_input_embeddings(), None, embeds,
                                     (1, 4, 4, 0), 0, None, 3, -1)
    assert len(ids) == 3
    # One prefill, then one pass per generated token.
    assert (trace.prefill_calls, trace.decode_calls) == (1, 3)
    # Everything but the rotary tables, which the loop builds before entering the first block.
    in_passes = trace.prefill_flops + trace.decode_flops
    assert 0 <= counter.total - in_passes < 1e-3 * counter.total
    assert trace.prefill_flops > trace.decode_flops / 3
    assert trace.prefill_seconds > 0.0 and trace.decode_seconds > 0.0
    assert "forward" not in llm.__dict__


def test_theoretical_flops_match_published_llava_figure() -> None:
    # LLaVA-1.5-7B with all 576 vision tokens: 3.82 TFLOPs in arXiv:2602.23699, Table 2.
    assert theoretical_flops([576] * 32, 4096, 11008) / 1e12 == pytest.approx(3.82, abs=0.005)
    assert theoretical_flops([0] * 32, 4096, 11008) == 0


def test_modules_partition_the_total_under_uncalled_containers() -> None:
    # Layers held in a ModuleList and called directly, as Llama and CORE's ConvNeXt do.
    class Stack(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(3))
            self.head = torch.nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    with torch.inference_mode(), FlopCounter() as counter:
        Stack()(torch.ones(1, 8))
    assert counter.by_module(depth=1) == {"blocks": 384, "head": 32}
    assert counter.by_module() == {"blocks.0": 128, "blocks.1": 128, "blocks.2": 128, "head": 32}
    assert sum(counter.by_module().values()) == counter.total


def test_bilinear_upsampling_is_not_flagged_as_uncounted() -> None:
    with torch.inference_mode(), FlopCounter() as counter:
        torch.nn.functional.interpolate(torch.ones(1, 1, 4, 4), scale_factor=2, mode="bilinear")
    assert counter.uncounted_ops == []
