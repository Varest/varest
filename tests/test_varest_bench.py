"""The Varest efficiency benchmark: loading, scoring, measurement and aggregation."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionConfig, LlamaConfig, LlamaForCausalLM, LlavaConfig, LlavaForConditionalGeneration
from torchvision.io import write_png

from llmoe.backbone import LLaVA
from llmoe.backbone.llava import LLaVAPromptPreprocessor
from llmoe.benchmarks import BENCHMARKS, DATASET_REQUIRED_BY_BENCHMARK
from llmoe.benchmarks.varest_bench import VarestBenchBenchmark, _MemorySampler, _VisionTokenTrace
from llmoe.compressor.varest.varest import vision_tokens_from_info
from llmoe.config import Config

BENCHMARK_DIR = Path("/workspace/output/Varest_Benchmark")


def _case(question_id: str, dataset: str, evaluator: str, text: str, answers: list[str], level: int) -> dict:
    return {
        "question_id": question_id,
        "image": f"images/{dataset}/{question_id}.png",
        "text": text,
        "answer": answers[0],
        "answers": answers,
        "category": "reference_answer_tokens",
        "level": level,
        "level_index": level - 1,
        "prompt_type": "short_answer",
        "evaluator": evaluator,
        "source": {"dataset": dataset},
        "workload": {"visual_tokens": 576},
    }


@pytest.fixture
def benchmark(tmp_path: Path) -> VarestBenchBenchmark:
    root = tmp_path / "input" / "Varest_Benchmark"
    cases = [
        _case("a", "DocVQA", "docvqa_anls", "who is cook?", ["ruby nell gay"], 1),
        _case("b", "ChartQA", "chartqa_relaxed_numeric", "How much?", ["53"], 1),
        _case("c", "AI2D", "ai2d_option_letter", "Which stage?\nOptions: pupa; adult", ["adult"], 2),
    ]
    for case in cases:
        path = root / case["image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        write_png(torch.zeros(3, 4, 4, dtype=torch.uint8), str(path))
    (root / "varest_benchmark.json").write_text(json.dumps(cases))
    config = Config("test", str(tmp_path), "Varest_Benchmark", _input_dir=str(tmp_path / "input"),
                    dataset_limit=3, custom_args={"varest_bench_warmup": 1, "varest_bench_repeats": 3})
    return VarestBenchBenchmark(config, LLaVA)


class _FakeBackbone:
    """Answers from a script and runs one tiny matmul inside a profiled 'generate'."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = 0
        self.model = torch.nn.Linear(8, 8)

    def __call__(self, *, images, prompts, max_new_tokens, generation_profiler=None, **kwargs) -> list[str]:
        self.calls += 1
        measure = getattr(generation_profiler, "measure_generate", None)
        with measure(self.model) if measure else nullcontext():
            for _ in range(3):  # one prefill, two decode steps
                self.model(torch.ones(1, 8))
        return [self.answers[(self.calls - 1) % len(self.answers)]]


def test_registered_as_a_performance_benchmark() -> None:
    assert BENCHMARKS["varest_bench"] is VarestBenchBenchmark
    assert DATASET_REQUIRED_BY_BENCHMARK[VarestBenchBenchmark] == "Varest_Benchmark"
    assert VarestBenchBenchmark.measures_performance


def test_rows_use_the_short_answer_prompt(benchmark: VarestBenchBenchmark) -> None:
    rows = list(benchmark.load_dataset())
    assert [row["question"] for row in rows] == [
        "who is cook?" + LLaVA.short_answer_suffix,
        "How much?" + LLaVA.short_answer_suffix,
        "Which stage?\nOptions: pupa; adult" + LLaVA.short_answer_suffix,
    ]
    assert rows[0]["image"].shape == (3, 4, 4)
    assert rows[2]["record_extra"]["dataset"] == "AI2D"


def test_responses_are_not_scored(benchmark: VarestBenchBenchmark) -> None:
    row = next(iter(benchmark.load_dataset()))
    assert benchmark.is_correct(row["answer"], row) is None


def test_generate_times_repeats_then_counts_flops(benchmark: VarestBenchBenchmark) -> None:
    row = next(iter(benchmark.load_dataset()))
    backbone = _FakeBackbone(["x"])
    assert benchmark.generate(backbone, [row], max_new_tokens=4) == ["x"]
    # 1 warmup + 3 timed + 1 FLOP pass; warmup happens once per run.
    assert backbone.calls == 5
    benchmark.generate(backbone, [row], max_new_tokens=4)
    assert backbone.calls == 9

    efficiency = row["record_extra"]["efficiency"]
    assert len(efficiency["latency_runs_seconds"]) == 3
    assert efficiency["latency_seconds"] == sorted(efficiency["latency_runs_seconds"])[1]
    assert efficiency["generated_tokens"] == 3
    assert efficiency["prefill_flops"] == efficiency["decode_flops"] // 2 == 2 * 8 * 8
    assert efficiency["flops"] == 3 * 2 * 8 * 8
    assert efficiency["outside_generate_flops"] == 0
    assert efficiency["flop_pass_matches_output"]

    with pytest.raises(RuntimeError, match="one query at a time"):
        benchmark.generate(backbone, [row, row], max_new_tokens=4)


def test_metrics_aggregate_per_level_and_dataset(benchmark: VarestBenchBenchmark) -> None:
    records = []
    backbone = _FakeBackbone(["x"])
    for row in benchmark.load_dataset():
        raw = benchmark.generate(backbone, [row], max_new_tokens=4)[0]
        records.append({
            "question_id": row["question_id"], "category": row["category"],
            "prediction": benchmark.parse_prediction(raw, row), **row["record_extra"],
        })
    metrics = benchmark.compute_metrics(records)
    assert metrics["metrics"]["n"] == 3
    assert "accuracy" not in metrics["metrics"]
    assert metrics["metrics"]["tflops_mean"] == pytest.approx(3 * 2 * 8 * 8 / 1e12)
    assert set(metrics["per_level"]["reference_answer_tokens"]) == {"1", "2"}
    assert metrics["per_level"]["reference_answer_tokens"]["1"]["n"] == 2
    assert set(metrics["per_dataset"]) == {"AI2D", "ChartQA", "DocVQA"}
    assert metrics["measurement"]["uncounted_ops"] == []
    assert benchmark.summary_line(metrics).startswith("latency=")


IMAGE_TOKEN = 199


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                        num_attention_heads=4, vocab_size=200)).eval()


def test_vision_tokens_are_traced_per_layer_for_llava() -> None:
    torch.manual_seed(0)
    model = LlavaForConditionalGeneration(LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=2, image_size=28, patch_size=14),
        text_config=LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                num_attention_heads=4, vocab_size=200),
        image_token_id=IMAGE_TOKEN, vision_feature_layer=-2,
    )).eval()
    input_ids = torch.tensor([[1, 5, *[IMAGE_TOKEN] * 4, 6, 7]])
    trace = _VisionTokenTrace(model)
    with torch.inference_mode(), trace:
        model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                       pixel_values=torch.randn(1, 3, 28, 28), max_new_tokens=3, do_sample=False, pad_token_id=0)
    # The 4 text tokens are excluded; each layer saw all 4 image tokens.
    assert trace.vision_tokens_per_layer == [4, 4]
    assert (trace.hidden_size, trace.intermediate_size) == (64, 128)


class _CoreShaped(torch.nn.Module):
    """CORE's layout: a LoRA-wrapped decoder as `.llm`, the image token behind `_image_token()`."""

    def __init__(self) -> None:
        super().__init__()
        self.llm = get_peft_model(_tiny_llama(), LoraConfig(r=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))

    def _image_token(self) -> int:
        return IMAGE_TOKEN

    def generate(self, image_tokens: int) -> None:
        # As in CoreVLM.forward: embed ids, scatter in visual features, generate from embeddings.
        input_ids = torch.tensor([[1, 5, *[IMAGE_TOKEN] * image_tokens, 6, 7, 8]])
        embeds = self.llm.get_input_embeddings()(input_ids)
        embeds[input_ids == IMAGE_TOKEN] = torch.randn(image_tokens, embeds.shape[-1])
        self.llm.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(input_ids),
                          max_new_tokens=3, do_sample=False, pad_token_id=0)


def test_vision_tokens_are_traced_per_layer_for_core() -> None:
    model = _CoreShaped().eval()
    trace = _VisionTokenTrace(model)
    with torch.inference_mode(), trace:
        model.generate(image_tokens=6)
    assert trace.vision_tokens_per_layer == [6, 6]
    # A fresh trace per prefill.
    with torch.inference_mode(), trace:
        model.generate(image_tokens=2)
    assert trace.vision_tokens_per_layer == [2, 2]


def test_vision_tokens_fall_back_to_the_backbone_report() -> None:
    # Varest feeds the decoder its own embeddings, so no image token reaches the embedding layer.
    model = _tiny_llama()
    embeds = model.get_input_embeddings()(torch.tensor([[1, 5, 6, 7, 8]]))

    def run(trace: _VisionTokenTrace) -> list[int] | None:
        with torch.inference_mode(), trace:
            model.generate(inputs_embeds=embeds, attention_mask=torch.ones(1, 5, dtype=torch.long),
                           max_new_tokens=2, do_sample=False, pad_token_id=0)
        return trace.vision_tokens_per_layer

    assert run(_VisionTokenTrace(model)) is None
    assert run(_VisionTokenTrace(model, lambda: [3, 1])) == [3, 1]
    assert run(_VisionTokenTrace(model, lambda: None)) is None


def test_varest_vision_tokens_follow_rel_and_halting() -> None:
    # W5: every visual token at every layer; W5RELJOINT: K' from the [REL] layer on; halting: none from its layer on.
    assert vision_tokens_from_info({"visual_tokens": 102}, 4) == [102] * 4
    assert vision_tokens_from_info({"visual_tokens": 102, "rel_layer": 2, "rel_kept": 26}, 4) == [102, 102, 26, 26]
    assert vision_tokens_from_info({"visual_tokens": 102, "rel_layer": 2, "rel_kept": 26, "halt_layer": 3},
                                   4) == [102, 102, 26, 0]
    assert vision_tokens_from_info({"visual_tokens": 102, "rel_layer": 2, "rel_kept": 0, "halt_layer": 1},
                                   4) == [102, 0, 0, 0]


def test_memory_mean_is_time_weighted() -> None:
    sampler = _MemorySampler(lambda: 0, 1.0)
    sampler.baseline_bytes = 100 * 1024 * 1024
    sampler.peak_bytes = 400 * 1024 * 1024
    mib = 1024 * 1024
    # 100 MiB for 1 s, 300 MiB for 2 s -> (100 + 600) / 3.
    sampler.samples = [(0.0, 100 * mib), (1.0, 300 * mib), (3.0, 100 * mib)]
    report = sampler.report()
    assert report["memory_mean_mib"] == pytest.approx(700 / 3)
    assert report["memory_mean_above_baseline_mib"] == pytest.approx(700 / 3 - 100)
    assert report["memory_peak_mib"] == 400
    assert report["memory_peak_above_baseline_mib"] == 300
    assert report["memory_samples"] == 3


def test_memory_sampler_polls_and_takes_the_counter_peak() -> None:
    reads = iter(range(10, 10_000))
    sampler = _MemorySampler(lambda: next(reads), 0.001, reset_peak=lambda: None, peak=lambda: 50_000)
    with sampler:
        time.sleep(0.02)
    assert sampler.baseline_bytes == 10
    assert len(sampler.samples) >= 3                       # entry, at least one poll, exit
    assert [t for t, _ in sampler.samples] == sorted(t for t, _ in sampler.samples)
    assert sampler.peak_bytes == 50_000                    # a spike between samples still counts
    assert sampler.mean_bytes >= 10


def test_no_trace_without_a_language_model(benchmark: VarestBenchBenchmark) -> None:
    row = next(iter(benchmark.load_dataset()))
    benchmark.generate(_FakeBackbone(["x"]), [row], max_new_tokens=4)
    efficiency = row["record_extra"]["efficiency"]
    assert efficiency["theoretical_tflops"] is None
    assert efficiency["vision_tokens_per_layer"] is None


@pytest.mark.skipif(not (BENCHMARK_DIR / "varest_benchmark.json").exists(), reason="benchmark not available")
def test_shipped_cases_reproduce_their_llava_token_levels() -> None:
    """The case ``text`` through LLMOE's own prompt path lands on every case's level."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf", local_files_only=True)
    except OSError:
        pytest.skip("LLaVA-1.5 tokenizer not cached")
    config = Config("test", "/workspace", "Varest_Benchmark", _input_dir=str(BENCHMARK_DIR.parent))
    benchmark = VarestBenchBenchmark(config, LLaVA)
    template = LLaVAPromptPreprocessor().prompt_template
    cases = json.loads((BENCHMARK_DIR / "varest_benchmark.json").read_text())
    for case, row in zip(cases, benchmark.load_dataset(), strict=True):
        # BOS plus the formatted text; the <image> placeholder is not a text token.
        tokens = len(tokenizer(template.format(prompt=row["question"])).input_ids) - 1
        assert tokens == case["workload"]["formatted_text_tokens"], case["question_id"]
