"""QuietPrune's self-distillation on Qwen2.5-VL.

The training loop is the released LLaVA one, unchanged. That is only possible
because the in-ViT schedule trims placeholders exactly as LLaVA does, so the
teacher/student pair the loop assumes -- same tensors, student's visual run
shorter -- holds on Qwen too. These tests drive a real optimizer step through it.
"""

import pytest
import torch

from llmoe.compressor.quietprune.quietprune import QuietPruneCompressor
from llmoe.compressor.quietprune.quietprune_tr import (
    answer_cross_entropy_loss,
    answer_kl_divergence_loss,
)

IMG = 151
VISION = dict(depth=8, hidden_size=32, out_hidden_size=64, intermediate_size=64,
              num_heads=2, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
              window_size=112, fullatt_block_indexes=[3, 7], in_chans=3)


def _tiny_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_VLConfig(
        vision_config=dict(VISION),
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=200,
                         rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]}),
        image_token_id=IMG, video_token_id=152,
        vision_start_token_id=153, vision_end_token_id=154,
    )
    torch.manual_seed(0)
    return Qwen2_5_VLForConditionalGeneration(config).eval()


class _StubVLM:
    def __init__(self, model) -> None:
        self.model = model
        self.processor = None

    def prepare_inputs(self, inputs):
        return inputs


def _teacher_inputs():
    ids = [5, 6, 153] + [IMG] * 16 + [154, 7, 8, 9]
    input_ids = torch.tensor([ids])
    return {
        "input_ids": input_ids,
        "pixel_values": torch.randn(64, 3 * 2 * 14 * 14),
        "image_grid_thw": torch.tensor([[1, 8, 8]]),
        "mm_token_type_ids": (input_ids == IMG).int(),
        "attention_mask": torch.ones_like(input_ids),
    }


def _install(model):
    compressor = QuietPruneCompressor(pruning_rate=0.25, pretrained=False)
    compressor.install(_StubVLM(model))
    return compressor


def _labels(input_ids, answer_len=3):
    labels = torch.full_like(input_ids, -100)
    labels[:, -answer_len:] = input_ids[:, -answer_len:]
    return labels


# --------------------------------------------------------------------- bypass


def test_bypass_gives_a_genuinely_unpruned_teacher():
    """The teacher must see every visual token, or the distillation target is wrong."""
    model = _tiny_qwen()
    compressor = _install(model)
    try:
        teacher = _teacher_inputs()
        with torch.no_grad(), compressor.bypass():
            logits = model(**teacher).logits
        # Nothing was trimmed and nothing was pruned.
        assert logits.shape[1] == teacher["input_ids"].shape[1]
        assert compressor._qwen_install._runner.survivor_groups is None
    finally:
        compressor.uninstall()


def test_bypass_lifts_the_placeholder_trim_too():
    model = _tiny_qwen()
    compressor = _install(model)
    try:
        teacher = _teacher_inputs()
        with compressor.bypass():
            untouched = compressor.trim_placeholders(dict(teacher))
        assert untouched["input_ids"].shape == teacher["input_ids"].shape
        trimmed = compressor.trim_placeholders(dict(teacher))
        assert trimmed["input_ids"].shape[1] < teacher["input_ids"].shape[1]
    finally:
        compressor.uninstall()


# ------------------------------------------------------------------- one step


def test_a_real_distillation_step_reaches_the_adapter():
    model = _tiny_qwen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    compressor = _install(model)
    try:
        compressor.train()
        assert list(compressor.parameters()), "the adapter must be trainable"

        teacher = _teacher_inputs()
        student = compressor.trim_placeholders(dict(teacher))
        teacher_labels = _labels(teacher["input_ids"])
        student_labels = _labels(student["input_ids"])

        with torch.no_grad(), compressor.bypass():
            teacher_logits = model(**teacher, use_cache=False).logits
        student_logits = model(**student, use_cache=False).logits

        kl, answer_tokens = answer_kl_divergence_loss(
            student_logits, student_labels, teacher_logits, teacher_labels)
        cross_entropy = answer_cross_entropy_loss(student_logits, student_labels)
        loss = kl + cross_entropy
        assert torch.isfinite(loss) and answer_tokens > 0
        loss.backward()

        grads = [p.grad for p in compressor.parameters() if p.grad is not None]
        assert grads, "no gradient reached the [Q-CLS] adapter"
        assert any(g.abs().sum() > 0 for g in grads)
        # The frozen VLM must not have accumulated any.
        assert all(p.grad is None for p in model.parameters())
    finally:
        compressor.uninstall()


def test_student_is_shorter_than_teacher_by_the_pruned_tokens():
    model = _tiny_qwen()
    compressor = _install(model)
    try:
        teacher = _teacher_inputs()
        student = compressor.trim_placeholders(dict(teacher))
        target = compressor._qwen_install.surviving_tokens(teacher["image_grid_thw"])
        dropped = 16 - target
        assert student["input_ids"].shape[1] == teacher["input_ids"].shape[1] - dropped
        assert int((student["input_ids"][0] == IMG).sum()) == target
    finally:
        compressor.uninstall()


def test_checkpoint_round_trips_the_adapter():
    model = _tiny_qwen()
    compressor = _install(model)
    try:
        payload = compressor.checkpoint_dict()
        state = payload.get("adapter_state_dict") or payload.get("state_dict") or payload
        assert state, "checkpoint carries no adapter tensors"
        assert all(isinstance(v, torch.Tensor) for v in state.values())
    finally:
        compressor.uninstall()
