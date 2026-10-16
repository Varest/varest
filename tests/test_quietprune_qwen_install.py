"""QuietPrune's in-ViT schedule installed on a real Qwen2.5-VL model.

Pruning inside the tower changes the visual-token count the decoder sees, so this
covers the three consequences a post-merger reduction never faces: the placeholder
trim, the M-RoPE gather for survivors, and the ordering problem where ``generate``
lays out positions before the tower has run.
"""

import pytest
import torch

from llmoe.compressor.quietprune.quietprune_qwen_install import QwenQuietPruneInstaller

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

    def prepare_inputs(self, inputs):
        return inputs


def _inputs(model, image_tokens):
    grid = torch.tensor([[1, 8, 8]])
    ids = [5, 6, 153] + [IMG] * image_tokens + [154, 7, 8]
    input_ids = torch.tensor([ids])
    return {
        "input_ids": input_ids,
        "pixel_values": torch.randn(64, 3 * 2 * 14 * 14),
        "image_grid_thw": grid,
        "mm_token_type_ids": (input_ids == IMG).int(),
        "attention_mask": torch.ones_like(input_ids),
    }


def _install(model, rate=0.25, aggregate=True):
    installer = QwenQuietPruneInstaller(
        rate=rate, aggregate=aggregate, normalize_aggregation=False)
    installer.install(_StubVLM(model))
    return installer


# ------------------------------------------------------------------ token count


def test_surviving_token_count_is_known_before_the_image():
    model = _tiny_qwen()
    installer = _install(model)
    try:
        # 8x8 patches -> 4x4 = 16 groups; three cuts at R=0.25 -> 12, 9, 7; +1 aggregate
        assert installer.surviving_tokens(torch.tensor([[1, 8, 8]])) == 8
    finally:
        installer.uninstall()


def test_zero_rate_keeps_every_token_and_adds_no_aggregate():
    model = _tiny_qwen()
    installer = _install(model, rate=0.0)
    try:
        assert installer.surviving_tokens(torch.tensor([[1, 8, 8]])) == 16
    finally:
        installer.uninstall()


# --------------------------------------------------------------- placeholders


def test_placeholders_are_trimmed_to_the_surviving_count():
    model = _tiny_qwen()
    vlm = _StubVLM(model)
    installer = QwenQuietPruneInstaller(rate=0.25, aggregate=True,
                                        normalize_aggregation=False)
    installer.install(vlm)
    try:
        raw = _inputs(model, image_tokens=16)
        trimmed = vlm.prepare_inputs(raw)
        target = installer.surviving_tokens(raw["image_grid_thw"])
        ids = trimmed["input_ids"][0]
        assert int((ids == IMG).sum()) == target
        # Companions must be cut in lockstep or M-RoPE reads the wrong layout.
        assert trimmed["mm_token_type_ids"].shape == trimmed["input_ids"].shape
        assert torch.equal((ids == IMG), trimmed["mm_token_type_ids"][0].bool())
    finally:
        installer.uninstall()


# ------------------------------------------------------------------- forward


def test_full_forward_runs_with_pruning_inside_the_tower():
    model = _tiny_qwen()
    vlm = _StubVLM(model)
    installer = QwenQuietPruneInstaller(rate=0.25, aggregate=True,
                                        normalize_aggregation=False)
    installer.install(vlm)
    try:
        target = installer.surviving_tokens(torch.tensor([[1, 8, 8]]))
        inputs = _inputs(model, image_tokens=target)
        with torch.no_grad():
            out = model(**inputs)
        assert torch.isfinite(out.logits).all()
        assert out.logits.shape[1] == inputs["input_ids"].shape[1]
        # The tower really did drop groups.
        assert installer._runner.survivor_groups.numel() < 16
    finally:
        installer.uninstall()


def test_generate_runs_despite_positions_being_laid_out_before_the_tower():
    """`generate` calls get_rope_index before the vision forward; priming covers it."""
    model = _tiny_qwen()
    vlm = _StubVLM(model)
    installer = QwenQuietPruneInstaller(rate=0.25, aggregate=True,
                                        normalize_aggregation=False)
    installer.install(vlm)
    try:
        target = installer.surviving_tokens(torch.tensor([[1, 8, 8]]))
        inputs = _inputs(model, image_tokens=target)
        installer._runner.survivor_groups = None      # what the first call sees
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        assert generated.shape[0] == 1
        assert generated.shape[1] == inputs["input_ids"].shape[1] + 3
    finally:
        installer.uninstall()


def test_survivors_get_their_own_grid_coordinates():
    model = _tiny_qwen()
    vlm = _StubVLM(model)
    installer = QwenQuietPruneInstaller(rate=0.25, aggregate=False,
                                        normalize_aggregation=False)
    installer.install(vlm)
    try:
        target = installer.surviving_tokens(torch.tensor([[1, 8, 8]]))
        inputs = _inputs(model, image_tokens=target)
        # Drive the whole model: [Q-CLS] comes from the prompt, so poking the
        # tower directly would skip the capture and silently disable pruning.
        with torch.no_grad():
            model(**inputs)
        survivors = installer._runner.survivor_groups
        assert survivors.numel() == target
        positions, _ = model.model.get_rope_index(
            inputs["input_ids"], inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"])
        mesh = model.model.get_vision_position_ids(
            3, inputs["image_grid_thw"][0], 1,
            model.config.vision_config.spatial_merge_size, 1)
        columns = (inputs["input_ids"][0] == IMG).nonzero(as_tuple=True)[0]
        assert torch.equal(positions[:, 0, columns], mesh[:, survivors])
    finally:
        installer.uninstall()


# ------------------------------------------------------------------- teardown


def test_uninstall_leaves_the_model_untouched():
    model = _tiny_qwen()
    vlm = _StubVLM(model)
    installer = QwenQuietPruneInstaller(rate=0.25, aggregate=True,
                                        normalize_aggregation=False)
    installer.install(vlm)
    assert {"get_image_features", "get_rope_index"} <= set(vars(model.model))
    installer.uninstall()
    assert "get_image_features" not in vars(model.model)
    assert "get_rope_index" not in vars(model.model)
    # And the stock path still runs.
    inputs = _inputs(model, image_tokens=16)
    with torch.no_grad():
        assert torch.isfinite(model(**inputs).logits).all()
