"""QuietPrune's query-guided ``[Q-CLS]`` path on Qwen2.5-VL.

The merged Qwen adaptation scores groups by received attention, which is
training-free and therefore not query-guided. These tests cover the trained
alternative: the adapter is ``visual.merger.mlp`` inverted, so it maps prompt
embeddings into the merger's own input space, and the score is that query's
similarity to each ``2x2`` group.
"""

import pytest
import torch

from llmoe.compressor.quietprune.quietprune_qwen_qcls import (
    MergerGroupCapture,
    QwenQCLSAdapter,
    qcls_group_scores,
)

VISION = dict(depth=4, hidden_size=32, out_hidden_size=64, intermediate_size=64,
              num_heads=2, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
              window_size=112, fullatt_block_indexes=[1, 3], in_chans=3)


def _tiny_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_VLConfig(
        vision_config=dict(VISION),
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=200,
                         rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]}),
        image_token_id=151, video_token_id=152,
        vision_start_token_id=153, vision_end_token_id=154,
    )
    torch.manual_seed(0)
    return Qwen2_5_VLForConditionalGeneration(config).eval()


def test_adapter_inverts_the_merger_into_group_space():
    """The adapter must land in the merger's *input* width, not the decoder's."""
    model = _tiny_qwen()
    visual = model.model.visual
    adapter = QwenQCLSAdapter.from_visual(visual)
    # merger input = context_dim * spatial_merge_size**2 = 32 * 4
    assert adapter.group_dim == VISION["hidden_size"] * VISION["spatial_merge_size"] ** 2
    prompt = torch.randn(1, 7, VISION["out_hidden_size"])
    assert adapter(prompt).shape == (1, adapter.group_dim)


def test_adapter_rejects_a_tower_without_a_merger():
    class _Bare:
        pass

    with pytest.raises(TypeError, match="visual.merger.mlp"):
        QwenQCLSAdapter.from_visual(_Bare())


def test_query_mask_excludes_masked_positions():
    model = _tiny_qwen()
    adapter = QwenQCLSAdapter.from_visual(model.model.visual)
    prompt = torch.randn(1, 6, VISION["out_hidden_size"])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    masked = adapter(prompt, mask)
    # Same as pooling only the two unmasked positions.
    direct = adapter(prompt[:, :2], torch.ones(1, 2))
    assert torch.allclose(masked, direct, atol=1e-5)


def test_all_masked_prompt_gives_a_neutral_query_not_a_blowup():
    model = _tiny_qwen()
    adapter = QwenQCLSAdapter.from_visual(model.model.visual)
    prompt = torch.randn(1, 4, VISION["out_hidden_size"])
    out = adapter(prompt, torch.zeros(1, 4))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_capture_returns_one_group_row_per_merged_token_in_raster_order():
    """The tower permutes groups; the capture must un-permute exactly as it does."""
    model = _tiny_qwen()
    visual = model.model.visual
    capture = MergerGroupCapture()
    capture.install(visual)
    try:
        grid = torch.tensor([[1, 8, 8]])
        patches = 8 * 8
        with torch.no_grad():
            out = visual(torch.randn(patches, 3 * 2 * 14 * 14), grid_thw=grid)
        groups = capture.raster(visual, grid, patches)
    finally:
        capture.remove()
    merged_tokens = out.pooler_output.shape[0]
    assert groups.shape == (merged_tokens,
                            VISION["hidden_size"] * VISION["spatial_merge_size"] ** 2)


def test_capture_raster_matches_the_towers_own_reordering():
    """Feeding the captured groups back through the merger MLP must reproduce it."""
    model = _tiny_qwen()
    visual = model.model.visual
    capture = MergerGroupCapture()
    capture.install(visual)
    try:
        grid = torch.tensor([[1, 8, 8]])
        patches = 8 * 8
        with torch.no_grad():
            out = visual(torch.randn(patches, 3 * 2 * 14 * 14), grid_thw=grid)
            groups = capture.raster(visual, grid, patches)
            replayed = visual.merger.mlp(groups)
    finally:
        capture.remove()
    # pooler_output is the merger's output already un-permuted into raster order.
    assert torch.allclose(replayed, out.pooler_output, atol=1e-4)


def test_capture_without_a_forward_is_an_error_not_silent_zeros():
    model = _tiny_qwen()
    capture = MergerGroupCapture()
    capture.install(model.model.visual)
    try:
        with pytest.raises(RuntimeError, match="captured no merger input"):
            capture.raster(model.model.visual, torch.tensor([[1, 8, 8]]), 64)
    finally:
        capture.remove()


def test_scores_are_cosine_and_rank_by_query_relevance():
    groups = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = torch.tensor([1.0, 0.0])
    scores = qcls_group_scores(query, groups)
    assert torch.allclose(scores, torch.tensor([1.0, 0.0, -1.0]), atol=1e-6)
    # Magnitude must not decide the ranking -- only direction.
    scaled = qcls_group_scores(query, groups * 100.0)
    assert torch.allclose(scores, scaled, atol=1e-6)


def test_scores_reject_a_width_mismatch():
    with pytest.raises(ValueError, match="does not match the merger group width"):
        qcls_group_scores(torch.randn(5), torch.randn(4, 7))


def test_score_is_a_drop_in_for_the_received_attention_signal():
    """Same shape and dtype as what `quiet_qwen_reduce` already consumes."""
    model = _tiny_qwen()
    visual = model.model.visual
    adapter = QwenQCLSAdapter.from_visual(visual)
    capture = MergerGroupCapture()
    capture.install(visual)
    try:
        grid = torch.tensor([[1, 8, 8]])
        patches = 8 * 8
        with torch.no_grad():
            visual(torch.randn(patches, 3 * 2 * 14 * 14), grid_thw=grid)
            groups = capture.raster(visual, grid, patches)
            q_cls = adapter(torch.randn(1, 9, VISION["out_hidden_size"]))
            scores = qcls_group_scores(q_cls, groups)
    finally:
        capture.remove()
    assert scores.shape == (patches // visual.spatial_merge_unit,)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
