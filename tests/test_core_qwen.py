"""CORE's object-centric merge applied to Qwen2.5-VL's visual tokens.

Eq. (1) needs a raster grid of features and masks at the same resolution; Qwen's
post-merger tokens are such a grid, so the merge should transpose verbatim. These
tests check that it does, and that the mask resolution gap is closed the paper's
own way (interpolate onto the feature grid).
"""

import pytest
import torch

from llmoe.compressor.core.core import ObjectCentricMerger
from llmoe.compressor.core.core_qwen import (
    QwenCoreMerge,
    centroid_grid_positions,
    merge_qwen_tokens,
)


def _masks(num, height, width, *, seed=0):
    torch.manual_seed(seed)
    return torch.rand(num, height, width)


def _disjoint_masks(height, width):
    """Two clean halves, so the expected merge is computable by hand."""
    masks = torch.zeros(2, height, width)
    masks[0, : height // 2] = 1.0
    masks[1, height // 2:] = 1.0
    return masks


# ------------------------------------------------------------------- the merge


def test_merge_produces_the_fixed_token_budget():
    tokens = torch.randn(64, 16)
    merger = ObjectCentricMerger(merge_mode="hard", rate="fixed", fixed_tokens=8)
    merged = merge_qwen_tokens(tokens, _masks(6, 8, 8), grid=(8, 8), merger=merger)
    assert merged.tokens.shape == (8, 16)
    assert merged.centroids.numel() == 8


def test_masks_are_interpolated_onto_the_token_grid():
    """The paper downsamples masks to the feature grid; here that grid is Qwen's."""
    tokens = torch.randn(16, 8)
    merger = ObjectCentricMerger(merge_mode="hard", rate="fixed", fixed_tokens=4)
    # Masks at 32x32 for a 4x4 token grid must still work.
    merged = merge_qwen_tokens(tokens, _masks(5, 32, 32), grid=(4, 4), merger=merger)
    assert merged.tokens.shape == (4, 8)


def test_merge_is_a_weighted_average_of_the_tokens_it_covers():
    """Eq. (1): t_n is the mask-weighted mean of the features under the mask."""
    height = width = 4
    tokens = torch.arange(height * width, dtype=torch.float32)[:, None].repeat(1, 3)
    merger = ObjectCentricMerger(merge_mode="hard", rate="adaptive")
    merged = merge_qwen_tokens(tokens, _disjoint_masks(height, width),
                               grid=(height, width), merger=merger)
    # Top half covers raster 0..7, bottom half 8..15.
    expected = torch.tensor([3.5, 11.5])
    assert torch.allclose(merged.tokens[:, 0].sort().values, expected, atol=1e-4)


def test_tokens_come_back_sorted_by_centroid():
    tokens = torch.randn(64, 8)
    merger = ObjectCentricMerger(merge_mode="hard", rate="fixed", fixed_tokens=6)
    merged = merge_qwen_tokens(tokens, _masks(5, 8, 8, seed=3), grid=(8, 8),
                               merger=merger)
    assert torch.equal(merged.centroids, merged.centroids.sort().values)


def test_grid_mismatch_is_refused():
    merger = ObjectCentricMerger(merge_mode="hard", rate="fixed", fixed_tokens=4)
    with pytest.raises(ValueError, match="expected .* tokens for a"):
        merge_qwen_tokens(torch.randn(63, 8), _masks(3, 8, 8), grid=(8, 8),
                          merger=merger)


def test_masks_must_be_three_dimensional():
    merger = ObjectCentricMerger(merge_mode="hard", rate="fixed", fixed_tokens=4)
    with pytest.raises(ValueError, match=r"masks must be \[N, H, W\]"):
        merge_qwen_tokens(torch.randn(16, 8), torch.rand(4, 4), grid=(4, 4),
                          merger=merger)


# ---------------------------------------------------------------- M-RoPE cells


def test_centroids_become_grid_cells_in_range():
    centroids = torch.tensor([0.0, 5.4, 15.0])
    cells = centroid_grid_positions(centroids, (4, 4))
    assert cells.tolist() == [[0, 0], [1, 1], [3, 3]]


def test_centroid_cells_are_clamped_inside_the_grid():
    cells = centroid_grid_positions(torch.tensor([-3.0, 999.0]), (4, 4))
    assert cells.min() >= 0
    assert cells[:, 0].max() < 4 and cells[:, 1].max() < 4


# --------------------------------------------------------------------- holder


def test_token_count_is_known_for_the_fixed_rate_and_not_for_adaptive():
    fixed = QwenCoreMerge(merger=ObjectCentricMerger(
        merge_mode="hard", rate="fixed", fixed_tokens=12))
    assert fixed.token_count == 12
    adaptive = QwenCoreMerge(merger=ObjectCentricMerger(
        merge_mode="hard", rate="adaptive"))
    assert adaptive.token_count is None


def test_apply_without_masks_is_an_error_not_silent_passthrough():
    holder = QwenCoreMerge(merger=ObjectCentricMerger(
        merge_mode="hard", rate="fixed", fixed_tokens=4))
    with pytest.raises(RuntimeError, match="no object masks"):
        holder.apply(torch.randn(16, 8), (4, 4))


def test_apply_records_centroids_for_the_rope_gather():
    holder = QwenCoreMerge(merger=ObjectCentricMerger(
        merge_mode="hard", rate="fixed", fixed_tokens=5))
    holder.masks = _masks(4, 8, 8, seed=7)
    out = holder.apply(torch.randn(64, 8), (8, 8))
    assert out.shape == (5, 8)
    assert holder.centroids is not None and holder.centroids.numel() == 5
    cells = centroid_grid_positions(holder.centroids, (8, 8))
    assert cells.shape == (5, 2)
