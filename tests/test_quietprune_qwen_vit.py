"""In-ViT group pruning for QuietPrune on Qwen2.5-VL.

The load-bearing claim of this module is that whole merger groups can be removed
from Qwen's flattened, window-permuted sequence without invalidating anything the
tower depends on. These tests check that against the real tower rather than a
mock: the layout is compared with what ``get_window_index`` actually produced, and
a no-op prune must reproduce the unpruned forward bit for bit.
"""

import pytest
import torch

from llmoe.compressor.quietprune.quietprune_qwen_vit import (
    group_layout,
    prune_groups,
    quietprune_layers,
)

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
        image_token_id=151, video_token_id=152,
        vision_start_token_id=153, vision_end_token_id=154,
    )
    torch.manual_seed(0)
    return Qwen2_5_VLForConditionalGeneration(config).eval()


def _layout_for(visual, grid):
    patches = int(grid[0].prod())
    _, cu_window = visual.get_window_index(grid)
    return group_layout(visual, grid, patches, cu_window), patches


# --------------------------------------------------------------- pruning depths


def test_pruning_depths_are_the_papers_quarters():
    """Sec. 5.1: three fixed layers at 1/4, 1/2 and 3/4 of the blocks."""
    layers = quietprune_layers(32)
    assert layers == (7, 15, 23)
    assert quietprune_layers(24) == (5, 11, 17)


def test_pruning_depths_stay_strictly_increasing_on_a_shallow_tower():
    layers = quietprune_layers(4)
    assert len(layers) == len(set(layers))
    assert list(layers) == sorted(layers)
    assert all(0 <= index <= 2 for index in layers)


def test_pruning_depths_reject_a_degenerate_tower():
    with pytest.raises(ValueError, match="at least 2 blocks"):
        quietprune_layers(1)


# ----------------------------------------------------------------- group layout


def test_layout_labels_every_group_with_the_towers_own_permutation():
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    layout, patches = _layout_for(visual, grid)
    window_index, _ = visual.get_window_index(grid)
    assert layout.num_groups == patches // visual.spatial_merge_unit
    assert torch.equal(layout.original, window_index)
    # Every group belongs to exactly one window, and the windows tile the groups.
    assert layout.window_of.numel() == layout.num_groups
    assert int(layout.window_of.max()) == layout.num_windows - 1


def test_layout_rejects_an_indivisible_patch_count():
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    _, cu_window = visual.get_window_index(grid)
    with pytest.raises(RuntimeError, match="divide by the merger unit"):
        group_layout(visual, grid, 63, cu_window)


# ------------------------------------------------------------------- no-op prune


def test_keeping_every_group_reproduces_the_unpruned_forward_exactly():
    """The strongest check: a no-op prune must change nothing at all."""
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    layout, patches = _layout_for(visual, grid)
    pixel_values = torch.randn(patches, 3 * 2 * 14 * 14)

    with torch.no_grad():
        baseline = visual(pixel_values, grid_thw=grid).pooler_output

    hidden = torch.randn(patches, VISION["hidden_size"])
    rotary = torch.randn(patches, 8)
    keep_all = torch.arange(layout.num_groups)
    pruned = prune_groups(hidden, rotary, layout, keep_all)

    assert torch.equal(pruned.hidden_states, hidden)
    assert torch.equal(pruned.rotary_pos_emb, rotary)
    assert torch.equal(pruned.layout.original, layout.original)
    assert int(pruned.cu_seqlens[-1]) == patches
    assert pruned.dropped_groups.numel() == 0
    # And the baseline forward is untouched by any of this.
    with torch.no_grad():
        again = visual(pixel_values, grid_thw=grid).pooler_output
    assert torch.equal(baseline, again)


def test_noop_prune_preserves_the_window_boundaries():
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    layout, patches = _layout_for(visual, grid)
    _, cu_window = visual.get_window_index(grid)
    expected = torch.unique_consecutive(torch.tensor(list(cu_window), dtype=torch.long))

    pruned = prune_groups(torch.randn(patches, 8), torch.randn(patches, 4),
                          layout, torch.arange(layout.num_groups))
    assert torch.equal(pruned.cu_window_seqlens.long(), expected.long())


# ---------------------------------------------------------------- real pruning


def test_pruning_removes_whole_groups_and_nothing_else():
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    layout, patches = _layout_for(visual, grid)
    unit = visual.spatial_merge_unit

    hidden = torch.arange(patches, dtype=torch.float32)[:, None].repeat(1, 6)
    rotary = torch.randn(patches, 4)
    keep = torch.tensor([0, 2, 5])
    pruned = prune_groups(hidden, rotary, layout, keep)

    assert pruned.hidden_states.shape[0] == keep.numel() * unit
    # Every surviving row is one of the kept groups' original rows, in order.
    expected_rows = torch.cat([torch.arange(g * unit, (g + 1) * unit) for g in keep])
    assert torch.equal(pruned.hidden_states[:, 0].long(), expected_rows)
    assert torch.equal(pruned.rotary_pos_emb, rotary[expected_rows])
    assert int(pruned.cu_seqlens[-1]) == keep.numel() * unit


def test_pruning_keeps_the_survivors_original_group_ids_for_re_sorting():
    """The merger's output is re-sorted by original id; dropping breaks argsort."""
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    layout, patches = _layout_for(visual, grid)
    keep = torch.tensor([1, 4, 7, 9])
    pruned = prune_groups(torch.randn(patches, 6), torch.randn(patches, 4), layout, keep)
    assert torch.equal(pruned.layout.original, layout.original[keep])
    assert torch.equal(pruned.dropped_groups,
                       torch.tensor([g for g in range(layout.num_groups)
                                     if g not in keep.tolist()]))


def test_window_boundaries_shrink_to_the_surviving_groups():
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    layout, patches = _layout_for(visual, grid)
    unit = visual.spatial_merge_unit
    keep = torch.arange(0, layout.num_groups, 2)      # drop every other group
    pruned = prune_groups(torch.randn(patches, 6), torch.randn(patches, 4), layout, keep)

    assert int(pruned.cu_window_seqlens[0]) == 0
    assert int(pruned.cu_window_seqlens[-1]) == keep.numel() * unit
    deltas = pruned.cu_window_seqlens[1:] - pruned.cu_window_seqlens[:-1]
    assert (deltas > 0).all()                          # no empty windows survive
    assert (deltas % unit == 0).all()                  # whole groups only


def test_pruning_everything_is_refused():
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    layout, patches = _layout_for(visual, grid)
    with pytest.raises(ValueError, match="at least one merger group"):
        prune_groups(torch.randn(patches, 6), torch.randn(patches, 4),
                     layout, torch.tensor([], dtype=torch.long))


def test_survivors_remain_valid_merger_inputs():
    """A surviving group is still a complete 2x2 block, so merger.mlp still applies."""
    model = _tiny_qwen()
    visual = model.model.visual
    grid = torch.tensor([[1, 8, 8]])
    layout, patches = _layout_for(visual, grid)
    hidden = torch.randn(patches, VISION["hidden_size"])
    keep = torch.tensor([0, 3, 6, 11])
    pruned = prune_groups(hidden, torch.randn(patches, 4), layout, keep)
    with torch.no_grad():
        merged = visual.merger(pruned.hidden_states)
    assert merged.shape == (keep.numel(), VISION["out_hidden_size"])
