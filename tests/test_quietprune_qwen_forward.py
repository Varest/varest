"""QuietPrune's schedule as a replacement for Qwen2.5-VL's vision forward.

The runner reimplements the tower's forward and inserts pruning between blocks, so
the first thing to establish is that the reimplementation is faithful: at ``R = 0``
it must reproduce the stock forward exactly. Everything else builds on that.
"""

import pytest
import torch

from llmoe.compressor.quietprune.quietprune_qwen_forward import (
    QuietPruneVisionRunner,
    groups_kept_after_schedule,
)
from llmoe.compressor.quietprune.quietprune_qwen_vit import quietprune_layers

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


def _inputs(grid_h=8, grid_w=8):
    grid = torch.tensor([[1, grid_h, grid_w]])
    return grid, torch.randn(grid_h * grid_w, 3 * 2 * 14 * 14)


# ------------------------------------------------------------------- schedule


def test_surviving_counts_are_known_before_the_image():
    """Placeholder trimming depends on the count being a pure function of G and R."""
    assert groups_kept_after_schedule(144, 0.25, 3) == [108, 81, 61]
    assert groups_kept_after_schedule(16, 0.0, 3) == [16, 16, 16]


def test_schedule_never_empties_the_sequence():
    assert groups_kept_after_schedule(2, 0.9, 5)[-1] >= 1


def test_schedule_rejects_a_rate_of_one():
    with pytest.raises(ValueError, match=r"rate must be in \[0, 1\)"):
        groups_kept_after_schedule(16, 1.0, 3)


# -------------------------------------------------------- faithful at R = 0


def test_zero_rate_reproduces_the_stock_vision_forward_exactly():
    """The runner reimplements the tower; at R=0 it must be indistinguishable."""
    model = _tiny_qwen()
    visual = model.model.visual
    grid, pixel_values = _inputs()

    with torch.no_grad():
        expected = visual(pixel_values, grid_thw=grid).pooler_output

    runner = QuietPruneVisionRunner(
        visual, pruning_layers=quietprune_layers(len(visual.blocks)), rate=0.0)
    q_cls = torch.randn(1, VISION["hidden_size"])
    with torch.no_grad():
        got = runner.forward(pixel_values, grid, q_cls)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)


def test_zero_rate_keeps_every_group_in_raster_order():
    model = _tiny_qwen()
    visual = model.model.visual
    grid, pixel_values = _inputs()
    runner = QuietPruneVisionRunner(
        visual, pruning_layers=quietprune_layers(len(visual.blocks)), rate=0.0)
    with torch.no_grad():
        runner.forward(pixel_values, grid, torch.randn(1, VISION["hidden_size"]))
    groups = 64 // visual.spatial_merge_unit
    assert torch.equal(runner.survivor_groups, torch.arange(groups))


# ------------------------------------------------------------- actual pruning


def test_pruning_shrinks_the_visual_tokens_by_the_scheduled_amount():
    model = _tiny_qwen()
    visual = model.model.visual
    grid, pixel_values = _inputs()
    groups = 64 // visual.spatial_merge_unit
    layers = quietprune_layers(len(visual.blocks))
    runner = QuietPruneVisionRunner(visual, pruning_layers=layers, rate=0.25,
                                    aggregate=False)
    with torch.no_grad():
        got = runner.forward(pixel_values, grid, torch.randn(1, VISION["hidden_size"]))
    expected = groups_kept_after_schedule(groups, 0.25, len(layers))[-1]
    assert got.shape[0] == expected
    assert runner.survivor_groups.numel() == expected


def test_survivors_stay_in_raster_order():
    """Order matters: the LLM scatters these into placeholders positionally."""
    model = _tiny_qwen()
    visual = model.model.visual
    grid, pixel_values = _inputs()
    runner = QuietPruneVisionRunner(
        visual, pruning_layers=quietprune_layers(len(visual.blocks)), rate=0.25,
        aggregate=False)
    with torch.no_grad():
        runner.forward(pixel_values, grid, torch.randn(1, VISION["hidden_size"]))
    survivors = runner.survivor_groups
    assert torch.equal(survivors, survivors.sort().values)
    assert survivors.unique().numel() == survivors.numel()


def test_aggregate_adds_exactly_one_token():
    model = _tiny_qwen()
    visual = model.model.visual
    grid, pixel_values = _inputs()
    layers = quietprune_layers(len(visual.blocks))
    groups = 64 // visual.spatial_merge_unit

    runner_off = QuietPruneVisionRunner(visual, pruning_layers=layers, rate=0.25,
                                        aggregate=False)
    runner_on = QuietPruneVisionRunner(visual, pruning_layers=layers, rate=0.25,
                                       aggregate=True)
    q_cls = torch.randn(1, VISION["hidden_size"])
    with torch.no_grad():
        without = runner_off.forward(pixel_values, grid, q_cls)
        with_agg = runner_on.forward(pixel_values, grid, q_cls)
    assert with_agg.shape[0] == without.shape[0] + 1
    # Sec. 4.3: "only one aggregated token is added", however many cuts ran.
    assert len(layers) > 1


def test_unnormalised_aggregate_is_the_papers_formula():
    """Normalising changes the aggregate; the default must be the paper's sum."""
    model = _tiny_qwen()
    visual = model.model.visual
    grid, pixel_values = _inputs()
    layers = quietprune_layers(len(visual.blocks))
    q_cls = torch.randn(1, VISION["hidden_size"])

    paper = QuietPruneVisionRunner(visual, pruning_layers=layers, rate=0.25,
                                   aggregate=True, normalize_aggregation=False)
    normed = QuietPruneVisionRunner(visual, pruning_layers=layers, rate=0.25,
                                    aggregate=True, normalize_aggregation=True)
    with torch.no_grad():
        a = paper.forward(pixel_values, grid, q_cls)
        b = normed.forward(pixel_values, grid, q_cls)
    assert a.shape == b.shape
    assert not torch.allclose(a[-1], b[-1], atol=1e-4)


def test_multiple_images_are_refused_rather_than_silently_wrong():
    model = _tiny_qwen()
    visual = model.model.visual
    runner = QuietPruneVisionRunner(
        visual, pruning_layers=quietprune_layers(len(visual.blocks)), rate=0.25)
    grid = torch.tensor([[1, 8, 8], [1, 4, 4]])
    with pytest.raises(NotImplementedError, match="one image per prompt"):
        runner.forward(torch.randn(80, 3 * 2 * 14 * 14), grid,
                       torch.randn(1, VISION["hidden_size"]))


def test_query_changes_which_groups_survive():
    """Query-guided means the query must actually matter."""
    model = _tiny_qwen()
    visual = model.model.visual
    grid, pixel_values = _inputs()
    layers = quietprune_layers(len(visual.blocks))
    torch.manual_seed(1)
    survivors = []
    for _ in range(2):
        runner = QuietPruneVisionRunner(visual, pruning_layers=layers, rate=0.5,
                                        aggregate=False)
        with torch.no_grad():
            runner.forward(pixel_values, grid,
                           torch.randn(1, VISION["hidden_size"]) * 5.0)
        survivors.append(runner.survivor_groups.tolist())
    assert survivors[0] != survivors[1]
