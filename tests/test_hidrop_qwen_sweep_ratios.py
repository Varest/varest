"""HiDrop's Qwen budgets, published and extrapolated.

The sweep scores every method at the same four budgets -- LLaVA's 224/128/64/32
tokens as fractions of 576. HiDrop publishes four *different* Qwen ratios, all
slightly looser, so running it on those would hand it more token-layers than the
methods it is compared against. These cover the extrapolation that closes that
gap, and the property the whole comparison rests on: the number is a layer
average, not a peak or a final count.
"""

import pytest

from llmoe.compressor.hidrop.config import (
    QWEN_PROFILES,
    QWEN_PUBLISHED_RATIOS,
    QWEN_SWEEP_RATIOS,
    RELEASED_FOUR_STAGE_CURVE,
    RELEASED_FIVE_STAGE_CURVE,
    derive_keep_ratios,
    visual_token_counts_by_layer,
)
from llmoe.compressor.hidrop.hidrop import BACKENDS

DEPTH = 28          # Qwen2.5-VL-7B's decoder
SWEEP = (0.3889, 0.2222, 0.1111, 0.0556)
UNITS = 100_000     # a large token count, so rounding does not blur the average


def _config(ratio: float):
    return BACKENDS["qwen"].build_config({"retain_ratio": ratio},
                                         batch_size_supported=True,
                                         num_decoder_layers=DEPTH)


@pytest.mark.parametrize("ratio", SWEEP)
def test_sweep_budget_realises_its_layer_average_exactly(ratio):
    """The contract: `retain_ratio` is the average over decoder layers.

    Not the peak (which is always 1.0, before pruning starts) and not the final
    count (0.0, after early exit) -- either reading would make the budgets
    incomparable across methods.
    """
    config = _config(ratio)
    counts = visual_token_counts_by_layer(
        config.late_entry_layer, config.early_exit_layer, config.layer_list,
        tuple(int(k * UNITS) for k in config.image_token_keep_ratio_list),
        visual_tokens=UNITS, num_decoder_layers=DEPTH)

    assert sum(counts) / DEPTH / UNITS == pytest.approx(ratio, abs=1e-4)
    assert max(counts) == UNITS            # full span before the first filter
    assert min(counts) == 0                # early exit drops the remainder


@pytest.mark.parametrize("ratio", SWEEP)
def test_keeps_are_usable_not_collapsed(ratio):
    """A schedule sitting on its curve's floor has a stage keeping ~nothing."""
    config = _config(ratio)
    keeps = config.image_token_keep_ratio_list
    assert all(0.0 < k <= 1.0 for k in keeps), keeps
    assert list(keeps) == sorted(keeps, reverse=True), "keeps must not increase"
    assert min(keeps) > 1e-3, f"a stage keeps essentially nothing: {keeps}"


def test_published_ratios_are_untouched():
    """The extrapolation must not perturb what upstream actually publishes."""
    for ratio, curve in ((0.375, RELEASED_FOUR_STAGE_CURVE),
                         (0.25, RELEASED_FOUR_STAGE_CURVE),
                         (0.125, RELEASED_FOUR_STAGE_CURVE),
                         (0.0625, RELEASED_FIVE_STAGE_CURVE)):
        assert QWEN_PROFILES[ratio] is curve
    assert QWEN_PUBLISHED_RATIOS == {0.375, 0.25, 0.125, 0.0625}
    assert not (QWEN_PUBLISHED_RATIOS & set(QWEN_SWEEP_RATIOS)), (
        "a sweep ratio must never shadow a published one")


def test_each_sweep_ratio_needs_the_curve_it_was_given():
    """Pin why the two lowest budgets do not reuse their neighbour's curve.

    The four-stage curve bottoms out at 0.113 and the five-stage at 0.0584, so
    0.1111 and 0.0556 respectively fall through them. If a future change makes a
    simpler curve reach that far, this test should be revisited, not deleted.
    """
    with pytest.raises(ValueError, match="below what this curve can reach"):
        derive_keep_ratios(RELEASED_FOUR_STAGE_CURVE, DEPTH, 0.1111)
    with pytest.raises(ValueError, match="below what this curve can reach"):
        derive_keep_ratios(RELEASED_FIVE_STAGE_CURVE, DEPTH, 0.0556)


def test_an_unsupported_ratio_still_lists_what_is_available():
    with pytest.raises(ValueError, match="Unsupported HiDrop retain_ratio"):
        _config(0.31415)
