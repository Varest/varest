"""A checkpoint's recorded layer average must describe its own decoder.

`duet_metadata` computed the average against a hardcoded 32 layers while writing
the boundaries resolved for the actual decoder. On LLaVA-1.5-13B (40 layers) that
recorded 39.6875 where the true average was 31.75: the weights were correct, the
number stored beside them was not.
"""
import pytest

from llmoe.compressor.duet.duet import DUETCompressor
from llmoe.compressor.duet.duet_tr import duet_metadata


def metadata_at(budget: int, depth: int | None) -> dict:
    compressor = DUETCompressor(max_tokens=budget)
    compressor._pdrop_layers, compressor._pdrop_keep_ratios = compressor._resolve_pdrop(
        "llava", depth
    )
    return duet_metadata(compressor, backbone="llava")


@pytest.mark.parametrize("budget", [32, 64, 128, 224])
@pytest.mark.parametrize("depth", [32, 40])
def test_average_matches_the_budget_at_every_depth(budget, depth):
    """The point of rescaling: the same budget on a deeper decoder."""
    meta = metadata_at(budget, depth)
    assert meta["decoder_layers"] == depth
    assert abs(meta["average_image_tokens"] - budget) <= 1.0


@pytest.mark.parametrize("budget", [32, 64, 128, 224])
def test_average_is_invariant_under_rescaling(budget):
    """Boundaries move with depth, the layer average does not."""
    assert (metadata_at(budget, 32)["average_image_tokens"]
            == metadata_at(budget, 40)["average_image_tokens"])


def test_the_13b_number_that_was_wrong():
    meta = metadata_at(32, 40)
    assert meta["pdrop_layers"] == [20, 30]      # rescaled from the published (16, 24)
    assert meta["average_image_tokens"] == pytest.approx(31.75)
    assert meta["average_image_tokens"] != pytest.approx(39.6875)


def test_the_misleading_key_is_gone():
    assert "average_image_tokens_32_layers" not in metadata_at(32, 40)


def test_unknown_depth_falls_back_to_the_published_reference():
    """`_resolve_pdrop(depth=None)` leaves the published boundaries alone, so the
    average must be reported against the depth they are stated at."""
    meta = metadata_at(32, None)
    assert meta["decoder_layers"] == 32
    assert meta["pdrop_layers"] == [16, 24]
    assert meta["average_image_tokens"] == pytest.approx(31.75)
