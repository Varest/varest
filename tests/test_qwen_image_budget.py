"""The Qwen collator's image-token budget.

Qwen sizes each image dynamically, so the 558K set contains images worth several
thousand visual tokens. One of those overran ``model_max_length=2048`` 43% into a
DUET run: truncation cut into the image span and the collator (correctly) refused
to continue. The budget bounds what a single image may cost so that cannot happen.
"""

import os

import pytest
import torch
from PIL import Image

from llmoe.compressor.hidrop.data import QwenHiDropCollator

REAL_QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
#: These need the *real* Qwen image processor -- its smart-resize behaviour is the
#: thing under test, so a stub would prove nothing. The cache is addressed with an
#: explicit ``cache_dir`` rather than ``HF_HOME``: the env var is read when
#: transformers is first imported, which another test module may already have
#: done, so setting it here would come too late and the tests would silently skip.
HUB = os.environ.get(
    "LLMOE_TEST_HF_HUB",
    "/space/hall7/sitestore/eccc/mrd/rpnet/awa001/workspace/output/model_weights/"
    "hf_cache/hub")


def _processor():
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(REAL_QWEN, local_files_only=True,
                                             cache_dir=HUB)
    except Exception as exc:  # noqa: BLE001 - any cache miss is the same story
        pytest.skip(f"{REAL_QWEN} is not in the local HF cache: {type(exc).__name__}")


def _tokens_for(processor, image):
    out = processor.image_processor(images=image, return_tensors="pt")
    grid = out["image_grid_thw"][0].tolist()
    merge = int(processor.image_processor.merge_size)
    return int(grid[0] * grid[1] * grid[2]) // (merge ** 2)


def test_an_uncapped_large_image_really_does_blow_the_context():
    """Establish the failure is real before checking the fix."""
    processor = _processor()
    big = Image.new("RGB", (2048, 2048), (128, 128, 128))
    assert _tokens_for(processor, big) > 2048


def test_default_cap_is_2048_when_the_context_can_hold_it():
    from llmoe.compressor.hidrop.data import DEFAULT_QWEN_IMAGE_TOKENS

    assert DEFAULT_QWEN_IMAGE_TOKENS == 2048
    processor = _processor()
    # 2048 + the 512 text reserve, so the full budget is actually available.
    QwenHiDropCollator(processor=processor, model_max_length=4096, stage="pretrain")
    ip = processor.image_processor
    per_token = (int(ip.patch_size) * int(ip.merge_size)) ** 2
    assert int(ip.max_pixels) == 2048 * per_token
    for side in (512, 1024, 2048, 4096):
        image = Image.new("RGB", (side, side), (100, 110, 120))
        assert _tokens_for(processor, image) <= 2048, side


def test_a_context_too_short_for_the_default_clamps_and_says_so():
    """At the released 2048-token context, 2048 image tokens do not fit.

    The clamp is correct -- an image may not fill the whole sequence -- but it
    must be audible, or a run silently trains at a budget nobody chose.
    """
    processor = _processor()
    with pytest.warns(UserWarning, match="clamped to"):
        collator = QwenHiDropCollator(processor=processor, model_max_length=2048,
                                      stage="pretrain")
    headroom = 2048 - collator.text_token_reserve
    for side in (1024, 4096):
        image = Image.new("RGB", (side, side), (100, 110, 120))
        assert _tokens_for(processor, image) <= headroom, side


def test_a_budget_that_would_not_fit_is_refused():
    processor = _processor()
    with pytest.raises(ValueError, match="leaves no room for text"):
        QwenHiDropCollator(processor=processor, model_max_length=2048,
                           stage="pretrain", image_max_tokens=4096)


def test_explicit_cap_is_honoured():
    processor = _processor()
    QwenHiDropCollator(processor=processor, model_max_length=2048, stage="pretrain",
                       image_max_tokens=256)
    image = Image.new("RGB", (4096, 4096), (10, 20, 30))
    assert _tokens_for(processor, image) <= 256


def test_cap_does_not_shrink_images_that_already_fit():
    """Only the outliers should move; ordinary rows must be untouched."""
    plain = _processor()
    small = Image.new("RGB", (336, 336), (5, 5, 5))
    before = _tokens_for(plain, small)

    capped = _processor()
    QwenHiDropCollator(processor=capped, model_max_length=2048, stage="pretrain")
    assert _tokens_for(capped, small) == before


def test_a_nonsense_budget_is_refused():
    processor = _processor()
    with pytest.raises(ValueError, match="image_max_tokens must be >= 1"):
        QwenHiDropCollator(processor=processor, model_max_length=2048,
                           stage="pretrain", image_max_tokens=0)


def test_budget_is_expressed_in_tokens_like_upstream():
    """Upstream's --max_pixels is tokens x (patch*merge)^2; match that conversion."""
    processor = _processor()
    QwenHiDropCollator(processor=processor, model_max_length=2048, stage="pretrain",
                       image_max_tokens=100)
    ip = processor.image_processor
    per_token = (int(ip.patch_size) * int(ip.merge_size)) ** 2
    assert per_token == 28 * 28
    assert int(ip.max_pixels) == 100 * per_token
