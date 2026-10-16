"""The Qwen recipes run a 4096-token context; LLaVA's stays pinned at 2048.

Both trainers refuse a `model_max_length` other than the released 2048, which is
right for LLaVA -- that pin is what keeps the reproduction faithful. Upstream
publishes no Qwen recipe for either method, and 2048 cannot hold a 2048-token
image plus its text, so the Qwen paths take 4096 and stay configurable.
"""

import pytest

from llmoe.compressor.duet import duet_tr
from llmoe.compressor.hidrop import hidrop_tr
from llmoe.compressor.hidrop.data import DEFAULT_QWEN_IMAGE_TOKENS


def test_the_two_lengths_are_what_they_claim():
    assert duet_tr.RELEASED_MODEL_MAX_LENGTH == 2048
    assert hidrop_tr.RELEASED_MODEL_MAX_LENGTH == 2048
    assert duet_tr.QWEN_MODEL_MAX_LENGTH == 4096
    assert hidrop_tr.QWEN_MODEL_MAX_LENGTH == 4096


def test_the_qwen_context_can_actually_hold_the_image_budget():
    """The whole point of 4096: 2048 image tokens plus room for text.

    At 2048 the image budget silently fell back to the text headroom, so the run
    trained at a cap nobody chose.
    """
    reserve = 512  # QwenHiDropCollator.text_token_reserve
    assert DEFAULT_QWEN_IMAGE_TOKENS + reserve <= duet_tr.QWEN_MODEL_MAX_LENGTH
    assert DEFAULT_QWEN_IMAGE_TOKENS + reserve > duet_tr.RELEASED_MODEL_MAX_LENGTH


def test_the_full_image_budget_is_available_at_4096():
    """No clamp, and so no warning, once the context is long enough."""
    import warnings

    from llmoe.compressor.hidrop.data import QwenHiDropCollator

    processor = _processor()
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any clamp warning fails here
        QwenHiDropCollator(processor=processor,
                           model_max_length=duet_tr.QWEN_MODEL_MAX_LENGTH,
                           stage="pretrain")
    ip = processor.image_processor
    per_token = (int(ip.patch_size) * int(ip.merge_size)) ** 2
    assert int(ip.max_pixels) == DEFAULT_QWEN_IMAGE_TOKENS * per_token


def _processor():
    import os

    from transformers import AutoProcessor

    hub = os.environ.get(
        "LLMOE_TEST_HF_HUB",
        "/space/hall7/sitestore/eccc/mrd/rpnet/awa001/workspace/output/"
        "model_weights/hf_cache/hub")
    try:
        return AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                             local_files_only=True, cache_dir=hub)
    except Exception as exc:  # noqa: BLE001 - a cache miss is the only story
        pytest.skip(f"Qwen processor not cached: {type(exc).__name__}")
