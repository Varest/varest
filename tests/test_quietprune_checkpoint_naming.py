"""QuietPrune's trained adapter is named for the backbone it inverts.

The adapter inverts *this* backbone's projector, so its shape is the decoder's:
4096 -> 1024 on LLaVA-1.5-7B, 5120 -> 1024 on the 13B, a different projector
again on Qwen. The trainer wrote a fixed `quietprune.pt`, so a 13B run would have
overwritten the 7B adapter -- a finished result, already published to the bucket
-- with weights that cannot load anywhere else. #22 fixed the same hazard for
VisionZip; this is the QuietPrune half.
"""

import pytest

from llmoe.weights import BACKBONE_SLUGS


def _name(key: str | None, *, finetune: bool = False) -> str:
    """The naming rule as `train_quietprune` applies it."""
    infix = "" if key in (None, "", "llava") else f"_{BACKBONE_SLUGS.get(key, key)}"
    stem = f"quietprune_finetuned{infix}" if finetune else f"quietprune{infix}"
    return f"{stem}.pt"


def test_the_released_backbone_keeps_its_original_filename():
    """Existing 7B checkpoints and configs must still resolve."""
    assert _name("llava") == "quietprune.pt"
    assert _name("llava", finetune=True) == "quietprune_finetuned.pt"


@pytest.mark.parametrize("key", ["llava-13b", "qwen-vl", "internvl", "llava-next-8b"])
def test_every_other_backbone_gets_its_own_filename(key):
    assert _name(key) == f"quietprune_{BACKBONE_SLUGS[key]}.pt"
    assert _name(key) != "quietprune.pt", "would overwrite the 7B adapter"


def test_no_two_backbones_share_a_filename():
    """The point of the change: distinct adapters cannot collide on disk."""
    names = [_name(key) for key in BACKBONE_SLUGS]
    assert len(names) == len(set(names)), names


def test_the_trainer_actually_applies_the_rule():
    """Pin it in the source, so the rule cannot be quietly reverted."""
    from pathlib import Path

    import llmoe.compressor.quietprune.quietprune_tr as module

    source = Path(module.__file__).read_text()
    assert "backbone_key(backbone)" in source
    assert 'options.get("checkpoint_name"' in source
    assert '"quietprune_finetuned.pt" if finetune else "quietprune.pt"' not in source, (
        "the unconditional filename is back")
