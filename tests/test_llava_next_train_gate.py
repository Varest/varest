"""`run_training` validates the compressor by name, so the gate must accept one.

`run_training` and `run_finetuning` call `backbone.validate_compressor(compressor)`
with the CompressorLiteral name -- the trainer builds the instance, so none exists
yet at that point. A `str` never carries `validate_llava_next`, so the gate raised
for every method, adapted or not, and QuietPrune/LLaVA-NeXT training could not
start:

    NotImplementedError: <Method> has not been adapted for llava-next-8b

Found while training QuietPrune from its in-review NeXT branch, but the gate is
on main and affects every method trained on this backbone; "VisionZip++" was
special-cased here for the same reason.

The knob-level checks still run: the trainer calls `validate_compressor` again
with the built compressor.
"""
import pytest

from llmoe.backbone.llava_next import LLaVANext8B
from llmoe.compressor import COMPRESSORS, COMPRESSOR_TRAINERS, COMPRESSOR_TUNERS

TRAINABLE = set(COMPRESSOR_TRAINERS) | set(COMPRESSOR_TUNERS)
# Accepted by name: adapted for NeXT *and* something this repo can train or tune.
ADAPTED = sorted(n for n, c in COMPRESSORS.items()
                 if callable(getattr(c, "validate_llava_next", None)) and n in TRAINABLE)
# Rejected: not adapted at all, or adapted for evaluation but with no trainer.
UNADAPTED = sorted(n for n, c in COMPRESSORS.items()
                   if not (callable(getattr(c, "validate_llava_next", None)) and n in TRAINABLE))


def gate():
    return LLaVANext8B.validate_compressor.__wrapped__ if hasattr(
        LLaVANext8B.validate_compressor, "__wrapped__") else LLaVANext8B.validate_compressor


@pytest.mark.parametrize("name", ADAPTED)
def test_named_adapted_compressor_passes(name):
    """The training path must reach the trainer for an adapted method."""
    gate()(object.__new__(LLaVANext8B), name)


@pytest.mark.parametrize("name", UNADAPTED)
def test_named_unadapted_compressor_still_fails(name):
    """The gate must keep its point: fail before loading 8B for an unported method."""
    with pytest.raises(NotImplementedError, match="has not been adapted"):
        gate()(object.__new__(LLaVANext8B), name)


def test_unknown_name_fails():
    with pytest.raises(NotImplementedError, match="has not been adapted"):
        gate()(object.__new__(LLaVANext8B), "NotARealCompressor")




def test_visionzip_plus_plus_still_passes_without_its_special_case():
    """`validate_compressor` used to hardcode `if compressor == "VisionZip++"`.

    The general rule replaces it, so this pins that the tuning preflight it was
    added for keeps working -- VisionZipPlusPlusCompressor implements the adapter,
    so it is accepted by name like any other adapted method.
    """
    assert "VisionZip++" in ADAPTED
    gate()(object.__new__(LLaVANext8B), "VisionZip++")


def test_inference_only_method_is_still_rejected_by_name():
    """`validate_llava_next` means the method runs on NeXT, not that it trains.

    Nuwa is adapted for evaluation and has no trainer, so naming it on a training
    path is still an error -- the behaviour test_vision_zip_llava_next pins.
    """
    assert "Nuwa" not in TRAINABLE
    with pytest.raises(NotImplementedError, match="has not been adapted"):
        gate()(object.__new__(LLaVANext8B), "Nuwa")
