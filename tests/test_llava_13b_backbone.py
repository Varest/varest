"""Registration tests for the named LLaVA-1.5-13B backbone."""

from llmoe.backbone import BACKBONES, LLaVA, LLaVA13B
from llmoe.resolvers import resolve_backbone


def test_llava_13b_is_a_named_llava_backbone() -> None:
    backbone = resolve_backbone("llava-13b", {})

    assert BACKBONES["llava-13b"] is LLaVA13B
    assert isinstance(backbone, LLaVA)
    assert backbone._model_id == "llava-hf/llava-1.5-13b-hf"


def test_llava_13b_keeps_custom_checkpoint_override() -> None:
    backbone = resolve_backbone(
        "llava-13b", {"model_name_or_path": "/models/custom-llava-13b"}
    )

    assert backbone._model_id == "/models/custom-llava-13b"


def test_llava_13b_reuses_the_llava_model_integration() -> None:
    """All structural LLaVA compressor routes see the same loaded model shape."""

    assert issubclass(LLaVA13B, LLaVA)
    assert LLaVA13B.load_model is LLaVA.load_model
