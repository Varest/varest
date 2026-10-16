"""Backbone-level contracts for Llama-3-LLaVA-NeXT-8B."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PreTrainedModel, PretrainedConfig, ProcessorMixin

from llmoe.backbone import BACKBONES
import llmoe.backbone.llava_next as llava_next_module
from llmoe.backbone.llava_next import (
    LLaVANext8B,
    LLaVANextImagePreprocessor,
    LLaVANextPromptPreprocessor,
)
from llmoe.resolvers import resolve_backbone


def test_llava_next_8b_is_registered_with_the_converted_checkpoint() -> None:
    backbone = resolve_backbone("llava-next-8b", {})

    assert BACKBONES["llava-next-8b"] is LLaVANext8B
    assert backbone._model_id == "llava-hf/llama3-llava-next-8b-hf"


def test_llava_next_8b_keeps_custom_checkpoint_override() -> None:
    backbone = resolve_backbone(
        "llava-next-8b",
        {"model_name_or_path": "/models/custom-llava-next-8b"},
    )

    assert backbone._model_id == "/models/custom-llava-next-8b"


def test_anyres_preprocessor_preserves_source_image_boundaries() -> None:
    images = torch.randn(2, 3, 8, 12)

    result = LLaVANextImagePreprocessor().forward(SimpleNamespace(), images)

    assert isinstance(result, list)
    assert len(result) == 2
    torch.testing.assert_close(result[0], images[0])
    torch.testing.assert_close(result[1], images[1])


def test_prompt_preprocessor_uses_the_checkpoint_chat_template() -> None:
    calls: list[tuple[list[dict], bool, bool]] = []

    class _Processor:
        def apply_chat_template(
            self,
            conversation: list[dict],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            calls.append((conversation, tokenize, add_generation_prompt))
            return f"formatted:{conversation[0]['content'][1]['text']}"

    vlm = SimpleNamespace(processor=_Processor())
    formatted = LLaVANextPromptPreprocessor().forward(vlm, ["first", "second"])

    assert formatted == ["formatted:first", "formatted:second"]
    assert all(not tokenize and generate for _, tokenize, generate in calls)
    assert all(call[0][0]["content"][0] == {"type": "image"} for call in calls)


def test_unsupported_compressors_fail_closed_without_an_anyres_adapter() -> None:
    backbone = LLaVANext8B()

    with pytest.raises(NotImplementedError, match="variable-length AnyRes"):
        backbone.validate_compressor(SimpleNamespace())


def test_loader_uses_native_next_class_and_completes_processor_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = object.__new__(ProcessorMixin)
    processor.patch_size = None
    processor.vision_feature_select_strategy = None
    processor.num_additional_image_tokens = None

    model = PreTrainedModel(PretrainedConfig())
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=14),
        vision_feature_select_strategy="default",
    )
    calls: dict[str, tuple[str, dict]] = {}

    def load_processor(model_id: str, **kwargs: object) -> ProcessorMixin:
        calls["processor"] = (model_id, kwargs)
        return processor

    class _NativeNextLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> PreTrainedModel:
            calls["model"] = (model_id, kwargs)
            return model

    monkeypatch.setattr(llava_next_module.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(
        "transformers.LlavaNextForConditionalGeneration",
        _NativeNextLoader,
    )

    backbone = LLaVANext8B(device=torch.device("cpu"))
    loaded_processor, loaded_model = backbone.load_model()

    assert loaded_processor is processor
    assert loaded_model is model
    assert calls["model"][1]["attn_implementation"] == "sdpa"
    assert processor.patch_size == 14
    assert processor.vision_feature_select_strategy == "default"
    assert processor.num_additional_image_tokens == 1
