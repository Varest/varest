from types import SimpleNamespace

import pytest

import llmoe.compressor.hidrop.hidrop as hidrop_module
from llmoe.compressor.hidrop import (
    QwenHiDropCollator as ExportedQwenHiDropCollator,
    QwenHiDropConfig as ExportedQwenHiDropConfig,
)
from llmoe.compressor.hidrop.data import QwenHiDropCollator
from llmoe.compressor.hidrop.config import CONFIG_METADATA_KEYS, QwenHiDropConfig
from llmoe.compressor.hidrop.hidrop_llama import IGNORE_INDEX


class _Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return 99 if token == "<|image_pad|>" else -1

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [1000 + ord(character) for character in text]}


def _collator(*, stage: str = "finetune") -> QwenHiDropCollator:
    processor = SimpleNamespace(
        image_token="<|image_pad|>",
        tokenizer=_Tokenizer(),
        image_processor=SimpleNamespace(merge_size=2, patch_size=14),
    )
    return QwenHiDropCollator(processor=processor, stage=stage)


@pytest.mark.parametrize("retain_ratio", [0.375, 0.25, 0.125, 0.0625])
def test_qwen_schedule_matches_requested_layer_average(retain_ratio: float) -> None:
    config = QwenHiDropConfig.for_backbone(retain_ratio, num_decoder_layers=28)

    assert config.computed_retain_ratio() == pytest.approx(retain_ratio, abs=1e-12)
    # The lowest profile deliberately prunes at entry; the others start one layer later.
    assert config.late_entry_layer <= config.layer_list[0]
    assert config.layer_list[-1] < config.early_exit_layer
    assert all(
        current >= following
        for current, following in zip(
            config.image_token_keep_ratio_list,
            config.image_token_keep_ratio_list[1:],
            strict=False,
        )
    )


def test_qwen_stage_budgets_use_the_original_visual_span_and_clamp() -> None:
    config = QwenHiDropConfig.for_backbone(0.25, num_decoder_layers=28)
    original = 101
    current = original
    keeps = []

    for layer, ratio in zip(
        config.layer_list, config.image_token_keep_ratio_list, strict=True
    ):
        expected = max(1, min(round(ratio * original), current))
        current = config.keep_k_for_layer(layer, current, original)
        keeps.append(current)
        assert current == expected

    assert keeps == sorted(keeps, reverse=True)


def test_qwen_config_metadata_round_trips_without_claiming_a_released_recipe() -> None:
    original = QwenHiDropConfig.for_backbone(0.25, num_decoder_layers=28)
    serialized = original.to_dict()

    assert serialized["released_recipe"] is False
    assert "trained" not in serialized
    for metadata_key in CONFIG_METADATA_KEYS:
        serialized.pop(metadata_key, None)

    restored = QwenHiDropConfig(**serialized)
    assert restored == original


def test_qwen_collator_splices_one_masked_contiguous_image_span() -> None:
    collator = _collator()
    input_ids, labels = collator._segments_to_tokens(
        [("prefix<image>suffix", False), ("answer", True)], image_tokens=3
    )

    image_positions = [index for index, token_id in enumerate(input_ids) if token_id == 99]
    assert image_positions == list(range(image_positions[0], image_positions[0] + 3))
    assert [labels[index] for index in image_positions] == [IGNORE_INDEX] * 3
    assert labels[-len("answer") :] == input_ids[-len("answer") :]
    assert all(label == IGNORE_INDEX for label in labels[: -len("answer")])


def test_qwen_public_exports_come_from_shared_hidrop_modules() -> None:
    assert ExportedQwenHiDropConfig is QwenHiDropConfig
    assert ExportedQwenHiDropCollator is QwenHiDropCollator
    assert QwenHiDropConfig.__module__ == "llmoe.compressor.hidrop.config"
    assert QwenHiDropCollator.__module__ == "llmoe.compressor.hidrop.data"


def test_llava_hidrop_uses_the_shared_weight_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_resolve_weights(method, budget, *, checkpoint_path=None, pretrained=None):
        calls.append((method, budget, checkpoint_path, pretrained))
        return None

    monkeypatch.setattr(hidrop_module, "resolve_weights", fake_resolve_weights)

    compressor = hidrop_module.HiDropCompressor(
        average_retained_tokens=64,
        checkpoint_path="custom/checkpoint",
        pretrained=True,
    )

    assert calls == [("HiDrop", 64, "custom/checkpoint", True)]
    assert compressor._checkpoint_dir is None


def test_qwen_hidrop_rejects_llava_released_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_resolve_weights(*args, **kwargs):
        raise AssertionError("Qwen must not resolve a LLaVA released checkpoint")

    monkeypatch.setattr(hidrop_module, "resolve_weights", unexpected_resolve_weights)

    compressor = hidrop_module.HiDropCompressor(retain_ratio=0.25, pretrained=False)
    assert compressor._checkpoint_dir is None

    with pytest.raises(ValueError, match="bundled `pretrained` weights are LLaVA-only"):
        hidrop_module.HiDropCompressor(retain_ratio=0.25, pretrained=True)
