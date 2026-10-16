"""LearnPruner contracts shared by LLaVA-1.5 7B/13B and LLaVA-NeXT."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from llmoe.backbone.llava_next import LLaVANext8B
from llmoe.compressor.hidrop.hidrop_llama import IGNORE_INDEX
from llmoe.compressor.learnpruner.learn_pruner import (
    LearnPrunerCompressor,
    pruning_boundary,
    ratio_stage_counts,
    stage_schedule,
)
from llmoe.compressor.learnpruner.learn_pruner_tr import (
    LlavaNextLearnPrunerCollator,
    _checkpoint_name,
)


def test_fixed_budgets_scale_the_boundary_but_not_the_public_budget() -> None:
    seven = stage_schedule(64, 32)
    thirteen = stage_schedule(64, 40)

    assert seven.boundaries == (12,)
    assert thirteen.boundaries == (15,)
    assert seven.n_visual == thirteen.n_visual == 111
    assert seven.keeps == thirteen.keeps == (37,)
    assert seven.layer_average == thirteen.layer_average == 64.75
    assert pruning_boundary(32) == 12
    assert pruning_boundary(40) == 15
    assert LearnPrunerCompressor(max_tokens=64).budget == 64


def test_next_ratio_resolves_against_each_complete_packed_span() -> None:
    stage1, stage2, achieved = ratio_stage_counts(2880, 1 / 9, 32, 12)

    assert (stage1, stage2) == (549, 183)
    assert achieved == pytest.approx(320.25 / 2880)
    assert ratio_stage_counts(777, 1.0, 32, 12) == (777, 777, 1.0)
    assert LearnPrunerCompressor(retain_ratio=1 / 9).budget == pytest.approx(1 / 9)


def test_budget_controls_are_backbone_specific() -> None:
    next_backbone = LLaVANext8B()

    next_backbone.validate_compressor("LearnPruner")
    next_backbone.validate_compressor(LearnPrunerCompressor(retain_ratio=1 / 9, for_training=True))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        next_backbone.validate_compressor(LearnPrunerCompressor(for_training=True))
    with pytest.raises(ValueError, match="only accepts retain_ratio"):
        next_backbone.validate_compressor(
            LearnPrunerCompressor(
                max_tokens=32,
                retain_ratio=1 / 9,
                for_training=True,
            )
        )


def test_default_checkpoint_names_are_backbone_specific() -> None:
    assert _checkpoint_name(SimpleNamespace(weights_key="llava"), {}) == "learnpruner_lpm.pt"
    assert (
        _checkpoint_name(SimpleNamespace(weights_key="llava-13b"), {})
        == "learnpruner_lpm_llava-1.5-13b.pt"
    )
    assert (
        _checkpoint_name(SimpleNamespace(weights_key="llava-next-8b"), {})
        == "learnpruner_lpm_llama3-llava-next-8b.pt"
    )


class _FakeAttention(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()

    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class _FakeLanguage(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_FakeLayer() for _ in range(depth))


class _FakeNextInner(nn.Module):
    def __init__(self, packed: torch.Tensor, calls: list[str]) -> None:
        super().__init__()
        self.vision_tower = nn.Identity()
        self.multi_modal_projector = nn.Linear(6, 6)
        self.language_model = _FakeLanguage(32)
        self._packed = packed
        self._calls = calls

        def native_get_image_features(*_: Any, **__: Any) -> BaseModelOutputWithPooling:
            self._calls.append("native-packed")
            return BaseModelOutputWithPooling(pooler_output=[self._packed])

        self.get_image_features = native_get_image_features


class _FakeOuter(nn.Module):
    def __init__(self, inner: nn.Module, config: Any) -> None:
        super().__init__()
        self.model = inner
        self.config = config


def test_13b_install_uses_the_fixed_profile_at_layer_15() -> None:
    inner = _FakeNextInner(torch.randn(1, 6), [])
    inner.multi_modal_projector = nn.Linear(1024, 6)
    inner.language_model = _FakeLanguage(40)
    config = SimpleNamespace(
        model_type="llava",
        image_seq_length=576,
        image_token_index=99,
        vision_config=SimpleNamespace(hidden_size=1024),
    )
    vlm = SimpleNamespace(
        model=_FakeOuter(inner, config),
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        prepare_inputs=lambda *_args, **_kwargs: {},
        weights_key="llava-13b",
    )
    compressor = LearnPrunerCompressor(max_tokens=64, for_training=True, hidden_dim=4)

    compressor.install(vlm)
    try:
        assert compressor.budget == 64
        assert compressor.retain_ratio is None
        assert compressor.feature_space == "vision_encoder"
        assert compressor.feature_dim == 1024
        assert compressor.resolved_pruning_layer == 15
        assert compressor._stage1_count == 111
        assert compressor._stage2_count == 37
    finally:
        compressor.uninstall()

    with pytest.raises(ValueError, match="constant max_tokens"):
        LearnPrunerCompressor(
            retain_ratio=0.25,
            for_training=True,
            hidden_dim=4,
        ).install(vlm)


def test_next_training_scores_only_the_native_post_join_sequence() -> None:
    packed = torch.randn(17, 6)
    calls: list[str] = []
    config = SimpleNamespace(
        model_type="llava_next",
        image_seq_length=12,
        image_token_index=99,
        text_config=SimpleNamespace(hidden_size=6),
        vision_config=SimpleNamespace(hidden_size=6),
    )
    inner = _FakeNextInner(packed, calls)
    outer = _FakeOuter(inner, config)
    prepared = {"input_ids": torch.tensor([[1, 99, 2]])}
    vlm = SimpleNamespace(
        model=outer,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        prepare_inputs=lambda *_args, **_kwargs: prepared,
        compressor_family="llava_next",
        weights_key="llava-next-8b",
    )
    compressor = LearnPrunerCompressor(
        retain_ratio=0.25,
        for_training=True,
        hidden_dim=4,
    )
    original_get = inner.get_image_features

    compressor.install(vlm)
    try:
        outputs = inner.get_image_features(torch.empty(0))

        assert calls == ["native-packed"]
        assert outputs.pooler_output[0] is packed
        assert compressor.feature_space == "packed_projected_anyres"
        assert compressor.feature_dim == 6
        assert len(compressor._training_masks) == 1
        assert compressor._training_masks[0].shape == (17,)
    finally:
        compressor.uninstall()

    assert inner.get_image_features is original_get


def test_next_inference_trims_for_the_globally_selected_packed_rows(
    tmp_path: Path,
) -> None:
    packed = torch.arange(72, dtype=torch.float32).reshape(12, 6)
    calls: list[str] = []
    config = SimpleNamespace(
        model_type="llava_next",
        image_seq_length=12,
        image_token_index=99,
        text_config=SimpleNamespace(hidden_size=6),
        vision_config=SimpleNamespace(hidden_size=6),
    )
    inner = _FakeNextInner(packed, calls)
    outer = _FakeOuter(inner, config)
    native_ids = torch.tensor([[7, *([99] * 12), 8, 9]])
    prepared = {
        "input_ids": native_ids,
        "attention_mask": torch.ones_like(native_ids),
    }
    vlm = SimpleNamespace(
        model=outer,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        prepare_inputs=lambda *_args, **_kwargs: {
            name: value.clone() for name, value in prepared.items()
        },
        compressor_family="llava_next",
        weights_key="llava-next-8b",
    )

    seed = LearnPrunerCompressor(
        retain_ratio=0.25,
        feature_dim=6,
        hidden_dim=4,
        for_training=True,
    )
    seed._feature_space = "packed_projected_anyres"
    seed._installed_backbone_key = "llava-next-8b"
    checkpoint = tmp_path / "next_lpm.pt"
    seed.save_checkpoint(checkpoint)
    compressor = LearnPrunerCompressor(
        retain_ratio=0.25,
        hidden_dim=4,
        checkpoint_path=checkpoint,
    )

    compressor.install(vlm)
    try:
        trimmed = vlm.prepare_inputs(None, None)
        outputs = inner.get_image_features(torch.empty(0))

        assert int((trimmed["input_ids"] == 99).sum()) == 6
        assert calls == ["native-packed"]
        assert outputs.pooler_output[0].shape == (6, 6)
        assert compressor._stage1_count == 6
        assert compressor._stage2_count == 2
        assert compressor._original_count == 12
    finally:
        compressor.uninstall()


class _ChatTokenizer:
    ids = {
        "<|start_header_id|>": 1,
        "<|end_header_id|>": 2,
        "<|eot_id|>": 3,
        "assistant": 4,
        "<image>": 99,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [self.ids[text]]


class _NextProcessor:
    image_token = "<image>"

    def __init__(self) -> None:
        self.tokenizer = _ChatTokenizer()
        self.messages: list[dict[str, Any]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        self.messages = messages
        return "rendered"

    def __call__(self, **_: Any) -> dict[str, torch.Tensor]:
        # user header/content, then assistant header/content
        ids = torch.tensor([[1, 8, 2, 99, 10, 3, 1, 4, 2, 20, 21, 3]])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "image_sizes": torch.tensor([[4, 4]]),
        }


def test_next_collator_uses_native_chat_template_and_assistant_labels() -> None:
    processor = _NextProcessor()
    item = {
        "image": torch.zeros(3, 4, 4),
        "has_image": True,
        "conversations": [
            {"role": "user", "value": "<image> question"},
            {"role": "assistant", "value": "answer"},
        ],
    }

    batch = LlavaNextLearnPrunerCollator(processor)([item])

    assert batch["labels"].tolist()[0] == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        20,
        21,
        3,
    ]
    assert processor.messages is not None
    assert processor.messages[0]["content"][0] == {"type": "image"}
