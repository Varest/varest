import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from llmoe.backbone import BACKBONES
from llmoe.backbone.llava_next import (
    LLAVA_NEXT_8B_MODEL_ID,
    LLAVA_NEXT_8B_REVISION,
    LLaVANext8B,
)
from llmoe.compressor.duet.duet import (
    DUETCompressor,
    DUET_LLAVA_NEXT_PROFILES,
    DUET_LLAVA_NEXT_PROFILE_RATIOS,
)
from llmoe.compressor.duet.duet_llava_next import (
    DUET_NEXT_IMPLEMENTATION_VERSION,
    LlavaNextDuetAdapter,
    next_budget_for_geometry,
    next_profile_per_view,
    round_half_up_ratio,
    retention_target,
    validate_duet_next_checkpoint,
)
import llmoe.compressor.duet.duet_next_tr as duet_next_tr_module
from llmoe.compressor.duet.duet_next_tr import (
    DUET_NEXT_FLASH_ATTN_VERSION,
    LlavaNextDuetCollator,
    LlavaNextParquetDataset,
    ParquetRowGroupSampler,
    build_next_training_args,
    duet_next_metadata,
    load_hf_llava_next_training_base,
    require_flash_attention_2_training_runtime,
)
from llmoe.compressor.duet.duet_tr import DuetTrainingLlavaForConditionalGeneration
from llmoe.compressor.llava_next import (
    geometry_for_image,
    placeholder_run_lengths,
    stitch_and_unpad_local_crops,
    trim_image_placeholders_to_lengths,
)


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="llava_next",
        image_token_index=99,
        image_grid_pinpoints=[[4, 8], [8, 4], [8, 8]],
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vision_config=SimpleNamespace(image_size=4, patch_size=2),
        text_config=SimpleNamespace(model_type="llama", num_hidden_layers=32),
    )


def _tiny_inner() -> SimpleNamespace:
    return SimpleNamespace(config=_tiny_config())


def test_backbone_is_explicitly_registered_and_fail_closed() -> None:
    assert BACKBONES["llava-next-8b"] is LLaVANext8B
    backbone = LLaVANext8B(device="cpu")
    backbone.validate_compressor("DUET")
    with pytest.raises(NotImplementedError, match="has not been adapted"):
        backbone.validate_compressor("ToMe")
    with pytest.raises(NotImplementedError, match="has not been adapted"):
        backbone.validate_compressor("GlimpsePrune")
    with pytest.raises(NotImplementedError, match="beam search"):
        LlavaNextDuetAdapter.validate_generation_kwargs({"num_beams": 2})
    with pytest.raises(NotImplementedError, match="one generated sequence"):
        LlavaNextDuetAdapter.validate_generation_kwargs({"num_return_sequences": 2})
    with pytest.raises(NotImplementedError, match="dynamic KV cache"):
        LlavaNextDuetAdapter.validate_generation_kwargs({"use_cache": False})
    with pytest.raises(NotImplementedError, match="cache_implementation='dynamic'"):
        LlavaNextDuetAdapter.validate_generation_kwargs({"cache_implementation": "static"})
    with pytest.raises(NotImplementedError, match="externally primed cache"):
        LlavaNextDuetAdapter.validate_generation_kwargs({"past_key_values": object()})


@pytest.mark.parametrize(
    ("profile_name", "profile", "ratio", "target", "average"),
    [
        (640, (850, 175), 1025 / 2880, 1025, 640.5),
        (320, (360, 150), 510 / 2880, 510, 318.75),
        (160, (225, 30), 255 / 2880, 255, 159.25),
    ],
)
def test_paper_profiles_define_ratio_and_mixture(
    profile_name: int,
    profile: tuple[int, int],
    ratio: float,
    target: int,
    average: float,
) -> None:
    assert DUET_LLAVA_NEXT_PROFILES[profile_name] == profile
    assert DUET_LLAVA_NEXT_PROFILE_RATIOS[profile_name] == ratio
    assert next_profile_per_view(*profile) == (
        profile[0] // 5, profile[1] // 5
    )
    compressor = DUETCompressor(
        retain_ratio=ratio,
        llava_next_profile=profile_name,
        pretrained=False,
    )
    compressor.validate_llava_next()
    compressor._llava_next_decoder_layers = 32
    geometry = SimpleNamespace(
        crop_count=5,
        patches_per_view=576,
        local_patch_count=2304,
        newline_count=48,
    )
    realized = next_budget_for_geometry(compressor, geometry)
    assert realized.original_content_tokens == 2880
    assert realized.requested_retention_ratio == ratio
    assert realized.target_content_tokens == target
    assert realized.realized_content_tokens == target
    assert realized.realized_retention_ratio == target / 2880
    assert realized.structural_tokens == 48
    assert realized.decoder_stage1_tokens == target + 48
    assert realized.stage_content_keep_counts == (target // 2, 0)
    assert realized.stage_total_keep_counts == (target // 2 + 48, 48)
    assert realized.content_layer_average == average
    assert realized.geometry_inclusive_layer_average == average + 48


def test_ratio_uses_each_samples_true_unpadded_content_count() -> None:
    compressor = DUETCompressor(
        retain_ratio=0.25, llava_next_profile=320, pretrained=False
    )
    compressor.validate_llava_next()
    compressor._llava_next_decoder_layers = 32
    first = next_budget_for_geometry(
        compressor,
        SimpleNamespace(
            crop_count=2,
            patches_per_view=576,
            local_patch_count=576,
            newline_count=24,
        ),
    )
    second = next_budget_for_geometry(
        compressor,
        SimpleNamespace(
            crop_count=5,
            patches_per_view=576,
            local_patch_count=1728,
            newline_count=36,
        ),
    )
    assert first.original_content_tokens == 1152
    assert first.target_content_tokens == 288
    assert second.original_content_tokens == 2304
    assert second.target_content_tokens == 576
    assert first.structural_tokens == 24
    assert first.decoder_stage1_tokens == 312
    assert round_half_up_ratio(1, 2) == 1
    assert retention_target(2, 0.25) == 1
    with pytest.raises(ValueError, match="divide exactly"):
        next_profile_per_view(101, 20)


def test_next_requires_default_feature_strategy_without_cls_decoder_token() -> None:
    compressor = DUETCompressor(
        retain_ratio=DUET_LLAVA_NEXT_PROFILE_RATIOS[320],
        llava_next_profile=320,
        pretrained=False,
    )
    assert compressor._include_cls_token is False
    assert compressor._text_rater == "last"
    compressor.validate_llava_next()
    with_cls = DUETCompressor(
        retain_ratio=DUET_LLAVA_NEXT_PROFILE_RATIOS[320],
        llava_next_profile=320,
        pretrained=False,
        include_cls_token=True,
    )
    with pytest.raises(ValueError, match="include_cls_token=false"):
        with_cls.validate_llava_next()
    missing_ratio = DUETCompressor(320, pretrained=False)
    with pytest.raises(ValueError, match="requires retain_ratio"):
        missing_ratio.validate_llava_next()
    wrong_views = DUETCompressor(
        retain_ratio=DUET_LLAVA_NEXT_PROFILE_RATIOS[320],
        llava_next_profile=320,
        include_cls_token=False,
        llava_next_reference_views=1,
        text_rater="last",
        pretrained=False,
    )
    with pytest.raises(ValueError, match="exactly five"):
        wrong_views.validate_llava_next()


def test_anyres_geometry_and_local_tile_major_to_raster_order() -> None:
    inner = _tiny_inner()
    size = torch.tensor([4, 8])
    geometry = geometry_for_image(inner, size)
    assert geometry.crop_count == 3
    assert (geometry.tile_rows, geometry.tile_columns) == (1, 2)
    assert (geometry.local_height, geometry.local_width) == (2, 4)
    assert geometry.newline_count == 2
    assert geometry.native_tokens == 14

    tiles = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    mosaic, shape = stitch_and_unpad_local_crops(inner, tiles, size)
    assert shape == (2, 4)
    torch.testing.assert_close(
        mosaic.squeeze(-1), torch.tensor([[0, 1, 4, 5, 2, 3, 6, 7]])
    )


def test_placeholder_trim_keeps_labels_masks_and_positions_aligned() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 99, 99, 99, 99, 2, 3]]),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, -100, -100, 2, 3]]),
        "position_ids": torch.arange(7).unsqueeze(0),
    }
    trimmed = trim_image_placeholders_to_lengths(
        inputs, token_id=99, lengths=(2,), pad_token_id=0
    )
    torch.testing.assert_close(trimmed["input_ids"], torch.tensor([[1, 99, 99, 2, 3]]))
    torch.testing.assert_close(trimmed["labels"], torch.tensor([[-100, -100, -100, 2, 3]]))
    torch.testing.assert_close(trimmed["attention_mask"], torch.ones(1, 5, dtype=torch.long))
    torch.testing.assert_close(trimmed["position_ids"], torch.arange(5).unsqueeze(0))
    assert placeholder_run_lengths(trimmed, token_id=99) == (2,)


class _IdentityProjector(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def test_stage1_emits_global_then_local_raster_newlines_and_sources() -> None:
    inner = _tiny_inner()
    inner.multi_modal_projector = _IdentityProjector()
    inner.image_newline = nn.Parameter(torch.tensor([99.0]))
    owner = SimpleNamespace(
        budget=2,
        _dominant=5,
        _contextual=0,
        _retain_ratio=0.25,
        _llava_next_profile=160,
        _llava_next_profile_counts=(225, 30),
        _cluster_width=4,
        _track_source=True,
        _llava_next_reference_views=5,
        _pdrop_layers=(16, 24),
        _pdrop_keep_ratios=(0.5, 0.0),
        _llava_next_decoder_layers=32,
        _benchmark_indices=None,
        _llava_next_run_metadata={},
        _source_history=[],
        _keep_counts=(),
    )
    adapter = object.__new__(LlavaNextDuetAdapter)
    adapter._owner = owner
    adapter._inner = inner
    adapter._feature_layer = -2
    adapter._logged = True
    # Crop features are [CLS, patch0..patch3].
    patches = torch.tensor(
        [
            [[0.0], [10.0], [11.0], [12.0], [13.0]],
            [[0.0], [20.0], [21.0], [22.0], [23.0]],
            [[0.0], [30.0], [31.0], [32.0], [33.0]],
        ]
    )
    outputs = SimpleNamespace(hidden_states=(patches, torch.zeros_like(patches)))
    adapter._attention = torch.tensor(
        [[0.1, 0.2, 0.9, 0.3], [0.1, 0.8, 0.2, 0.1], [0.1, 0.2, 0.9, 0.1]]
    )
    adapter._metric = torch.arange(12, dtype=torch.float32).view(3, 4, 1)
    size = torch.tensor([4, 8])
    geometry = geometry_for_image(inner, size)
    budget = next_budget_for_geometry(owner, geometry)

    packed = adapter._transform(outputs, size, geometry, budget)

    # Global winner patch2, local mosaic winners index1 (tile0 patch1) and
    # index6 (tile1 patch2), with a newline after each local row.
    torch.testing.assert_close(
        packed.squeeze(-1), torch.tensor([12.0, 21.0, 99.0, 32.0, 99.0])
    )
    torch.testing.assert_close(
        owner._benchmark_indices, torch.tensor([[2, 5, 8, 11, 13]])
    )
    assert owner._stage1_source.shape == (5, 14)
    assert owner._source_history[0].shape == (14, 14)
    assert owner._llava_next_run_metadata["feature_order"].startswith("global_duet")
    assert {
        "original_content_tokens",
        "requested_retention_ratio",
        "target_content_tokens",
        "realized_content_tokens",
        "realized_retention_ratio",
        "structural_tokens",
    } <= owner._llava_next_run_metadata.keys()

    # K=1 belongs to the global view; an empty local content target is valid and
    # the two local row newlines still survive as structural rows.
    owner._retain_ratio = 0.01
    one_budget = next_budget_for_geometry(owner, geometry)
    one = adapter._transform(outputs, size, geometry, one_budget)
    torch.testing.assert_close(one.squeeze(-1), torch.tensor([12.0, 99.0, 99.0]))
    torch.testing.assert_close(
        owner._llava_next_structural_positions, torch.tensor([1, 2])
    )


class _HookableAttention(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def test_adapter_install_and_uninstall_restore_both_wrappers() -> None:
    layers = [SimpleNamespace(self_attn=_HookableAttention()) for _ in range(2)]
    inner = SimpleNamespace(config=_tiny_config(), get_image_features=lambda *a, **k: None)
    outer = SimpleNamespace(model=inner, config=_tiny_config())
    vlm = SimpleNamespace(
        model=outer,
        _model_id=LLAVA_NEXT_8B_MODEL_ID,
        prepare_inputs=lambda *a, **k: None,
        generate=lambda *a, **k: None,
    )
    owner = SimpleNamespace(
        _resolve_feature_layer=lambda _vlm: -2,
        _vision_encoder_layers=lambda _vlm: layers,
        _dominant=225,
        _contextual=30,
        _retain_ratio=255 / 2880,
        _llava_next_profile=160,
        _llava_next_profile_counts=(225, 30),
        _llava_next_reference_views=5,
        _cluster_width=4,
        _pdrop_layers=(16, 24),
        _pdrop_keep_ratios=(0.5, 0.0),
        _text_rater="last",
        _score_reduction="mean",
        budget=160,
    )
    original_features = inner.get_image_features
    original_prepare = vlm.prepare_inputs
    original_generate = vlm.generate
    adapter = LlavaNextDuetAdapter(owner, vlm)
    adapter.install()
    assert inner.get_image_features is not original_features
    assert vlm.prepare_inputs is not original_prepare
    assert vlm.generate is not original_generate
    adapter.uninstall()
    assert inner.get_image_features is original_features
    assert vlm.prepare_inputs is original_prepare
    assert vlm.generate is original_generate


def test_tiny_native_llava_next_prefill_decode_and_per_layer_cache() -> None:
    """Exercise the real Transformers model class with tiny random modules."""

    from transformers import (
        CLIPVisionConfig,
        LlamaConfig,
        LlavaNextConfig,
        LlavaNextForConditionalGeneration,
    )

    vision = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=4,
        patch_size=2,
        projection_dim=8,
    )
    text = LlamaConfig(
        vocab_size=120,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=25,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    config = LlavaNextConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=99,
        image_grid_pinpoints=[[4, 8], [8, 4], [8, 8]],
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        image_seq_length=4,
    )
    model = LlavaNextForConditionalGeneration(config).eval()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        @staticmethod
        def convert_tokens_to_ids(_token: str) -> int:
            return 99

    class Backbone:
        compressor_family = "llava_next"
        _model_id = LLAVA_NEXT_8B_MODEL_ID

        def __init__(self) -> None:
            self.model = model
            self.processor = SimpleNamespace(image_token="<image>", tokenizer=Tokenizer())

        @staticmethod
        def prepare_inputs(*_args, **_kwargs):
            return None

        @staticmethod
        def generate(*_args, **_kwargs):
            return None

    backbone = Backbone()
    compressor = DUETCompressor(
        retain_ratio=0.25,
        llava_next_profile=160,
        include_cls_token=False,
        text_rater="last",
        pretrained=False,
    )
    # The production path intentionally accepts only the three paper budgets.
    # This tiny random model uses a synthetic profile solely to exercise the
    # adapter/cache mechanics without allocating the 8B checkpoint.
    compressor.validate_llava_next = lambda: None
    original_features = model.model.get_image_features
    compressor.install(backbone)
    input_ids = torch.tensor([[1, 99, 99, 99, 99, 99, 10]])
    output = model(
        input_ids=input_ids,
        pixel_values=torch.randn(1, 3, 3, 4, 4),
        image_sizes=torch.tensor([[4, 8]]),
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
    )
    assert output.logits.shape == (1, 4, 120)
    assert compressor.final_keep_index.shape == (1, 4)
    assert compressor.final_keep_index[0, 0] == 0
    assert compressor.final_keep_index[0, -1] == 6
    assert compressor._llava_next_run_metadata["stage_1_content_tokens"] == 0
    assert compressor._llava_next_run_metadata["stage_1_structural_tokens"] == 2
    assert compressor._llava_next_run_metadata["stage_1_total_visual_tokens"] == 2
    cache = output.past_key_values
    assert [cache.layers[index].keys.shape[-2] for index in (0, 16, 24)] == [7, 5, 4]

    decoded = model(
        input_ids=torch.tensor([[11]]),
        past_key_values=cache,
        attention_mask=torch.ones(1, 8, dtype=torch.long),
        use_cache=True,
    )
    assert decoded.logits.shape == (1, 1, 120)
    assert [cache.layers[index].keys.shape[-2] for index in (0, 16, 24)] == [8, 6, 5]
    compressor.uninstall()
    assert model.model.get_image_features == original_features


class _FakeRotary(nn.Module):
    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        shape = (hidden.shape[0], 1, hidden.shape[1], 2)
        return torch.ones(shape), torch.zeros(shape)


class _FakeLlamaAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)
        self.head_dim = 2
        self.scaling = 2 ** -0.5
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)


def test_llama3_gqa_positions_labels_and_cache_bookkeeping() -> None:
    compressor = DUETCompressor(160, pretrained=False, text_rater="last")
    compressor._image_start = torch.tensor([1])
    compressor._visual_count = 3
    compressor._keep_counts = (1, 0)
    compressor._rater_index = torch.tensor([[0]])
    compressor._rater_valid = torch.tensor([[True]])
    compressor._language_model = SimpleNamespace(rotary_emb=_FakeRotary())
    layer = SimpleNamespace(
        input_layernorm=nn.Identity(), self_attn=_FakeLlamaAttention()
    )
    hidden = torch.randn(1, 6, 8)
    scores = compressor._boundary_scores(
        layer, hidden, None, torch.ones(1, 6, dtype=torch.long), torch.tensor([1])
    )
    assert scores.shape == (1, 3)
    assert torch.isfinite(scores).all()

    compressor._boundary_scores = lambda *args, **kwargs: torch.tensor([[0.1, 0.9, 0.2]])
    original = hidden.clone()
    kwargs = {
        "position_ids": torch.arange(6).unsqueeze(0),
        "attention_mask": torch.tril(torch.ones(1, 1, 6, 6, dtype=torch.bool)),
        "cache_position": torch.arange(6),
    }
    pruned = compressor._prune(layer, hidden, kwargs, 0)
    torch.testing.assert_close(pruned, original[:, [0, 2, 4, 5]])
    torch.testing.assert_close(compressor.final_keep_index, torch.tensor([[0, 2, 4, 5]]))
    torch.testing.assert_close(
        compressor._stage_state[0]["position_ids"], torch.tensor([[0, 2, 4, 5]])
    )
    labels = torch.tensor([[10, 11, 12, 13, 14, 15]])
    torch.testing.assert_close(
        labels.gather(1, compressor.final_keep_index), torch.tensor([[10, 12, 14, 15]])
    )
    compressor._advance_decode_step(torch.zeros(1, 1, 8), 0)
    assert compressor._stage_state[0]["kv_len"] == 5
    torch.testing.assert_close(
        compressor._stage_state[0]["cache_position"], torch.tensor([4])
    )
    with pytest.raises(RuntimeError, match="Beam search"):
        compressor._advance_decode_step(torch.zeros(2, 1, 8), 0)


def test_training_wrapper_gathers_pruned_labels_for_causal_loss() -> None:
    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace()
            self.generation_config = SimpleNamespace()

        def forward(self, **_kwargs):
            return SimpleNamespace(logits=torch.randn(1, 4, 17, requires_grad=True))

    compressor = SimpleNamespace(final_keep_index=torch.tensor([[0, 2, 4, 5]]))
    wrapper = DuetTrainingLlavaForConditionalGeneration(Base(), compressor)
    result = wrapper(
        input_ids=torch.ones(1, 6, dtype=torch.long),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        labels=torch.tensor([[1, 2, 3, 4, 5, 6]]),
    )
    assert result.loss.ndim == 0
    assert torch.isfinite(result.loss)


def _metadata_owner() -> DUETCompressor:
    owner = DUETCompressor(
        retain_ratio=DUET_LLAVA_NEXT_PROFILE_RATIOS[320],
        llava_next_profile=320,
        pretrained=False,
    )
    owner.validate_llava_next()
    owner._metadata_backbone = "llava_next"
    owner._training_stage = "finetune"
    owner._llava_next_decoder_layers = 32
    owner._vision_feature_layer = -2
    owner._llava_next_image_grid_pinpoints = [
        [336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]
    ]
    owner._llava_next_vision_geometry = {
        "image_size": 336,
        "patch_size": 14,
        "patches_per_view": 576,
    }
    return owner


def _checkpoint_vlm(path: Path) -> SimpleNamespace:
    config = SimpleNamespace(
        vision_feature_layer=-2,
        image_grid_pinpoints=[
            [336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]
        ],
        vision_config=SimpleNamespace(image_size=336, patch_size=14),
    )
    return SimpleNamespace(_model_id=str(path), model=SimpleNamespace(config=config))


def test_checkpoint_metadata_accepts_exact_and_rejects_other_budget(tmp_path: Path) -> None:
    owner = _metadata_owner()
    payload = duet_next_metadata(owner)
    assert payload["schema_version"] == 5
    assert payload["training_attention_backend"] == "flash_attention_2"
    assert payload["training_flash_attn_version"] == DUET_NEXT_FLASH_ATTN_VERSION
    assert payload["inference_attention_backend"] == "sdpa"
    (tmp_path / "duet_config.json").write_text(json.dumps(payload), encoding="utf-8")
    vlm = _checkpoint_vlm(tmp_path)
    validate_duet_next_checkpoint(owner, vlm)
    payload["requested_retention_ratio"] = DUET_LLAVA_NEXT_PROFILE_RATIOS[160]
    (tmp_path / "duet_config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        validate_duet_next_checkpoint(owner, vlm)
    payload = duet_next_metadata(owner)
    payload["training_attention_backend"] = "sdpa"
    (tmp_path / "duet_config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="training_attention_backend"):
        validate_duet_next_checkpoint(owner, vlm)
    payload = duet_next_metadata(owner)
    payload["training_stage"] = "pretrain"
    (tmp_path / "duet_config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="training_stage"):
        validate_duet_next_checkpoint(owner, vlm)


def test_nonstock_checkpoint_without_metadata_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without duet_config"):
        validate_duet_next_checkpoint(_metadata_owner(), _checkpoint_vlm(tmp_path))


def test_lazy_parquet_dataset_and_locality_sampler(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    buffer = BytesIO()
    Image.new("RGB", (3, 2), color=(1, 2, 3)).save(buffer, format="PNG")
    rows = [
        {
            "id": str(index),
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion?"},
                {"from": "gpt", "value": "Answer."},
            ],
            "data_source": "unit",
            "image": {"bytes": buffer.getvalue(), "path": f"{index}.png"},
        }
        for index in range(3)
    ]
    parquet.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet", row_group_size=2)
    dataset = LlavaNextParquetDataset(tmp_path)
    assert len(dataset) == 3
    assert dataset[1]["image"].size == (3, 2)
    sampler = ParquetRowGroupSampler(dataset, seed=7)
    assert sorted(iter(sampler)) == [0, 1, 2]


def test_training_stage_chains_only_matching_projector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    next_data = input_root / "LLaVA-NeXT-Data" / "data"
    next_data.mkdir(parents=True)
    # The builder checks existence; the lazy dataset validates Parquet later.
    (next_data / "train.parquet").touch()
    projector_dir = tmp_path / "projector"
    projector_dir.mkdir()
    projector = projector_dir / "mm_projector.bin"
    projector.touch()
    owner = _metadata_owner()
    owner._training_stage = "pretrain"
    (projector_dir / "duet_config.json").write_text(
        json.dumps(duet_next_metadata(owner)), encoding="utf-8"
    )
    config = SimpleNamespace(
        custom_args={
            "training_args": {
                "data_path": str(input_root / "LLaVA-NeXT-Data"),
                "pretrain_mm_mlp_adapter": str(projector),
                "gradient_accumulation_steps": 32,
            }
        },
        input_dir=str(input_root),
        output_dir=str(tmp_path / "output"),
        dataset_limit=None,
    )
    backbone = SimpleNamespace(
        _model_id=LLAVA_NEXT_8B_MODEL_ID,
        _model_revision=LLAVA_NEXT_8B_REVISION,
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    args = build_next_training_args(backbone, config, "finetune", owner)
    assert args.pretrain_mm_mlp_adapter == str(projector)
    assert args.global_batch_size == 128
    assert args.gradient_accumulation_steps == 32
    assert args.attn_implementation == "flash_attention_2"
    assert args.bf16 is True

    config.custom_args["training_args"]["attn_implementation"] = "sdpa"
    with pytest.raises(ValueError, match="FlashAttention-2 only"):
        build_next_training_args(backbone, config, "finetune", owner)
    del config.custom_args["training_args"]["attn_implementation"]
    config.custom_args["training_args"]["bf16"] = False
    with pytest.raises(ValueError, match="requires bf16=true"):
        build_next_training_args(backbone, config, "finetune", owner)
    del config.custom_args["training_args"]["bf16"]

    payload = duet_next_metadata(owner)
    payload["requested_retention_ratio"] = DUET_LLAVA_NEXT_PROFILE_RATIOS[160]
    (projector_dir / "duet_config.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incompatible"):
        build_next_training_args(backbone, config, "finetune", owner)


def test_training_loader_rejects_non_fa2_and_non_bf16_before_model_loading() -> None:
    with pytest.raises(ValueError, match="FlashAttention-2 only"):
        load_hf_llava_next_training_base(
            "unused",
            revision=LLAVA_NEXT_8B_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    with pytest.raises(ValueError, match="dtype=torch.bfloat16"):
        load_hf_llava_next_training_base(
            "unused",
            revision=LLAVA_NEXT_8B_REVISION,
            dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )


def test_training_runtime_requires_pinned_compiled_fa2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(duet_next_tr_module, "distribution_version", lambda _name: "2.8.3")
    with pytest.raises(RuntimeError, match="requires flash-attn==2.8.3.post1"):
        require_flash_attention_2_training_runtime()

    def missing(_name: str) -> str:
        raise duet_next_tr_module.PackageNotFoundError

    monkeypatch.setattr(duet_next_tr_module, "distribution_version", missing)
    with pytest.raises(RuntimeError, match="install the pinned Conrad wheel"):
        require_flash_attention_2_training_runtime()


def test_configs_pin_fa2_for_training_and_sdpa_for_inference() -> None:
    import yaml

    config_root = Path(__file__).parents[1] / "configs" / "duet"
    training_configs = sorted(config_root.glob("llava_next_8b_pretrain_*.yaml"))
    training_configs += sorted(config_root.glob("llava_next_8b_finetune_*.yaml"))
    benchmark_configs = sorted(config_root.glob("llava_next_8b_benchmark_*.yaml"))
    assert len(training_configs) == 6
    assert len(benchmark_configs) == 3
    for path in training_configs:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["configuration_metadata"]["schema_version"] == 5
        assert "retain_ratio" in payload["compressor_args"]
        assert "llava_next_profile" in payload["compressor_args"]
        assert "max_tokens" not in payload["compressor_args"]
        assert payload["training_args"]["attn_implementation"] == "flash_attention_2"
        assert payload["training_args"]["bf16"] is True
        assert payload["configuration_metadata"]["attention_backend"] == "flash_attention_2"
        assert payload["configuration_metadata"]["flash_attn_version"] == "2.8.3.post1"
    for path in benchmark_configs:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["configuration_metadata"]["schema_version"] == 5
        assert payload["backbone_args"]["attn_implementation"] == "sdpa"
        assert payload["configuration_metadata"]["attention_backend"] == "sdpa"


def test_cached_processor_collator_matches_exact_anyres_budget() -> None:
    snapshot = Path(
        "/workspace/output/hf_cache/models--llava-hf--llama3-llava-next-8b-hf/"
        "snapshots/b041c0d0ea0dd0196d147206c210c8d1752fc2da"
    )
    if not snapshot.is_dir():
        pytest.skip("Pinned processor snapshot is not available")
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    compressor = _metadata_owner()
    collator = LlavaNextDuetCollator(processor, config, compressor)
    sample = {
        "has_image": True,
        "image": Image.new("RGB", (640, 480), color=(10, 20, 30)),
        "conversations": [
            {"role": "user", "value": "<image>\nWhat is shown?"},
            {"role": "assistant", "value": "A test image."},
        ],
    }
    batch = collator([sample])
    assert tuple(batch["pixel_values"].shape[:2]) == (1, 5)
    # 640x480 produces 2,304 content rows after exact unpadding. Profile 320's
    # 510/2,880 ratio requests 408 content rows; 36 newlines remain structural.
    assert int((batch["input_ids"] == config.image_token_index).sum()) == 444
    assert bool((batch["labels"] != -100).any())

    # The native 2,340-row expansion plus this long prompt exceeds 4,096, but
    # the DUET-compressed sequence fits. Assistant labels must not be discarded
    # by premature native-span truncation.
    long_sample = dict(sample)
    long_sample["conversations"] = [
        {"role": "user", "value": "<image>\n" + "object " * 1900},
        {"role": "assistant", "value": "Still supervised."},
    ]
    long_batch = collator([long_sample])
    assert long_batch["input_ids"].shape[1] < 4096
    assert bool((long_batch["labels"] != -100).any())

    pretrain_collator = LlavaNextDuetCollator(
        processor, config, compressor, stage="pretrain"
    )
    pretrain_batch = pretrain_collator([sample])
    image_positions = (
        pretrain_batch["input_ids"][0] == config.image_token_index
    ).nonzero(as_tuple=True)[0]
    assert bool((pretrain_batch["labels"][0, : image_positions[-1] + 1] == -100).all())
    assert bool((pretrain_batch["labels"][0, image_positions[-1] + 1 :] != -100).any())
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        collator([sample, sample])

    # Visualization uses the native processor pixels, not model weights. Its
    # atlas contains the 24x24 global grid over the 36x48 unpadded local mosaic;
    # 36 newline source columns are intentionally absent from the pixel atlas.
    from llmoe.compressor.duet.duet_llava_next import duet_visualization_view

    fake_model = SimpleNamespace(
        compressor_family="llava_next",
        model=SimpleNamespace(model=SimpleNamespace(config=config)),
        processor=processor,
    )
    view = duet_visualization_view(fake_model, batch, processor.image_processor)
    assert view.canvas.shape == (3, 60 * 14, 48 * 14)
    assert tuple(view.source_columns.shape) == (2304,)
    assert tuple(view.canvas_columns.shape) == (2304,)
    assert int(view.source_columns.max()) < 2340
    assert int(view.canvas_columns.max()) < 60 * 48


def test_metadata_records_pinned_provenance() -> None:
    payload = duet_next_metadata(_metadata_owner())
    assert payload["implementation_version"] == DUET_NEXT_IMPLEMENTATION_VERSION
    assert payload["base_model_id"] == LLAVA_NEXT_8B_MODEL_ID
    assert payload["base_model_revision"] == LLAVA_NEXT_8B_REVISION
    assert payload["include_cls_in_decoder"] is False
    assert payload["reference_content_layer_average"] == 318.75
    assert payload["reference_target_content_tokens"] == 510
