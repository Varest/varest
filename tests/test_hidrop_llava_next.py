"""CPU contracts for HiDrop on the pinned Llama-3 LLaVA-NeXT-8B layout."""

from fractions import Fraction
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoProcessor,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    PreTrainedModel,
    PretrainedConfig,
    ProcessorMixin,
    TrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.llava_next.modeling_llava_next import LlavaNextModel
from transformers.models.llama.modeling_llama import LlamaModel

from llmoe.backbone import BACKBONES
import llmoe.backbone.llava_next as backbone_module
from llmoe.backbone.llava_next import (
    LLAVA_NEXT_8B_MODEL_ID,
    LLAVA_NEXT_8B_REVISION,
    LLaVANext8B,
    LLaVANextImagePreprocessor,
)
from llmoe.compressor.hidrop.hidrop import (
    IGNORE_INDEX,
    HiDropCompressor,
    HiDropConfig,
    HiDropForwardContext,
    HiDropLlamaModel,
    VisualTokenSpan,
    backend_for_training,
)
from llmoe.compressor.hidrop.hidrop_llava_next import (
    LLAVA_NEXT_IMAGE_GRID_PINPOINTS,
    LlavaNextVisualLayout,
    llava_next_visualization_views,
    resolve_scaled_schedule,
    resolve_visual_layout,
)
from llmoe.compressor.hidrop.hidrop_next_tr import (
    HiDropTrainingLlavaNextForConditionalGeneration,
    LlavaNextHiDropCollator,
    LlavaNextParquetDataset,
    _assistant_labels,
    _require_flash_attention_2,
    configure_llava_next_trainable_parameters,
    load_hf_llava_next_training_base,
)
from llmoe.compressor.hidrop.hidrop_tr import HiDropTrainer, validate_hidrop_artifact
import llmoe.compressor.hidrop.provenance as provenance_module
from llmoe.compressor.hidrop.provenance import is_pinned_llava_next_snapshot
from llmoe.resolvers import resolve_backbone


CACHE_SNAPSHOT = Path(
    "/workspace/output/hf_cache/models--llava-hf--llama3-llava-next-8b-hf/"
    "snapshots/b041c0d0ea0dd0196d147206c210c8d1752fc2da"
)
DATA_ROOT = Path("/workspace/input/LLaVA-NeXT-Data")


def _layout(image_size: tuple[int, int]) -> LlavaNextVisualLayout:
    return resolve_visual_layout(
        image_size,
        grid_pinpoints=LLAVA_NEXT_IMAGE_GRID_PINPOINTS,
        vision_image_size=336,
        vision_patch_size=14,
    )


def _synthetic_layout() -> LlavaNextVisualLayout:
    # Base 2x2, then a 2x4 local mosaic with one newline after each row.
    return LlavaNextVisualLayout(
        version=1,
        image_height=28,
        image_width=56,
        crop_count=3,
        tile_rows=1,
        tile_columns=2,
        patch_rows_per_crop=2,
        patch_columns_per_crop=2,
        local_top_offset=0,
        local_left_offset=0,
        local_height=2,
        local_width=4,
        native_tokens=14,
        patch_tokens=12,
        global_patch_tokens=4,
        local_patch_tokens=8,
        newline_tokens=2,
        newline_indices=(8, 13),
        source_view_by_token=(0, 0, 0, 0, 1, 1, 2, 2, -1, 1, 1, 2, 2, -1),
        source_patch_by_token=(0, 1, 2, 3, 0, 1, 0, 1, -1, 2, 3, 2, 3, -1),
        local_row_by_token=(-1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
    )


@pytest.mark.parametrize(
    ("image_size", "crop_count", "tile_grid", "unpad", "patches", "newlines", "native"),
    [
        ((336, 336), 3, (1, 2), (0, 12, 24, 24), 1152, 24, 1176),
        ((500, 500), 5, (2, 2), (0, 0, 48, 48), 2880, 48, 2928),
        ((400, 800), 3, (1, 2), (0, 0, 24, 48), 1728, 24, 1752),
        ((800, 400), 3, (2, 1), (0, 0, 48, 24), 1728, 48, 1776),
        ((100, 1000), 4, (1, 3), (8, 0, 8, 72), 1152, 8, 1160),
    ],
)
def test_anyres_geometry_matches_transformers_packing(
    image_size: tuple[int, int],
    crop_count: int,
    tile_grid: tuple[int, int],
    unpad: tuple[int, int, int, int],
    patches: int,
    newlines: int,
    native: int,
) -> None:
    layout = _layout(image_size)

    assert layout.crop_count == crop_count
    assert (layout.tile_rows, layout.tile_columns) == tile_grid
    assert (
        layout.local_top_offset,
        layout.local_left_offset,
        layout.local_height,
        layout.local_width,
    ) == unpad
    assert (layout.patch_tokens, layout.newline_tokens, layout.native_tokens) == (
        patches,
        newlines,
        native,
    )
    assert len(layout.source_view_by_token) == native
    assert len(layout.patch_indices) == patches
    assert len(layout.newline_indices) == newlines


def test_tile_major_crops_are_rearranged_into_spatial_raster_order() -> None:
    layout = _layout((500, 500))

    assert set(layout.source_view_by_token[:576]) == {0}
    assert layout.source_view_by_token[576:624] == (1,) * 24 + (2,) * 24
    assert layout.source_patch_by_token[576:624] == tuple(range(24)) + tuple(range(24))
    assert layout.newline_indices[0] == 624
    assert layout.source_view_by_token[625:673] == (1,) * 24 + (2,) * 24
    assert layout.source_patch_by_token[625:673] == tuple(range(24, 48)) + tuple(range(24, 48))
    first_second_tile_row = 576 + 24 * 49
    assert layout.source_view_by_token[first_second_tile_row : first_second_tile_row + 48] == (
        (3,) * 24 + (4,) * 24
    )


def test_layout_provenance_matches_transformers_pack_image_features_order() -> None:
    layout = resolve_visual_layout(
        (28, 56),
        grid_pinpoints=((28, 56),),
        vision_image_size=28,
        vision_patch_size=14,
        observed_crop_count=3,
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=28, patch_size=14),
            image_grid_pinpoints=[[28, 56]],
        )
    )
    features = torch.tensor(
        [[[view * 100 + patch] for patch in range(4)] for view in range(3)],
        dtype=torch.float32,
    )
    packed, lengths = LlavaNextModel.pack_image_features(
        owner,
        [features],
        torch.tensor([[28, 56]]),
        "default",
        image_newline=torch.tensor([-1.0]),
    )
    expected = [
        -1 if view < 0 else view * 100 + patch
        for view, patch in zip(
            layout.source_view_by_token, layout.source_patch_by_token, strict=True
        )
    ]

    assert lengths.tolist() == [layout.native_tokens]
    assert packed[0][:, 0].tolist() == expected


def test_single_view_layout_has_one_trailing_newline(monkeypatch: pytest.MonkeyPatch) -> None:
    import llmoe.compressor.hidrop.hidrop_llava_next as geometry_module

    monkeypatch.setattr(geometry_module, "image_size_to_num_patches", lambda *_args, **_kwargs: 1)
    layout = geometry_module.resolve_visual_layout(
        (336, 336),
        grid_pinpoints=LLAVA_NEXT_IMAGE_GRID_PINPOINTS,
        vision_image_size=336,
        vision_patch_size=14,
        observed_crop_count=1,
    )

    assert layout.native_tokens == 577
    assert layout.patch_tokens == 576
    assert layout.newline_indices == (576,)
    assert layout.source_view_by_token[-1] == -1


def test_crop_count_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="crop-count mismatch"):
        resolve_visual_layout(
            (500, 500),
            grid_pinpoints=LLAVA_NEXT_IMAGE_GRID_PINPOINTS,
            vision_image_size=336,
            vision_patch_size=14,
            observed_crop_count=4,
        )


@pytest.mark.parametrize("budget", [32, 64, 128, 224])
def test_integer_view_budget_scaling_preserves_decoder_layer_average(budget: int) -> None:
    config = HiDropConfig(backbone_family="llava_next", average_retained_tokens=budget)
    schedule = resolve_scaled_schedule(config, _layout((500, 500)))

    assert schedule.requested_scaled_patch_average == Fraction(budget * 5, 1)
    assert schedule.realized_patch_average == Fraction(budget * 5, 1)
    assert schedule.realized_structural_average == Fraction(24, 1)
    assert schedule.realized_total_average == Fraction(budget * 5 + 24, 1)
    assert schedule.patch_keep_list == tuple(value * 5 for value in config.image_token_keep_list)


def test_nonintegral_geometry_reports_exact_rounding_drift() -> None:
    layout = _layout((101, 137))
    schedule = resolve_scaled_schedule(
        HiDropConfig(backbone_family="llava_next", retain_ratio=0.1111), layout
    )
    metadata = schedule.to_metadata()

    assert layout.patch_tokens == 1344
    assert schedule.patch_keep_list == (501, 23, 12, 2)
    assert schedule.requested_scaled_patch_average == Fraction(149, 1)
    assert schedule.realized_patch_average == Fraction(2387, 16)
    assert schedule.realized_total_average == Fraction(2579, 16)
    assert metadata["requested_scaled_patch_average"]["denominator"] == 1
    assert metadata["realized_patch_average"]["denominator"] == 16
    assert metadata["rounding"] == "target_and_stages_use_nearest_integer_half_up"


def test_ratio_budget_uses_each_images_true_unpadded_content_count() -> None:
    config = HiDropConfig(backbone_family="llava_next", retain_ratio=0.25)
    first = resolve_scaled_schedule(config, _layout((336, 336)))
    second = resolve_scaled_schedule(config, _layout((101, 137)))

    assert first.native_patch_tokens == 1152
    assert first.target_content_tokens == 288
    assert second.native_patch_tokens == 1344
    assert second.target_content_tokens == 336
    assert first.requested_retention_ratio == second.requested_retention_ratio == Fraction(1, 4)
    # A single target anchors the whole official curve. It is not multiplied by
    # 0.25 again at every stage.
    assert first.patch_keep_list[0] > first.target_content_tokens


def test_budget_metadata_reports_content_and_structure_separately() -> None:
    schedule = resolve_scaled_schedule(
        HiDropConfig(backbone_family="llava_next", retain_ratio=0.25),
        _layout((336, 336)),
    )
    metadata = schedule.to_metadata()

    assert metadata["original_content_tokens"] == 1152
    assert metadata["requested_retention_ratio"] == 0.25
    assert metadata["target_content_tokens"] == 288
    assert metadata["realized_content_tokens"] == float(schedule.realized_patch_average)
    assert metadata["realized_retention_ratio"] == float(
        schedule.realized_patch_average / 1152
    )
    assert metadata["structural_tokens"] == 24
    assert metadata["target_content_tokens"] == round(0.25 * 1152)
    with pytest.raises(ValueError, match="assembled content count"):
        resolve_scaled_schedule(
            HiDropConfig(backbone_family="llava_next", retain_ratio=0.25),
            _layout((336, 336)),
            observed_content_tokens=1151,
        )


def test_public_next_ratio_routes_to_next_not_qwen_and_surfaces_evaluation_fields() -> None:
    compressor = HiDropCompressor(
        backbone_family="llava_next", retain_ratio=0.25, pretrained=False
    )
    assert isinstance(compressor.config, HiDropConfig)
    assert compressor.config.backbone_family == "llava_next"
    assert compressor.config.retain_ratio == 0.25
    assert compressor.budget == 0.25
    assert (
        backend_for_training(
            {"backbone_family": "llava_next", "retain_ratio": 0.25},
            LLAVA_NEXT_8B_MODEL_ID,
        ).key
        == "llava"
    )

    _unused, _wrapper, context, *_rest = _direct_pruning_fixture()
    context.resolved_schedule = resolve_scaled_schedule(compressor.config, _synthetic_layout())
    compressor.last_context = context
    diagnostics = compressor.hidrop_config_dict()
    for key in (
        "original_content_tokens",
        "requested_retention_ratio",
        "target_content_tokens",
        "realized_content_tokens",
        "realized_retention_ratio",
        "structural_tokens",
    ):
        assert diagnostics[key] == diagnostics["last_resolved_budget"][key]


def test_legacy_llava15_profile_and_public_metadata_remain_unchanged() -> None:
    config = HiDropConfig(average_retained_tokens=64)
    public = config.to_dict()
    checkpoint = config.hidrop_metadata()

    assert config.layer_list == (10, 14, 16, 18)
    assert config.image_token_keep_list == (215, 10, 5, 1)
    assert config.computed_average_retained_tokens() == 64.0
    assert len(config.visual_token_counts_by_layer()) == 32
    assert public["trained_with_hf_llava"] is True
    assert checkpoint["trained_with_hf_llava"] is True
    for next_only in (
        "backbone_family",
        "backbone_model_id",
        "backbone_revision",
        "image_grid_pinpoints",
        "llava_next_implementation_version",
        "llava_next_layout_version",
        "budget_convention",
        "newline_policy",
        "native_visual_layout",
        "training_stage",
        "vision_architecture",
        "multimodal_projector",
    ):
        assert next_only not in public
        assert next_only not in checkpoint


def _direct_pruning_fixture() -> tuple[
    HiDropCompressor,
    HiDropLlamaModel,
    HiDropForwardContext,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    compressor = HiDropCompressor(
        backbone_family="llava_next", average_retained_tokens=64, pretrained=False
    )
    base = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=32, _attn_implementation="sdpa"),
        embed_tokens=nn.Embedding(32, 4),
        layers=nn.ModuleList([nn.Identity() for _ in range(32)]),
        norm=nn.Identity(),
        rotary_emb=nn.Identity(),
        gradient_checkpointing=False,
    )
    wrapper = HiDropLlamaModel(base, compressor)
    layout = _synthetic_layout()
    hidden = torch.arange(18 * 4, dtype=torch.float32).view(1, 18, 4)
    positions = torch.arange(18).unsqueeze(0)
    mask = torch.ones((1, 18), dtype=torch.bool)
    labels = torch.full((1, 18), IGNORE_INDEX, dtype=torch.long)
    labels[0, -1] = 7
    span = VisualTokenSpan(start=2, end=16, input_seq_len=18)
    context = HiDropForwardContext(
        span=span,
        spans=(span,),
        original_hidden_states=hidden.clone(),
        original_attention_mask=mask.clone(),
        original_position_ids=positions.clone(),
        original_labels=labels.clone(),
        active_visual_count=layout.native_tokens,
        active_patch_count=layout.patch_tokens,
        llava_next_layout=layout,
        resolved_schedule=resolve_scaled_schedule(compressor.config, layout),
        preserved_visual_rows=set(),
        pruning_events=[],
    )
    wrapper._project_qk = MethodType(  # type: ignore[method-assign]
        lambda self, _layer, states, _positions: (
            states.new_zeros((1, 1, states.shape[1], 1)),
            states.new_zeros((1, 1, states.shape[1], 1)),
            1.0,
        ),
        wrapper,
    )
    return compressor, wrapper, context, hidden, mask, positions, labels


def test_joint_global_local_selection_preserves_newlines_positions_labels_and_sources() -> None:
    compressor, wrapper, context, hidden, mask, positions, labels = _direct_pruning_fixture()
    # Highest patch scores include one base patch (relative 0) and three local patches.
    scores = torch.zeros(14)
    scores[[0, 4, 10, 12]] = torch.tensor([100.0, 90.0, 80.0, 70.0])
    wrapper._visual_scores_from_qk = MethodType(  # type: ignore[method-assign]
        lambda self, *_args, **_kwargs: scores,
        wrapper,
    )

    pruned, pruned_mask, pruned_positions, pruned_labels = wrapper._prune_visual_tokens(
        nn.Identity(), hidden, mask, positions, labels, context, rank_layer=10
    )

    expected_visual_positions = torch.tensor([2, 6, 10, 12, 15])
    assert torch.equal(pruned_positions[0, 2:7], expected_visual_positions)
    assert 2 in expected_visual_positions.tolist()  # selected global row
    assert 12 in expected_visual_positions.tolist()  # selected local row
    assert {10, 15}.issubset(set(expected_visual_positions.tolist()))  # both newlines
    assert tuple(pruned.shape) == (1, 9, 4)
    assert tuple(pruned_mask.shape) == (1, 9)
    assert pruned_labels[0, -1] == 7
    assert context.active_patch_count == 3
    assert context.active_visual_count == 5
    assert context.pruning_events[-1]["structural_tokens_after"] == 2
    history = compressor.get_source_history()
    assert [tuple(source.shape) for source in history] == [(14, 14), (5, 14)]
    assert torch.equal(torch.where(history[-1].sum(dim=0) > 0)[0], expected_visual_positions - 2)

    # A later stage identifies structural rows from persistent original positions,
    # not from their now-shifted current offsets.
    wrapper._visual_scores_from_qk = MethodType(  # type: ignore[method-assign]
        lambda self, *_args, **_kwargs: torch.arange(
            context.active_visual_count, dtype=torch.float32
        ),
        wrapper,
    )
    second = wrapper._prune_visual_tokens(
        nn.Identity(),
        pruned,
        pruned_mask,
        pruned_positions,
        pruned_labels,
        context,
        rank_layer=14,
    )
    assert context.active_patch_count == 1
    assert context.active_visual_count == 3
    assert {10, 15}.issubset(set(second[2][0, 2:5].tolist()))
    assert compressor.get_source_history()[-1].shape == (3, 14)

    exited = wrapper._remove_visual_span(
        second[0], second[1], second[2], second[3], context.spans, context.active_visual_count
    )
    assert torch.equal(exited[2], torch.tensor([[0, 1, 16, 17]]))
    assert exited[3][0, -1] == 7


def test_late_injection_restores_original_rows_masks_positions_and_labels() -> None:
    _, wrapper, context, hidden, mask, positions, labels = _direct_pruning_fixture()
    removed = wrapper._remove_visual_span(
        hidden, mask, positions, labels, context.spans, context.span.count
    )
    restored = wrapper._late_inject_visual_span(*removed, context)

    torch.testing.assert_close(restored[0], hidden)
    assert torch.equal(restored[1], mask)
    assert torch.equal(restored[2], positions)
    assert torch.equal(restored[3], labels)


def test_next_selector_ignores_causal_assistant_turns_before_the_image() -> None:
    _, wrapper, context, hidden, _, _, _ = _direct_pruning_fixture()
    labels = torch.full((1, 18), IGNORE_INDEX, dtype=torch.long)
    labels[0, 1] = 3
    labels[0, 17] = 7

    queries = wrapper._query_indices_for_scoring(labels, hidden, context)

    assert queries[0] is not None
    assert torch.equal(queries[0], torch.tensor([16]))


def test_eager_cache_mask_uses_each_decoder_layers_own_length() -> None:
    class _RaggedCache:
        @staticmethod
        def get_seq_length(layer_idx: int) -> int:
            return {3: 11, 10: 5}[layer_idx]

    hidden = torch.zeros((1, 2, 4), dtype=torch.float32)
    deep = HiDropLlamaModel._eager_causal_mask(hidden, None, _RaggedCache(), 3)
    pruned = HiDropLlamaModel._eager_causal_mask(hidden, None, _RaggedCache(), 10)

    assert deep.shape == (1, 1, 2, 13)
    assert pruned.shape == (1, 1, 2, 7)
    assert deep[0, 0, 0, -1] < -1e20
    assert pruned[0, 0, 1, -1] == 0


def test_dynamic_cache_tracks_ragged_layers_and_persistent_decode_positions() -> None:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    base = LlamaModel(config).eval()
    original_rotary = base.rotary_emb
    seen_positions: list[torch.Tensor] = []

    class _RecordingRotary(nn.Module):
        def forward(self, hidden_states, position_ids):
            seen_positions.append(position_ids.detach().clone())
            return original_rotary(hidden_states, position_ids)

    base.rotary_emb = _RecordingRotary()
    compressor = HiDropCompressor(
        backbone_family="llava_next", average_retained_tokens=64, pretrained=False
    )
    compressor._image_token_id = 99
    compressor._pending_llava_next_layouts = (_synthetic_layout(),)
    wrapper = HiDropLlamaModel(base, compressor).eval()
    input_ids = torch.tensor([[1, 5, *([99] * 14), 6, 7]])

    with torch.no_grad():
        prefill = wrapper(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
        )
    cache = prefill.past_key_values
    expected_prefill_lengths = [4] * 8 + [18] * 2 + [9] * 4 + [7] * 10 + [4] * 8
    assert [cache.get_seq_length(layer) for layer in range(32)] == expected_prefill_lengths

    # The shallow cache contains only four text tokens, so inferring the next
    # RoPE position from that cache would silently produce 4 instead of 18.
    with pytest.raises(ValueError, match="persistent prefill positions"):
        wrapper(input_ids=torch.tensor([[8]]), past_key_values=cache, use_cache=True)

    seen_positions.clear()
    with torch.no_grad():
        decoded = wrapper(
            input_ids=torch.tensor([[8]]),
            attention_mask=torch.ones((1, 19), dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
    assert decoded.last_hidden_state.shape == (1, 1, 8)
    assert seen_positions and all(
        torch.equal(item, torch.tensor([[18]])) for item in seen_positions
    )
    assert [cache.get_seq_length(layer) for layer in range(32)] == [
        value + 1 for value in expected_prefill_lengths
    ]


def test_hf_generation_preparation_sends_one_token_after_ragged_prefill() -> None:
    """Exercise the real NeXT GenerationMixin path, not only direct decoder calls."""
    vision = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    text = LlamaConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlavaNextForConditionalGeneration(
        LlavaNextConfig(
            vision_config=vision,
            text_config=text,
            image_token_index=99,
            image_grid_pinpoints=[[8, 8]],
        )
    ).eval()
    model.model.language_model.config._attn_implementation = "sdpa"
    packed = torch.randn(14, 8)
    model.model.get_image_features = lambda *_args, **_kwargs: BaseModelOutputWithPooling(
        pooler_output=[packed]
    )
    compressor = HiDropCompressor(
        backbone_family="llava_next", average_retained_tokens=64, pretrained=False
    )
    compressor._image_token_id = 99
    compressor._pending_llava_next_layouts = (_synthetic_layout(),)
    input_ids = torch.tensor([[1, 5, *([99] * 14), 6, 7]])
    compressor._latest_input_ids = input_ids
    model.model.language_model = HiDropLlamaModel(model.model.language_model, compressor)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=torch.zeros((1, 3, 3, 8, 8)),
            image_sizes=torch.tensor([[8, 8]]),
            do_sample=False,
            use_cache=True,
            min_new_tokens=2,
            max_new_tokens=2,
        )

    assert generated.shape == (1, input_ids.shape[1] + 2)
    assert compressor.last_context is not None
    assert compressor.last_visual_token_counts_by_layer == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        14,
        14,
        5,
        5,
        5,
        5,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


def test_context_requires_one_captured_layout_and_exact_placeholder_span() -> None:
    compressor, _, _, hidden, mask, positions, labels = _direct_pruning_fixture()
    compressor._image_token_id = 99
    ids = torch.tensor([[1, 1, *([99] * 14), 2, 2]])
    with pytest.raises(ValueError, match="captured AnyRes layout"):
        compressor._build_hidrop_context(hidden, ids, mask, positions, labels)

    compressor._pending_llava_next_layouts = (_synthetic_layout(),)
    context = compressor._build_hidrop_context(hidden, ids, mask, positions, labels)
    assert context is not None and context.span.count == 14

    wrong_ids = torch.tensor([[1, 1, *([99] * 13), 2, 2, 2]])
    with pytest.raises(ValueError, match="placeholder expansion differs"):
        compressor._build_hidrop_context(hidden, wrong_ids, mask, positions, labels)

    with pytest.raises(NotImplementedError, match="exactly one source image"):
        compressor.configure_llava_next_layouts(
            SimpleNamespace(),
            torch.tensor([[28, 56], [28, 56]]),
        )


def _cached_config():
    if not CACHE_SNAPSHOT.is_dir():
        pytest.skip("Pinned LLaVA-NeXT config cache is unavailable.")
    return AutoConfig.from_pretrained(CACHE_SNAPSHOT, local_files_only=True)


def test_pinned_cached_config_satisfies_exact_llama3_and_anyres_contract() -> None:
    config = _cached_config()
    compressor = HiDropCompressor(
        backbone_family="llava_next", average_retained_tokens=64, pretrained=False
    )
    compressor._validate_llava_next_model(SimpleNamespace(config=config))

    config.text_config.num_key_value_heads = 4
    with pytest.raises(ValueError, match="Llama-3-8B decoder signature"):
        compressor._validate_llava_next_model(SimpleNamespace(config=config))

    config = _cached_config()
    config.vision_config.hidden_size = 768
    with pytest.raises(ValueError, match="CLIP-L/14-336 signature"):
        compressor._validate_llava_next_model(SimpleNamespace(config=config))


class _FakeLanguageModel(nn.Module):
    def __init__(self, text_config) -> None:
        super().__init__()
        self.config = text_config
        self.embed_tokens = nn.Embedding(128320, 2)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(32)])
        self.norm = nn.Identity()
        self.rotary_emb = nn.Identity()
        self.gradient_checkpointing = False


class _FakeInnerNext(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.language_model = _FakeLanguageModel(config.text_config)


class _FakeOuterNext(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeInnerNext(config)


def test_install_uninstall_is_idempotent_and_guards_unsupported_generation() -> None:
    config = _cached_config()
    config.text_config._attn_implementation = "sdpa"
    outer = _FakeOuterNext(config)
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _token: 128256),
        patch_size=14,
        vision_feature_select_strategy="default",
        num_additional_image_tokens=1,
    )

    def prepare_inputs(images, prompts, *_args, **_kwargs):
        return images, prompts

    def generate(inputs, *_args, **_kwargs):
        return inputs

    vlm = SimpleNamespace(
        compressor_family="llava_next",
        _model_id=LLAVA_NEXT_8B_MODEL_ID,
        _model_revision=LLAVA_NEXT_8B_REVISION,
        _model=outer,
        _processor=processor,
        model=outer,
        processor=processor,
        prepare_inputs=prepare_inputs,
        generate=generate,
    )
    original_language_model = outer.model.language_model
    compressor = HiDropCompressor(
        backbone_family="llava_next", average_retained_tokens=64, pretrained=False
    )
    with pytest.warns(UserWarning, match="untrained base model"):
        compressor.install(vlm)
    compressor.install(vlm)

    assert isinstance(outer.model.language_model, HiDropLlamaModel)
    with pytest.raises(ValueError, match="batch_size=1"):
        vlm.prepare_inputs([torch.zeros(3, 2, 2)] * 2, ["a", "b"])
    with pytest.raises(NotImplementedError, match="beam search"):
        vlm.generate({"input_ids": torch.ones((1, 2), dtype=torch.long)}, num_beams=2)
    with pytest.raises(NotImplementedError, match="use_cache=True"):
        vlm.generate({"input_ids": torch.ones((1, 2), dtype=torch.long)}, use_cache=False)
    outer.generation_config = SimpleNamespace(num_beams=2, use_cache=True)
    with pytest.raises(NotImplementedError, match="beam search"):
        vlm.generate({"input_ids": torch.ones((1, 2), dtype=torch.long)})
    outer.generation_config = SimpleNamespace(num_beams=1, use_cache=False)
    with pytest.raises(NotImplementedError, match="use_cache=True"):
        vlm.generate({"input_ids": torch.ones((1, 2), dtype=torch.long)})

    compressor.uninstall()
    compressor.uninstall()
    assert outer.model.language_model is original_language_model
    assert vlm.prepare_inputs is prepare_inputs
    assert vlm.generate is generate
    assert compressor._vlm is None


def test_next_config_refuses_legacy_weights_and_wrong_backbone() -> None:
    # A LLaVA-NeXT checkpoint is released now, so `pretrained: true` is a valid
    # request rather than an error; it resolves per backbone at install time, and
    # a backbone with no release still fails there by name.
    HiDropCompressor(backbone_family="llava_next", pretrained=True)
    compressor = HiDropCompressor(backbone_family="llava_next", pretrained=False)
    with pytest.raises(TypeError, match="requires compressor_args.backbone_family"):
        LLaVANext8B().validate_compressor(HiDropCompressor(pretrained=False))
    backbone = LLaVANext8B()
    backbone.validate_compressor("HiDrop")
    backbone.validate_compressor(compressor)
    with pytest.raises(ValueError, match="pinned target model"):
        compressor.validate_llava_next_backbone(
            LLaVANext8B(model_name_or_path="different/llava-next")
        )


def test_checkpoint_metadata_rejects_a_different_model_revision(tmp_path: Path) -> None:
    config = HiDropConfig(backbone_family="llava_next", average_retained_tokens=64)
    metadata = config.hidrop_metadata()
    metadata["backbone_revision"] = "wrong-revision"
    (tmp_path / "hidrop_config.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned to model revision"):
        validate_hidrop_artifact(tmp_path, config, artifact_name="test checkpoint")


class _ChatTokenizer:
    pad_token_id = 0

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return {
            "<|start_header_id|>": 10,
            "<|end_header_id|>": 11,
            "<|eot_id|>": 12,
            "<image>": 99,
        }.get(token, -1)

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return {"assistant": [20], "user": [21]}.get(text, [30])


def test_llama3_labels_supervise_only_assistant_content_and_eot() -> None:
    ids = torch.tensor([10, 21, 11, 31, 12, 10, 20, 11, 41, 42, 12, 99, 99])
    labels = _assistant_labels(ids, _ChatTokenizer(), 99)

    assert torch.equal(labels[8:11], torch.tensor([41, 42, 12]))
    assert bool((labels[:8] == IGNORE_INDEX).all())
    assert bool((labels[11:] == IGNORE_INDEX).all())


def test_collator_preserves_the_turn_and_text_position_of_the_image_marker() -> None:
    captured: list[list[dict]] = []

    class _Processor:
        image_token = "<image>"
        tokenizer = _ChatTokenizer()

        @staticmethod
        def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool):
            assert not tokenize and not add_generation_prompt
            captured.append(messages)
            return "rendered"

        @staticmethod
        def __call__(**_kwargs):
            ids = torch.tensor([[10, 21, 11, 31, 12, 10, 20, 11, 41, 12, 99, 99]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros((1, 1, 3, 2, 2)),
                "image_sizes": torch.tensor([[2, 2]]),
            }

    item = {
        "conversations": [
            {"role": "user", "value": "first"},
            {"role": "assistant", "value": "ok"},
            {"role": "user", "value": "look <image> now"},
            {"role": "assistant", "value": "answer"},
        ],
        "image": Image.new("RGB", (2, 2)),
    }
    result = LlavaNextHiDropCollator(_Processor())([item])

    assert captured[0][0]["content"] == [{"type": "text", "text": "first"}]
    assert captured[0][2]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image"},
        {"type": "text", "text": "now"},
    ]
    assert result["labels"][0, 8] == 41
    assert bool((result["labels"][result["input_ids"] == 99] == IGNORE_INDEX).all())


def test_text_only_collator_uses_dummy_pixels_without_visual_placeholders() -> None:
    captured: dict[str, object] = {}

    class _Processor:
        image_token = "<image>"
        tokenizer = _ChatTokenizer()

        @staticmethod
        def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool):
            assert not tokenize and not add_generation_prompt
            captured["messages"] = messages
            return "rendered"

        @staticmethod
        def __call__(**kwargs):
            image = kwargs["images"]
            assert isinstance(image, Image.Image)
            captured["dummy_size"] = image.size
            captured["dummy_pixel"] = image.getpixel((0, 0))
            ids = torch.tensor([[10, 21, 11, 31, 12, 10, 20, 11, 41, 12]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros((1, 3, 3, 336, 336)),
                "image_sizes": torch.tensor([[336, 336]]),
            }

    item = {
        "conversations": [
            {"role": "user", "value": "say hello"},
            {"role": "assistant", "value": "hello"},
        ],
        "image": None,
    }
    result = LlavaNextHiDropCollator(_Processor())([item])

    assert captured["dummy_size"] == (336, 336)
    assert captured["dummy_pixel"] == (0, 0, 0)
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert all(part["type"] == "text" for message in messages for part in message["content"])
    assert not bool((result["input_ids"] == 99).any())
    assert result["labels"][0, 8] == 41


def test_local_dataset_is_lazy_and_reads_embedded_image_rows() -> None:
    if not (DATA_ROOT / "data").is_dir():
        pytest.skip("LLaVA-NeXT-Data is unavailable.")
    dataset = LlavaNextParquetDataset(DATA_ROOT, dataset_limit=1)
    row = dataset[0]

    assert len(dataset) == 1
    assert len(dataset.shards) == 250
    assert row["image"].mode == "RGB"
    assert row["conversations"][0]["role"] == "user"
    assert any(turn["role"] == "assistant" for turn in row["conversations"])


def test_local_dataset_preserves_released_text_only_rows() -> None:
    if not (DATA_ROOT / "data").is_dir():
        pytest.skip("LLaVA-NeXT-Data is unavailable.")
    # The complete annotation audit found the first text-only row at this exact
    # global index; preserving it proves that None is not treated as a missing
    # extracted image path.
    dataset = LlavaNextParquetDataset(DATA_ROOT, dataset_limit=738_602)
    row = dataset[738_601]

    assert row["id"] == "i6IyJda_0"
    assert row["image"] is None
    assert sum(turn["value"].count("<image>") for turn in row["conversations"]) == 0


def test_cached_processor_placeholder_expansion_equals_independent_layout() -> None:
    if not CACHE_SNAPSHOT.is_dir():
        pytest.skip("Pinned LLaVA-NeXT processor cache is unavailable.")
    processor = AutoProcessor.from_pretrained(CACHE_SNAPSHOT, local_files_only=True)
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "describe"}],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    encoded = processor(
        images=Image.new("RGB", (500, 500), "white"), text=prompt, return_tensors="pt"
    )
    layout = _layout((500, 500))
    image_id = int(processor.tokenizer.convert_tokens_to_ids("<image>"))

    assert int((encoded["input_ids"] == image_id).sum()) == layout.native_tokens
    assert int(encoded["pixel_values"].shape[1]) == layout.crop_count
    assert encoded["image_sizes"].tolist() == [[500, 500]]


def test_visualization_splits_base_and_unpadded_local_sources_without_newlines() -> None:
    layout = _synthetic_layout()
    processor = SimpleNamespace(image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0])
    pixels = torch.rand((1, 3, 3, 28, 28))
    views = llava_next_visualization_views(pixels, processor, layout)

    assert [view.name for view in views] == ["base", "local_mosaic"]
    assert views[0].canvas.shape == (3, 28, 28)
    assert views[1].canvas.shape == (3, 28, 56)
    assert views[0].source_columns.tolist() == list(layout.global_indices)
    assert views[1].source_columns.tolist() == list(layout.local_indices)


def test_tiny_llama3_training_wrapper_runs_native_packing_pruning_and_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vision = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    text = LlamaConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = LlavaNextConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=99,
        image_grid_pinpoints=[[8, 8]],
    )
    base_model = LlavaNextForConditionalGeneration(config)
    base_model.model.language_model.config._attn_implementation = "sdpa"
    packed = torch.randn(36, 8, requires_grad=True)
    base_model.model.get_image_features = lambda **_kwargs: BaseModelOutputWithPooling(
        pooler_output=[packed]
    )
    monkeypatch.setattr(HiDropCompressor, "_validate_llava_next_model", lambda *_args: None)
    wrapper = HiDropTrainingLlavaNextForConditionalGeneration(
        base_model,
        HiDropConfig(backbone_family="llava_next", average_retained_tokens=64),
    )
    input_ids = torch.tensor([[1, *([99] * 36), 5, 6]])
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, -1] = 6
    output = wrapper(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        pixel_values=torch.zeros((1, 2, 3, 8, 8)),
        image_sizes=torch.tensor([[8, 8]]),
    )

    assert output.logits.shape == (1, 3, 128)
    assert output.image_hidden_states.shape == (36, 8)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert [event["layer"] for event in wrapper.compressor.last_context.pruning_events] == [
        10,
        14,
        16,
        18,
    ]
    assert wrapper.compressor.last_context.active_visual_count == 0
    assert wrapper.compressor.last_visual_token_counts_by_layer[8] == 36
    assert wrapper.compressor.last_visual_token_counts_by_layer[24] == 0

    # Text-only samples still execute dummy vision/projector features, but an
    # empty slice makes their contribution exactly zero and leaves the native
    # Llama sequence untouched. The zero gradient remains connected to every
    # projected row, which is required for distributed stage-1 graph parity.
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    text_ids = torch.tensor([[1, 5, 6]])
    text_labels = torch.full_like(text_ids, IGNORE_INDEX)
    text_labels[0, -1] = 6
    text_output = wrapper(
        input_ids=text_ids,
        attention_mask=torch.ones_like(text_ids),
        labels=text_labels,
        pixel_values=torch.zeros((1, 2, 3, 8, 8)),
        image_sizes=torch.tensor([[8, 8]]),
    )
    assert text_output.logits.shape == (1, 3, 128)
    assert text_output.loss is not None and torch.isfinite(text_output.loss)
    text_output.loss.backward()
    assert packed.grad is not None
    assert torch.count_nonzero(packed.grad) == 0
    assert wrapper.compressor.last_context is None
    assert wrapper.compressor._pending_llava_next_layouts == ()


def test_training_stage_trainability_is_projector_only_then_lm_plus_projector() -> None:
    class _TinyTrainingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.model = nn.Module()
            self.base_model.model.vision_tower = nn.Linear(2, 2)
            self.base_model.model.multi_modal_projector = nn.Linear(2, 2)
            self.base_model.model.language_model = nn.Linear(2, 2)

    model = _TinyTrainingModel()
    configure_llava_next_trainable_parameters(model, "pretrain")
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert all("multi_modal_projector" in name for name in trainable)

    configure_llava_next_trainable_parameters(model, "finetune")
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert any("language_model" in name for name in trainable)
    assert any("multi_modal_projector" in name for name in trainable)
    assert not any("vision_tower" in name for name in trainable)


def test_next_trainer_step_checkpoint_uses_native_save_and_processor(tmp_path: Path) -> None:
    class _CheckpointModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.hidrop_config = HiDropConfig(
                backbone_family="llava_next",
                average_retained_tokens=64,
            )
            self.saved_keys: set[str] | None = None

        def forward(self, input_ids=None, labels=None):
            del input_ids, labels
            return {"loss": self.weight.sum()}

        def save_pretrained(self, output_dir, *, state_dict, safe_serialization):
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            self.saved_keys = set(state_dict)
            (output / "native_model_save.txt").write_text(
                str(bool(safe_serialization)), encoding="utf-8"
            )

    class _Processor:
        @staticmethod
        def save_pretrained(output_dir) -> None:
            (Path(output_dir) / "processor_config.json").write_text("{}", encoding="utf-8")

    class _Collator:
        processor = _Processor()

        def __call__(self, rows):
            return rows

    model = _CheckpointModel()
    trainer = HiDropTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            use_cpu=True,
            report_to=[],
        ),
        data_collator=_Collator(),
    )
    checkpoint = tmp_path / "checkpoint-1"
    trainer._save(str(checkpoint), state_dict=model.state_dict())

    assert model.saved_keys == {"weight"}
    assert (checkpoint / "native_model_save.txt").read_text(encoding="utf-8") == "True"
    assert (checkpoint / "processor_config.json").is_file()
    assert (checkpoint / "training_args.bin").is_file()
    metadata = json.loads((checkpoint / "hidrop_config.json").read_text(encoding="utf-8"))
    assert metadata["training_stage"] == "finetune"


def test_backbone_registration_revision_and_loader_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = resolve_backbone("llava-next-8b", {})
    assert BACKBONES["llava-next-8b"] is LLaVANext8B
    assert backbone._model_id == LLAVA_NEXT_8B_MODEL_ID
    assert backbone._model_revision == LLAVA_NEXT_8B_REVISION
    assert is_pinned_llava_next_snapshot(CACHE_SNAPSHOT)
    assert not is_pinned_llava_next_snapshot(CACHE_SNAPSHOT.parent / "not-the-revision")
    images = torch.randn(2, 3, 8, 12)
    processed = LLaVANextImagePreprocessor().forward(SimpleNamespace(), images)
    assert len(processed) == 2
    with pytest.raises(ValueError, match="Expected 4 dimensions"):
        LLaVANextImagePreprocessor().forward(SimpleNamespace(), torch.randn(1, 2, 3, 8, 12))
    frames = [torch.randn(2, 3, 8, 12)]
    assert LLaVANextImagePreprocessor().forward(SimpleNamespace(), frames) == frames

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

    def load_processor(model_id: str, **kwargs):
        calls["processor"] = (model_id, kwargs)
        return processor

    class _Loader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls["model"] = (model_id, kwargs)
            return model

    monkeypatch.setattr(backbone_module.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr("transformers.LlavaNextForConditionalGeneration", _Loader)
    loaded_processor, loaded_model = LLaVANext8B(device=torch.device("cpu")).load_model()

    assert loaded_processor is processor and loaded_model is model
    assert calls["processor"][1]["revision"] == LLAVA_NEXT_8B_REVISION
    assert calls["model"][1]["revision"] == LLAVA_NEXT_8B_REVISION
    assert calls["model"][1]["attn_implementation"] == "sdpa"
    assert processor.patch_size == 14
    assert processor.vision_feature_select_strategy == "default"
    assert processor.num_additional_image_tokens == 1


def test_training_is_fa2_only_while_inference_remains_sdpa() -> None:
    assert _require_flash_attention_2("flash_attention_2") == "flash_attention_2"
    with pytest.raises(ValueError, match="FlashAttention-2 only"):
        _require_flash_attention_2("sdpa")
    with pytest.raises(ValueError, match="FlashAttention-2 only"):
        load_hf_llava_next_training_base(
            LLAVA_NEXT_8B_MODEL_ID,
            model_revision=LLAVA_NEXT_8B_REVISION,
            dtype=None,
            attn_implementation="sdpa",
            local_files_only=True,
        )


def test_fixed_training_runtime_rejects_the_wrong_fa2_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        **provenance_module.FIXED_RUNTIME_RELEASES,
        "deepspeed": "0.18.7",
        "flash-attn": "2.8.2",
    }
    monkeypatch.setattr(
        provenance_module,
        "_distribution_version",
        lambda package: versions.get(package),
    )
    monkeypatch.setattr(provenance_module.torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="flash-attn==2.8.3.post1"):
        provenance_module.validate_fixed_training_environment(require_distributed_dependencies=True)

    versions["flash-attn"] = "2.8.3.post1"
    resolved = provenance_module.validate_fixed_training_environment(
        require_distributed_dependencies=True
    )
    assert resolved["flash-attn"] == "2.8.3.post1"
    assert (
        provenance_module.LLAVA_NEXT_TRAINING_SEMANTICS["attention_backend"]
        == "flash_attention_2_only"
    )
