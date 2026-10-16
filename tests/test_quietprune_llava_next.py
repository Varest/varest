"""Tiny native-Transformers tests for QuietPrune's AnyRes token lifecycle."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    BatchFeature,
    CLIPVisionConfig,
    GenerationConfig,
    LlamaConfig,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
)

from llmoe.backbone.llava_next import LLAVA_NEXT_8B_MODEL_ID, LLAVA_NEXT_8B_REVISION
from llmoe.compressor.quietprune import QuietPruneCompressor
from llmoe.compressor.quietprune.quietprune import build_plan
from llmoe.compressor.quietprune.quietprune_next import (
    QUIETPRUNE_NEXT_FLASH_ATTN_VERSION,
    QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION,
    _aggregate_dropped,
    _trim_one_visual_run,
    build_anyres_layout,
    layer_pruning_rate_for_average,
    llava_next_visualization_views,
    pack_quietprune_features,
    plan_for_content_retention,
    target_content_tokens,
)
from llmoe.viz import make_visualization


def _tiny_model(*, image_size: int = 8, vision_layers: int = 4, decoder_layers: int = 3):
    vision = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=vision_layers,
        num_attention_heads=2,
        image_size=image_size,
        patch_size=2,
    )
    text = LlamaConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=decoder_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=256,
        rope_theta=500000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = LlavaNextConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=99,
        image_grid_pinpoints=[[image_size, image_size]],
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
    )
    return LlavaNextForConditionalGeneration(config).eval()


def _tiny_vlm():
    torch.manual_seed(3)
    model = _tiny_model()
    native_tokens = 36  # 16 base + 4x(4 local + newline)

    def prepare_inputs(*_args, **_kwargs):
        input_ids = torch.tensor([[1, *([99] * native_tokens), 3, 4]])
        return BatchFeature(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.randn(1, 2, 3, 8, 8),
                "image_sizes": torch.tensor([[8, 8]]),
            }
        )

    def generate(inputs, **kwargs):
        return model.generate(**inputs, **kwargs)

    vlm = SimpleNamespace(
        compressor_family="llava_next",
        model=model,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        prepare_inputs=prepare_inputs,
        generate=generate,
        _model_id=LLAVA_NEXT_8B_MODEL_ID,
        _model_revision=LLAVA_NEXT_8B_REVISION,
    )
    return model, vlm


def _canonical_training_metadata() -> dict[str, object]:
    return {
        "stage": "tr",
        "recipe": "adapter_only_self_distillation_kl_plus_answer_ce",
        "attention_implementation": QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION,
        "effective_text_attention_implementation": (QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION),
        "effective_vision_attention_implementation": (QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION),
        "flash_attn_version": QUIETPRUNE_NEXT_FLASH_ATTN_VERSION,
    }


def _training_compressor(rate: float = 0.25, **kwargs) -> QuietPruneCompressor:
    compressor = QuietPruneCompressor(pruning_rate=rate, pretrained=False, **kwargs)
    compressor._next_training_rate_mode = True
    return compressor


def test_corrected_rstar_schedule_does_not_invert_pruning_meaning() -> None:
    rate = layer_pruning_rate_for_average(0.50, 3)
    assert rate == pytest.approx(0.2062994740159002)
    assert 1.0 - (1.0 - rate) ** 3 == pytest.approx(0.50)

    evaluation = build_plan(
        (24, 24),
        group_size=2,
        pruning_layers=(6, 12, 18),
        pruning_rate=rate,
        keep_border_tokens=True,
        aggregate=True,
        include_cls=False,
    )
    assert evaluation.patches_after_layer == (460, 368, 296)
    assert 1.0 - evaluation.patches_after_layer[-1] / 576 == pytest.approx(35 / 72)

    training = build_plan(
        (24, 24),
        group_size=2,
        pruning_layers=(6, 12, 18),
        pruning_rate=0.25,
        keep_border_tokens=True,
        aggregate=True,
        include_cls=False,
    )
    assert training.patches_after_layer == (432, 324, 244)
    assert training.final_tokens == 245


def test_retention_target_uses_each_native_content_count_once_and_excludes_structure() -> None:
    assert target_content_tokens(1152, 0.25) == 288
    assert target_content_tokens(1176, 0.25) == 294
    assert target_content_tokens(3, 0.5) == 2  # Python round: 1.5 -> 2.

    plan, target, estimated = plan_for_content_retention(
        original_content_tokens=1152,
        retain_ratio=0.25,
        grid=(24, 24),
        group_size=2,
        pruning_layers=(6, 12, 18),
        keep_border_tokens=True,
        aggregate=True,
    )
    assert target == 288
    assert abs(estimated - target) <= 4
    # The 25% request is a final target, not 25% retention at every stage.
    assert plan.patches_after_layer[-1] > 576 * 0.25**3


def test_same_crop_count_gets_different_budget_from_true_unpadded_content() -> None:
    inner = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=336, patch_size=14),
            image_grid_pinpoints=[[336, 672], [672, 336]],
        )
    )
    square = build_anyres_layout(inner, torch.tensor([336, 336]), crop_count=3)
    landscape = build_anyres_layout(inner, torch.tensor([336, 672]), crop_count=3)

    assert square.crop_count == landscape.crop_count == 3
    assert square.original_content_tokens == 1152
    assert landscape.original_content_tokens == 1728
    assert square.native_length == square.original_content_tokens + square.structural_tokens
    assert landscape.native_length == (
        landscape.original_content_tokens + landscape.structural_tokens
    )
    assert target_content_tokens(square.original_content_tokens, 0.25) == 288
    assert target_content_tokens(landscape.original_content_tokens, 0.25) == 432


def test_paper_aggregate_is_unnormalized_and_ignores_unpadded_local_tokens() -> None:
    tokens = torch.tensor([[[2.0], [10.0]]])
    relevance = torch.tensor([[0.25, 0.50]])
    originals = torch.tensor([[0, 1]])
    validity = torch.tensor([[True, False]])

    aggregate = _aggregate_dropped(
        tokens,
        relevance,
        originals,
        validity,
        original_patches=2,
        normalize=False,
        eps=1e-6,
    )
    torch.testing.assert_close(aggregate.token, torch.tensor([[0.5]]))
    torch.testing.assert_close(aggregate.mass, torch.tensor([0.25]))
    torch.testing.assert_close(aggregate.source_mass, torch.tensor([[0.25, 0.0]]))


def test_rectangular_anyres_layout_matches_native_unpadding_and_order() -> None:
    inner = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=8, patch_size=2),
            image_grid_pinpoints=[[8, 16]],
        )
    )
    layout = build_anyres_layout(inner, torch.tensor([4, 16]), crop_count=3)

    assert layout.patch_grid == (4, 4)
    assert layout.tile_grid == (1, 2)
    assert layout.local_grid == (2, 8)
    assert layout.native_length == 34
    assert layout.newline_positions == (24, 33)
    assert int(layout.valid_patch_mask[0].sum()) == 16
    assert int(layout.valid_patch_mask[1:].sum()) == 16


def test_sparse_packer_keeps_base_then_mosaic_rows_newlines_and_aggregate() -> None:
    inner = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=8, patch_size=2),
            image_grid_pinpoints=[[8, 8]],
        )
    )
    layout = build_anyres_layout(inner, torch.tensor([8, 8]), crop_count=2)
    indices = torch.tensor([[0, 5, 10, 15], [0, 5, 10, 15]])
    tokens = torch.arange(2 * 4 * 2).reshape(2, 4, 2).float()
    aggregate_source = torch.zeros(1, layout.native_length)
    aggregate_source[0, 1] = 1.0
    packed = pack_quietprune_features(
        tokens,
        indices,
        layout=layout,
        image_newline=torch.tensor([99.0, 99.0]),
        aggregate=(torch.tensor([[77.0, 77.0]]), aggregate_source),
    )

    assert packed.native_positions.tolist() == [0, 5, 10, 15, 16, 20, 22, 25, 28, 30, 34, 35, -1]
    assert packed.tokens[5].tolist() == [99.0, 99.0]
    assert packed.tokens[-1].tolist() == [77.0, 77.0]
    assert packed.patch_tokens == 8
    assert packed.newline_tokens == 4
    assert packed.aggregate_tokens == 1
    torch.testing.assert_close(packed.source.sum(dim=1), torch.ones(13))


def test_identity_packer_is_exactly_native_transformers_anyres() -> None:
    model = _tiny_model()
    inner = model.model
    inner.config.image_grid_pinpoints = [[8, 16]]
    image_size = torch.tensor([4, 16])
    crops = 3
    patches = 16
    features = torch.arange(crops * patches * 8).reshape(crops, patches, 8).float()
    newline = torch.arange(8).float() + 10_000

    native, native_lengths = inner.pack_image_features(
        [features],
        image_size.unsqueeze(0),
        vision_feature_select_strategy="default",
        image_newline=newline,
    )
    layout = build_anyres_layout(inner, image_size, crop_count=crops)
    packed = pack_quietprune_features(
        features,
        torch.arange(patches).expand(crops, -1),
        layout=layout,
        image_newline=newline,
        aggregate=None,
    )

    assert int(native_lengths[0]) == layout.native_length
    torch.testing.assert_close(packed.tokens, native[0])
    torch.testing.assert_close(
        packed.native_positions,
        torch.arange(layout.native_length),
    )


def test_single_view_keeps_its_native_terminal_newline() -> None:
    inner = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=8, patch_size=2),
            image_grid_pinpoints=[[8, 8]],
        )
    )
    layout = build_anyres_layout(inner, torch.tensor([8, 8]), crop_count=1)
    packed = pack_quietprune_features(
        torch.randn(1, 2, 3),
        torch.tensor([[1, 7]]),
        layout=layout,
        image_newline=torch.ones(3),
        aggregate=None,
    )
    assert layout.native_length == 17
    assert packed.native_positions.tolist() == [1, 7, 16]
    assert packed.newline_tokens == 1


def test_anyres_visualization_exposes_base_and_unpadded_local_canvases() -> None:
    _model, vlm = _tiny_vlm()
    views = llava_next_visualization_views(
        vlm,
        vlm.prepare_inputs(),
        SimpleNamespace(image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0]),
    )

    assert [view.name for view in views] == ["base", "local"]
    assert [tuple(view.canvas.shape) for view in views] == [(3, 8, 8), (3, 8, 8)]
    assert views[0].source_columns.tolist() == list(range(16))
    assert views[1].source_columns.numel() == 16
    native_source = torch.eye(36)
    for view in views:
        source = native_source.index_select(1, view.source_columns)
        source = source[source.abs().sum(dim=1) > 0]
        rendered = make_visualization(view.canvas, source, patch_size=2)
        assert rendered.shape == view.canvas.shape


def test_trim_mutates_ids_masks_labels_and_positions_together() -> None:
    ids = torch.tensor([[1, 9, 9, 9, 9, 2, 3]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 2, 3]])
    prepared = _trim_one_visual_run(
        {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(ids.shape[1]).unsqueeze(0),
        },
        image_token_id=9,
        native_length=4,
        target_length=2,
        pad_token_id=0,
        labels=labels,
    )
    assert prepared["input_ids"].tolist() == [[1, 9, 9, 2, 3]]
    assert prepared["labels"].tolist() == [[-100, -100, -100, 2, 3]]
    assert prepared["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert prepared["position_ids"].tolist() == [[0, 1, 2, 3, 4]]


def test_tiny_native_model_prefill_decode_labels_sources_and_lifecycle() -> None:
    model, vlm = _tiny_vlm()
    original_get = model.model.get_image_features
    original_prepare = vlm.prepare_inputs
    original_generate = vlm.generate
    compressor = QuietPruneCompressor(retain_ratio=0.75, pretrained=False, track_source=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    try:
        compressor.install(vlm)
        compressor.train()
        teacher = original_prepare()
        labels = torch.full_like(teacher["input_ids"], -100)
        labels[:, -2:] = teacher["input_ids"][:, -2:]
        captured_masks: list[torch.Tensor] = []
        hook = compressor.adapter.register_forward_pre_hook(
            lambda _module, args: captured_masks.append(args[1].detach().clone())
        )
        student = compressor.prepare_training_inputs(teacher, labels)
        hook.remove()

        assert int((student["input_ids"] == 99).sum()) == 29
        assert student["labels"].shape == student["input_ids"].shape
        assert not bool(captured_masks[0][student["input_ids"].new_tensor([0]), -2:].any())
        assert not bool(captured_masks[0][0, 1:37].any())

        output = model(**student, use_cache=True)
        output.loss.backward()
        assert tuple(output.image_hidden_states.shape) == (29, 8)
        assert output.logits.shape[1] == student["input_ids"].shape[1]
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(parameter.grad is not None for parameter in compressor.adapter.parameters())

        cache = output.past_key_values
        prefill_len = student["input_ids"].shape[1]
        assert all(cache.get_seq_length(layer) == prefill_len for layer in range(3))
        attention_mask = torch.ones(1, prefill_len + 1, dtype=torch.long)
        decoded = model(
            input_ids=torch.tensor([[5]]),
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        assert all(
            decoded.past_key_values.get_seq_length(layer) == prefill_len + 1 for layer in range(3)
        )

        report = compressor.last_budget_report
        assert report is not None
        assert report["original_content_tokens"] == 32
        assert report["requested_retention_ratio"] == pytest.approx(0.75)
        assert report["target_content_tokens"] == 24
        assert report["realized_content_tokens"] == 25
        assert report["realized_retention_ratio"] == pytest.approx(25 / 32)
        assert report["structural_tokens"] == 4
        assert report["native_packed_tokens"] == 36
        assert report["stage_patch_tokens_all_views"] == (24, 24)
        assert report["realized_vit_patch_pruning_rate"] == pytest.approx(0.25)
        assert report["realized_packed_tokens"] == 29
        assert report["realized_decoder_pruning_rate"] == pytest.approx(7 / 36)
        assert report["protected_newline_tokens"] == 4
        assert [tuple(source.shape) for source in compressor.get_source_history()] == [
            (36, 36),
            (29, 36),
            (29, 36),
        ]
    finally:
        compressor.uninstall()

    assert model.model.get_image_features == original_get
    assert vlm.prepare_inputs is original_prepare
    assert vlm.generate is original_generate


def test_bfloat16_aggregate_is_cast_back_before_projection() -> None:
    model, vlm = _tiny_vlm()
    model.to(dtype=torch.bfloat16)
    original_prepare = vlm.prepare_inputs

    def prepare_bfloat16(*args, **kwargs):
        inputs = original_prepare(*args, **kwargs)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    vlm.prepare_inputs = prepare_bfloat16
    compressor = QuietPruneCompressor(retain_ratio=0.75, pretrained=False)
    try:
        compressor.install(vlm)
        prepared = vlm.prepare_inputs()
        assert compressor._next_adapter.pending_output.pooler_output[0].dtype == torch.bfloat16
        assert int((prepared["input_ids"] == 99).sum()) == 29
    finally:
        compressor.uninstall()


def test_generation_modes_and_dynamic_inputs_fail_closed() -> None:
    _model, vlm = _tiny_vlm()
    compressor = QuietPruneCompressor(retain_ratio=0.75, pretrained=False)
    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs()
        with pytest.raises(NotImplementedError, match="beam search"):
            vlm.generate(inputs, num_beams=2)
        with pytest.raises(NotImplementedError, match="num_return_sequences=1"):
            vlm.generate(inputs, num_return_sequences=2)
        with pytest.raises(NotImplementedError, match="use_cache=True"):
            vlm.generate(inputs, use_cache=False)
        with pytest.raises(NotImplementedError, match="beam search"):
            vlm.generate(inputs, generation_config=GenerationConfig(num_beams=2))
        with pytest.raises(NotImplementedError, match="pre-existing KV cache"):
            vlm.generate(inputs, past_key_values=object())
        compressor._next_adapter.pending_output = None

        batched = deepcopy(inputs)
        batched["input_ids"] = batched["input_ids"].expand(2, -1).clone()
        batched["attention_mask"] = batched["attention_mask"].expand(2, -1).clone()
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            compressor._next_adapter.prepare_training_inputs(
                batched, torch.full_like(batched["input_ids"], -100)
            )

        multi_image = deepcopy(inputs)
        multi_image["image_sizes"] = torch.tensor([[8, 8], [8, 8]])
        with pytest.raises(NotImplementedError, match="exactly one source image"):
            compressor._next_adapter.prepare_training_inputs(
                multi_image, torch.full_like(multi_image["input_ids"], -100)
            )

        bad = deepcopy(inputs)
        bad["pixel_values"] = torch.zeros(1, 1, 2, 3, 8, 8)
        with pytest.raises(NotImplementedError, match="video"):
            compressor._next_adapter.prepare_training_inputs(
                bad, torch.full_like(bad["input_ids"], -100)
            )

        extra_crop = deepcopy(inputs)
        extra_crop["pixel_values"] = torch.cat(
            [extra_crop["pixel_values"], extra_crop["pixel_values"][:, :1]], dim=1
        )
        with pytest.raises(ValueError, match="exactly 2 views"):
            compressor._next_adapter.prepare_training_inputs(
                extra_crop, torch.full_like(extra_crop["input_ids"], -100)
            )
    finally:
        compressor.uninstall()


def test_checkpoint_provenance_rejects_other_backbones_and_revisions(tmp_path) -> None:
    _model, vlm = _tiny_vlm()
    compressor = _training_compressor()
    try:
        compressor.install(vlm)
        path = tmp_path / "adapter.pt"
        compressor._training_metadata = _canonical_training_metadata()
        compressor.save_checkpoint(path)
        payload = torch.load(path, weights_only=True)
        assert payload["metadata"]["backbone"]["model_revision"] == LLAVA_NEXT_8B_REVISION
        assert payload["metadata"]["training"]["stage"] == "tr"
        compressor.load_checkpoint(path)

        wrong = deepcopy(payload)
        wrong["metadata"]["backbone"]["model_id"] = "llava-hf/llava-1.5-7b-hf"
        wrong_path = tmp_path / "wrong.pt"
        torch.save(wrong, wrong_path)
        with pytest.raises(ValueError, match="model_id"):
            compressor.load_checkpoint(wrong_path)

        wrong_geometry = deepcopy(payload)
        wrong_geometry["metadata"]["geometry"]["group_size"] = 4
        geometry_path = tmp_path / "wrong_geometry.pt"
        torch.save(wrong_geometry, geometry_path)
        with pytest.raises(ValueError, match="group geometry"):
            compressor.load_checkpoint(geometry_path)

        wrong_recipe = deepcopy(payload)
        wrong_recipe["metadata"]["schedule"]["per_stage_pruning_rate"] = 0.40
        recipe_path = tmp_path / "wrong_recipe.pt"
        torch.save(wrong_recipe, recipe_path)
        with pytest.raises(ValueError, match="canonical R=0.25"):
            compressor.load_checkpoint(recipe_path)

        wrong_attention = deepcopy(payload)
        wrong_attention["metadata"]["training"]["effective_vision_attention_implementation"] = (
            "sdpa"
        )
        attention_path = tmp_path / "wrong_attention.pt"
        torch.save(wrong_attention, attention_path)
        with pytest.raises(ValueError, match="FlashAttention 2"):
            compressor.load_checkpoint(attention_path)

        wrong_flash_version = deepcopy(payload)
        wrong_flash_version["metadata"]["training"]["flash_attn_version"] = "2.8.2"
        flash_version_path = tmp_path / "wrong_flash_version.pt"
        torch.save(wrong_flash_version, flash_version_path)
        with pytest.raises(ValueError, match="package version"):
            compressor.load_checkpoint(flash_version_path)

        legacy_path = tmp_path / "legacy.pt"
        torch.save({"format_version": 1, "state_dict": payload["state_dict"]}, legacy_path)
        with pytest.raises(ValueError, match="provenance"):
            compressor.load_checkpoint(legacy_path)
    finally:
        compressor.uninstall()


def test_checkpoint_adapter_is_budget_agnostic_but_backbone_specific(tmp_path) -> None:
    _model, vlm = _tiny_vlm()
    path = tmp_path / "adapter.pt"
    training = _training_compressor()
    try:
        training.install(vlm)
        training._training_metadata = _canonical_training_metadata()
        training.save_checkpoint(path)
    finally:
        training.uninstall()

    evaluation = QuietPruneCompressor(retain_ratio=0.50, checkpoint_path=path)
    try:
        evaluation.install(vlm)
        assert evaluation.budget == pytest.approx(0.50)
    finally:
        evaluation.uninstall()

    wrong_training_rate = _training_compressor(0.40)
    try:
        wrong_training_rate.install(vlm)
        wrong_training_rate._training_metadata = _canonical_training_metadata()
        with pytest.raises(ValueError, match="trained at R=0.25"):
            wrong_training_rate.checkpoint_dict()
    finally:
        wrong_training_rate.uninstall()


def test_next_rejects_fixed_token_budget_and_noncanonical_groups() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        QuietPruneCompressor(max_tokens=64, pretrained=False).validate_llava_next()
    with pytest.raises(ValueError, match="2x2"):
        QuietPruneCompressor(
            retain_ratio=0.25, group_size=3, pretrained=False
        ).validate_llava_next()
    with pytest.raises(ValueError, match="evaluation requires retain_ratio"):
        QuietPruneCompressor(pruning_rate=0.25, pretrained=False).validate_llava_next()
