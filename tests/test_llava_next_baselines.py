"""Canonical training-free baseline contracts for LLaVA-NeXT."""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    BatchFeature,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from llmoe.backbone.llava_next import LLaVANext8B
from llmoe.compressor.apet import ApETCompressor
from llmoe.compressor.apet.apet_llava_next import (
    layer_average_schedule as apet_layer_average_schedule,
)
from llmoe.compressor.divprune import DivPruneCompressor
from llmoe.compressor.fastv import FastVCompressor
from llmoe.compressor.llava_next import (
    layer_average_keep,
    retained_tokens,
)
from llmoe.compressor.must_drop import MustDropCompressor
from llmoe.compressor.must_drop.must_drop_llava_next import (
    layer_average_schedule as mustdrop_layer_average_schedule,
)
from llmoe.compressor.nuwa import NuwaCompressor
from llmoe.compressor.nuwa.nuwa_llava_next import layer_average_schedule
from llmoe.compressor.otprune import OTPruneCompressor
from llmoe.compressor.pyramiddrop import PyramidDropCompressor
from llmoe.compressor.sparse_vlm import SparseVLMCompressor
from llmoe.compressor.sparse_vlm.sparse_vlm_llava_next import (
    layer_average_schedule as sparsevlm_layer_average_schedule,
)
from llmoe.compressor.tome import ToMeCompressor
from llmoe.compressor.vflowopt import VFlowOptCompressor
from llmoe.compressor.vflowopt.vflowopt_llava_next import (
    layer_average_schedule as vflowopt_layer_average_schedule,
)
from llmoe.compressor.vision_zip import VisionZipCompressor

CompressorFactory = Callable[
    [],
    ApETCompressor
    | DivPruneCompressor
    | FastVCompressor
    | MustDropCompressor
    | NuwaCompressor
    | OTPruneCompressor
    | PyramidDropCompressor
    | SparseVLMCompressor
    | ToMeCompressor
    | VFlowOptCompressor
    | VisionZipCompressor,
]


def _ragged_vlm() -> tuple[SimpleNamespace, object, object]:
    sequences = [torch.randn(8, 6), torch.randn(5, 6)]

    def get_image_features(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(pooler_output=[sequence.clone() for sequence in sequences])

    inner = SimpleNamespace(get_image_features=get_image_features)
    model = SimpleNamespace(
        model=inner,
        config=SimpleNamespace(model_type="llava_next", image_token_index=99),
    )

    def prepare_inputs(*_args: object, **_kwargs: object) -> BatchFeature:
        first = [1, *([99] * 8), 2]
        second = [0, 0, 0, 1, *([99] * 5), 2]
        input_ids = torch.tensor([first, second])
        return BatchFeature(
            {
                "input_ids": input_ids,
                "attention_mask": torch.tensor([[1] * 10, [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]),
            }
        )

    vlm = SimpleNamespace(
        compressor_family="llava_next",
        model=model,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        prepare_inputs=prepare_inputs,
    )
    return vlm, get_image_features, prepare_inputs


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: DivPruneCompressor(retain_ratio=0.5, track_source=True),
            id="DivPrune",
        ),
        pytest.param(
            lambda: OTPruneCompressor(retain_ratio=0.5, track_source=True),
            id="OTPrune",
        ),
    ],
)
def test_static_selectors_share_one_ragged_packed_span_adapter(
    factory: CompressorFactory,
) -> None:
    torch.manual_seed(0)
    vlm, original_get_image_features, original_prepare_inputs = _ragged_vlm()
    compressor = factory()

    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        assert [int((row == 99).sum()) for row in inputs["input_ids"]] == [4, 2]

        outputs = vlm.model.model.get_image_features()
        assert [int(sequence.shape[0]) for sequence in outputs.pooler_output] == [4, 2]
        history = compressor.get_source_history()
        assert [tuple(source.shape) for source in history] == [(8, 8), (4, 8)]
    finally:
        compressor.uninstall()

    assert vlm.model.model.get_image_features is original_get_image_features
    assert vlm.prepare_inputs is original_prepare_inputs


def test_retain_ratio_has_one_floor_based_meaning() -> None:
    assert retained_tokens(8, 0.5) == 4
    assert retained_tokens(5, 0.5) == 2
    assert retained_tokens(3, 0.01) == 1


def test_decoder_keep_respects_the_same_layer_average_ratio() -> None:
    keep = layer_average_keep(
        100, retain_ratio=0.4, full_layers=3, decoder_layers=32, label="FastV"
    )
    achieved = (3 * 100 + 29 * keep) / (32 * 100)
    one_token_step = 29 / (32 * 100)

    assert keep == 33
    assert achieved <= 0.4
    assert 0.4 - achieved < one_token_step


def test_backbone_accepts_only_canonical_supported_configs() -> None:
    backbone = LLaVANext8B()
    backbone.validate_compressor(DivPruneCompressor(retain_ratio=0.25))
    backbone.validate_compressor(OTPruneCompressor(retain_ratio=0.25))

    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(DivPruneCompressor())
    with pytest.raises(ValueError, match="only accepts retain_ratio"):
        backbone.validate_compressor(DivPruneCompressor(max_tokens=32, retain_ratio=0.25))
    with pytest.raises(ValueError, match="fixes gamma"):
        backbone.validate_compressor(OTPruneCompressor(gamma=0.1, retain_ratio=0.25))
    backbone.validate_compressor(FastVCompressor(retain_ratio=0.25))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(FastVCompressor())
    # An explicit boundary is a deliberate, logged adaptation (the NeXT sweep's
    # retain_ratio 0.0556 is only reachable with k=1).
    backbone.validate_compressor(FastVCompressor(k=1, retain_ratio=0.0556))

    backbone.validate_compressor(PyramidDropCompressor(retain_ratio=0.25))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(PyramidDropCompressor())
    with pytest.raises(ValueError, match="only accepts retain_ratio"):
        backbone.validate_compressor(PyramidDropCompressor(max_tokens=64, retain_ratio=0.25))

    backbone.validate_compressor(VisionZipCompressor(retain_ratio=0.5))
    backbone.validate_compressor(VisionZipCompressor(retain_ratio=0.01))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(VisionZipCompressor())
    with pytest.raises(ValueError, match="split internally"):
        backbone.validate_compressor(VisionZipCompressor(retain_ratio=0.5, contextual_ratio=0.1))

    backbone.validate_compressor(NuwaCompressor(retain_ratio=0.25))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(NuwaCompressor())
    with pytest.raises(ValueError, match="only accepts retain_ratio"):
        backbone.validate_compressor(NuwaCompressor(max_tokens=32, retain_ratio=0.25))
    with pytest.raises(ValueError, match="only retain_ratio is configurable"):
        backbone.validate_compressor(NuwaCompressor(retain_ratio=0.25, distance=100))
    with pytest.raises(ValueError, match="must be <= 0.55"):
        backbone.validate_compressor(NuwaCompressor(retain_ratio=0.75))

    backbone.validate_compressor(ApETCompressor(retain_ratio=0.5))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(ApETCompressor(budget=192))
    with pytest.raises(ValueError, match="only retain_ratio is configurable"):
        backbone.validate_compressor(ApETCompressor(retain_ratio=0.5, basis_tokens=8))
    with pytest.raises(ValueError, match="must be <= 2/3"):
        backbone.validate_compressor(ApETCompressor(retain_ratio=0.8))

    backbone.validate_compressor(SparseVLMCompressor(retain_ratio=0.5))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(SparseVLMCompressor())
    with pytest.raises(ValueError, match="only accepts retain_ratio"):
        backbone.validate_compressor(SparseVLMCompressor(max_tokens=64, retain_ratio=0.5))
    with pytest.raises(ValueError, match="only retain_ratio and pruning_layers are configurable"):
        backbone.validate_compressor(SparseVLMCompressor(retain_ratio=0.5, recycling=False))
    with pytest.raises(ValueError, match="profile's maximum"):
        backbone.validate_compressor(SparseVLMCompressor(retain_ratio=0.8))
    # Canonical (2, 6, 15) cannot reach 0.0556; the first cut a layer earlier can.
    with pytest.raises(ValueError):
        backbone.validate_compressor(SparseVLMCompressor(retain_ratio=0.0556))
    backbone.validate_compressor(SparseVLMCompressor(retain_ratio=0.0556, pruning_layers=[0, 6, 15]))

    backbone.validate_compressor(ToMeCompressor(retain_ratio=0.5))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(ToMeCompressor())
    with pytest.raises(ValueError, match="only retain_ratio is configurable"):
        backbone.validate_compressor(ToMeCompressor(retain_ratio=0.5, prop_attention=False))
    backbone.validate_compressor(ToMeCompressor(retain_ratio=0.0556, schedule="front_loaded"))

    backbone.validate_compressor(MustDropCompressor(retain_ratio=0.5))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(MustDropCompressor())
    with pytest.raises(ValueError, match="only retain_ratio is configurable"):
        backbone.validate_compressor(MustDropCompressor(retain_ratio=0.5, window_size=2))

    backbone.validate_compressor(VFlowOptCompressor(retain_ratio=0.5, verbose=False))
    with pytest.raises(ValueError, match="requires retain_ratio"):
        backbone.validate_compressor(VFlowOptCompressor(max_tokens=128, verbose=False))
    with pytest.raises(ValueError, match="fixes the entropy weighting"):
        backbone.validate_compressor(
            VFlowOptCompressor(retain_ratio=0.5, entropy_mode="raw", verbose=False)
        )
    # A profile calibrated for this backbone is accepted; the canonical
    # LLaVA-OneVision one is infeasible below ~0.22 retention, so a searched
    # strategy is the only way to reach the tighter operating points.
    backbone.validate_compressor(
        VFlowOptCompressor(retain_ratio=0.5, verbose=False,
                           strategy={"r1": 0.6, "r2": 0.8, "r3": 0.7, "t": 20.0, "a": 20.0})
    )
    # ... but it must be a NeXT profile, not a fixed-grid LLaVA-1.5 one.
    with pytest.raises(ValueError, match="needs"):
        backbone.validate_compressor(
            VFlowOptCompressor(retain_ratio=0.5, verbose=False, strategy={"t": 20.0})
        )


@pytest.mark.parametrize(
    ("factory", "decoder_layers", "image_tokens", "output_length"),
    [
        pytest.param(lambda: DivPruneCompressor(retain_ratio=0.5), 2, 4, 6, id="DivPrune"),
        pytest.param(lambda: OTPruneCompressor(retain_ratio=0.5), 2, 4, 6, id="OTPrune"),
        pytest.param(lambda: FastVCompressor(retain_ratio=0.8), 4, 8, 3, id="FastV"),
        pytest.param(lambda: PyramidDropCompressor(retain_ratio=0.8), 4, 8, 6, id="PyramidDrop"),
    ],
)
def test_adapters_satisfy_transformers_placeholder_and_decoder_contracts(
    factory: CompressorFactory,
    decoder_layers: int,
    image_tokens: int,
    output_length: int,
) -> None:
    vision_config = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=8,
        patch_size=2,
    )
    text_config = LlamaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=decoder_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = LlavaNextConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=99,
        image_grid_pinpoints=[[8, 8]],
    )
    model = LlavaNextForConditionalGeneration(config).eval()
    packed_features = torch.randn(8, text_config.hidden_size)

    def get_image_features(*_args: object, **_kwargs: object) -> BaseModelOutputWithPooling:
        return BaseModelOutputWithPooling(pooler_output=[packed_features])

    model.model.get_image_features = get_image_features

    def prepare_inputs(*_args: object, **_kwargs: object) -> BatchFeature:
        input_ids = torch.tensor([[1, *([99] * 8), 2]])
        return BatchFeature({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})

    vlm = SimpleNamespace(
        compressor_family="llava_next",
        model=model,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        prepare_inputs=prepare_inputs,
    )
    compressor = factory()
    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        outputs = model(
            **inputs,
            pixel_values=torch.zeros(1, 1, 3, 8, 8),
            image_sizes=torch.tensor([[8, 8]]),
            use_cache=False,
        )
    finally:
        compressor.uninstall()

    assert tuple(outputs.image_hidden_states.shape) == (image_tokens, text_config.hidden_size)
    assert tuple(outputs.logits.shape) == (1, output_length, text_config.vocab_size)


def _native_crop_vlm(
    *,
    image_size: int,
    native_tokens: int,
    vision_layers: int,
    decoder_layers: int,
) -> tuple[LlavaNextForConditionalGeneration, SimpleNamespace]:
    vision_config = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=vision_layers,
        num_attention_heads=2,
        image_size=image_size,
        patch_size=2,
    )
    text_config = LlamaConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=decoder_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = LlavaNextConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=99,
        image_grid_pinpoints=[[image_size, image_size]],
    )
    model = LlavaNextForConditionalGeneration(config).eval()

    def prepare_inputs(*_args: object, **_kwargs: object) -> BatchFeature:
        input_ids = torch.tensor([[1, *([99] * native_tokens), 2]])
        return BatchFeature(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.randn(1, 2, 3, image_size, image_size),
                "image_sizes": torch.tensor([[image_size, image_size]]),
            }
        )

    vlm = SimpleNamespace(
        compressor_family="llava_next",
        model=model,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        prepare_inputs=prepare_inputs,
    )
    return model, vlm


def test_visionzip_runs_crop_local_before_the_projector_on_native_next() -> None:
    torch.manual_seed(0)
    model, vlm = _native_crop_vlm(
        image_size=8,
        native_tokens=36,
        vision_layers=2,
        decoder_layers=2,
    )
    original_get_image_features = model.model.get_image_features
    original_prepare_inputs = vlm.prepare_inputs
    compressor = VisionZipCompressor(retain_ratio=0.5, track_source=True)

    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        assert int((inputs["input_ids"] == 99).sum()) == 18
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        compressor.uninstall()

    assert tuple(outputs.image_hidden_states.shape) == (18, 8)
    assert tuple(outputs.logits.shape) == (1, 20, 128)
    assert [tuple(source.shape) for source in compressor.get_source_history()] == [
        (34, 34),
        (18, 34),
    ]
    assert model.model.get_image_features == original_get_image_features
    assert vlm.prepare_inputs is original_prepare_inputs


def test_tome_runs_key_matching_per_crop_on_native_next() -> None:
    torch.manual_seed(0)
    model, vlm = _native_crop_vlm(
        image_size=8,
        native_tokens=36,
        vision_layers=4,
        decoder_layers=2,
    )
    original_get_image_features = model.model.get_image_features
    original_prepare_inputs = vlm.prepare_inputs
    compressor = ToMeCompressor(retain_ratio=0.5, track_source=True)

    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        assert int((inputs["input_ids"] == 99).sum()) == 18
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        compressor.uninstall()

    assert tuple(outputs.image_hidden_states.shape) == (18, 8)
    assert tuple(outputs.logits.shape) == (1, 20, 128)
    history = compressor.get_source_history()
    assert [tuple(source.shape) for source in history] == [
        (36, 36),
        (18, 36),
    ]
    assert torch.equal(history[-1].sum(dim=0), torch.ones(36))
    assert not bool(history[-1][:7, 16:].any())
    assert not bool(history[-1][7:, :16].any())
    assert model.model.get_image_features == original_get_image_features
    assert vlm.prepare_inputs is original_prepare_inputs


def test_nuwa_native_next_runs_both_released_stages() -> None:
    assert layer_average_schedule(78, 0.25, 16, 17) == (20, 2)
    torch.manual_seed(0)
    model, vlm = _native_crop_vlm(
        image_size=12,
        native_tokens=78,
        vision_layers=24,
        decoder_layers=17,
    )
    original_get_image_features = model.model.get_image_features
    original_prepare_inputs = vlm.prepare_inputs
    compressor = NuwaCompressor(retain_ratio=0.25, track_source=True)

    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        assert int((inputs["input_ids"] == 99).sum()) == 20
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        compressor.uninstall()

    assert tuple(outputs.image_hidden_states.shape) == (20, 8)
    assert tuple(outputs.logits.shape) == (1, 4, 128)
    assert [tuple(source.shape) for source in compressor.get_source_history()] == [
        (78, 78),
        (20, 78),
        (2, 78),
    ]
    assert model.model.get_image_features == original_get_image_features
    assert vlm.prepare_inputs is original_prepare_inputs


def test_apet_native_next_runs_crop_local_and_decoder_stages() -> None:
    assert apet_layer_average_schedule(136, 0.5, 17) == (70, 23)
    torch.manual_seed(0)
    model, vlm = _native_crop_vlm(
        image_size=16,
        native_tokens=136,
        vision_layers=2,
        decoder_layers=17,
    )
    original_get_image_features = model.model.get_image_features
    original_prepare_inputs = vlm.prepare_inputs
    compressor = ApETCompressor(retain_ratio=0.5, track_source=True)

    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        assert int((inputs["input_ids"] == 99).sum()) == 70
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        compressor.uninstall()

    assert tuple(outputs.image_hidden_states.shape) == (70, 8)
    assert tuple(outputs.logits.shape) == (1, 25, 128)
    assert [tuple(source.shape) for source in compressor.get_source_history()] == [
        (136, 136),
        (70, 136),
        (23, 136),
    ]
    assert model.model.get_image_features == original_get_image_features
    assert vlm.prepare_inputs is original_prepare_inputs


def test_sparsevlm_native_next_runs_all_released_stages() -> None:
    schedule, counts = sparsevlm_layer_average_schedule(36, 0.5, 17)
    assert schedule == (17, 11, 6)
    assert counts == (18, 12, 7)
    torch.manual_seed(0)
    model, vlm = _native_crop_vlm(
        image_size=8,
        native_tokens=36,
        vision_layers=2,
        decoder_layers=17,
    )
    original_get_image_features = model.model.get_image_features
    original_prepare_inputs = vlm.prepare_inputs
    compressor = SparseVLMCompressor(retain_ratio=0.5, track_source=True)

    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        assert int((inputs["input_ids"] == 99).sum()) == 36
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        compressor.uninstall()

    assert tuple(outputs.image_hidden_states.shape) == (36, 8)
    assert tuple(outputs.logits.shape) == (1, 9, 128)
    assert [tuple(source.shape) for source in compressor.get_source_history()] == [
        (36, 36),
        (18, 36),
        (12, 36),
        (7, 36),
    ]
    assert model.model.get_image_features == original_get_image_features
    assert vlm.prepare_inputs is original_prepare_inputs


def test_mustdrop_native_next_runs_crop_local_lsm_and_four_decoder_cuts() -> None:
    assert mustdrop_layer_average_schedule(36, 0.5, 17, key_count=4) == (27, (22, 18, 14, 12))
    torch.manual_seed(0)
    model, vlm = _native_crop_vlm(
        image_size=8,
        native_tokens=36,
        vision_layers=2,
        decoder_layers=17,
    )
    original_get_image_features = model.model.get_image_features
    original_prepare_inputs = vlm.prepare_inputs
    compressor = MustDropCompressor(retain_ratio=0.5)

    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        assert int((inputs["input_ids"] == 99).sum()) == 27
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        compressor.uninstall()

    assert tuple(outputs.image_hidden_states.shape) == (27, 8)
    assert tuple(outputs.logits.shape) == (1, 14, 128)
    assert [tuple(source.shape) for source in compressor.get_source_history()] == [
        (36, 36),
        (27, 36),
        (22, 36),
        (18, 36),
        (14, 36),
        (12, 36),
    ]
    assert model.model.get_image_features == original_get_image_features
    assert vlm.prepare_inputs is original_prepare_inputs


def test_vflowopt_native_next_recycles_crops_and_runs_both_decoder_cuts() -> None:
    stages, counts = vflowopt_layer_average_schedule(36, 0.5, 17)
    assert (stages.l1, stages.l2, stages.l3) == (5, 5, 7)
    assert counts == (25, 17, 12)
    torch.manual_seed(0)
    model, vlm = _native_crop_vlm(
        image_size=8,
        native_tokens=36,
        vision_layers=2,
        decoder_layers=17,
    )
    original_get_image_features = model.model.get_image_features
    original_prepare_inputs = vlm.prepare_inputs
    compressor = VFlowOptCompressor(
        retain_ratio=0.5,
        track_source=True,
        verbose=False,
    )

    try:
        compressor.install(vlm)
        inputs = vlm.prepare_inputs(None, None)
        assert int((inputs["input_ids"] == 99).sum()) == 25
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        compressor.uninstall()

    assert tuple(outputs.image_hidden_states.shape) == (25, 8)
    assert tuple(outputs.logits.shape) == (1, 14, 128)
    assert [tuple(source.shape) for source in compressor.get_source_history()] == [
        (36, 36),
        (25, 36),
        (17, 36),
        (12, 36),
    ]
    assert model.model.get_image_features == original_get_image_features
    assert vlm.prepare_inputs is original_prepare_inputs
