"""LLaVA-1.5-13B artifact resolution.

Trained baselines were previously unusable on 13B: `ARTIFACTS` held only
`llava`-keyed entries, so a 13B run either failed or, worse, could be pointed at
4096-wide weights meant for a 5120-wide decoder. These tests pin the released
13B artifacts to their bucket paths and keep the resolver honest about the
methods that still have no 13B release.
"""

import pytest

from llmoe.backbone import BACKBONES
from llmoe.compressor.vflowopt.vflowopt import StageLayers
from llmoe.weights import ARTIFACTS, artifact_for, backbones_for

BUDGETS = (32, 64, 128, 224)


@pytest.mark.parametrize("method,expected", [
    ("HiDrop", ["llava", "llava-13b", "llava-next-8b"]),
    ("QuietPrune", ["llava", "llava-13b", "llava-next-8b", "qwen-vl"]),
    ("VisionZip", ["llava", "llava-13b"]),
    ("VFlowOpt", ["llava", "llava-13b"]),
    ("DUET", ["llava", "llava-13b", "llava-next-8b"]),
    ("CORE", ["core", "core-13b", "core-llama3-8b"]),
])
def test_released_backbones(method, expected):
    assert backbones_for(method) == expected


@pytest.mark.parametrize("budget", BUDGETS)
def test_thirteen_b_paths_are_backbone_scoped(budget):
    """A 13B artifact must never resolve to a 7B path."""
    for method in ("HiDrop", "VisionZip", "VFlowOpt"):
        path = artifact_for(method, budget, "llava-13b").path
        assert "llava-1.5-13b" in path, (method, path)
        assert "llava-1.5-7b" not in path, (method, path)


def test_quietprune_is_budget_independent_per_backbone():
    """One adapter serves every budget, but the adapters are not interchangeable."""
    seven = {artifact_for("QuietPrune", b, "llava").path for b in BUDGETS}
    thirteen = {artifact_for("QuietPrune", b, "llava-13b").path for b in BUDGETS}
    assert len(seven) == len(thirteen) == 1
    assert seven != thirteen


def test_core_13b_is_its_own_backbone():
    """CORE swaps only the decoder; its stage-2 release is decoder-shaped."""
    assert "core-13b" in BACKBONES
    assert BACKBONES["core-13b"].weights_key == "core-13b"
    assert "vicuna-13b-v1.5" in artifact_for("CORE", 1024, "core-13b").path
    assert "vicuna-7b-v1.5" in artifact_for("CORE", 1024, "core").path


def test_ambiguous_backbone_is_refused():
    """Two backbones with releases means the caller must say which."""
    with pytest.raises(KeyError, match="pass backbone="):
        artifact_for("HiDrop", 64)


def test_methods_without_a_13b_release_fail_loudly():
    """Falling through to the 7B weights would be an ablation wearing the label.

    Every trained method now has a 13B release, so this guards the *mechanism*
    with a backbone that has none rather than with a method.
    """
    # HiDrop now has a LLaVA-NeXT release, so the mechanism is guarded with a
    # backbone it has never been trained for.
    with pytest.raises(KeyError, match="qwen-vl"):
        artifact_for("HiDrop", 64, "qwen-vl")


@pytest.mark.parametrize("depth,expected,prune", [
    (28, (9, 9, 10), (9, 18)),    # LLaVA-OneVision-7B / Qwen2-VL-7B, per the paper
    (32, (10, 10, 12), (10, 20)),  # LLaVA-NeXT-7B and Vicuna-7B, per the paper
    (40, (13, 13, 14), (13, 26)),  # LLaVA-1.5-13B
])
def test_vflowopt_stage_split_follows_decoder_depth(depth, expected, prune):
    """Eq. 5 is depth-relative: this is why 13B needs its own calibration."""
    stages = StageLayers.even(depth)
    assert (stages.l1, stages.l2, stages.l3) == expected
    assert stages.prune_layers == prune
    assert stages.total == depth


def test_every_artifact_key_names_a_real_backbone():
    for method, backbone, _ in ARTIFACTS:
        assert backbone in BACKBONES, (method, backbone)


def test_core_compressor_selects_its_release_by_backbone():
    """Adding a second CORE release made the implicit resolution ambiguous.

    `CORECompressor` is built by `resolve_compressor` at CLI parse time, before
    any backbone exists, so it cannot infer the decoder. It used to call
    `resolve_weights` with no backbone, which worked only while CORE had exactly
    one release; a second one turned every CORE run into a KeyError.
    """
    from llmoe.compressor.core.core import CORECompressor

    seven = CORECompressor(max_tokens=32, pretrained=True)
    thirteen = CORECompressor(max_tokens=32, pretrained=True, backbone="core-13b")
    assert "vicuna-7b-v1.5" in str(seven._adapter_dir)
    assert "vicuna-13b-v1.5" in str(thirteen._adapter_dir)


def test_vflowopt_compressor_selects_its_strategy_by_backbone():
    """Same trap as CORE: a second release makes implicit resolution ambiguous.

    `VFlowOptCompressor` resolves its strategy in `__init__`, before any backbone
    exists. With releases for both `llava` and `llava-13b`, resolving with no
    backbone raises -- which would have broken `pretrained: true` on the 7B too,
    not just the new backbone.
    """
    from llmoe.compressor.vflowopt.vflowopt import VFlowOptCompressor

    seven = VFlowOptCompressor(max_tokens=64, pretrained=True)
    thirteen = VFlowOptCompressor(max_tokens=64, pretrained=True, backbone="llava-13b")
    assert seven.strategy is not None and thirteen.strategy is not None
    # 32 layers split 10/10/12; 40 split 13/13/14.
    assert seven.strategy.stages.prune_layers == (10, 20)
    assert thirteen.strategy.stages.prune_layers == (13, 26)


def test_core_13b_declares_its_decoder_width():
    """`CoreConfig.llm_hidden_size` defaults to the 7B's 4096.

    `load_causal_lm` checks that value against the decoder it just loaded, so a
    13B run that inherits the default dies with "Vicuna hidden size 5120 does not
    match CORE projector output 4096". The released 13B projector's fc2 is
    [5120, 5120] and its core_config records llm_hidden_size 5120.
    """
    from llmoe.compressor.core.core import CoreConfig, VICUNA_13B_HIDDEN_SIZE

    assert CoreConfig().llm_hidden_size == 4096
    assert VICUNA_13B_HIDDEN_SIZE == 5120


def test_core_prompt_preprocessor_reads_the_config_off_the_backbone():
    """`CorePromptPreprocessor.forward` read `self.core_config`, which it never had.

    `core_config` is a property of `CoreVLM`, and the preprocessor stores only the
    image token, so every CORE run raised `AttributeError: 'CorePromptPreprocessor'
    object has no attribute 'core_config'` at prompt formatting -- on any backbone,
    not just the 13B one. Introduced in #16 when the prompt gained an LLM family.
    """
    from types import SimpleNamespace

    from llmoe.compressor.core.core import CorePromptPreprocessor

    vlm = SimpleNamespace(core_config=SimpleNamespace(llm_family="vicuna"))
    out = CorePromptPreprocessor("<image>").forward(vlm, ["describe this"])
    assert len(out) == 1 and "describe this" in out[0]
    # The preprocessor itself must not be the one carrying the config.
    assert not hasattr(CorePromptPreprocessor("<image>"), "core_config")


@pytest.mark.parametrize("budget", BUDGETS)
def test_duet_13b_is_backbone_scoped(budget):
    """DUET's 13B checkpoints are compression-aware finetunes of the 40-layer
    decoder, not the 7B ones: `duet_config` records decoder_layers 40 and
    pdrop_layers [20, 30] against the 7B's [16, 24] over 32 layers.
    """
    thirteen = artifact_for("DUET", budget, "llava-13b").path
    seven = artifact_for("DUET", budget, "llava").path
    assert "llava-1.5-13b" in thirteen and "llava-1.5-7b" not in thirteen
    assert "llava-1.5-7b" in seven


def test_next_releases_are_backbone_scoped():
    """QuietPrune, CORE and VisionZip++ all have LLaVA-NeXT releases.

    Plain ``VisionZip`` does not, and must not borrow the ++ projector: on this
    backbone the two are different compressors, not one with a flag.
    """
    assert "llama3-llava-next-8b" in artifact_for("QuietPrune", 64, "llava-next-8b").path
    assert "llama3-llava-next-8b" in artifact_for("CORE", 1024, "core-llama3-8b").path
    assert "llama3-llava-next-8b" in artifact_for("VisionZip++", 128, "llava-next-8b").path
    with pytest.raises(KeyError, match="llava-next-8b"):
        artifact_for("VisionZip", 128, "llava-next-8b")


@pytest.mark.parametrize("budget,ratio", [(32, "0.0556"), (64, "0.1111"),
                                          (128, "0.2222"), (224, "0.3889")])
def test_visionzip_pp_next_is_keyed_by_ratio(budget, ratio):
    """The NeXT projectors are named for the ratio they were tuned at.

    A token count would be meaningless on AnyRes, whose visual span varies per
    image. The published ``r0.25`` projector is deliberately unregistered: it
    matches none of the sweep's four operating points, so putting it in the
    0.2222 column would credit the method with 12.5% more budget than the
    baselines beside it.
    """
    path = artifact_for("VisionZip++", budget, "llava-next-8b").path
    assert path.endswith(f"vision_zip_pp_llava_next_r{ratio}.pt"), path
    assert "r0.25.pt" not in path


def test_visionzip_and_visionzip_pp_do_not_share_backbones():
    """Keeping them separate is what stops a 7B-style `pretrained` flag from
    silently selecting a NeXT projector, or vice versa."""
    assert backbones_for("VisionZip") == ["llava", "llava-13b"]
    assert backbones_for("VisionZip++") == ["llava-next-8b"]


def test_quietprune_adapters_differ_per_backbone():
    """One adapter per backbone: each inverts a different projector."""
    paths = {b: artifact_for("QuietPrune", 64, b).path
             for b in ("llava", "llava-13b", "llava-next-8b", "qwen-vl")}
    assert len(set(paths.values())) == 4, paths


@pytest.mark.parametrize("profile,mixture,ratio", [
    (160, (225, 30), 0.08854166666666667),
    (320, (360, 150), 0.177083333333333333),
    (640, (850, 175), 0.35590277777777773),
    (1120, (1750, 40), 0.6215277777777778),
])
def test_duet_next_profiles_match_their_checkpoints(profile, mixture, ratio):
    """Each NeXT profile's mixture must reproduce the ratio its checkpoint records.

    The releases are named for their *layer average* over the 2880-token maximum
    AnyRes span, while `retain_ratio` is Stage-1 content retention, which two
    pdrop cuts then decay. Deriving one from the other is what keeps the arm from
    silently running ~45% under budget.
    """
    from llmoe.compressor.duet.duet import (
        DUET_LLAVA_NEXT_PROFILES, DUET_LLAVA_NEXT_PROFILE_RATIOS)

    assert DUET_LLAVA_NEXT_PROFILES[profile] == mixture
    assert DUET_LLAVA_NEXT_PROFILE_RATIOS[profile] == pytest.approx(ratio, abs=1e-12)
    assert artifact_for("DUET", profile, "llava-next-8b").path.endswith(f"budget_{profile}")


def test_duet_next_profile_sets_its_own_budget():
    """`llava_next_profile` must set the budget `install` resolves the checkpoint
    with; leaving it at the constructor default made every profile load one
    checkpoint while reporting four different budgets."""
    from llmoe.compressor.duet.duet import DUETCompressor, DUET_LLAVA_NEXT_PROFILE_RATIOS

    seen = set()
    for profile in (160, 320, 640, 1120):
        compressor = DUETCompressor(retain_ratio=DUET_LLAVA_NEXT_PROFILE_RATIOS[profile],
                                    llava_next_profile=profile)
        assert int(compressor._max_tokens) == profile
        seen.add(artifact_for("DUET", int(compressor._max_tokens), "llava-next-8b").path)
    assert len(seen) == 4, seen


@pytest.mark.parametrize("budget,directory", [
    (32, "budget_160"), (64, "budget_320"), (128, "budget_640"), (224, "budget_1120"),
])
def test_hidrop_next_key_and_directory_differ_on_purpose(budget, directory):
    """HiDrop/NeXT is keyed by `average_retained_tokens`, stored under the budget spent.

    The compressor resolves with `_weight_budget`, which takes the checkpoint's
    `average_retained_tokens` (32/64/128/224) on this backbone -- so that must be
    the key. The directory is named for what the budget actually spends: a full
    five-view NeXT image is 2880 content tokens against LLaVA-1.5's 576, so
    `retain_ratio 0.0556` is 160 tokens, not 32.

    `average_retained_tokens` is not a token count here; it is the denominator
    that rescales the stored per-stage curve, so it must stay 32 while the
    directory says 160.
    """
    assert artifact_for("HiDrop", budget, "llava-next-8b").path.endswith(directory)
    # the LLaVA-1.5 backbones keep count-named directories
    assert artifact_for("HiDrop", budget, "llava").path.endswith(f"budget_{budget}")
    assert artifact_for("HiDrop", budget, "llava-13b").path.endswith(f"budget_{budget}")


def test_hidrop_next_resolves_with_average_retained_tokens():
    """The compressor's resolve key on NeXT is the budget, not the ratio."""
    from llmoe.compressor.hidrop import HiDropCompressor

    seen = set()
    for budget, ratio in ((32, 0.0556), (64, 0.1111), (128, 0.2222), (224, 0.3889)):
        compressor = HiDropCompressor(retain_ratio=ratio, average_retained_tokens=budget,
                                      backbone_family="llava_next", pretrained=True)
        assert compressor._weight_budget == budget
        seen.add(artifact_for("HiDrop", compressor._weight_budget, "llava-next-8b").path)
    assert len(seen) == 4, seen
