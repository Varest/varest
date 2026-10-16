"""HiDrop schedules, diagnostics, and token-selection policy for both backbones.

A HiDrop schedule is defined by four things:

* `late_entry_layer` -- the 1-indexed decoder layer that first sees the 576
  projected visual tokens (layers before it run text-only);
* `layer_list` -- the 1-indexed layers after which a pruning stage fires;
* `image_token_keep_list` -- how many visual tokens survive each of those stages;
* `early_exit_layer` -- the layer before which every remaining visual token is
  dropped.

The reported budget is the **layer-averaged** visual-token count
(`sum(tokens held by each decoder layer) / depth`), not the final retained count --
so a "64-token" HiDrop still hands layer 9 all 576 tokens. The four named profiles
below are immutable: a schedule that merely reproduces the same average is
rejected, because the projector/model checkpoints are trained per schedule.

They are stated against Vicuna-7B's 32 layers, which is the *only* thing about
them that is not fixed: dividing by the depth means the same numbers describe a
different budget on a deeper decoder. `profile_schedule` therefore resolves a
profile for the decoder actually loaded -- moving the window and filtering layers
to the same fractions of it and re-solving the keeps to hold the label -- and
`HiDropConfig.resolved_for_depth` is what `install` calls to do it. Immutability
still holds, now per (budget, depth): the resolved schedule is checked the same
way, and a config that came from a checkpoint refuses to be re-solved at all,
since its weights were trained for one exact schedule.
"""

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any
import json

import torch

from llmoe.compressor.hidrop.hidrop_llava_next import (
    LLAVA_NEXT_HIDROP_IMPLEMENTATION_VERSION,
    LLAVA_NEXT_IMAGE_GRID_PINPOINTS,
    LLAVA_NEXT_LAYOUT_VERSION,
    LLAVA_NEXT_MODEL_ID,
    LLAVA_NEXT_MODEL_REVISION,
)
from llmoe.utils.depth import REFERENCE_DEPTH, LayerSchedule, scale_layer


HIDROP_IMPLEMENTATION_VERSION = 3
HIDROP_UPSTREAM_COMMIT = "5dfc02b7de75aabe565c2040ad50498b70086185"
HIDROP_PROFILE_FAMILY_VERSION = 1


HIDROP_PROFILES: dict[int, dict[str, object]] = {
    224: {
        "late_entry_layer": 9,
        "early_exit_layer": 25,
        "layer_list": (10, 14, 16, 18),
        "image_token_keep_list": (474, 415, 412, 411),
        "average_retained_tokens": 224.0,
        "source": "HiDrop-derived exact profile: scale the released 64-token drop curve toward the "
                  "288-token unpruned vision window",
    },
    128: {
        "late_entry_layer": 9,
        "early_exit_layer": 25,
        "layer_list": (10, 14, 16, 18),
        "image_token_keep_list": (318, 173, 168, 165),
        "average_retained_tokens": 128.0,
        "source": "HiDrop-derived exact profile: scale the released 64-token drop curve toward the "
                  "288-token unpruned vision window",
    },
    64: {
        "late_entry_layer": 9,
        "early_exit_layer": 25,
        "layer_list": (10, 14, 16, 18),
        "image_token_keep_list": (215, 10, 5, 1),
        "average_retained_tokens": 64.0,
        "source": "released HiDrop 64-token profile",
    },
    32: {
        # The released [10, 14, 16, 18] window bottoms out at a 36.4375-token
        # average even when every stage keeps a single token, so 32 needs one
        # extra filter at the entry layer itself.
        "late_entry_layer": 9,
        "early_exit_layer": 25,
        "layer_list": (9, 10, 14, 16, 18),
        "image_token_keep_list": (132, 70, 10, 5, 1),
        "average_retained_tokens": 32.0,
        "source": "HiDrop-derived exact profile: add a layer-9 emergency filter while retaining all "
                  "released filtering layers",
    },
}

# LLaVA-NeXT exposes ratios publicly, but each ratio still selects one immutable
# HiDrop curve.  Four-decimal spellings are the benchmark-facing labels; exact
# profile fractions are accepted as well.  The per-image target is always
# computed from the caller's ratio rather than from these nominal 576-row labels.
LLAVA_NEXT_HIDROP_RATIO_PROFILES: dict[float, int] = {
    0.3889: 224,
    0.2222: 128,
    0.1111: 64,
    0.0556: 32,
}


def llava_next_profile_for_ratio(retain_ratio: float) -> int:
    """Return the nearest immutable curve shape for a NeXT retention ratio.

    The ratio itself remains authoritative. The selected profile supplies only
    HiDrop's concave stage shape; runtime rescales that shape to the exact
    per-image ``K_target``. Ratios above one half are unreachable because the
    official late-injection/early-exit window spans only half the decoder.
    """

    ratio = float(retain_ratio)
    if not 0.0 < ratio <= 0.5:
        raise ValueError(
            f"retain_ratio must be in (0, 0.5] for HiDrop's 16/32-layer visual window, "
            f"got {ratio}."
        )
    candidates = {
        **LLAVA_NEXT_HIDROP_RATIO_PROFILES,
        **{budget / 576: budget for budget in HIDROP_PROFILES},
    }
    _label, budget = min(candidates.items(), key=lambda item: abs(item[0] - ratio))
    return int(budget)

# Descriptive keys `to_dict()` adds on top of the dataclass fields. Accepted and
# dropped by the compressor and the trainers, so a config round-tripped through a
# checkpoint or a results JSON can be passed straight back in as `compressor_args`.
CONFIG_METADATA_KEYS = (
    "backbone",
    "computed_retain_ratio",
    "debug_warnings",
    "finetuning_image_aspect_ratio",
    "finetuning_template",
    "last_visual_token_counts_by_layer",
    "method",
    "position_encoding",
    "pretraining_image_aspect_ratio",
    "pretraining_template",
    "profile_family_version",
    "profile_source",
    "pruning_history",
    "released_recipe",
    "selector_semantics",
    "source",
    "supported_retained_tokens",
    "topk",
    "trained",
    "trained_with_hf_llava",
    "upstream_commit",
    "visual_token_counts_by_layer",
    "visual_token_ratios_by_layer",
)

CONFIG_METADATA_KEYS += (
    "last_patch_token_counts_by_layer",
    "trained_with_hf_llava_next",
    "llava_next_implementation_version",
    "llava_next_layout_version",
    "budget_convention",
    "newline_policy",
    "native_visual_layout",
    "decoder_architecture",
    "source_commits",
    "training_stage",
    "vision_architecture",
    "multimodal_projector",
    "base_view_reference_patch_counts_by_layer",
    "visual_token_counts_by_layer_semantics",
    "last_llava_next_layout",
    "last_resolved_budget",
    "original_content_tokens",
    "requested_retention_ratio",
    "target_content_tokens",
    "realized_content_tokens",
    "realized_retention_ratio",
    "structural_tokens",
)

# Generation settings the released HiDrop evaluation runs with. Mirror these into
# a `--custom_args` file; `batch_size: 1` is mandatory (see `HiDropCompressor`).
HIDROP_EVAL_ARGS: dict[str, object] = {
    "batch_size": 1,
    "max_new_tokens": 128,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": None,
    "num_beams": 1,
    "use_cache": True,
}


def layer_schedule(late_entry_layer: int, early_exit_layer: int, layer_list: tuple[int, ...],
                   image_token_keep_list: tuple[int, ...], *, visual_tokens: int = 576,
                   num_decoder_layers: int = 32) -> LayerSchedule:
    """This schedule as a :class:`~llmoe.utils.depth.LayerSchedule`.

    HiDrop states its window 1-indexed and its filtering layers as "the layer a
    stage fires *after*", so 1-indexed layer ``L`` still holds the pre-cut count
    and 1-indexed ``L + 1`` -- 0-indexed ``L`` -- is the first to run reduced.
    That makes ``layer_list`` already the shared 0-indexed boundary convention,
    and only the window ends need shifting down by one.
    """
    if len(layer_list) != len(image_token_keep_list):
        raise ValueError("layer_list and image_token_keep_list must have the same length.")
    return LayerSchedule(
        boundaries=tuple(int(layer) for layer in layer_list),
        keeps=tuple(int(keep) for keep in image_token_keep_list),
        depth=int(num_decoder_layers),
        n_visual=int(visual_tokens),
        entry=int(late_entry_layer) - 1,
        exit_layer=int(early_exit_layer) - 1,
    )


def schedule_fields(schedule: LayerSchedule) -> dict[str, Any]:
    """The inverse of :func:`layer_schedule`: HiDrop's own field names and indexing."""
    return {
        "late_entry_layer": schedule.entry + 1,
        "early_exit_layer": schedule.exit_layer + 1,
        "layer_list": schedule.boundaries,
        "image_token_keep_list": tuple(int(keep) for keep in schedule.keeps),
        "num_decoder_layers": schedule.depth,
    }


def visual_token_counts_by_layer(late_entry_layer: int, early_exit_layer: int, layer_list: tuple[int, ...],
                                 image_token_keep_list: tuple[int, ...], *, visual_tokens: int = 576,
                                 num_decoder_layers: int = 32) -> list[int]:
    """Visual tokens held by each 1-indexed decoder layer under this schedule."""
    return [int(count) for count in layer_schedule(
        late_entry_layer, early_exit_layer, layer_list, image_token_keep_list,
        visual_tokens=visual_tokens, num_decoder_layers=num_decoder_layers,
    ).counts_by_layer()]


def average_retained_tokens(late_entry_layer: int, early_exit_layer: int, layer_list: tuple[int, ...],
                            image_token_keep_list: tuple[int, ...], *, visual_tokens: int = 576,
                            num_decoder_layers: int = 32) -> float:
    """The reported budget: visual tokens averaged over every decoder layer."""
    return layer_schedule(
        late_entry_layer, early_exit_layer, layer_list, image_token_keep_list,
        visual_tokens=visual_tokens, num_decoder_layers=num_decoder_layers,
    ).layer_average


def profile_schedule(budget: int, *, depth: int = REFERENCE_DEPTH,
                     visual_tokens: int = 576) -> LayerSchedule:
    """The named profile for `budget`, resolved onto a `depth`-layer decoder.

    The profiles are published against Vicuna-7B's 32 layers. On any other depth
    -- LLaVA-1.5-13B's 40, say -- the filtering window and layers move to the same
    fractions of the decoder and the keeps are re-solved so the budget label still
    describes the run. At depth 32 this returns the profile verbatim.
    """
    profile = HIDROP_PROFILES[int(budget)]
    reference = layer_schedule(
        int(profile["late_entry_layer"]),
        int(profile["early_exit_layer"]),
        tuple(int(layer) for layer in profile["layer_list"]),
        tuple(int(keep) for keep in profile["image_token_keep_list"]),
        visual_tokens=int(visual_tokens),
        num_decoder_layers=REFERENCE_DEPTH,
    )
    return reference.rescaled_to(int(depth), label=f"HiDrop's budget-{int(budget)} profile")


@dataclass(frozen=True)
class HiDropConfig:
    implementation_version: int = HIDROP_IMPLEMENTATION_VERSION
    average_retained_tokens: int = 64
    retain_ratio: float | None = None
    late_entry_layer: int | None = None
    early_exit_layer: int | None = None
    layer_list: tuple[int, ...] = field(default_factory=tuple)
    image_token_keep_list: tuple[int, ...] = field(default_factory=tuple)
    visual_tokens: int = 576
    num_decoder_layers: int = REFERENCE_DEPTH
    dtopk_lambda: float = 1.0
    dtopk_normalize: bool = True
    dtopk_mask_threshold: float = 0.5
    allow_nonstandard_vision_length: bool = False
    batch_size_supported: bool = False
    debug: bool = False
    enabled: bool = True
    checkpoint_path: str | None = None
    # The default remains the released LLaVA-1.5 path. LLaVA-NeXT must be
    # requested explicitly so incompatible released 7B/Vicuna artifacts cannot
    # be selected by accident.
    backbone_family: str = "llava"
    backbone_model_id: str | None = None
    backbone_revision: str | None = None
    image_grid_pinpoints: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    vision_image_size: int | None = None
    vision_patch_size: int | None = None
    vision_feature_select_strategy: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "implementation_version", int(self.implementation_version))
        if self.implementation_version != HIDROP_IMPLEMENTATION_VERSION:
            raise ValueError(
                "Incompatible HiDrop implementation version "
                f"{self.implementation_version}; this code requires version {HIDROP_IMPLEMENTATION_VERSION}. "
                "Restart projector pretraining and full finetuning with the current implementation."
            )
        family = str(self.backbone_family).replace("-", "_").lower()
        if family not in {"llava", "llava_next"}:
            raise ValueError(f"Unsupported HiDrop backbone_family={self.backbone_family!r}.")
        object.__setattr__(self, "backbone_family", family)
        if family == "llava":
            if self.retain_ratio is not None:
                raise ValueError(
                    "retain_ratio is the LLaVA-NeXT budget interface; LLaVA-1.5 keeps its "
                    "released average_retained_tokens interface."
                )
        else:
            ratio = (
                float(self.retain_ratio)
                if self.retain_ratio is not None
                else int(self.average_retained_tokens) / int(self.visual_tokens)
            )
            budget = llava_next_profile_for_ratio(ratio)
            object.__setattr__(self, "retain_ratio", ratio)
            object.__setattr__(self, "average_retained_tokens", budget)
        if family == "llava_next":
            if self.batch_size_supported:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT requires per-device batch size 1 because AnyRes spans are ragged."
                )
            object.__setattr__(
                self, "backbone_model_id", self.backbone_model_id or LLAVA_NEXT_MODEL_ID
            )
            object.__setattr__(
                self, "backbone_revision", self.backbone_revision or LLAVA_NEXT_MODEL_REVISION
            )
            pinpoints = self.image_grid_pinpoints or LLAVA_NEXT_IMAGE_GRID_PINPOINTS
            object.__setattr__(
                self,
                "image_grid_pinpoints",
                tuple((int(point[0]), int(point[1])) for point in pinpoints),
            )
            object.__setattr__(self, "vision_image_size", int(self.vision_image_size or 336))
            object.__setattr__(self, "vision_patch_size", int(self.vision_patch_size or 14))
            object.__setattr__(
                self,
                "vision_feature_select_strategy",
                self.vision_feature_select_strategy or "default",
            )
            if self.backbone_model_id != LLAVA_NEXT_MODEL_ID:
                raise ValueError(
                    f"HiDrop/LLaVA-NeXT is pinned to {LLAVA_NEXT_MODEL_ID}; "
                    f"got backbone_model_id={self.backbone_model_id!r}."
                )
            if self.backbone_revision != LLAVA_NEXT_MODEL_REVISION:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT is pinned to model revision "
                    f"{LLAVA_NEXT_MODEL_REVISION}; got {self.backbone_revision!r}."
                )
            if self.image_grid_pinpoints != LLAVA_NEXT_IMAGE_GRID_PINPOINTS:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT image_grid_pinpoints differ from the pinned checkpoint."
                )
            if self.vision_image_size != 336 or self.vision_patch_size != 14:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT requires CLIP image_size=336 and patch_size=14."
                )
            if self.vision_feature_select_strategy != "default":
                raise ValueError(
                    "HiDrop/LLaVA-NeXT supports vision_feature_select_strategy='default' only."
                )
            if int(self.visual_tokens) != 576:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT uses visual_tokens=576 as its base-view schedule unit; "
                    "the native packed length is resolved per image."
                )
        profile = HIDROP_PROFILES.get(int(self.average_retained_tokens))
        if profile is None and not self.layer_list:
            supported = ", ".join(str(k) for k in sorted(HIDROP_PROFILES))
            raise ValueError(
                f"Unsupported HiDrop budget {self.average_retained_tokens}. Supported budgets: {supported}."
            )
        if profile is None and (self.late_entry_layer is None or self.early_exit_layer is None):
            raise ValueError(
                "A custom HiDrop schedule must set late_entry_layer and early_exit_layer explicitly."
            )
        object.__setattr__(self, "average_retained_tokens", int(self.average_retained_tokens))
        object.__setattr__(self, "num_decoder_layers", int(self.num_decoder_layers))
        # Defaults come from the profile *resolved for this decoder's depth*, so a
        # backbone that is not 32 layers deep gets the schedule that still spends
        # the labelled budget rather than the 32-layer one run out of position.
        resolved = (
            schedule_fields(profile_schedule(int(self.average_retained_tokens),
                                             depth=self.num_decoder_layers,
                                             visual_tokens=self.visual_tokens))
            if profile is not None else {}
        )
        object.__setattr__(self, "late_entry_layer",
                           int(self.late_entry_layer or resolved["late_entry_layer"]))
        object.__setattr__(self, "early_exit_layer",
                           int(self.early_exit_layer or resolved["early_exit_layer"]))
        object.__setattr__(self, "layer_list",
                           tuple(int(x) for x in (self.layer_list or resolved["layer_list"])))
        object.__setattr__(
            self,
            "image_token_keep_list",
            tuple(int(x) for x in (self.image_token_keep_list or resolved["image_token_keep_list"])),
        )
        if profile is not None:
            # Named profiles are immutable: a checkpoint is trained for one exact
            # schedule, so another schedule with the same average is not a
            # substitute for it. On a decoder that is not the profile's own depth
            # the schedule is a *function* of that depth -- the boundaries move
            # with it and the keeps are re-solved to hold the label -- so the
            # signature is checked against the resolved form, which is still one
            # schedule per (budget, depth) and still not hand-editable.
            expected = profile_schedule(int(self.average_retained_tokens),
                                        depth=self.num_decoder_layers,
                                        visual_tokens=self.visual_tokens)
            expected_signature = tuple(schedule_fields(expected)[field] for field in
                                       ("late_entry_layer", "early_exit_layer",
                                        "layer_list", "image_token_keep_list"))
            actual_signature = (
                self.late_entry_layer,
                self.early_exit_layer,
                self.layer_list,
                self.image_token_keep_list,
            )
            if actual_signature != expected_signature:
                depth_note = (
                    "" if self.num_decoder_layers == REFERENCE_DEPTH
                    else f" resolved for a {self.num_decoder_layers}-layer decoder"
                )
                raise ValueError(
                    f"Budget {self.average_retained_tokens} is an immutable HiDrop profile"
                    f"{depth_note}; expected entry/exit/layers/keeps={expected_signature}, "
                    f"got {actual_signature}."
                )
        if len(self.layer_list) != len(self.image_token_keep_list):
            raise ValueError("HiDrop layer_list length must match image_token_keep_list length.")
        if tuple(sorted(self.layer_list)) != self.layer_list:
            raise ValueError(
                "HiDrop layer_list must be sorted in increasing 1-indexed layer order."
            )
        if self.late_entry_layer < 1 or self.early_exit_layer > self.num_decoder_layers + 1:
            raise ValueError(
                "HiDrop late_entry_layer/early_exit_layer are outside the decoder depth."
            )
        if self.late_entry_layer >= self.early_exit_layer:
            raise ValueError("HiDrop late_entry_layer must be earlier than early_exit_layer.")
        if self.layer_list and (
            self.layer_list[0] < self.late_entry_layer
            or self.layer_list[-1] >= self.early_exit_layer
        ):
            raise ValueError(
                "HiDrop pruning layers must fall inside [late_entry_layer, early_exit_layer)."
            )
        if any(keep < 1 for keep in self.image_token_keep_list):
            raise ValueError(
                "HiDrop pruning stages must retain at least one visual token; early exit removes the remainder."
            )
        if any(
            current < following
            for current, following in zip(
                self.image_token_keep_list, self.image_token_keep_list[1:]
            )
        ):
            raise ValueError("HiDrop image-token keep counts must be monotonically non-increasing.")
        actual = self.computed_average_retained_tokens()
        expected = (
            float(profile["average_retained_tokens"])
            if profile is not None
            else float(self.average_retained_tokens)
        )
        # Exact at the depth the profile was solved at. Re-solving for another depth
        # rounds each stage's keep to a whole token, which moves the average by well
        # under one token -- more than that means the schedule is mislabelled.
        tolerance = 1e-6 if self.num_decoder_layers == REFERENCE_DEPTH else 1.0
        if abs(actual - expected) > tolerance:
            raise ValueError(f"HiDrop profile average mismatch: computed {actual}, expected {expected}.")

    @property
    def source(self) -> str:
        profile = HIDROP_PROFILES.get(self.average_retained_tokens)
        return str(profile["source"]) if profile is not None else "custom"

    def keep_k_for_layer(self, rank_layer: int, current_visual_tokens: int) -> int:
        if rank_layer not in self.layer_list:
            raise ValueError(f"Layer {rank_layer} is not a HiDrop pruning layer.")
        keep_k = int(self.image_token_keep_list[self.layer_list.index(rank_layer)])
        if keep_k < 0 or keep_k > int(current_visual_tokens):
            raise ValueError(
                f"HiDrop keep count {keep_k} is invalid for {current_visual_tokens} active visual tokens."
            )
        return keep_k

    def visual_token_counts_by_layer(self) -> list[int]:
        return visual_token_counts_by_layer(
            self.late_entry_layer,
            self.early_exit_layer,
            self.layer_list,
            self.image_token_keep_list,
            visual_tokens=self.visual_tokens,
            num_decoder_layers=self.num_decoder_layers,
        )

    def computed_average_retained_tokens(self) -> float:
        return float(sum(self.visual_token_counts_by_layer()) / self.num_decoder_layers)

    def resolved_for_depth(self, depth: int) -> "HiDropConfig":
        """This budget's schedule, re-solved for a `depth`-layer decoder.

        Returns ``self`` when the depth already matches. Clearing the schedule
        fields is what re-derives them: ``__post_init__`` fills them from the
        profile resolved for the new depth, and then re-checks the label.
        """
        depth = int(depth)
        if depth == self.num_decoder_layers:
            return self
        if self.checkpoint_path is not None:
            raise ValueError(
                f"This HiDrop config came from checkpoint {self.checkpoint_path!r}, whose weights "
                f"were trained for its exact {self.num_decoder_layers}-layer schedule; it cannot "
                f"be re-solved for a {depth}-layer decoder. Train HiDrop for this backbone, or "
                f"run it untrained with `pretrained: false`."
            )
        if int(self.average_retained_tokens) not in HIDROP_PROFILES:
            raise ValueError(
                f"This HiDrop schedule is custom, so it has no depth-independent definition to "
                f"re-solve for a {depth}-layer decoder. Pass a `layer_list` / "
                f"`image_token_keep_list` built for this depth, with `num_decoder_layers={depth}`."
            )
        fields = asdict(self)
        fields.update(
            late_entry_layer=None,
            early_exit_layer=None,
            layer_list=(),
            image_token_keep_list=(),
            num_decoder_layers=depth,
        )
        return HiDropConfig(**fields)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.backbone_family == "llava":
            for key in (
                "retain_ratio",
                "backbone_family",
                "backbone_model_id",
                "backbone_revision",
                "image_grid_pinpoints",
                "vision_image_size",
                "vision_patch_size",
                "vision_feature_select_strategy",
            ):
                data.pop(key, None)
        else:
            data["image_grid_pinpoints"] = [list(point) for point in self.image_grid_pinpoints]
        data["method"] = "hidrop"
        data["layer_list"] = list(self.layer_list)
        data["image_token_keep_list"] = list(self.image_token_keep_list)
        data["position_encoding"] = "persistent"
        data["topk"] = "released_rank_sigmoid_hard_threshold_exact_k_guard"
        data["trained_with_hf_llava"] = True
        data["upstream_commit"] = HIDROP_UPSTREAM_COMMIT
        data["selector_semantics"] = "detached_next_layer_qk_full_causal_softmax_hard_indices"
        data["profile_family_version"] = HIDROP_PROFILE_FAMILY_VERSION
        data["pretraining_template"] = "llava_plain"
        data["finetuning_template"] = "vicuna_v1"
        data["pretraining_image_aspect_ratio"] = "square"
        data["finetuning_image_aspect_ratio"] = "pad"
        data["profile_source"] = self.source
        data["visual_token_counts_by_layer"] = self.visual_token_counts_by_layer()
        if self.backbone_family == "llava_next":
            data.update(self._llava_next_metadata())
        return data

    def _llava_next_metadata(self) -> dict[str, Any]:
        return {
            "trained_with_hf_llava": False,
            "trained_with_hf_llava_next": True,
            "llava_next_implementation_version": LLAVA_NEXT_HIDROP_IMPLEMENTATION_VERSION,
            "llava_next_layout_version": LLAVA_NEXT_LAYOUT_VERSION,
            "backbone_family": "llava_next",
            "backbone_model_id": self.backbone_model_id,
            "backbone_revision": self.backbone_revision,
            "image_grid_pinpoints": [list(point) for point in self.image_grid_pinpoints],
            "vision_image_size": self.vision_image_size,
            "vision_patch_size": self.vision_patch_size,
            "vision_feature_select_strategy": self.vision_feature_select_strategy,
            "pretraining_template": "llama3_chat_compression_adaptation",
            "finetuning_template": "llama3_chat",
            "pretraining_image_aspect_ratio": "anyres_spatial_unpad",
            "finetuning_image_aspect_ratio": "anyres_spatial_unpad",
            "budget_convention": "per_image_native_content_decoder_layer_average",
            "retain_ratio": self.retain_ratio,
            "base_view_reference_patch_counts_by_layer": self.visual_token_counts_by_layer(),
            "visual_token_counts_by_layer_semantics": (
                "576_patch_base_view_reference_only; native per_image counts are resolved from geometry"
            ),
            "newline_policy": "structural_passthrough_until_early_exit",
            "native_visual_layout": "base_then_unpadded_local_mosaic_raster_with_row_newlines",
            "decoder_architecture": {
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "layers": 32,
                "attention_heads": 32,
                "key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "max_position_embeddings": 8192,
            },
            "vision_architecture": {
                "model_type": "clip_vision_model",
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "layers": 24,
                "attention_heads": 16,
                "image_size": 336,
                "patch_size": 14,
                "feature_layer": -2,
                "feature_select_strategy": "default",
            },
            "multimodal_projector": {
                "hidden_act": "gelu",
                "learned_image_newline": True,
            },
            "source_commits": {
                "hidrop": HIDROP_UPSTREAM_COMMIT,
                "projectneura_hidrop_reference": "1d17b954dc555faab69f2cf30dec5faedb2d9283",
                "nuwa": "175b7d57a7890b9818dd26d82f2d91d9868b0d80",
                "projectneura_nuwa": "23a838a4aa2bf6d3285cee9c08ac01c112feed36",
                "varest_llava15_foundation": "475fc2d768132d39dd8974a41e4da3df1e572ef6",
                "transformers": "5.5.4",
            },
        }

    def hidrop_metadata(self) -> dict[str, Any]:
        """The subset written to a checkpoint's `hidrop_config.json`."""
        data = {
            "method": "hidrop",
            "implementation_version": self.implementation_version,
            "upstream_commit": HIDROP_UPSTREAM_COMMIT,
            "selector_semantics": "detached_next_layer_qk_full_causal_softmax_hard_indices",
            "profile_family_version": HIDROP_PROFILE_FAMILY_VERSION,
            "pretraining_template": "llava_plain",
            "finetuning_template": "vicuna_v1",
            "pretraining_image_aspect_ratio": "square",
            "finetuning_image_aspect_ratio": "pad",
            "average_retained_tokens": self.average_retained_tokens,
            "late_entry_layer": self.late_entry_layer,
            "early_exit_layer": self.early_exit_layer,
            "layer_list": list(self.layer_list),
            "image_token_keep_list": list(self.image_token_keep_list),
            "visual_tokens": self.visual_tokens,
            "num_decoder_layers": self.num_decoder_layers,
            "position_encoding": "persistent",
            "topk": "released_rank_sigmoid_hard_threshold_exact_k_guard",
            "dtopk_lambda": self.dtopk_lambda,
            "dtopk_normalize": self.dtopk_normalize,
            "dtopk_mask_threshold": self.dtopk_mask_threshold,
            "trained_with_hf_llava": True,
        }
        if self.backbone_family == "llava_next":
            data.update(self._llava_next_metadata())
        return data

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, **overrides: Any) -> "HiDropConfig":
        """Load the schedule a HiDrop checkpoint was trained with.

        A checkpoint whose `hidrop_config.json` is missing or records another
        implementation version is rejected rather than silently re-scheduled: the
        selector and preprocessing semantics it was trained under are part of the
        weights.
        """
        path = Path(checkpoint_path) / "hidrop_config.json"
        if not path.exists():
            if overrides:
                return cls(checkpoint_path=str(checkpoint_path), **overrides)
            raise FileNotFoundError(f"Could not find {path}.")
        raw = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_version = raw.get("implementation_version")
        if checkpoint_version != HIDROP_IMPLEMENTATION_VERSION:
            found = "missing" if checkpoint_version is None else repr(checkpoint_version)
            raise ValueError(
                f"Checkpoint {path} has HiDrop implementation_version={found}; "
                f"version {HIDROP_IMPLEMENTATION_VERSION} is required. The checkpoint was produced with "
                "incompatible selector/preprocessing semantics. Restart projector pretraining and full "
                "finetuning from clean Vicuna and CLIP weights."
            )
        checkpoint_family = str(raw.get("backbone_family", "llava")).replace("-", "_").lower()
        if checkpoint_family == "llava_next":
            next_version = raw.get("llava_next_implementation_version")
            layout_version = raw.get("llava_next_layout_version")
            if next_version != LLAVA_NEXT_HIDROP_IMPLEMENTATION_VERSION:
                raise ValueError(
                    f"Checkpoint {path} has LLaVA-NeXT HiDrop implementation version "
                    f"{next_version!r}; expected {LLAVA_NEXT_HIDROP_IMPLEMENTATION_VERSION}."
                )
            if layout_version != LLAVA_NEXT_LAYOUT_VERSION:
                raise ValueError(
                    f"Checkpoint {path} has LLaVA-NeXT layout version {layout_version!r}; "
                    f"expected {LLAVA_NEXT_LAYOUT_VERSION}."
                )
        kwargs = {
            "implementation_version": int(checkpoint_version),
            "average_retained_tokens": int(
                raw.get("average_retained_tokens", overrides.get("average_retained_tokens", 64))
            ),
            "retain_ratio": raw.get("retain_ratio", overrides.get("retain_ratio")),
            "late_entry_layer": int(
                raw.get("late_entry_layer", overrides.get("late_entry_layer", 9))
            ),
            "early_exit_layer": int(
                raw.get("early_exit_layer", overrides.get("early_exit_layer", 25))
            ),
            "layer_list": tuple(raw.get("layer_list", overrides.get("layer_list", ()))),
            "image_token_keep_list": tuple(
                raw.get("image_token_keep_list", overrides.get("image_token_keep_list", ()))
            ),
            "visual_tokens": int(raw.get("visual_tokens", overrides.get("visual_tokens", 576))),
            "num_decoder_layers": int(
                raw.get("num_decoder_layers", overrides.get("num_decoder_layers", 32))
            ),
            "dtopk_lambda": float(raw.get("dtopk_lambda", overrides.get("dtopk_lambda", 1.0))),
            "dtopk_normalize": bool(
                raw.get("dtopk_normalize", overrides.get("dtopk_normalize", True))
            ),
            "dtopk_mask_threshold": float(
                raw.get("dtopk_mask_threshold", overrides.get("dtopk_mask_threshold", 0.5))
            ),
            "checkpoint_path": str(checkpoint_path),
            "backbone_family": str(
                raw.get("backbone_family", overrides.get("backbone_family", "llava"))
            ),
            "backbone_model_id": raw.get("backbone_model_id", overrides.get("backbone_model_id")),
            "backbone_revision": raw.get("backbone_revision", overrides.get("backbone_revision")),
            "image_grid_pinpoints": tuple(
                tuple(point)
                for point in raw.get(
                    "image_grid_pinpoints", overrides.get("image_grid_pinpoints", ())
                )
            ),
            "vision_image_size": raw.get("vision_image_size", overrides.get("vision_image_size")),
            "vision_patch_size": raw.get("vision_patch_size", overrides.get("vision_patch_size")),
            "vision_feature_select_strategy": raw.get(
                "vision_feature_select_strategy",
                overrides.get("vision_feature_select_strategy"),
            ),
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)

# ---------------------------------------------------------------------------
# Qwen2.5-VL ratio schedules
# ---------------------------------------------------------------------------

# Reference geometry used to derive Qwen schedules from the LLaVA-1.5 curves.
LLAVA_REFERENCE_DEPTH = REFERENCE_DEPTH
LLAVA_REFERENCE_VISUAL_TOKENS = 576


@dataclass(frozen=True)
class ReleasedCurve:
    """A LLaVA-1.5 keep curve, in the units `config.HIDROP_PROFILES` states it in."""

    late_entry_layer: int
    early_exit_layer: int
    layer_list: tuple[int, ...]
    image_token_keep_list: tuple[int, ...]

    def keep_fractions(self) -> tuple[Fraction, ...]:
        return tuple(
            Fraction(int(keep), LLAVA_REFERENCE_VISUAL_TOKENS) for keep in self.image_token_keep_list
        )


# The released 64-token profile. Its four filtering layers are the local maxima of the
# paper's ILVAS curve, and every budget above the window floor reuses them.
RELEASED_FOUR_STAGE_CURVE = ReleasedCurve(
    late_entry_layer=9,
    early_exit_layer=25,
    layer_list=(10, 14, 16, 18),
    image_token_keep_list=(215, 10, 5, 1),
)
# The derived 32-token profile, which adds an emergency filter at the entry layer
# because the four released layers cannot reach that low on their own.
RELEASED_FIVE_STAGE_CURVE = ReleasedCurve(
    late_entry_layer=9,
    early_exit_layer=25,
    layer_list=(9, 10, 14, 16, 18),
    image_token_keep_list=(132, 70, 10, 5, 1),
)

# Ratios supported by the in-repo Qwen adaptation. These are derived from the released
# LLaVA curves; upstream does not publish a Qwen recipe or checkpoint.
# The sweep's lowest budget sits below the five-stage curve's reach (0.0584 at
# depth 28), so it takes one more filter, in the gap the released layers leave
# between 10 and 14. This is the same answer upstream gave when four stages could
# not reach 32 tokens; the early keeps are tightened so 0.0556 clears the floor
# (0.0539) with margin instead of landing on it, where the last stage's keep would
# collapse to zero.
DERIVED_SIX_STAGE_CURVE = ReleasedCurve(
    late_entry_layer=9,
    early_exit_layer=25,
    layer_list=(9, 10, 12, 14, 16, 18),
    image_token_keep_list=(90, 60, 30, 10, 5, 1),
)

#: Ratios upstream itself publishes for Qwen2.5-VL.
QWEN_PUBLISHED_RATIOS: frozenset[float] = frozenset({0.375, 0.25, 0.125, 0.0625})

#: The sweep's budgets: LLaVA's 224/128/64/32 tokens as fractions of 576, so
#: HiDrop is scored at the same budgets as the methods beside it rather than at
#: its own nearby-but-looser ones.
#:
#: An IN-REPO EXTRAPOLATION, not an upstream recipe. Three reuse the curve their
#: nearest published ratio uses and differ only in the target `derive_keep_ratios`
#: solves for -- which is already how 0.375, 0.25 and 0.125 relate to each other.
#: The lowest needs a sixth filter; see DERIVED_SIX_STAGE_CURVE.
QWEN_SWEEP_RATIOS: dict[float, ReleasedCurve] = {
    0.3889: RELEASED_FOUR_STAGE_CURVE,   # 224/576, nearest published 0.375
    0.2222: RELEASED_FOUR_STAGE_CURVE,   # 128/576, nearest published 0.25
    0.1111: RELEASED_FIVE_STAGE_CURVE,   #  64/576; four-stage bottoms out at 0.113
    0.0556: DERIVED_SIX_STAGE_CURVE,     #  32/576; five-stage bottoms out at 0.0584
}

QWEN_PROFILES: dict[float, ReleasedCurve] = {
    0.375: RELEASED_FOUR_STAGE_CURVE,
    0.25: RELEASED_FOUR_STAGE_CURVE,
    0.125: RELEASED_FOUR_STAGE_CURVE,
    0.0625: RELEASED_FIVE_STAGE_CURVE,
    **QWEN_SWEEP_RATIOS,
}


def stage_layer_weights(late_entry_layer: int, early_exit_layer: int, layer_list: tuple[int, ...],
                        depth: int) -> tuple[int, list[int]]:
    """How many decoder layers hold each value the schedule takes.

    Mirrors `config.visual_token_counts_by_layer`: a layer inside the window holds
    whatever the previous stage left, and the stage listed *at* that layer fires after
    it. Returns the count of layers holding the full span, then one count per stage.

    :return: ``(leading_layers, per_stage_layers)``.
    """
    stage_of_layer = {int(layer): index for index, layer in enumerate(layer_list)}
    leading = 0
    per_stage = [0] * len(layer_list)
    stage = -1
    for rank_layer in range(1, int(depth) + 1):
        if rank_layer < late_entry_layer or rank_layer >= early_exit_layer:
            continue
        if stage < 0:
            leading += 1
        else:
            per_stage[stage] += 1
        if rank_layer in stage_of_layer:
            stage = stage_of_layer[rank_layer]
    return leading, per_stage


def derive_keep_ratios(curve: ReleasedCurve, depth: int, retain_ratio: float
                       ) -> tuple[int, int, tuple[int, ...], tuple[float, ...]]:
    """Resolve a reference curve onto `depth` layers at the requested layer-average.

    The layer positions are scaled first, then the keep curve is interpolated between
    the reference shape and "keep everything" until the resulting layer-average equals
    `retain_ratio` exactly (see the module docstring for the solved expression).

    :return: ``(late_entry_layer, early_exit_layer, layer_list, keep_ratios)``.
    """
    depth = int(depth)
    late_entry_layer = scale_layer(curve.late_entry_layer, depth)
    early_exit_layer = scale_layer(curve.early_exit_layer, depth)
    layer_list = tuple(scale_layer(layer, depth) for layer in curve.layer_list)
    if len(set(layer_list)) != len(layer_list):
        raise ValueError(
            f"HiDrop's reference filtering layers {curve.layer_list} collapse onto "
            f"{layer_list} at depth {depth}; the schedule needs distinct layers. This decoder "
            f"is too shallow for the Qwen HiDrop profiles."
        )

    leading, per_stage = stage_layer_weights(late_entry_layer, early_exit_layer, layer_list, depth)
    if leading < 1:
        raise ValueError(
            f"HiDrop schedule at depth {depth} leaves no layer holding the full visual span "
            f"(late_entry={late_entry_layer}, first filter={layer_list[0]})."
        )
    ceiling = Fraction(leading + sum(per_stage), depth)
    target = Fraction(str(retain_ratio))
    if target >= ceiling:
        raise ValueError(
            f"retain_ratio={retain_ratio:g} is unreachable: layers "
            f"[{late_entry_layer}, {early_exit_layer}) are the only ones holding visual tokens, "
            f"so the layer-average cannot exceed {float(ceiling):g} even with no pruning at all. "
            f"Lower retain_ratio, or widen the schedule with an explicit early_exit_layer."
        )

    gaps = [1 - fraction for fraction in curve.keep_fractions()]
    denominator = sum(count * gap for count, gap in zip(per_stage, gaps, strict=True))
    if denominator <= 0:
        raise ValueError("HiDrop keep curve is degenerate: no stage layer carries a pruned value.")
    t = (ceiling - target) * depth / denominator
    keep_ratios = tuple(1 - t * gap for gap in gaps)
    if any(ratio <= 0 for ratio in keep_ratios):
        raise ValueError(
            f"retain_ratio={retain_ratio:g} is below what this curve can reach at depth {depth}: "
            f"stage keeps would go to {[float(ratio) for ratio in keep_ratios]}, and every stage "
            f"must retain at least one visual token. Use a curve with an extra filtering layer."
        )
    return late_entry_layer, early_exit_layer, layer_list, tuple(float(ratio) for ratio in keep_ratios)


def visual_token_ratios_by_layer(late_entry_layer: int, early_exit_layer: int,
                                 layer_list: tuple[int, ...],
                                 image_token_keep_ratio_list: tuple[float, ...],
                                 *, num_decoder_layers: int) -> list[float]:
    """Fraction of the image's visual tokens held by each 1-indexed decoder layer."""
    if len(layer_list) != len(image_token_keep_ratio_list):
        raise ValueError("layer_list and image_token_keep_ratio_list must have the same length.")
    keep_by_layer = dict(zip(layer_list, image_token_keep_ratio_list, strict=True))
    current = 1.0
    ratios: list[float] = []
    for rank_layer in range(1, int(num_decoder_layers) + 1):
        if rank_layer < late_entry_layer or rank_layer >= early_exit_layer:
            ratios.append(0.0)
            continue
        ratios.append(current)
        if rank_layer in keep_by_layer:
            current = float(keep_by_layer[rank_layer])
    return ratios


@dataclass(frozen=True)
class QwenHiDropConfig:
    """A HiDrop schedule resolved for one Qwen2.5-VL decoder.

    Built by :meth:`for_backbone` from a supported `retain_ratio` and the model's own
    depth. Every field can also be given explicitly, which skips the profile lookup.
    Unlike :class:`HiDropConfig`, this does not enforce an upstream checkpoint profile:
    HiDrop publishes no Qwen recipe or checkpoint. Checkpoints trained with this in-repo
    adaptation are validated separately through their provenance metadata.
    """

    retain_ratio: float = 0.25
    num_decoder_layers: int = 28
    late_entry_layer: int | None = None
    early_exit_layer: int | None = None
    layer_list: tuple[int, ...] = field(default_factory=tuple)
    image_token_keep_ratio_list: tuple[float, ...] = field(default_factory=tuple)
    dtopk_lambda: float = 1.0
    dtopk_normalize: bool = True
    dtopk_mask_threshold: float = 0.5
    # Batched forwards are a training-only capability: the trainer sets this from
    # `per_device_train_batch_size`, while generation stays batch-size-1 because the
    # per-image token counts make a padded batch's keep-sets incomparable.
    batch_size_supported: bool = False
    debug: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "retain_ratio", float(self.retain_ratio))
        object.__setattr__(self, "num_decoder_layers", int(self.num_decoder_layers))
        if not 0.0 < self.retain_ratio < 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1), got {self.retain_ratio}.")
        if not self.layer_list:
            curve = QWEN_PROFILES.get(self.retain_ratio)
            if curve is None:
                supported = ", ".join(f"{ratio:g}" for ratio in sorted(QWEN_PROFILES, reverse=True))
                raise ValueError(
                    f"Unsupported HiDrop retain_ratio {self.retain_ratio:g} for Qwen2.5-VL. "
                    f"Supported ratios: {supported}. Pass layer_list and "
                    f"image_token_keep_ratio_list explicitly to run a custom schedule."
                )
            late, exit_, layers, keeps = derive_keep_ratios(
                curve, self.num_decoder_layers, self.retain_ratio
            )
            object.__setattr__(self, "late_entry_layer", late)
            object.__setattr__(self, "early_exit_layer", exit_)
            object.__setattr__(self, "layer_list", layers)
            object.__setattr__(self, "image_token_keep_ratio_list", keeps)
        else:
            if self.late_entry_layer is None or self.early_exit_layer is None:
                raise ValueError(
                    "A custom Qwen HiDrop schedule must set late_entry_layer and early_exit_layer."
                )
            object.__setattr__(self, "late_entry_layer", int(self.late_entry_layer))
            object.__setattr__(self, "early_exit_layer", int(self.early_exit_layer))
            object.__setattr__(self, "layer_list", tuple(int(x) for x in self.layer_list))
            object.__setattr__(
                self,
                "image_token_keep_ratio_list",
                tuple(float(x) for x in self.image_token_keep_ratio_list),
            )

        if len(self.layer_list) != len(self.image_token_keep_ratio_list):
            raise ValueError(
                "HiDrop layer_list length must match image_token_keep_ratio_list length."
            )
        if tuple(sorted(self.layer_list)) != self.layer_list:
            raise ValueError("HiDrop layer_list must be sorted in increasing 1-indexed layer order.")
        if self.late_entry_layer < 1 or self.early_exit_layer > self.num_decoder_layers + 1:
            raise ValueError("HiDrop late_entry_layer/early_exit_layer are outside the decoder depth.")
        if self.late_entry_layer >= self.early_exit_layer:
            raise ValueError("HiDrop late_entry_layer must be earlier than early_exit_layer.")
        if self.layer_list and (self.layer_list[0] < self.late_entry_layer
                                or self.layer_list[-1] >= self.early_exit_layer):
            raise ValueError("HiDrop pruning layers must fall inside [late_entry_layer, early_exit_layer).")
        if any(ratio <= 0.0 or ratio > 1.0 for ratio in self.image_token_keep_ratio_list):
            raise ValueError(
                "HiDrop keep ratios must lie in (0, 1]; every stage retains at least one visual "
                "token and early exit removes the remainder."
            )
        if any(current < following for current, following
               in zip(self.image_token_keep_ratio_list, self.image_token_keep_ratio_list[1:])):
            raise ValueError("HiDrop keep ratios must be monotonically non-increasing.")

    @property
    def source(self) -> str:
        curve = QWEN_PROFILES.get(self.retain_ratio)
        if curve is None:
            return "custom"
        stages = len(curve.layer_list)
        return (
            f"HiDrop-derived Qwen profile: released {stages}-stage LLaVA curve scaled to "
            f"{self.num_decoder_layers} layers at a {self.retain_ratio:g} layer-average"
        )

    def keep_k_for_layer(self, rank_layer: int, current_visual_tokens: int,
                         original_visual_tokens: int) -> int:
        """Visual tokens surviving the stage that fires after `rank_layer`.

        The ratio is taken against the image's *original* token count, mirroring the
        LLaVA schedule where each stage is an absolute count out of 576 rather than a
        fraction of whatever the previous stage left. Clamped into
        ``[1, current_visual_tokens]``: a stage always keeps at least one token, and
        rounding must never ask for more tokens than are still present.
        """
        if rank_layer not in self.layer_list:
            raise ValueError(f"Layer {rank_layer} is not a HiDrop pruning layer.")
        ratio = self.image_token_keep_ratio_list[self.layer_list.index(rank_layer)]
        keep_k = int(round(ratio * int(original_visual_tokens)))
        return max(1, min(keep_k, int(current_visual_tokens)))

    def visual_token_ratios_by_layer(self) -> list[float]:
        return visual_token_ratios_by_layer(
            self.late_entry_layer,
            self.early_exit_layer,
            self.layer_list,
            self.image_token_keep_ratio_list,
            num_decoder_layers=self.num_decoder_layers,
        )

    def computed_retain_ratio(self) -> float:
        return float(sum(self.visual_token_ratios_by_layer()) / self.num_decoder_layers)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["method"] = "hidrop"
        data["backbone"] = "qwen2_5_vl"
        data["layer_list"] = list(self.layer_list)
        data["image_token_keep_ratio_list"] = list(self.image_token_keep_ratio_list)
        data["position_encoding"] = "persistent_mrope"
        data["topk"] = "released_rank_sigmoid_hard_threshold_exact_k_guard"
        data["selector_semantics"] = "detached_next_layer_qk_full_causal_softmax_hard_indices"
        data["released_recipe"] = False
        data["profile_source"] = self.source
        data["visual_token_ratios_by_layer"] = self.visual_token_ratios_by_layer()
        data["computed_retain_ratio"] = self.computed_retain_ratio()
        return data

    @classmethod
    def for_backbone(cls, retain_ratio: float, *, num_decoder_layers: int,
                     **overrides: Any) -> "QwenHiDropConfig":
        """Resolve a supported ratio against a decoder of `num_decoder_layers` layers."""
        return cls(
            retain_ratio=float(retain_ratio),
            num_decoder_layers=int(num_decoder_layers),
            **{key: value for key, value in overrides.items() if value is not None},
        )


# ---------------------------------------------------------------------------
# Runtime diagnostics and DTop-K selection
# ---------------------------------------------------------------------------



class HiDropDebugger:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled: bool = bool(enabled)
        self.warnings: list[str] = []

    def log(self, message: str) -> None:
        if self.enabled:
            print(f"[HIDROP] {message}", flush=True)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.enabled:
            print(f"[HIDROP][WARN] {message}", flush=True)

    def shape(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        if torch.is_tensor(value):
            self.log(f"{name} shape={tuple(value.shape)}")
        elif value is None:
            self.log(f"{name}=None")
        else:
            self.log(f"{name} type={type(value).__name__}")


@dataclass
class DTopKResult:
    indices: torch.Tensor
    soft_mask: torch.Tensor
    corrected: bool


def rank_normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"DTop-K expects a 1D score vector, got {tuple(scores.shape)}.")
    comparisons = (scores[:, None] >= scores[None, :]).to(scores.dtype)
    # Match the released HiDrop/DMS rank normalization. The comparisons are
    # deliberately discrete; production HiDrop consumes hard indices rather
    # than pretending that the unused soft mask trains the selector.
    return comparisons.mean(dim=1)


def dtopk(scores: torch.Tensor, k: int, *, lambda_: float = 1.0, normalize: bool = True,
          mask_threshold: float = 0.5, debugger: HiDropDebugger | None = None) -> DTopKResult:
    if scores.ndim != 1:
        raise ValueError(f"DTop-K expects scores [N], got {tuple(scores.shape)}.")
    n = int(scores.numel())
    k = int(k)
    if k < 0 or k > n:
        raise ValueError(f"DTop-K keep count k={k} must be in [0, {n}].")
    if k == 0:
        return DTopKResult(
            indices=torch.empty(0, dtype=torch.long, device=scores.device),
            soft_mask=torch.zeros_like(scores),
            corrected=False,
        )
    rank_scores = rank_normalize_scores(scores) if normalize else scores
    threshold = 1.0 - (float(k) / float(n))
    soft_mask = torch.sigmoid((rank_scores - threshold) * float(n) * float(lambda_))
    selected = torch.where(soft_mask > float(mask_threshold))[0]
    corrected = int(selected.numel()) != k
    if corrected:
        if debugger is not None:
            debugger.warn(
                f"DTop-K soft threshold selected {int(selected.numel())} tokens for k={k}; "
                "correcting with hard top-k."
            )
        _, selected = torch.topk(scores, k, dim=0)
    selected = torch.sort(selected).values
    if int(selected.numel()) != k:
        raise RuntimeError(f"DTop-K exact-k guard failed: selected {int(selected.numel())}, expected {k}.")
    return DTopKResult(indices=selected, soft_mask=soft_mask, corrected=corrected)


class DifferentiableTopK:
    """Released HiDrop rank/sigmoid selector with a hard forward decision.

    The upstream class is named ``DifferentiableTopK``, so the public name is
    retained for checkpoint/source compatibility. Token removal itself is not
    differentiable because the model consumes integer indices.
    """

    def __init__(self, lambda_: float = 1.0, normalize: bool = True, mask_threshold: float = 0.5,
                 debugger: HiDropDebugger | None = None) -> None:
        self.lambda_: float = float(lambda_)
        self.normalize: bool = bool(normalize)
        self.mask_threshold: float = float(mask_threshold)
        self.debugger: HiDropDebugger | None = debugger

    def apply_k(self, scores: torch.Tensor, k: int) -> DTopKResult:
        return dtopk(scores, k, lambda_=self.lambda_, normalize=self.normalize,
                     mask_threshold=self.mask_threshold, debugger=self.debugger)
