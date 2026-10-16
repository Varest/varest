"""DUET-VLM on native Transformers LLaVA-NeXT with Llama-3.

The paper calibrates LLaVA-NeXT at five full CLIP views (one base view plus a
2x2 local grid), whose 2,880 content patches define the profile ratios. Runtime
budgets are instead computed from each image's exact native content count after
AnyRes stitching and unpadding. The profile determines DUET's dominant/contextual
mixture; it does not impose a fixed token count on variable-resolution images.

Nuwa is used only as a verified layout reference: the base view remains first,
while local views are stitched into the native spatial mosaic and unpadded.
DUET's own dominant/contextual algorithm still performs every reduction. Local
dominant rows retain raster order, a learned newline is inserted after every
unpadded row exactly as Transformers 5.5.4 does, and contextual summaries follow
the local raster block as in the released DUET/VisionZip NeXT packer.

Budget convention
-----------------
For one image, ``N_original`` is the base-view patch count plus the exact number
of unpadded local-mosaic patches. Given ``retain_ratio=r``, Stage 1 targets
``K=max(1, round_half_up(r*N_original))`` content tokens. Learned image-newline
rows are structural: they are excluded from ``N_original`` and ``K``, preserved
at both PyramidDrop boundaries, and reported separately. The target is split
proportionally across the base view and local mosaic, then each share is split
with the selected paper profile's dominant/contextual mixture.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from typing import Any

import torch
from transformers.models.llava_next.modeling_llava_next import unpad_image

from llmoe.compressor.prototype import VisualizationView
from llmoe.visualization import pixel_values_to_canvas
from llmoe.compressor.duet.stage1 import Stage1Plan, duet_stage1_plan
from llmoe.compressor.duet.stage2 import average_image_tokens
from llmoe.compressor.llava_next import (
    anyres_grid_shape,
    crop_counts,
    geometry_for_image,
    image_token_id,
    patch_grid,
    placeholder_run_lengths,
    split_crop_tensor,
    stitch_and_unpad_local_crops,
    tokenizer_pad_token_id,
    trim_image_placeholders_to_lengths,
)


DUET_NEXT_REFERENCE_VIEWS = 5
DUET_NEXT_IMPLEMENTATION_VERSION = "duet-llava-next-v3"
DUET_NEXT_TRAINING_ATTENTION_BACKEND = "flash_attention_2"
DUET_NEXT_FLASH_ATTN_VERSION = "2.8.3.post1"
DUET_NEXT_INFERENCE_ATTENTION_BACKEND = "sdpa"
DUET_NEXT_BUDGET_SEMANTICS = (
    "per-image Stage-1 content retention: K=max(1, round_half_up(r*N_original)); "
    "N_original is global patches plus exact unpadded local patches; learned "
    "geometry newlines are structural, excluded from K, and always preserved"
)
DUET_NEXT_FEATURE_ORDER = (
    "global dominant+contextual; local dominant raster rows with one learned "
    "newline after each unpadded row; local contextual summaries"
)
DUET_NEXT_VIEW_POLICY = (
    "global reduced separately; local crops stitched tile-major to an unpadded "
    "raster mosaic and reduced jointly; K is split proportional to the two exact "
    "content capacities and each share uses the paper profile's DUET mixture"
)
DUET_NEXT_ROUNDING_RULE = "nearest integer, half up; clamp K to [1, N_original]"
DUET_NEXT_CONTENT_DEFINITION = (
    "base-view patch rows plus exact stitched-and-unpadded local-mosaic patch rows"
)
DUET_NEXT_STRUCTURAL_POLICY = (
    "learned image-newline rows are excluded from original/target/realized content "
    "counts and preserved at every decoder stage"
)
DUET_NEXT_SOURCE_COMMITS = {
    "paper": "arXiv:2602.18846v1",
    "official_duet": "cd69e489f39a2a51e3b278a470dc8dc32c00e0fb",
    "projectneura_reference": "dd8eb6af0e7b5f1fd970190aea579bfaa84c0301",
    "projectneura_trained_behavior": "e1886b4",
    "official_nuwa": "175b7d57a7890b9818dd26d82f2d91d9868b0d80",
    "varest_base": "475fc2d768132d39dd8974a41e4da3df1e572ef6",
    "conrad_launcher_reference": "llmoe_baseline_hidrop@1d17b95",
}


@dataclass(frozen=True)
class DuetNextBudget:
    paper_profile: int
    original_content_tokens: int
    requested_retention_ratio: float
    target_content_tokens: int
    realized_content_tokens: int
    realized_retention_ratio: float
    structural_tokens: int
    crop_count: int
    global_content_tokens: int
    local_content_tokens: int
    global_target_tokens: int
    local_target_tokens: int
    global_dominant_tokens: int
    global_contextual_tokens: int
    local_dominant_tokens: int
    local_contextual_tokens: int
    decoder_stage1_tokens: int
    stage_content_keep_counts: tuple[int, ...]
    stage_total_keep_counts: tuple[int, ...]
    content_layer_average: float
    geometry_inclusive_layer_average: float


def next_profile_per_view(
    dominant: int,
    contextual: int,
    *,
    reference_views: int = DUET_NEXT_REFERENCE_VIEWS,
) -> tuple[int, int]:
    """Return the exact published per-view split; reject lossy division."""

    views = int(reference_views)
    if views < 1:
        raise ValueError(f"reference_views must be positive, got {reference_views}")
    if int(dominant) % views or int(contextual) % views:
        raise ValueError(
            "LLaVA-NeXT DUET profiles are reference-image totals and must divide "
            f"exactly across {views} views; got dominant={dominant}, contextual={contextual}."
        )
    return int(dominant) // views, int(contextual) // views


def round_half_up_ratio(numerator: int, denominator: int) -> int:
    """Round a non-negative exact ratio to nearest integer, with ties upward."""

    numerator = int(numerator)
    denominator = int(denominator)
    if numerator < 0 or denominator < 1:
        raise ValueError("round_half_up_ratio requires numerator >= 0 and denominator >= 1")
    return (2 * numerator + denominator) // (2 * denominator)


def retention_target(original: int, ratio: float) -> int:
    """Per-sample ``max(1, round_half_up(r*N_original))`` without float drift."""

    original = int(original)
    if original < 1:
        raise ValueError("N_original must be positive")
    exact = Decimal(str(float(ratio))) * Decimal(original)
    rounded = int(exact.to_integral_value(rounding=ROUND_HALF_UP))
    return max(1, min(original, rounded))


def _proportional_split(total: int, first_capacity: int, second_capacity: int) -> tuple[int, int]:
    """Split an exact target by capacity while retaining both non-empty views."""

    total = int(total)
    first_capacity = int(first_capacity)
    second_capacity = int(second_capacity)
    capacity = first_capacity + second_capacity
    if not 0 <= total <= capacity or first_capacity < 0 or second_capacity < 0:
        raise ValueError("Invalid proportional target/capacities")
    if second_capacity == 0:
        return total, 0
    if total == 1:
        return 1, 0
    first = round_half_up_ratio(total * first_capacity, capacity)
    first = min(first_capacity, max(1, first))
    second = total - first
    if second < 1:
        second = 1
        first = total - 1
    if second > second_capacity:
        second = second_capacity
        first = total - second
    return first, second


def _profile_split(total: int, dominant: int, contextual: int) -> tuple[int, int]:
    """Nearest feasible dominant/contextual split that sums exactly to ``total``."""

    contextual_target = round_half_up_ratio(int(total) * int(contextual), dominant + contextual)
    return int(total) - contextual_target, contextual_target


def next_budget_for_geometry(owner: Any, geometry: Any) -> DuetNextBudget:
    global_content = int(geometry.patches_per_view)
    local_content = int(geometry.local_patch_count)
    original = global_content + local_content
    ratio = float(owner._retain_ratio)
    target = retention_target(original, ratio)
    global_target, local_target = _proportional_split(
        target, global_content, local_content
    )
    profile_dominant, profile_contextual = owner._llava_next_profile_counts
    global_dominant, global_contextual = _profile_split(
        global_target, profile_dominant, profile_contextual
    )
    local_dominant, local_contextual = _profile_split(
        local_target, profile_dominant, profile_contextual
    )
    structural = int(geometry.newline_count)
    total = target + structural
    content_keeps = tuple(
        max(0, int(target * float(stage_ratio)))
        for stage_ratio in owner._pdrop_keep_ratios
    )
    total_keeps = tuple(keep + structural for keep in content_keeps)
    content_average = average_image_tokens(
        target,
        owner._pdrop_layers,
        owner._pdrop_keep_ratios,
        owner._llava_next_decoder_layers,
    )
    # Structural rows survive every decoder layer, unlike content rows.
    inclusive_average = content_average + structural
    return DuetNextBudget(
        paper_profile=int(owner._llava_next_profile),
        original_content_tokens=original,
        requested_retention_ratio=ratio,
        target_content_tokens=target,
        realized_content_tokens=target,
        realized_retention_ratio=target / original,
        structural_tokens=structural,
        crop_count=int(geometry.crop_count),
        global_content_tokens=global_content,
        local_content_tokens=local_content,
        global_target_tokens=global_target,
        local_target_tokens=local_target,
        global_dominant_tokens=global_dominant,
        global_contextual_tokens=global_contextual,
        local_dominant_tokens=local_dominant,
        local_contextual_tokens=local_contextual,
        decoder_stage1_tokens=total,
        stage_content_keep_counts=content_keeps,
        stage_total_keep_counts=total_keeps,
        content_layer_average=content_average,
        geometry_inclusive_layer_average=inclusive_average,
    )


def _realize_plan(
    features: torch.Tensor,
    plan: Stage1Plan,
    *,
    track_source: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply :func:`duet_stage1_plan` while retaining its representative indices."""

    batch, token_count, channels = features.shape
    dominant = int(plan.dominant_index.shape[1])
    contextual = int(plan.anchor_index.shape[1])
    dominant_tokens = features.gather(
        1,
        plan.dominant_index.unsqueeze(-1).expand(batch, dominant, channels),
    )
    source: torch.Tensor | None = None
    if track_source:
        source = torch.zeros(
            batch,
            dominant,
            token_count,
            dtype=torch.float32,
            device=features.device,
        )
        if dominant:
            source.scatter_(2, plan.dominant_index.unsqueeze(-1), 1.0)
    if not contextual:
        return dominant_tokens, source

    anchors = features.gather(
        1,
        plan.anchor_index.unsqueeze(-1).expand(batch, contextual, channels),
    )
    if plan.assign_one_hot is None or plan.counts is None:
        contextual_tokens = anchors
    else:
        merged = features.gather(
            1,
            plan.merge_index.unsqueeze(-1).expand(
                batch, plan.merge_index.shape[1], channels
            ),
        )
        contextual_tokens = anchors + torch.bmm(
            plan.assign_one_hot.transpose(1, 2), merged
        ) / plan.counts
    if source is not None:
        contextual_source = torch.zeros(
            batch,
            contextual,
            token_count,
            dtype=torch.float32,
            device=features.device,
        )
        contextual_source.scatter_(2, plan.anchor_index.unsqueeze(-1), 1.0)
        if plan.assign_one_hot is not None and plan.counts is not None:
            contextual_source.scatter_add_(
                2,
                plan.merge_index.unsqueeze(1).expand(
                    batch, contextual, plan.merge_index.shape[1]
                ),
                plan.assign_one_hot.transpose(1, 2).float() / plan.counts.float(),
            )
        source = torch.cat((source, contextual_source), dim=1)
    return torch.cat((dominant_tokens, contextual_tokens), dim=1), source


class LlavaNextDuetAdapter:
    """Pre-projector DUET reduction and native AnyRes packing lifecycle."""

    def __init__(self, owner: Any, vlm: Any) -> None:
        self._owner = owner
        self._vlm = vlm
        self._inner = vlm.model.model
        self._feature_layer = int(owner._resolve_feature_layer(vlm))
        self._attention: torch.Tensor | None = None
        self._metric: torch.Tensor | None = None
        self._capture_handle: Any | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._original_generate: Any | None = None
        self._pending: tuple[tuple[int, int], int, int] | None = None
        self._logged = False

    def install(self) -> None:
        config = self._vlm.model.config
        if getattr(config, "model_type", None) != "llava_next":
            raise TypeError("DUET's LLaVA-NeXT adapter requires LlavaNextForConditionalGeneration")
        if getattr(getattr(config, "text_config", None), "model_type", None) != "llama":
            raise TypeError("DUET/LLaVA-NeXT requires the target Llama-3 decoder")
        validate_duet_next_checkpoint(self._owner, self._vlm)
        strategy = getattr(config, "vision_feature_select_strategy", "default")
        if strategy != "default":
            raise ValueError(
                "DUET/LLaVA-NeXT requires vision_feature_select_strategy='default'; "
                "CLS is a ranking signal but is not inserted into the stock decoder sequence."
            )
        layers = self._owner._vision_encoder_layers(self._vlm)
        selected = (
            layers[self._feature_layer]
            if self._feature_layer < 0
            else layers[self._feature_layer - 1]
        )
        original = getattr(self._inner, "get_image_features", None)
        if not callable(original):
            raise TypeError("LLaVA-NeXT model has no get_image_features method")

        self._original_get_image_features = original
        self._original_prepare_inputs = self._vlm.prepare_inputs
        self._original_generate = self._vlm.generate
        try:
            self._capture_handle = selected.self_attn.register_forward_pre_hook(
                self._capture_attention,
                with_kwargs=True,
            )
            self._inner.get_image_features = self._wrapped_get_image_features
            self._vlm.prepare_inputs = self._wrapped_prepare_inputs
            self._vlm.generate = self._wrapped_generate
        except Exception:
            self.uninstall()
            raise

    def uninstall(self) -> None:
        if self._capture_handle is not None:
            self._capture_handle.remove()
        if self._original_get_image_features is not None:
            self._inner.get_image_features = self._original_get_image_features
        if self._original_prepare_inputs is not None:
            self._vlm.prepare_inputs = self._original_prepare_inputs
        if self._original_generate is not None:
            self._vlm.generate = self._original_generate
        self._capture_handle = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._original_generate = None
        self._pending = None
        self._attention = None
        self._metric = None

    @staticmethod
    def validate_generation_kwargs(kwargs: Mapping[str, Any]) -> None:
        """Reject generation modes whose token flow DUET does not implement."""

        if int(kwargs.get("num_beams", 1)) != 1:
            raise NotImplementedError("DUET/LLaVA-NeXT does not support beam search")
        if int(kwargs.get("num_return_sequences", 1)) != 1:
            raise NotImplementedError(
                "DUET/LLaVA-NeXT supports one generated sequence per input"
            )
        if kwargs.get("use_cache", True) is False:
            raise NotImplementedError("DUET/LLaVA-NeXT requires the dynamic KV cache")
        cache_implementation = kwargs.get("cache_implementation")
        if cache_implementation not in (None, "dynamic"):
            raise NotImplementedError(
                "DUET/LLaVA-NeXT requires cache_implementation='dynamic'"
            )
        if kwargs.get("past_key_values") is not None:
            raise NotImplementedError(
                "DUET/LLaVA-NeXT does not support an externally primed cache"
            )

    def _wrapped_generate(self, inputs: Any, **kwargs: Any) -> torch.Tensor:
        if self._original_generate is None:
            raise RuntimeError("DUET/LLaVA-NeXT adapter is not installed")
        self.validate_generation_kwargs(kwargs)
        return self._original_generate(inputs, **kwargs)

    def _capture_attention(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            self._attention = self._metric = None
            return
        batch, sequence, _ = hidden.shape
        queries = module.q_proj(hidden).view(
            batch, sequence, module.num_heads, module.head_dim
        ).transpose(1, 2)
        keys = module.k_proj(hidden).view(
            batch, sequence, module.num_heads, module.head_dim
        ).transpose(1, 2)
        logits = (
            queries[:, :, :1].float() @ keys.float().transpose(-1, -2)
        ) * float(module.scale)
        self._attention = logits.softmax(dim=-1)[:, :, 0, 1:].sum(dim=1).detach()
        self._metric = keys.mean(dim=1)[:, 1:].contiguous().detach()

    @staticmethod
    def _argument(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str,
        position: int,
    ) -> Any:
        value = kwargs.get(name)
        if value is None and len(args) > position:
            value = args[position]
        return value

    def _target(self, image_size: torch.Tensor) -> tuple[Any, DuetNextBudget]:
        geometry = geometry_for_image(self._inner, image_size)
        budget = next_budget_for_geometry(self._owner, geometry)
        if budget.global_target_tokens > geometry.patches_per_view:
            raise ValueError(
                f"DUET requests {budget.global_target_tokens} global rows from "
                f"{geometry.patches_per_view} base-view patches"
            )
        if budget.local_target_tokens > geometry.local_patch_count:
            raise ValueError(
                f"DUET requests {budget.local_target_tokens} local rows after AnyRes unpadding leaves "
                f"only {geometry.local_patch_count} patches"
            )
        if budget.decoder_stage1_tokens > geometry.native_tokens:
            raise ValueError(
                f"DUET-{self._owner.budget} needs {budget.decoder_stage1_tokens} decoder rows "
                f"for this {geometry.crop_count}-view image, but native LLaVA-NeXT has only "
                f"{geometry.native_tokens}. Use a smaller published profile."
            )
        return geometry, budget

    def _wrapped_prepare_inputs(self, *args: Any, **kwargs: Any) -> Any:
        if self._original_prepare_inputs is None:
            raise RuntimeError("DUET/LLaVA-NeXT is not installed")
        inputs = self._original_prepare_inputs(*args, **kwargs)
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        image_sizes = inputs.get("image_sizes") if hasattr(inputs, "get") else None
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("DUET/LLaVA-NeXT requires tokenized input_ids")
        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                "DUET/LLaVA-NeXT physical decoder pruning supports batch_size=1 only"
            )
        if not isinstance(image_sizes, torch.Tensor) or tuple(image_sizes.shape) != (1, 2):
            raise NotImplementedError(
                "DUET/LLaVA-NeXT supports exactly one source image and no video/multi-image input"
            )
        token_id = image_token_id(self._vlm, label="DUET/LLaVA-NeXT")
        lengths = placeholder_run_lengths(inputs, token_id=token_id)
        if len(lengths) != 1:
            raise NotImplementedError(
                "DUET/LLaVA-NeXT requires exactly one contiguous image-placeholder run"
            )
        geometry, budget = self._target(image_sizes[0])
        if lengths[0] != geometry.native_tokens:
            raise RuntimeError(
                f"Processor expanded {lengths[0]} image tokens, while Transformers geometry "
                f"requires {geometry.native_tokens}. Check processor revision and metadata."
            )
        self._pending = (
            tuple(int(value) for value in image_sizes[0].detach().cpu().tolist()),
            lengths[0],
            budget.decoder_stage1_tokens,
        )
        return trim_image_placeholders_to_lengths(
            inputs,
            token_id=token_id,
            lengths=(budget.decoder_stage1_tokens,),
            pad_token_id=tokenizer_pad_token_id(self._vlm),
        )

    def _wrapped_get_image_features(self, *args: Any, **kwargs: Any) -> Any:
        if self._original_get_image_features is None:
            raise RuntimeError("DUET/LLaVA-NeXT is not installed")
        pixel_values = self._argument(args, kwargs, "pixel_values", 0)
        image_sizes = self._argument(args, kwargs, "image_sizes", 1)
        if not isinstance(image_sizes, torch.Tensor) or tuple(image_sizes.shape) != (1, 2):
            raise NotImplementedError(
                "DUET/LLaVA-NeXT supports one source image per forward"
            )
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim not in (4, 5):
            raise ValueError("LLaVA-NeXT pixel_values must have four or five dimensions")
        feature_layer = self._argument(args, kwargs, "vision_feature_layer", 2)
        if feature_layer is not None and int(feature_layer) != self._feature_layer:
            raise ValueError(
                f"DUET was installed for CLIP feature layer {self._feature_layer}, "
                f"but the forward requested {feature_layer}."
            )
        strategy = self._argument(args, kwargs, "vision_feature_select_strategy", 3)
        if strategy is not None and strategy != "default":
            raise ValueError("DUET/LLaVA-NeXT supports the checkpoint's default patch strategy")

        try:
            geometry, budget = self._target(image_sizes[0])
            if self._owner._image_start is None \
                    or self._owner._visual_count != budget.decoder_stage1_tokens:
                raise NotImplementedError(
                    "DUET/LLaVA-NeXT needs input_ids with the already-trimmed contiguous "
                    "image span; inputs_embeds-only and pixel-on-decode forwards are unsupported"
                )
            counts = crop_counts(self._inner, image_sizes)
            if pixel_values.ndim == 5:
                if pixel_values.shape[0] != 1 or pixel_values.shape[1] < counts[0]:
                    raise ValueError(
                        f"Pixel batch {tuple(pixel_values.shape[:2])} cannot supply "
                        f"the required {counts[0]} crops"
                    )
                flattened_pixels = pixel_values[0, : counts[0]]
            else:
                flattened_pixels = pixel_values
            if int(flattened_pixels.shape[0]) != sum(counts):
                raise RuntimeError(
                    f"LLaVA-NeXT expected {sum(counts)} crops but received "
                    f"{flattened_pixels.shape[0]}"
                )
            # This is the stock Transformers 5.5.4 get_image_features prefix,
            # stopped before its full-token projector/packer. Projecting all
            # native rows and then discarding them doubles Stage-1 memory during
            # training; DUET projects only the rows it actually emits.
            vision_kwargs = dict(kwargs)
            for key in (
                "pixel_values",
                "image_sizes",
                "vision_feature_layer",
                "vision_feature_select_strategy",
                "output_hidden_states",
                "return_dict",
            ):
                vision_kwargs.pop(key, None)
            outputs = self._inner.vision_tower(
                flattened_pixels,
                output_hidden_states=True,
                return_dict=True,
                **vision_kwargs,
            )
            if self._pending is not None:
                size, native_count, target = self._pending
                actual_size = tuple(int(value) for value in image_sizes[0].detach().cpu().tolist())
                if (size, native_count, target) != (
                    actual_size,
                    geometry.native_tokens,
                    budget.decoder_stage1_tokens,
                ):
                    raise RuntimeError(
                        "The image processed for features does not match the placeholder budget"
                    )
            packed = self._transform(outputs, image_sizes[0], geometry, budget)
            if int(packed.shape[0]) != budget.decoder_stage1_tokens:
                raise RuntimeError(
                    f"DUET packed {packed.shape[0]} rows; expected {budget.decoder_stage1_tokens}"
                )
            outputs.pooler_output = [packed]
            return outputs
        finally:
            self._pending = None
            self._attention = None
            self._metric = None

    def _transform(
        self,
        outputs: Any,
        image_size: torch.Tensor,
        geometry: Any,
        budget: DuetNextBudget,
    ) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states or self._attention is None or self._metric is None:
            raise RuntimeError(
                "DUET/LLaVA-NeXT did not capture CLIP features, CLS attention, and key metric"
            )
        raw = hidden_states[self._feature_layer][:, 1:]
        counts = crop_counts(self._inner, image_size.unsqueeze(0))
        features = split_crop_tensor(raw, counts, label="DUET features")[0]
        attention = split_crop_tensor(
            self._attention, counts, label="DUET CLS attention"
        )[0]
        metric = split_crop_tensor(self._metric, counts, label="DUET key metric")[0]
        patch_h, patch_w = patch_grid(self._inner)
        patches = patch_h * patch_w
        if tuple(features.shape[:2]) != (geometry.crop_count, patches):
            raise RuntimeError(
                f"DUET captured crop tensor {tuple(features.shape[:2])}; expected "
                f"({geometry.crop_count}, {patches})"
            )

        global_plan = duet_stage1_plan(
            attention[:1],
            metric[:1],
            dominant=budget.global_dominant_tokens,
            contextual=budget.global_contextual_tokens,
            cluster_width=self._owner._cluster_width,
            dtype=features.dtype,
        )
        global_tokens, global_source = _realize_plan(
            features[:1], global_plan, track_source=self._owner._track_source
        )
        projected_global = self._inner.multi_modal_projector(global_tokens)[0]

        pieces = [projected_global]
        structural_positions: list[int] = []
        packed_cursor = int(projected_global.shape[0])
        benchmark_parts = [
            torch.cat((global_plan.dominant_index[0], global_plan.anchor_index[0]))
        ]
        source_parts: list[torch.Tensor] = []
        if global_source is not None:
            block = torch.zeros(
                global_source.shape[1], geometry.native_tokens, dtype=torch.float32
            )
            block[:, :patches] = global_source[0].detach().float().cpu()
            source_parts.append(block)

        if geometry.crop_count > 1:
            local_features, local_shape = stitch_and_unpad_local_crops(
                self._inner, features[1:], image_size
            )
            local_attention, _ = stitch_and_unpad_local_crops(
                self._inner, attention[1:], image_size
            )
            local_metric, _ = stitch_and_unpad_local_crops(
                self._inner, metric[1:], image_size
            )
            if local_shape != (geometry.local_height, geometry.local_width):
                raise RuntimeError("DUET local mosaic shape does not match budget geometry")
            assembled_original = patches + int(local_features.shape[1])
            if assembled_original != budget.original_content_tokens:
                raise RuntimeError(
                    "DUET's precomputed AnyRes budget does not match the content rows "
                    f"captured after assembly/unpadding: {budget.original_content_tokens} "
                    f"!= {assembled_original}"
                )
            local_dominant = budget.local_dominant_tokens
            local_contextual = budget.local_contextual_tokens
            if budget.local_target_tokens:
                local_plan = duet_stage1_plan(
                    local_attention.squeeze(-1)
                    if local_attention.ndim == 3 else local_attention,
                    local_metric,
                    dominant=local_dominant,
                    contextual=local_contextual,
                    cluster_width=self._owner._cluster_width,
                    dtype=local_features.dtype,
                )
                local_tokens, local_source = _realize_plan(
                    local_features, local_plan, track_source=self._owner._track_source
                )
                projected_local = self._inner.multi_modal_projector(local_tokens)[0]
            else:
                empty = torch.empty(1, 0, dtype=torch.long, device=features.device)
                local_plan = Stage1Plan(empty, empty, empty, None, None)
                projected_local = projected_global.new_empty(
                    0, projected_global.shape[-1]
                )
                local_source = (
                    torch.zeros(
                        1, 0, geometry.local_patch_count, dtype=torch.float32,
                        device=features.device,
                    )
                    if self._owner._track_source else None
                )
            dominant_indices = local_plan.dominant_index[0]
            cursor = 0
            for row in range(geometry.local_height):
                count = int((dominant_indices // geometry.local_width == row).sum())
                if count:
                    selected = dominant_indices[cursor : cursor + count]
                    pieces.append(projected_local[cursor : cursor + count])
                    packed_cursor += count
                    benchmark_parts.append(patches + selected + selected // geometry.local_width)
                    if local_source is not None:
                        block = torch.zeros(
                            count, geometry.native_tokens, dtype=torch.float32
                        )
                        local_columns = (
                            patches
                            + torch.arange(geometry.local_patch_count)
                            + torch.arange(geometry.local_patch_count) // geometry.local_width
                        ).long()
                        block[:, local_columns] = local_source[
                            0, cursor : cursor + count
                        ].detach().float().cpu()
                        source_parts.append(block)
                    cursor += count
                newline_column = patches + (row + 1) * geometry.local_width + row
                structural_positions.append(packed_cursor)
                pieces.append(self._inner.image_newline[None].to(projected_global))
                packed_cursor += 1
                benchmark_parts.append(
                    torch.tensor([newline_column], device=features.device, dtype=torch.long)
                )
                if local_source is not None:
                    newline_source = torch.zeros(
                        1, geometry.native_tokens, dtype=torch.float32
                    )
                    newline_source[0, newline_column] = 1.0
                    source_parts.append(newline_source)
            if cursor != local_dominant:
                raise RuntimeError(
                    f"DUET placed {cursor} of {local_dominant} local dominant rows"
                )
            if local_contextual:
                pieces.append(projected_local[local_dominant:])
                packed_cursor += local_contextual
                anchors = local_plan.anchor_index[0]
                benchmark_parts.append(patches + anchors + anchors // geometry.local_width)
                if local_source is not None:
                    block = torch.zeros(
                        local_contextual, geometry.native_tokens, dtype=torch.float32
                    )
                    local_columns = (
                        patches
                        + torch.arange(geometry.local_patch_count)
                        + torch.arange(geometry.local_patch_count) // geometry.local_width
                    ).long()
                    block[:, local_columns] = local_source[
                        0, local_dominant:
                    ].detach().float().cpu()
                    source_parts.append(block)
        else:
            if patches != budget.original_content_tokens:
                raise RuntimeError(
                    "DUET's single-view budget does not match captured patch content"
                )
            newline_column = patches
            structural_positions.append(packed_cursor)
            pieces.append(self._inner.image_newline[None].to(projected_global))
            packed_cursor += 1
            benchmark_parts.append(
                torch.tensor([newline_column], device=features.device, dtype=torch.long)
            )
            if self._owner._track_source:
                newline_source = torch.zeros(1, geometry.native_tokens, dtype=torch.float32)
                newline_source[0, newline_column] = 1.0
                source_parts.append(newline_source)

        packed = torch.cat(pieces, dim=0)
        if packed_cursor != int(packed.shape[0]):
            raise RuntimeError("DUET/LLaVA-NeXT packed-cursor bookkeeping diverged")
        if len(structural_positions) != budget.structural_tokens:
            raise RuntimeError(
                f"DUET packed {len(structural_positions)} structural rows; expected "
                f"{budget.structural_tokens}"
            )
        self._owner._llava_next_structural_positions = torch.tensor(
            structural_positions, dtype=torch.long, device=packed.device
        )
        self._owner._llava_next_content_count = budget.realized_content_tokens
        # `_capture_span` runs before feature assembly and initially sees total
        # decoder rows. Replace that provisional schedule with content-only
        # targets now that exact AnyRes geometry and structural positions exist.
        self._owner._keep_counts = budget.stage_content_keep_counts
        benchmark = torch.cat(benchmark_parts).long()
        self._owner._benchmark_indices = benchmark.unsqueeze(0)
        self._owner._llava_next_run_metadata = {
            **asdict(budget),
            "image_size": [int(value) for value in image_size.detach().cpu().tolist()],
            "native_tokens": geometry.native_tokens,
            "patches_per_view": geometry.patches_per_view,
            "tile_rows": geometry.tile_rows,
            "tile_columns": geometry.tile_columns,
            "local_height": geometry.local_height,
            "local_width": geometry.local_width,
            "feature_order": "global_duet_then_local_raster_newlines_then_local_context",
        }
        if self._owner._track_source:
            source = torch.cat(source_parts, dim=0)
            if tuple(source.shape) != (budget.decoder_stage1_tokens, geometry.native_tokens):
                raise RuntimeError(
                    f"DUET source map has shape {source.shape}; expected "
                    f"({budget.decoder_stage1_tokens}, {geometry.native_tokens})"
                )
            identity = torch.eye(geometry.native_tokens, dtype=torch.float32)
            self._owner._stage1_source = source
            self._owner._source_history = [identity, source]
            self._owner._composed_source = source
        if not self._logged:
            self._logged = True
            print(
                f"[DUET/LLaVA-NeXT] first image: native={geometry.native_tokens}, "
                f"views={geometry.crop_count}, original_content={budget.original_content_tokens}, "
                f"target_content={budget.target_content_tokens}, "
                f"structural={budget.structural_tokens}, decoder_stage1={packed.shape[0]}, "
                f"realized_average={budget.geometry_inclusive_layer_average:g}",
                flush=True,
            )
        return packed


def duet_visualization_view(
    model: Any,
    inputs: Mapping[str, Any],
    image_processor: Any,
) -> VisualizationView:
    """Describe DUET's packed NeXT sequence on a base/local image atlas."""

    pixel_values = inputs.get("pixel_values")
    image_sizes = inputs.get("image_sizes")
    if (
        not isinstance(pixel_values, torch.Tensor)
        or pixel_values.ndim != 5
        or pixel_values.shape[0] != 1
    ):
        raise ValueError("LLaVA-NeXT visualization requires [1, crops, C, H, W] pixels")
    if not isinstance(image_sizes, torch.Tensor) or tuple(image_sizes.shape) != (1, 2):
        raise ValueError("LLaVA-NeXT visualization requires one source image size")

    inner = model.model.model
    geometry = geometry_for_image(inner, image_sizes[0])
    count = crop_counts(inner, image_sizes)[0]
    crops = pixel_values[0, :count]
    patch_size = int(inner.config.vision_config.patch_size)
    global_canvas = pixel_values_to_canvas(crops[0], image_processor)
    atlas_width = max(geometry.patch_width, geometry.local_width)
    atlas_height = geometry.patch_height + geometry.local_height
    atlas = torch.zeros(3, atlas_height * patch_size, atlas_width * patch_size)
    atlas[:, : global_canvas.shape[1], : global_canvas.shape[2]] = global_canvas

    if geometry.crop_count > 1:
        tile_h, tile_w = anyres_grid_shape(inner, image_sizes[0])
        local = crops[1:].reshape(
            tile_h, tile_w, crops.shape[1], crops.shape[2], crops.shape[3]
        )
        local = local.permute(2, 0, 3, 1, 4).contiguous()
        local = local.flatten(1, 2).flatten(2, 3)
        local = unpad_image(local, image_sizes[0])
        local_canvas = pixel_values_to_canvas(local, image_processor)
        top = geometry.patch_height * patch_size
        atlas[:, top : top + local_canvas.shape[1], : local_canvas.shape[2]] = local_canvas

    native_columns: list[int] = []
    atlas_columns: list[int] = []
    for row in range(geometry.patch_height):
        for column in range(geometry.patch_width):
            native_columns.append(row * geometry.patch_width + column)
            atlas_columns.append(row * atlas_width + column)
    native_cursor = geometry.patches_per_view
    for row in range(geometry.local_height):
        for column in range(geometry.local_width):
            native_columns.append(native_cursor + column)
            atlas_columns.append((geometry.patch_height + row) * atlas_width + column)
        native_cursor += geometry.local_width + 1
    if geometry.crop_count == 1:
        native_cursor += 1
    if native_cursor != geometry.native_tokens:
        raise RuntimeError(
            f"NeXT visualization mapped {native_cursor} of {geometry.native_tokens} native rows"
        )
    return VisualizationView(
        name="native_atlas",
        canvas=atlas,
        source_columns=torch.tensor(native_columns, dtype=torch.long),
        canvas_columns=torch.tensor(atlas_columns, dtype=torch.long),
    )


def validate_duet_next_checkpoint(owner: Any, vlm: Any) -> None:
    """Fail closed for local checkpoints trained under another token flow.

    The pinned stock model is intentionally allowed without ``duet_config.json``
    as the clearly labelled training-free ablation. Any other local/HF model
    must carry compatibility metadata written by the NeXT trainer.
    """

    from llmoe.backbone.llava_next import LLAVA_NEXT_8B_MODEL_ID, LLAVA_NEXT_8B_REVISION

    model_id = str(getattr(vlm, "_model_id", ""))
    revision = str(getattr(vlm, "_model_revision", LLAVA_NEXT_8B_REVISION))
    if revision != LLAVA_NEXT_8B_REVISION:
        raise ValueError(
            f"DUET/LLaVA-NeXT requires revision {LLAVA_NEXT_8B_REVISION}; got {revision}"
        )
    path = Path(model_id).expanduser()
    metadata_path = path / "duet_config.json" if path.is_dir() else None
    if metadata_path is None or not metadata_path.is_file():
        is_stock = model_id == LLAVA_NEXT_8B_MODEL_ID or (
            LLAVA_NEXT_8B_REVISION in model_id
            and "models--llava-hf--llama3-llava-next-8b-hf" in model_id
        )
        if is_stock:
            owner._metadata_backbone = "llava_next"
            return
        raise ValueError(
            "DUET/LLaVA-NeXT refuses a non-stock checkpoint without duet_config.json; "
            "the file must identify the backbone revision, schedule, view scaling, and "
            f"budget convention. Missing beside {model_id!r}."
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = vlm.model.config
    vision = config.vision_config
    expected: dict[str, Any] = {
        "schema_version": 5,
        "implementation_version": DUET_NEXT_IMPLEMENTATION_VERSION,
        "backend": "llava-next-8b-llama3",
        "base_model_id": LLAVA_NEXT_8B_MODEL_ID,
        "base_model_revision": LLAVA_NEXT_8B_REVISION,
        "transformers_version": "5.5.4",
        "training_attention_backend": DUET_NEXT_TRAINING_ATTENTION_BACKEND,
        "training_flash_attn_version": DUET_NEXT_FLASH_ATTN_VERSION,
        "inference_attention_backend": DUET_NEXT_INFERENCE_ATTENTION_BACKEND,
        "paper_profile": int(owner._llava_next_profile),
        "requested_retention_ratio": float(owner._retain_ratio),
        "budget_semantics": DUET_NEXT_BUDGET_SEMANTICS,
        "rounding_rule": DUET_NEXT_ROUNDING_RULE,
        "content_definition": DUET_NEXT_CONTENT_DEFINITION,
        "structural_token_policy": DUET_NEXT_STRUCTURAL_POLICY,
        "profile_dominant_reference_total": int(owner._dominant),
        "profile_contextual_reference_total": int(owner._contextual),
        "reference_views": int(owner._llava_next_reference_views),
        "reference_content_tokens": 2880,
        "cluster_width": int(owner._cluster_width),
        "pdrop_layers": list(owner._pdrop_layers),
        "pdrop_keep_ratios": list(owner._pdrop_keep_ratios),
        "text_rater": owner._text_rater,
        "score_reduction": owner._score_reduction,
        "vision_feature_layer": owner._resolve_feature_layer(vlm),
        "vision_feature_select_strategy": "default",
        "include_cls_in_decoder": False,
        "feature_order": DUET_NEXT_FEATURE_ORDER,
        "view_policy": DUET_NEXT_VIEW_POLICY,
        "use_stage1": True,
        "use_stage2": True,
        "training_stage": "finetune",
        "source_commits": dict(DUET_NEXT_SOURCE_COMMITS),
        "image_grid_pinpoints": [list(map(int, point)) for point in config.image_grid_pinpoints],
        "vision_geometry": {
            "image_size": int(vision.image_size),
            "patch_size": int(vision.patch_size),
            "patches_per_view": int(vision.image_size // vision.patch_size) ** 2,
        },
    }
    mismatches = [
        f"{key}: checkpoint={payload.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Incompatible DUET/LLaVA-NeXT checkpoint metadata:\n  "
            + "\n  ".join(mismatches)
        )
    owner._metadata_backbone = "llava_next"


__all__ = [
    "DUET_NEXT_BUDGET_SEMANTICS",
    "DUET_NEXT_CONTENT_DEFINITION",
    "DUET_NEXT_FLASH_ATTN_VERSION",
    "DUET_NEXT_FEATURE_ORDER",
    "DUET_NEXT_IMPLEMENTATION_VERSION",
    "DUET_NEXT_INFERENCE_ATTENTION_BACKEND",
    "DUET_NEXT_REFERENCE_VIEWS",
    "DUET_NEXT_ROUNDING_RULE",
    "DUET_NEXT_SOURCE_COMMITS",
    "DUET_NEXT_STRUCTURAL_POLICY",
    "DUET_NEXT_VIEW_POLICY",
    "DUET_NEXT_TRAINING_ATTENTION_BACKEND",
    "DuetNextBudget",
    "LlavaNextDuetAdapter",
    "next_budget_for_geometry",
    "next_profile_per_view",
    "round_half_up_ratio",
    "retention_target",
    "validate_duet_next_checkpoint",
]
