"""Ratio-budgeted VisionZip++ policy on native Transformers LLaVA-NeXT.

The released VisionZip NeXT implementation does *not* prune one already-packed
Transformers visual span. It compresses the base view and every AnyRes crop
independently before projection, then emits:

``global [CLS, dominant, context]`` -> ``local dominant in mosaic raster order``
-> ``local CLS in tile order`` -> ``local context in tile order``.

The released crop-local reduction is retained, but its image-wide target is
anchored to the exact native Transformers content sequence: the 576 base-view
patches plus the local AnyRes mosaic *after* native unpadding. Learned newline
rows are structural, are preserved at every native row boundary, and never
enter the retention denominator. Placeholder, label, mask, and position
compaction is performed before the Llama-3 prefill, so the native decoder and
DynamicCache see one ordinary compact sequence thereafter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

import torch


TARGET_MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
TARGET_MODEL_REVISION = "b041c0d0ea0dd0196d147206c210c8d1752fc2da"
TRAINING_ATTN_IMPLEMENTATION = "flash_attention_2"
FLASH_ATTN_VERSION = "2.8.3.post1"
TARGET_TRANSFORMERS_VERSION = "5.5.4"
TARGET_IMAGE_GRID_PINPOINTS = (
    (336, 672),
    (672, 336),
    (672, 672),
    (1008, 336),
    (336, 1008),
)

VISIONZIP_REPOSITORY = "https://github.com/JIA-Lab-research/VisionZip"
VISIONZIP_COMMIT = "8f86b55c6f000eb033e6912538af2dd7dcb30502"
PROJECTNEURA_REPOSITORY = "https://github.com/ProjectNeura/llmoe_baseline_vision_zip"
PROJECTNEURA_COMMIT = "3038b4a109aba0bd83aa728faa757680e7b8acd2"
NUWA_REPOSITORY = "https://github.com/Man-PaperRejected/Nuwa"
NUWA_COMMIT = "175b7d57a7890b9818dd26d82f2d91d9868b0d80"
VAREST_BASE_COMMIT = "475fc2d768132d39dd8974a41e4da3df1e572ef6"
VAREST_LLAVA_NEXT_HISTORY_COMMIT = "bc12414657741e7862e01ff8c7444b4643d77674"

IMPLEMENTATION_VERSION = "2.0.0"
CHECKPOINT_FORMAT_VERSION = 4
TRAINING_STAGE = "efficient_tuning"
TRAINING_SCOPE = "multi_modal_projector_only"
PROJECTOR_MODULE_PATH = "model.multi_modal_projector"

BUDGET_SEMANTICS = "per_image_ratio_of_native_unpadded_content_tokens"
VIEW_POLICY = "image_target_waterfilled_over_global_and_unpadded_local_crop_regions"
OUTPUT_ORDER = (
    "global_then_local_mosaic_dominant_with_native_newlines_then_local_cls_then_local_context"
)
NEWLINE_POLICY = "preserve_every_native_image_newline_excluded_from_content_budget"
ROUNDING_POLICY = "nearest_integer_half_up_max_1"

# Fields that define an evaluation-eligible projector. Runtime-only details
# such as the dataset mount and DDP accumulation may vary while preserving the
# same effective recipe; those remain recorded, but are not compatibility keys.
FINAL_TRAINING_RECIPE: dict[str, Any] = {
    "stage": "efficient_tuning",
    "dataset": "LLaVA-NeXT-Data image-bearing train split",
    "dataset_total_rows": 779289,
    "dataset_image_rows": 738601,
    "dataset_shards": 250,
    "subset_frac": 0.1,
    "num_train_epochs": 1.0,
    "learning_rate": 2e-5,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "warmup_rounding": "floor_int_like_projectneura",
    "lr_scheduler_type": "cosine",
    "global_batch_size": 128,
    "model_max_length": 2048,
    "deepspeed_stage": 0,
    "dataset_limit": None,
    "max_steps": -1,
    "stop_after_steps": None,
    "precision": "bf16",
    "attn_implementation": TRAINING_ATTN_IMPLEMENTATION,
    "flash_attn_version": FLASH_ATTN_VERSION,
    "initial_checkpoint_path": None,
    "seed": 42,
}


@dataclass(frozen=True)
class LlavaNextVisionZipLayout:
    """AnyRes views and output geometry for one source image."""

    view_count: int
    local_crop_count: int
    tile_rows: int
    tile_columns: int
    original_content_tokens: int
    requested_retention_ratio: float
    target_content_tokens: int
    realized_content_tokens: int
    structural_tokens: int
    crop_capacities: tuple[int, ...]
    crop_budgets: tuple[int, ...]
    unpadded_rows: int
    unpadded_columns: int

    @property
    def realized_total(self) -> int:
        return self.realized_content_tokens + self.structural_tokens

    @property
    def realized_retention_ratio(self) -> float:
        return self.realized_content_tokens / self.original_content_tokens


@dataclass(frozen=True)
class LlavaNextBudgetReport:
    """Requested and realized budget from the latest prefill."""

    layouts: tuple[LlavaNextVisionZipLayout, ...]

    @property
    def total_views(self) -> int:
        return sum(layout.view_count for layout in self.layouts)

    @property
    def original_content_tokens(self) -> int:
        return sum(layout.original_content_tokens for layout in self.layouts)

    @property
    def requested_retention_ratio(self) -> float:
        ratios = {layout.requested_retention_ratio for layout in self.layouts}
        if len(ratios) != 1:
            raise RuntimeError("A budget report contains multiple retention ratios")
        return next(iter(ratios))

    @property
    def target_content_tokens(self) -> int:
        return sum(layout.target_content_tokens for layout in self.layouts)

    @property
    def realized_content_tokens(self) -> int:
        return sum(layout.realized_content_tokens for layout in self.layouts)

    @property
    def structural_tokens(self) -> int:
        return sum(layout.structural_tokens for layout in self.layouts)

    @property
    def realized_retention_ratio(self) -> float:
        return self.realized_content_tokens / self.original_content_tokens

    @property
    def realized_total(self) -> int:
        return sum(layout.realized_total for layout in self.layouts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_semantics": BUDGET_SEMANTICS,
            "original_content_tokens": self.original_content_tokens,
            "requested_retention_ratio": self.requested_retention_ratio,
            "target_content_tokens": self.target_content_tokens,
            "realized_content_tokens": self.realized_content_tokens,
            "realized_retention_ratio": self.realized_retention_ratio,
            "structural_tokens": self.structural_tokens,
            "realized_total": self.realized_total,
            "total_views": self.total_views,
            "view_policy": VIEW_POLICY,
            "output_order": OUTPUT_ORDER,
            "newline_policy": NEWLINE_POLICY,
            "rounding_policy": ROUNDING_POLICY,
            "layouts": [asdict(layout) for layout in self.layouts],
        }


class CropReduction(NamedTuple):
    tokens: torch.Tensor
    dominant_patch_indices: torch.Tensor
    contextual_count: int
    source: torch.Tensor | None


@dataclass(frozen=True)
class _NativeGeometry:
    """Exact native content domain and raw-crop mappings for one image."""

    capacities: tuple[int, ...]
    eligible_patch_indices: tuple[torch.Tensor, ...]
    local_mosaic_positions: tuple[torch.Tensor, ...]
    unpadded_rows: int
    unpadded_columns: int
    structural_tokens: int

    @property
    def original_content_tokens(self) -> int:
        return sum(self.capacities)


def _native_geometry(
    inner: Any,
    image_size: torch.Tensor | Sequence[int],
    views: int,
) -> _NativeGeometry:
    """Map native unpadded content rows back to their CLIP crop patches."""

    from llmoe.compressor.llava_next import stitch_and_unpad_local_crops

    patches = 24 * 24
    all_base = torch.arange(patches, dtype=torch.long)
    if int(views) == 1:
        return _NativeGeometry(
            capacities=(patches,),
            eligible_patch_indices=(all_base,),
            local_mosaic_positions=(),
            unpadded_rows=0,
            unpadded_columns=0,
            structural_tokens=1,
        )

    local_count = int(views) - 1
    encoded = torch.arange(local_count * patches, dtype=torch.long).reshape(local_count, patches)
    mosaic, (height, width) = stitch_and_unpad_local_crops(inner, encoded, image_size)
    flattened = mosaic[0].reshape(-1).to(torch.long)
    eligible: list[torch.Tensor] = [all_base]
    positions: list[torch.Tensor] = []
    capacities = [patches]
    for tile in range(local_count):
        selected_positions = ((flattened // patches) == tile).nonzero(as_tuple=True)[0]
        raw_indices = flattened.index_select(0, selected_positions) % patches
        eligible.append(raw_indices)
        positions.append(selected_positions)
        capacities.append(int(raw_indices.numel()))
    return _NativeGeometry(
        capacities=tuple(capacities),
        eligible_patch_indices=tuple(eligible),
        local_mosaic_positions=tuple(positions),
        unpadded_rows=height,
        unpadded_columns=width,
        structural_tokens=height,
    )


def retention_target(original_content_tokens: int, retain_ratio: float) -> int:
    """Return ``max(1, round_half_up(r * N_original))`` for one image."""

    import math

    from llmoe.compressor.llava_next import validate_retain_ratio

    count = int(original_content_tokens)
    ratio = validate_retain_ratio(retain_ratio, label="VisionZip++/LLaVA-NeXT")
    if count < 1:
        raise ValueError("VisionZip++ cannot budget an empty native content sequence")
    return max(1, min(count, math.floor(ratio * count + 0.5)))


def _crop_split(budget: int) -> tuple[int, int]:
    """Scale the released 54:10 split to one dynamically allocated crop budget."""

    budget = int(budget)
    if budget <= 0:
        return 0, 0
    contextual = min(budget - 1, int(budget * 10 / 64 + 0.5))
    return budget - contextual, contextual


def is_llava_next_backbone(vlm: Any) -> bool:
    if getattr(vlm, "compressor_family", None) == "llava_next":
        return True
    model = getattr(vlm, "_model", None)
    if model is None:
        return False
    return getattr(getattr(model, "config", None), "model_type", None) == "llava_next"


def crop_count(config: Any, image_size: torch.Tensor | Sequence[int]) -> int:
    from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

    return int(
        image_size_to_num_patches(
            image_size=image_size,
            grid_pinpoints=config.image_grid_pinpoints,
            patch_size=config.vision_config.image_size,
        )
    )


def anyres_grid_shape(config: Any, image_size: torch.Tensor | Sequence[int]) -> tuple[int, int]:
    from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape

    rows, columns = get_anyres_image_grid_shape(
        image_size,
        config.image_grid_pinpoints,
        config.vision_config.image_size,
    )
    return int(rows), int(columns)


def reduce_crop(
    features: torch.Tensor,
    attention: torch.Tensor,
    keys: torch.Tensor,
    *,
    dominant: int,
    contextual: int,
    eligible_patch_indices: torch.Tensor | None = None,
    track_source: bool = False,
) -> CropReduction:
    """Apply the released CLS-attention selection and key-similarity merge."""

    if features.ndim != 2 or attention.ndim != 1 or keys.ndim != 2:
        raise ValueError(
            "VisionZip crop inputs must be [CLS+patches, hidden], [patches], and "
            f"[patches, metric], got {tuple(features.shape)}, {tuple(attention.shape)}, "
            f"and {tuple(keys.shape)}"
        )
    patch_count = int(features.shape[0]) - 1
    dominant_patches = int(dominant) - 1
    contextual = int(contextual)
    if attention.shape[0] != patch_count or keys.shape[0] != patch_count:
        raise ValueError("VisionZip attention/key rows do not match the CLIP patch rows")
    if int(dominant) == 0 and contextual == 0:
        empty = features[:0]
        source = (
            torch.zeros(0, patch_count + 1, dtype=torch.float32, device=features.device)
            if track_source
            else None
        )
        return CropReduction(
            tokens=empty,
            dominant_patch_indices=torch.empty(0, dtype=torch.long, device=features.device),
            contextual_count=0,
            source=source,
        )
    if dominant_patches < 0 or contextual < 0:
        raise ValueError("VisionZip dominant must include CLS and contextual must be non-negative")
    eligible = (
        torch.arange(patch_count, dtype=torch.long, device=features.device)
        if eligible_patch_indices is None
        else eligible_patch_indices.to(device=features.device, dtype=torch.long)
    )
    if eligible.ndim != 1 or eligible.numel() != eligible.unique().numel():
        raise ValueError("VisionZip eligible patch indices must be a unique 1D tensor")
    if eligible.numel() and (int(eligible.min()) < 0 or int(eligible.max()) >= patch_count):
        raise ValueError("VisionZip eligible patch indices are outside the CLIP patch grid")
    eligible_count = int(eligible.numel())
    if dominant_patches + contextual > eligible_count:
        raise ValueError(
            f"VisionZip split {dominant}+{contextual} cannot fit CLS+"
            f"{eligible_count} eligible patches"
        )

    device = features.device
    if dominant_patches:
        local_selected = attention.index_select(0, eligible).topk(dominant_patches).indices
        selected_indices = eligible.index_select(0, local_selected).sort().values
        selected = features[1:].index_select(0, selected_indices)
    else:
        selected_indices = torch.empty(0, dtype=torch.long, device=device)
        selected = features[:0]
    kept = torch.zeros(patch_count, dtype=torch.bool, device=device)
    kept[selected_indices] = True
    pieces = [features[:1], selected]

    source: torch.Tensor | None = None
    if track_source:
        source = torch.zeros(
            int(dominant) + contextual,
            patch_count + 1,
            dtype=torch.float32,
            device=device,
        )
        source[0, 0] = 1.0
        if dominant_patches:
            source[
                torch.arange(1, 1 + dominant_patches, device=device),
                selected_indices + 1,
            ] = 1.0

    if contextual:
        leftover = eligible[~kept.index_select(0, eligible)]
        remaining_features = features[1:].index_select(0, leftover)
        metric = keys.index_select(0, leftover).float()
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        remaining = int(leftover.numel())
        step = max(1, remaining // contextual)
        targets = torch.arange(0, remaining, step, device=device)[:contextual]
        if targets.numel() != contextual:
            raise RuntimeError("VisionZip could not form the requested contextual targets")
        target_mask = torch.zeros(remaining, dtype=torch.bool, device=device)
        target_mask[targets] = True
        merge_mask = ~target_mask

        target_metric = metric.index_select(0, targets)
        merge_metric = metric[merge_mask]
        if merge_metric.shape[0]:
            assignment = (merge_metric @ target_metric.transpose(0, 1)).argmax(dim=1)
            assignment_matrix = torch.zeros(
                merge_metric.shape[0],
                contextual,
                dtype=features.dtype,
                device=device,
            )
            assignment_matrix.scatter_(1, assignment.unsqueeze(-1), 1)
            counts = assignment_matrix.sum(dim=0).clamp_min(1).unsqueeze(-1)
            aggregate = assignment_matrix.transpose(0, 1) @ remaining_features[merge_mask]
            aggregate = aggregate / counts
        else:
            assignment_matrix = torch.zeros(0, contextual, dtype=features.dtype, device=device)
            counts = torch.ones(contextual, 1, dtype=features.dtype, device=device)
            aggregate = torch.zeros_like(remaining_features[targets])
        contextual_tokens = remaining_features[targets] + aggregate
        pieces.append(contextual_tokens.to(features.dtype))

        if source is not None:
            rows = torch.arange(int(dominant), int(dominant) + contextual, device=device)
            target_global = leftover[targets] + 1
            source[rows, target_global] = 1.0
            merge_global = leftover[merge_mask] + 1
            if merge_global.numel():
                source[rows.unsqueeze(1), merge_global.unsqueeze(0)] += (
                    assignment_matrix.transpose(0, 1).float() / counts.float()
                )

    return CropReduction(
        tokens=torch.cat(pieces, dim=0),
        dominant_patch_indices=selected_indices,
        contextual_count=contextual,
        source=source,
    )


def pack_official_order(
    projected: torch.Tensor,
    reductions: Sequence[CropReduction],
    *,
    tile_rows: int,
    tile_columns: int,
    patch_rows: int,
    patch_columns: int,
    image_newline: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack reduced views in the exact order used by the VisionZip release.

    ``row_order`` indexes a conceptual concatenation of the per-view reduced
    rows. ``-1`` denotes the learned single-view newline.
    """

    if projected.ndim != 3 or len(reductions) != projected.shape[0]:
        raise ValueError("Projected views and VisionZip crop reductions disagree")
    view_count = int(projected.shape[0])
    budget = int(projected.shape[1])
    device = projected.device
    if view_count == 1:
        output = projected[0]
        order = torch.arange(budget, dtype=torch.long, device=device)
        if image_newline is not None:
            output = torch.cat((output, image_newline[None].to(output)), dim=0)
            order = torch.cat((order, torch.tensor([-1], dtype=torch.long, device=device)))
        return output, order

    local_count = view_count - 1
    if local_count != int(tile_rows) * int(tile_columns):
        raise ValueError(
            f"VisionZip received {local_count} local crops for a "
            f"{tile_rows}x{tile_columns} AnyRes grid"
        )

    pieces = [projected[0]]
    orders = [torch.arange(budget, dtype=torch.long, device=device)]
    spatial_positions: list[torch.Tensor] = []
    spatial_rows: list[torch.Tensor] = []
    spatial_order: list[torch.Tensor] = []
    cls_rows: list[torch.Tensor] = []
    cls_order: list[torch.Tensor] = []
    context_rows: list[torch.Tensor] = []
    context_order: list[torch.Tensor] = []

    for local_index, reduction in enumerate(reductions[1:]):
        tensor = projected[local_index + 1]
        offset = (local_index + 1) * budget
        selected = reduction.dominant_patch_indices
        selected_count = int(selected.numel())
        if selected_count:
            patch_y = selected // int(patch_columns)
            patch_x = selected % int(patch_columns)
            tile_y, tile_x = divmod(local_index, int(tile_columns))
            position = (
                (tile_y * int(patch_rows) + patch_y) * (int(tile_columns) * int(patch_columns))
                + tile_x * int(patch_columns)
                + patch_x
            )
            spatial_positions.append(position)
            spatial_rows.append(tensor[1 : 1 + selected_count])
            spatial_order.append(offset + torch.arange(1, 1 + selected_count, device=device))
        cls_rows.append(tensor[:1])
        cls_order.append(torch.tensor([offset], device=device))
        if reduction.contextual_count:
            context_rows.append(tensor[-reduction.contextual_count :])
            context_order.append(
                offset
                + torch.arange(
                    budget - reduction.contextual_count,
                    budget,
                    device=device,
                )
            )

    if spatial_rows:
        positions = torch.cat(spatial_positions)
        sorting = positions.argsort(stable=True)
        pieces.append(torch.cat(spatial_rows)[sorting])
        orders.append(torch.cat(spatial_order)[sorting])
    pieces.append(torch.cat(cls_rows))
    orders.append(torch.cat(cls_order))
    if context_rows:
        pieces.append(torch.cat(context_rows))
        orders.append(torch.cat(context_order))
    return torch.cat(pieces), torch.cat(orders)


def pack_ratio_order(
    projected: Sequence[torch.Tensor],
    reductions: Sequence[CropReduction],
    geometry: _NativeGeometry,
    *,
    image_newline: torch.Tensor,
    track_source: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pack crop-local reductions while retaining every native newline row.

    Global output remains first. Local dominant patches are restored to the
    unpadded mosaic's raster order and each native row keeps its newline.
    Crop CLS representatives and contextual summaries follow the spatial block
    in tile-major order, matching VisionZip's released LLaVA-NeXT policy.
    """

    if len(projected) != len(reductions) or len(projected) != len(geometry.capacities):
        raise ValueError("Projected views, reductions, and native geometry disagree")
    if not projected:
        raise ValueError("VisionZip requires at least the global image view")
    device = projected[0].device
    dtype = projected[0].dtype
    hidden = int(projected[0].shape[-1])
    raw_width = len(projected) * 577
    source_width = raw_width + geometry.structural_tokens
    pieces: list[torch.Tensor] = []
    source_pieces: list[torch.Tensor] = []

    def append(rows: torch.Tensor, source: torch.Tensor | None, view: int) -> None:
        if not rows.numel():
            return
        pieces.append(rows)
        if track_source:
            if source is None:
                raise RuntimeError("VisionZip source tracking was requested but not captured")
            expanded = torch.zeros(source.shape[0], source_width, dtype=torch.float32)
            start = int(view) * 577
            expanded[:, start : start + 577] = source.detach().cpu().float()
            source_pieces.append(expanded)

    append(projected[0], reductions[0].source, 0)

    if len(projected) == 1:
        pieces.append(image_newline[None].to(device=device, dtype=dtype))
        if track_source:
            structural = torch.zeros(1, source_width, dtype=torch.float32)
            structural[0, raw_width] = 1.0
            source_pieces.append(structural)
    else:
        spatial: list[tuple[int, torch.Tensor, torch.Tensor | None, int]] = []
        tail: list[tuple[torch.Tensor, torch.Tensor | None, int]] = []
        for local_index, (tensor, reduction) in enumerate(
            zip(projected[1:], reductions[1:], strict=True)
        ):
            selected_count = int(reduction.dominant_patch_indices.numel())
            if selected_count:
                lookup = torch.full((576,), -1, dtype=torch.long)
                lookup[geometry.eligible_patch_indices[local_index + 1]] = (
                    geometry.local_mosaic_positions[local_index]
                )
                positions = lookup.index_select(0, reduction.dominant_patch_indices.detach().cpu())
                if bool((positions < 0).any()):
                    raise RuntimeError("VisionZip selected a padded local patch")
                for row_index, position in enumerate(positions.tolist(), start=1):
                    row_source = (
                        None
                        if reduction.source is None
                        else reduction.source[row_index : row_index + 1]
                    )
                    spatial.append(
                        (position, tensor[row_index : row_index + 1], row_source, local_index + 1)
                    )
            # Reduced order is CLS, dominant patches, contextual summaries.
            if tensor.shape[0]:
                tail.append(
                    (
                        tensor[:1],
                        None if reduction.source is None else reduction.source[:1],
                        local_index + 1,
                    )
                )
            if reduction.contextual_count:
                tail.append(
                    (
                        tensor[-reduction.contextual_count :],
                        None
                        if reduction.source is None
                        else reduction.source[-reduction.contextual_count :],
                        local_index + 1,
                    )
                )

        spatial.sort(key=lambda item: item[0])
        cursor = 0
        for mosaic_row in range(geometry.unpadded_rows):
            boundary = (mosaic_row + 1) * geometry.unpadded_columns
            while cursor < len(spatial) and spatial[cursor][0] < boundary:
                _, rows, source, view = spatial[cursor]
                append(rows, source, view)
                cursor += 1
            pieces.append(image_newline[None].to(device=device, dtype=dtype))
            if track_source:
                structural = torch.zeros(1, source_width, dtype=torch.float32)
                structural[0, raw_width + mosaic_row] = 1.0
                source_pieces.append(structural)
        if cursor != len(spatial):
            raise RuntimeError("VisionZip local mosaic positions exceed native unpadded geometry")
        for rows, source, view in tail:
            append(rows, source, view)

    packed = (
        torch.cat(pieces, dim=0) if pieces else torch.empty(0, hidden, device=device, dtype=dtype)
    )
    source = torch.cat(source_pieces, dim=0) if track_source else None
    return packed, source


def _contiguous_runs(positions: torch.Tensor) -> list[torch.Tensor]:
    if positions.numel() == 0:
        return []
    cuts = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=True)[0] + 1
    return list(torch.tensor_split(positions, cuts.tolist()))


def replace_image_placeholder_span(
    inputs: Any,
    *,
    image_token_id: int,
    target_length: int,
) -> Any:
    """Replace the sole image run and keep every aligned sequence field exact."""

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("VisionZip/LLaVA-NeXT requires input_ids shaped [batch, sequence]")
    if input_ids.shape[0] != 1:
        raise NotImplementedError(
            f"VisionZip/LLaVA-NeXT supports batch_size=1, got {input_ids.shape[0]}"
        )
    if int(target_length) < 1:
        raise ValueError("VisionZip visual target length must be positive")
    positions = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=True)[0]
    runs = _contiguous_runs(positions)
    if len(runs) != 1:
        raise NotImplementedError(
            "VisionZip/LLaVA-NeXT supports exactly one image and one contiguous visual span"
        )
    run = runs[0]
    first, last = int(run[0]), int(run[-1]) + 1
    old_length = int(run.numel())
    target_length = int(target_length)
    if old_length == target_length:
        return inputs

    old_width = int(input_ids.shape[1])
    for key, value in list(inputs.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if key == "cache_position" and (
            (value.ndim == 1 and value.shape[0] == old_width)
            or (value.ndim == 2 and value.shape == input_ids.shape)
        ):
            compact_cache = torch.arange(
                old_width - old_length + target_length,
                dtype=value.dtype,
                device=value.device,
            )
            inputs[key] = compact_cache if value.ndim == 1 else compact_cache.unsqueeze(0)
            continue
        if value.ndim < 2 or value.shape[0] != 1 or value.shape[1] != old_width:
            continue
        prefix, suffix = value[:, :first], value[:, last:]
        shape = (1, target_length, *value.shape[2:])
        if key == "input_ids":
            middle = torch.full(shape, image_token_id, dtype=value.dtype, device=value.device)
        elif key == "labels":
            middle = torch.full(shape, -100, dtype=value.dtype, device=value.device)
        elif key == "attention_mask":
            middle = torch.ones(shape, dtype=value.dtype, device=value.device)
        elif old_length:
            middle = value[:, first : first + 1].expand(shape).clone()
        else:  # pragma: no cover - a run cannot be empty
            middle = torch.zeros(shape, dtype=value.dtype, device=value.device)
        inputs[key] = torch.cat((prefix, middle, suffix), dim=1)

    attention = inputs.get("attention_mask")
    position_ids = inputs.get("position_ids")
    if isinstance(position_ids, torch.Tensor):
        if isinstance(attention, torch.Tensor) and attention.shape == inputs["input_ids"].shape:
            compact = (attention.long().cumsum(dim=-1) - 1).masked_fill(attention == 0, 0)
        else:
            compact = torch.arange(
                inputs["input_ids"].shape[1],
                device=position_ids.device,
                dtype=position_ids.dtype,
            ).unsqueeze(0)
        inputs["position_ids"] = compact.to(
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
    return inputs


def checkpoint_metadata(
    *,
    model_id: str,
    model_revision: str,
    retain_ratio: float,
    image_grid_pinpoints: Sequence[Sequence[int]],
    training_schedule: Mapping[str, Any],
    artifact_role: str,
    training_complete: bool,
    global_step: int | None,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "method": "VisionZip++",
        "algorithm": "VisionZip",
        "variant": "efficient_tuning",
        "artifact_role": str(artifact_role),
        "training_complete": bool(training_complete),
        "global_step": global_step,
        "base_model": str(model_id),
        "base_model_revision": str(model_revision),
        "architecture": "LlavaNextForConditionalGeneration",
        "decoder": "Llama-3-8B",
        "decoder_layers": 32,
        "decoder_hidden_size": 4096,
        "decoder_attention_heads": 32,
        "decoder_kv_heads": 8,
        "decoder_rms_norm_eps": 1e-5,
        "decoder_rope_theta": 500000.0,
        "decoder_max_position_embeddings": 8192,
        "position_policy": "compact_contiguous_llama3_positions",
        "cache_policy": "native_transformers_prefill_and_decode_dynamic_cache",
        "transformers_version": TARGET_TRANSFORMERS_VERSION,
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
        "vision_image_size": 336,
        "vision_patch_size": 14,
        "patch_grid": [24, 24],
        "raw_rows_per_view": 577,
        "image_grid_pinpoints": [list(point) for point in image_grid_pinpoints],
        "requested_retention_ratio": float(retain_ratio),
        "target_formula": "max(1, round_half_up(retain_ratio * original_content_tokens))",
        "original_content_definition": (
            "native 576 base patches plus unpadded local AnyRes mosaic patches"
        ),
        "structural_token_definition": "native learned image-newline rows, excluded and preserved",
        "crop_split_policy": "dynamic_released_54_to_10_dominant_contextual_proportion",
        "budget_semantics": BUDGET_SEMANTICS,
        "view_policy": VIEW_POLICY,
        "output_order": OUTPUT_ORDER,
        "newline_policy": NEWLINE_POLICY,
        "rounding_policy": ROUNDING_POLICY,
        "global_view_policy": "compress_independently_and_emit_first",
        "local_crop_policy": "compress_independently_then_restore_dominant_mosaic_order",
        "native_unpadding_policy": "apply_before_budget_and_exclude_padded_local_patches",
        "training_stage": TRAINING_STAGE,
        "training_scope": TRAINING_SCOPE,
        "module_path": PROJECTOR_MODULE_PATH,
        "training_schedule": dict(training_schedule),
        "visionzip_repository": VISIONZIP_REPOSITORY,
        "visionzip_commit": VISIONZIP_COMMIT,
        "projectneura_repository": PROJECTNEURA_REPOSITORY,
        "projectneura_commit": PROJECTNEURA_COMMIT,
        "nuwa_repository": NUWA_REPOSITORY,
        "nuwa_commit": NUWA_COMMIT,
        "varest_base_commit": VAREST_BASE_COMMIT,
        "varest_llava_next_history_commit": VAREST_LLAVA_NEXT_HISTORY_COMMIT,
    }


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    retain_ratio: float,
    model_id: str,
    model_revision: str | None,
    image_grid_pinpoints: Sequence[Sequence[int]],
    allow_incomplete: bool = False,
    expected_training_schedule: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "method": "VisionZip++",
        "algorithm": "VisionZip",
        "variant": "efficient_tuning",
        "base_model": TARGET_MODEL_ID,
        "base_model_revision": TARGET_MODEL_REVISION,
        "architecture": "LlavaNextForConditionalGeneration",
        "decoder": "Llama-3-8B",
        "decoder_layers": 32,
        "decoder_hidden_size": 4096,
        "decoder_attention_heads": 32,
        "decoder_kv_heads": 8,
        "decoder_rms_norm_eps": 1e-5,
        "decoder_rope_theta": 500000.0,
        "decoder_max_position_embeddings": 8192,
        "position_policy": "compact_contiguous_llama3_positions",
        "cache_policy": "native_transformers_prefill_and_decode_dynamic_cache",
        "transformers_version": TARGET_TRANSFORMERS_VERSION,
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
        "vision_image_size": 336,
        "vision_patch_size": 14,
        "patch_grid": [24, 24],
        "raw_rows_per_view": 577,
        "image_grid_pinpoints": [list(point) for point in TARGET_IMAGE_GRID_PINPOINTS],
        "requested_retention_ratio": float(retain_ratio),
        "target_formula": "max(1, round_half_up(retain_ratio * original_content_tokens))",
        "original_content_definition": (
            "native 576 base patches plus unpadded local AnyRes mosaic patches"
        ),
        "structural_token_definition": "native learned image-newline rows, excluded and preserved",
        "crop_split_policy": "dynamic_released_54_to_10_dominant_contextual_proportion",
        "budget_semantics": BUDGET_SEMANTICS,
        "view_policy": VIEW_POLICY,
        "output_order": OUTPUT_ORDER,
        "newline_policy": NEWLINE_POLICY,
        "rounding_policy": ROUNDING_POLICY,
        "global_view_policy": "compress_independently_and_emit_first",
        "local_crop_policy": "compress_independently_then_restore_dominant_mosaic_order",
        "native_unpadding_policy": "apply_before_budget_and_exclude_padded_local_patches",
        "training_stage": TRAINING_STAGE,
        "training_scope": TRAINING_SCOPE,
        "module_path": PROJECTOR_MODULE_PATH,
        "visionzip_commit": VISIONZIP_COMMIT,
        "visionzip_repository": VISIONZIP_REPOSITORY,
        "projectneura_commit": PROJECTNEURA_COMMIT,
        "projectneura_repository": PROJECTNEURA_REPOSITORY,
        "nuwa_commit": NUWA_COMMIT,
        "nuwa_repository": NUWA_REPOSITORY,
        "varest_base_commit": VAREST_BASE_COMMIT,
        "varest_llava_next_history_commit": VAREST_LLAVA_NEXT_HISTORY_COMMIT,
    }
    mismatches = [
        f"{key}: checkpoint={metadata.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if model_id != TARGET_MODEL_ID:
        mismatches.append(f"loaded model id={model_id!r}, expected={TARGET_MODEL_ID!r}")
    if model_revision is not None and str(model_revision) != TARGET_MODEL_REVISION:
        mismatches.append(f"loaded revision={model_revision!r}, expected={TARGET_MODEL_REVISION!r}")
    runtime_grid = tuple(tuple(int(value) for value in point) for point in image_grid_pinpoints)
    if runtime_grid != TARGET_IMAGE_GRID_PINPOINTS:
        mismatches.append(
            f"runtime image_grid_pinpoints={runtime_grid!r}, "
            f"expected={TARGET_IMAGE_GRID_PINPOINTS!r}"
        )
    if version("transformers") != TARGET_TRANSFORMERS_VERSION:
        mismatches.append(
            f"runtime transformers={version('transformers')!r}, "
            f"expected={TARGET_TRANSFORMERS_VERSION!r}"
        )
    if metadata.get("artifact_role") not in {
        "periodic_resume",
        "incomplete_run",
        "final_evaluation",
    }:
        mismatches.append(f"invalid artifact_role={metadata.get('artifact_role')!r}")
    if not allow_incomplete and (
        metadata.get("artifact_role") != "final_evaluation"
        or metadata.get("training_complete") is not True
    ):
        mismatches.append("checkpoint is not a completed final_evaluation artifact")
    schedule = metadata.get("training_schedule")
    if not isinstance(schedule, Mapping):
        mismatches.append("training_schedule is missing or is not a mapping")
    elif not allow_incomplete:
        mismatches.extend(
            f"training_schedule.{key}: checkpoint={schedule.get(key)!r}, expected={value!r}"
            for key, value in FINAL_TRAINING_RECIPE.items()
            if schedule.get(key) != value
        )
    if expected_training_schedule is not None:
        if schedule != dict(expected_training_schedule):
            mismatches.append("training_schedule does not match the run being resumed")
    if mismatches:
        raise ValueError("Incompatible LLaVA-NeXT VisionZip++ checkpoint: " + "; ".join(mismatches))


def load_projector_checkpoint(
    path: str | Path,
    projector: Any,
    *,
    retain_ratio: float,
    model_id: str,
    model_revision: str | None,
    image_grid_pinpoints: Sequence[Sequence[int]],
    allow_incomplete: bool = False,
    expected_training_schedule: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    checkpoint = Path(path)
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError(
            "LLaVA-NeXT VisionZip++ requires a metadata-bearing .pt checkpoint; "
            "raw safetensors and LLaVA-1.5 projectors are incompatible"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("VisionZip++ checkpoint must contain state_dict and config mappings")
    metadata = payload.get("config")
    if not isinstance(metadata, Mapping):
        raise ValueError("VisionZip++ checkpoint is missing provenance config")
    if payload.get("format_version") != metadata.get("format_version"):
        raise ValueError("VisionZip++ checkpoint format versions disagree")
    if payload.get("compressor") != "VisionZipPlusPlusCompressor":
        raise ValueError("VisionZip++ checkpoint has an invalid compressor identity")
    validate_checkpoint_metadata(
        metadata,
        retain_ratio=retain_ratio,
        model_id=model_id,
        model_revision=model_revision,
        image_grid_pinpoints=image_grid_pinpoints,
        allow_incomplete=allow_incomplete,
        expected_training_schedule=expected_training_schedule,
    )
    missing, unexpected = projector.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"VisionZip++ projector state mismatch (missing={list(missing)}, "
            f"unexpected={list(unexpected)})"
        )
    return metadata


class LlavaNextVisionZipAdapter:
    """Reversible pre-projector VisionZip adapter for one-image NeXT prefills."""

    def __init__(self, owner: Any, vlm: Any) -> None:
        self.owner = owner
        self.vlm = vlm
        self.inner = vlm.model.model
        self.feature_layer = int(owner._resolve_feature_layer(vlm))
        self.attention: torch.Tensor | None = None
        self.keys: torch.Tensor | None = None
        self.hidden: torch.Tensor | None = None
        self.attention_handle: Any | None = None
        self.hidden_handle: Any | None = None
        self.original_get_image_features: Any | None = None
        self.original_prepare_inputs: Any | None = None
        self.original_generate: Any | None = None
        self.image_token_id: int | None = None
        self.last_budget_report: LlavaNextBudgetReport | None = None

    @property
    def budget_report(self) -> LlavaNextBudgetReport | None:
        return self.last_budget_report

    def _validate_model(self) -> None:
        config = self.vlm.model.config
        if getattr(config, "model_type", None) != "llava_next":
            raise TypeError("VisionZip's LLaVA-NeXT adapter requires LlavaNextModel")
        if self.feature_layer != -2:
            raise ValueError(
                "The released LLaVA-NeXT VisionZip policy is pinned to feature "
                f"layer -2, got {self.feature_layer}"
            )
        if getattr(config, "vision_feature_select_strategy", None) != "default":
            raise ValueError(
                "VisionZip/LLaVA-NeXT requires vision_feature_select_strategy='default'"
            )
        vision = config.vision_config
        actual_grid = tuple(tuple(int(v) for v in point) for point in config.image_grid_pinpoints)
        if int(vision.image_size) != 336 or int(vision.patch_size) != 14:
            raise ValueError("VisionZip/LLaVA-NeXT requires the target 336/14 CLIP geometry")
        if actual_grid != TARGET_IMAGE_GRID_PINPOINTS:
            raise ValueError(
                f"Unexpected image_grid_pinpoints={actual_grid}; this implementation "
                "is pinned to the target checkpoint"
            )
        text = getattr(config, "text_config", None)
        if text is not None and (
            getattr(text, "model_type", None) != "llama"
            or int(getattr(text, "num_hidden_layers", -1)) != 32
        ):
            raise ValueError("VisionZip/LLaVA-NeXT requires the 32-layer Llama-3 decoder")

    def install(self) -> None:
        self._validate_model()
        if self.original_get_image_features is not None:
            return
        layers = self.owner._vision_encoder_layers(self.vlm)
        target = (
            layers[self.feature_layer] if self.feature_layer < 0 else layers[self.feature_layer - 1]
        )
        self.attention_handle = target.self_attn.register_forward_pre_hook(
            self._capture_attention,
            with_kwargs=True,
        )
        self.hidden_handle = target.register_forward_hook(self._capture_hidden)
        self.original_get_image_features = self.inner.get_image_features
        self.original_prepare_inputs = self.vlm.prepare_inputs
        self.original_generate = self.vlm.generate
        self.image_token_id = int(self.vlm.model.config.image_token_id)

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            return self._get_image_features(*args, **kwargs)

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            prepared = self.original_prepare_inputs(*args, **kwargs)
            return self._prepare_inputs(prepared)

        def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
            self._validate_generation(kwargs)
            return self.original_generate(*args, **kwargs)

        self.inner.get_image_features = wrapped_get_image_features
        self.vlm.prepare_inputs = wrapped_prepare_inputs
        self.vlm.generate = wrapped_generate

    def uninstall(self) -> None:
        if self.attention_handle is not None:
            self.attention_handle.remove()
        if self.hidden_handle is not None:
            self.hidden_handle.remove()
        if self.original_get_image_features is not None:
            self.inner.get_image_features = self.original_get_image_features
        if self.original_prepare_inputs is not None:
            self.vlm.prepare_inputs = self.original_prepare_inputs
        if self.original_generate is not None:
            self.vlm.generate = self.original_generate
        self.attention_handle = None
        self.hidden_handle = None
        self.original_get_image_features = None
        self.original_prepare_inputs = None
        self.original_generate = None
        self.attention = None
        self.keys = None
        self.hidden = None
        self.last_budget_report = None

    def _capture_attention(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            return
        batch, sequence, _ = hidden.shape
        # VisionZip only needs CLS-to-patch saliency. Projecting the sole CLS
        # query avoids materializing unused patch queries under FA2/SDPA.
        queries = module.q_proj(hidden[:, :1])
        keys = module.k_proj(hidden)
        q_norm, k_norm = getattr(module, "q_norm", None), getattr(module, "k_norm", None)
        if callable(q_norm):
            queries = q_norm(queries)
        if callable(k_norm):
            keys = k_norm(keys)
        queries = queries.view(batch, 1, module.num_heads, module.head_dim).transpose(1, 2)
        keys = keys.view(batch, sequence, module.num_heads, module.head_dim).transpose(1, 2)
        logits = (queries.float() @ keys.float().transpose(-1, -2)) * module.scale
        self.attention = logits.softmax(dim=-1)[:, :, 0, 1:].sum(dim=1)
        self.keys = keys.mean(dim=1)[:, 1:].contiguous()

    def _capture_hidden(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        del module, args
        self.hidden = output[0] if isinstance(output, tuple) else output

    def _layout(
        self,
        image_size: torch.Tensor | Sequence[int],
        views: int,
    ) -> LlavaNextVisionZipLayout:
        from llmoe.compressor.llava_next import balanced_group_budgets

        geometry = _native_geometry(self.inner, image_size, views)
        ratio = float(self.owner._retain_ratio)
        target = retention_target(geometry.original_content_tokens, ratio)
        budgets = balanced_group_budgets(
            target,
            geometry.capacities,
            minimum=0,
            label="VisionZip++/LLaVA-NeXT",
        )
        rows, columns = anyres_grid_shape(self.inner.config, image_size) if views > 1 else (0, 0)
        if views > 1 and rows * columns != views - 1:
            raise ValueError(f"AnyRes grid {rows}x{columns} disagrees with {views - 1} local crops")
        return LlavaNextVisionZipLayout(
            view_count=views,
            local_crop_count=max(0, views - 1),
            tile_rows=rows,
            tile_columns=columns,
            original_content_tokens=geometry.original_content_tokens,
            requested_retention_ratio=ratio,
            target_content_tokens=target,
            realized_content_tokens=sum(budgets),
            structural_tokens=geometry.structural_tokens,
            crop_capacities=geometry.capacities,
            crop_budgets=budgets,
            unpadded_rows=geometry.unpadded_rows,
            unpadded_columns=geometry.unpadded_columns,
        )

    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        feature_layer = self.feature_layer if vision_feature_layer is None else vision_feature_layer
        strategy = (
            "default" if vision_feature_select_strategy is None else vision_feature_select_strategy
        )
        if feature_layer != self.feature_layer or strategy != "default":
            raise ValueError(
                "VisionZip/LLaVA-NeXT cannot override the pinned feature layer -2/default strategy"
            )
        if not isinstance(image_sizes, torch.Tensor) or image_sizes.shape != (1, 2):
            raise NotImplementedError(
                "VisionZip/LLaVA-NeXT supports one source image per prefill; "
                "image_sizes must be [1, 2]"
            )
        views = crop_count(self.inner.config, image_sizes[0])
        if pixel_values.ndim == 5:
            if pixel_values.shape[0] != 1 or pixel_values.shape[1] != views:
                raise ValueError(
                    f"Expected one AnyRes stack with exactly {views} views, "
                    f"got {tuple(pixel_values.shape)}"
                )
            pixels = pixel_values[0]
        elif pixel_values.ndim == 4:
            if pixel_values.shape[0] != views:
                raise ValueError(
                    f"Expected {views} concatenated views, got {pixel_values.shape[0]}"
                )
            pixels = pixel_values
        else:
            raise ValueError(f"pixel_values must be 4D or 5D, got {tuple(pixel_values.shape)}")

        layout = self._layout(image_sizes[0], views)
        geometry = _native_geometry(self.inner, image_sizes[0], views)
        tower_parameter = next(self.inner.vision_tower.parameters(), None)
        if tower_parameter is not None:
            pixels = pixels.to(dtype=tower_parameter.dtype)
        vision_kwargs = dict(kwargs)
        vision_kwargs.pop("return_dict", None)
        self.attention = self.keys = self.hidden = None
        try:
            with torch.no_grad():
                image_outputs = self.inner.vision_tower(
                    pixels,
                    output_hidden_states=bool(output_hidden_states),
                    return_dict=True,
                    **vision_kwargs,
                )
                if self.hidden is None or self.attention is None or self.keys is None:
                    raise RuntimeError("VisionZip did not capture the selected CLIP layer")
                if self.hidden.shape[:2] != (views, 577):
                    raise ValueError(
                        f"Expected {views} CLIP views with CLS+576 patches, "
                        f"got {tuple(self.hidden.shape)}"
                    )
                reductions: list[CropReduction] = []
                for index, budget in enumerate(layout.crop_budgets):
                    dominant, contextual = _crop_split(budget)
                    reductions.append(
                        reduce_crop(
                            self.hidden[index],
                            self.attention[index],
                            self.keys[index],
                            dominant=dominant,
                            contextual=contextual,
                            eligible_patch_indices=geometry.eligible_patch_indices[index],
                            track_source=self.owner._track_source,
                        )
                    )
                reduced = torch.cat([reduction.tokens for reduction in reductions], dim=0).detach()

            projected_flat = self.inner.multi_modal_projector(reduced)
            projected = list(torch.split(projected_flat, layout.crop_budgets, dim=0))
            packed, source = pack_ratio_order(
                projected,
                reductions,
                geometry,
                image_newline=self.inner.image_newline,
                track_source=self.owner._track_source,
            )
            if packed.shape[0] != layout.realized_total:
                raise RuntimeError(
                    f"VisionZip packed {packed.shape[0]} rows, expected {layout.realized_total}"
                )
            image_outputs.pooler_output = [packed]
            self.last_budget_report = LlavaNextBudgetReport(
                layouts=(layout,),
            )
            self.owner._last_budget_report = self.last_budget_report
            if self.owner._track_source and source is not None:
                self.owner._source_history = [
                    torch.eye(source.shape[1], dtype=torch.float32),
                    source,
                ]
            return image_outputs
        finally:
            self.attention = self.keys = self.hidden = None

    def _prepare_inputs(self, inputs: Any) -> Any:
        image_sizes = inputs.get("image_sizes") if hasattr(inputs, "get") else None
        pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else None
        if not isinstance(image_sizes, torch.Tensor) or image_sizes.shape != (1, 2):
            raise NotImplementedError("VisionZip/LLaVA-NeXT requires exactly one image per prompt")
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("VisionZip/LLaVA-NeXT expected pixel_values from the processor")
        views = crop_count(self.inner.config, image_sizes[0])
        layout = self._layout(image_sizes[0], views)
        self.last_budget_report = LlavaNextBudgetReport(
            layouts=(layout,),
        )
        self.owner._last_budget_report = self.last_budget_report
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
            positions = (input_ids[0] == int(self.image_token_id)).nonzero(as_tuple=True)[0]
            runs = _contiguous_runs(positions)
            native_length = layout.original_content_tokens + layout.structural_tokens
            if len(runs) != 1 or int(runs[0].numel()) != native_length:
                actual = 0 if len(runs) != 1 else int(runs[0].numel())
                raise ValueError(
                    "LLaVA-NeXT processor placeholder span does not match the native "
                    f"unpacked geometry: got {actual}, expected {native_length} "
                    f"({layout.original_content_tokens} content + "
                    f"{layout.structural_tokens} structural)."
                )
        return replace_image_placeholder_span(
            inputs,
            image_token_id=int(self.image_token_id),
            target_length=layout.realized_total,
        )

    @staticmethod
    def _validate_generation(kwargs: Mapping[str, Any]) -> None:
        if int(kwargs.get("num_beams", 1)) != 1:
            raise NotImplementedError("VisionZip/LLaVA-NeXT does not support beam search")
        if int(kwargs.get("num_return_sequences", 1)) != 1:
            raise NotImplementedError("VisionZip/LLaVA-NeXT supports one generated sequence")
        if kwargs.get("past_key_values") is not None:
            raise NotImplementedError(
                "VisionZip/LLaVA-NeXT does not accept an externally supplied prefill cache"
            )
        for name in ("videos", "pixel_values_videos", "video_values", "video_grid_thw"):
            if kwargs.get(name) is not None:
                raise NotImplementedError("VisionZip/LLaVA-NeXT does not support video inputs")
