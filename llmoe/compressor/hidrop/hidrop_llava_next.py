"""LLaVA-NeXT AnyRes geometry and dynamic HiDrop budget resolution.

Transformers 5.5.4 packs one source image in this exact decoder order:

1. all patch rows of the low-resolution/base view;
2. the high-resolution crops, rearranged from tile-major order into one spatial
   mosaic and unpadded to the source aspect ratio; and
3. one learned ``image_newline`` row after every surviving mosaic row.

HiDrop keeps that native order.  Patch rows from the base and local mosaic
jointly compete in every DTop-K stage, while newline rows are structural and
pass through the complete vision-processing window.  This is deliberately not
Nuwa's separate base/local stage-1 reducer: only Nuwa's verified layout facts
are reused, while HiDrop's defining decoder-internal selector remains intact.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

import torch
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
)

from llmoe.compressor.prototype import VisualizationView
from llmoe.visualization import pixel_values_to_canvas


LLAVA_NEXT_LAYOUT_VERSION = 1
LLAVA_NEXT_HIDROP_IMPLEMENTATION_VERSION = 2
LLAVA_NEXT_MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
LLAVA_NEXT_MODEL_REVISION = "b041c0d0ea0dd0196d147206c210c8d1752fc2da"
LLAVA_NEXT_IMAGE_GRID_PINPOINTS = (
    (336, 672),
    (672, 336),
    (672, 672),
    (1008, 336),
    (336, 1008),
)


def _pair(value: torch.Tensor | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if len(value) != 2:
        raise ValueError(f"Expected an image size [height, width], got {value!r}.")
    height, width = (int(value[0]), int(value[1]))
    if height < 1 or width < 1:
        raise ValueError(f"Image dimensions must be positive, got {(height, width)}.")
    return height, width


def _unpad_window(
    full_height: int,
    full_width: int,
    original_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return ``(top, left, height, width)`` matching HF ``unpad_image``.

    The seven-digit rounding and symmetric floor padding intentionally mirror
    ``transformers.models.llava_next.modeling_llava_next.unpad_image`` rather
    than an algebraically similar implementation; odd padding therefore keeps
    HF's extra trailing row/column.
    """

    original_height, original_width = original_size
    original_aspect = original_width / original_height
    current_aspect = full_width / full_height
    if original_aspect > current_aspect:
        scale = full_width / original_width
        new_height = int(round(original_height * scale, 7))
        padding = (full_height - new_height) // 2
        return padding, 0, full_height - 2 * padding, full_width
    scale = full_height / original_height
    new_width = int(round(original_width * scale, 7))
    padding = (full_width - new_width) // 2
    return 0, padding, full_height, full_width - 2 * padding


@dataclass(frozen=True)
class LlavaNextVisualLayout:
    """Provenance for every row in one native packed decoder image span."""

    version: int
    image_height: int
    image_width: int
    crop_count: int
    tile_rows: int
    tile_columns: int
    patch_rows_per_crop: int
    patch_columns_per_crop: int
    local_top_offset: int
    local_left_offset: int
    local_height: int
    local_width: int
    native_tokens: int
    patch_tokens: int
    global_patch_tokens: int
    local_patch_tokens: int
    newline_tokens: int
    newline_indices: tuple[int, ...]
    # For packed patch rows, the source view is 0 for the base image and 1..
    # for local crops in processor tile-major order. Newlines use -1.
    source_view_by_token: tuple[int, ...]
    source_patch_by_token: tuple[int, ...]
    local_row_by_token: tuple[int, ...]

    @property
    def patch_indices(self) -> tuple[int, ...]:
        newline = set(self.newline_indices)
        return tuple(index for index in range(self.native_tokens) if index not in newline)

    @property
    def global_indices(self) -> tuple[int, ...]:
        return tuple(range(self.global_patch_tokens))

    @property
    def local_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(self.global_patch_tokens, self.native_tokens)
            if self.source_view_by_token[index] > 0
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "layout_version": self.version,
            "image_size": [self.image_height, self.image_width],
            "crop_count_including_base": self.crop_count,
            "local_tile_grid": [self.tile_rows, self.tile_columns],
            "crop_patch_grid": [self.patch_rows_per_crop, self.patch_columns_per_crop],
            "unpad_patch_window": [
                self.local_top_offset,
                self.local_left_offset,
                self.local_height,
                self.local_width,
            ],
            "native_visual_tokens": self.native_tokens,
            "patch_tokens": self.patch_tokens,
            "global_patch_tokens": self.global_patch_tokens,
            "local_patch_tokens": self.local_patch_tokens,
            "structural_newline_tokens": self.newline_tokens,
            "packing": "base_then_unpadded_local_mosaic_raster_with_row_newlines",
            "selection_scope": "joint_global_and_local_patch_rows",
            "newline_policy": "structural_passthrough_until_early_exit",
        }



def llava_next_visualization_views(
    pixel_values: torch.Tensor,
    image_processor: Any,
    layout: LlavaNextVisualLayout,
) -> list[VisualizationView]:
    """Map HiDrop's native base/local source columns to two image canvases."""

    if pixel_values.ndim == 5:
        if int(pixel_values.shape[0]) != 1:
            raise ValueError("LLaVA-NeXT visualization supports one source image at a time.")
        crops = pixel_values[0]
    elif pixel_values.ndim == 4:
        crops = pixel_values
    else:
        raise ValueError(
            "LLaVA-NeXT pixel_values must be [1, crops, C, H, W] or [crops, C, H, W], "
            f"got {tuple(pixel_values.shape)}."
        )
    if int(crops.shape[0]) != int(layout.crop_count):
        raise ValueError(
            f"LLaVA-NeXT visualization expected {layout.crop_count} views, got {crops.shape[0]}."
        )

    canvases = [pixel_values_to_canvas(crop, image_processor) for crop in crops]
    result = [
        VisualizationView(
            name="base",
            canvas=canvases[0],
            source_columns=torch.tensor(layout.global_indices, dtype=torch.long),
        )
    ]
    if int(layout.crop_count) == 1:
        return result

    rows = []
    for tile_row in range(int(layout.tile_rows)):
        start = 1 + tile_row * int(layout.tile_columns)
        rows.append(torch.cat(canvases[start : start + int(layout.tile_columns)], dim=-1))
    mosaic = torch.cat(rows, dim=-2)
    patch_size = int(canvases[0].shape[-2] // layout.patch_rows_per_crop)
    top = int(layout.local_top_offset) * patch_size
    left = int(layout.local_left_offset) * patch_size
    height = int(layout.local_height) * patch_size
    width = int(layout.local_width) * patch_size
    local = mosaic[:, top : top + height, left : left + width]
    result.append(
        VisualizationView(
            name="local_mosaic",
            canvas=local,
            source_columns=torch.tensor(layout.local_indices, dtype=torch.long),
        )
    )
    return result

def resolve_visual_layout(
    image_size: torch.Tensor | Sequence[int],
    *,
    grid_pinpoints: Sequence[Sequence[int]],
    vision_image_size: int,
    vision_patch_size: int,
    observed_crop_count: int | None = None,
) -> LlavaNextVisualLayout:
    """Resolve the exact packed rows produced by HF for one source image."""

    original_size = _pair(image_size)
    vision_image_size = int(vision_image_size)
    vision_patch_size = int(vision_patch_size)
    if vision_image_size % vision_patch_size:
        raise ValueError(
            f"vision image size {vision_image_size} is not divisible by patch size {vision_patch_size}."
        )
    patch_side = vision_image_size // vision_patch_size
    pinpoints = [list(map(int, point)) for point in grid_pinpoints]
    crop_count = int(
        image_size_to_num_patches(
            original_size,
            grid_pinpoints=pinpoints,
            patch_size=vision_image_size,
        )
    )
    if observed_crop_count is not None and int(observed_crop_count) != crop_count:
        raise ValueError(
            "LLaVA-NeXT processor/model crop-count mismatch: "
            f"geometry expects {crop_count}, pixel_values expose {observed_crop_count}."
        )

    global_patches = patch_side * patch_side
    if crop_count == 1:
        newline_index = global_patches
        native = global_patches + 1
        return LlavaNextVisualLayout(
            version=LLAVA_NEXT_LAYOUT_VERSION,
            image_height=original_size[0],
            image_width=original_size[1],
            crop_count=1,
            tile_rows=0,
            tile_columns=0,
            patch_rows_per_crop=patch_side,
            patch_columns_per_crop=patch_side,
            local_top_offset=0,
            local_left_offset=0,
            local_height=0,
            local_width=0,
            native_tokens=native,
            patch_tokens=global_patches,
            global_patch_tokens=global_patches,
            local_patch_tokens=0,
            newline_tokens=1,
            newline_indices=(newline_index,),
            source_view_by_token=tuple([0] * global_patches + [-1]),
            source_patch_by_token=tuple(list(range(global_patches)) + [-1]),
            local_row_by_token=tuple([-1] * native),
        )

    tile_rows, tile_columns = get_anyres_image_grid_shape(
        original_size,
        grid_pinpoints=pinpoints,
        patch_size=vision_image_size,
    )
    tile_rows, tile_columns = int(tile_rows), int(tile_columns)
    expected = 1 + tile_rows * tile_columns
    if crop_count != expected:
        raise RuntimeError(
            f"LLaVA-NeXT crop geometry is inconsistent: count={crop_count}, grid={tile_rows}x{tile_columns}."
        )
    full_height = tile_rows * patch_side
    full_width = tile_columns * patch_side
    top, left, local_height, local_width = _unpad_window(
        full_height,
        full_width,
        original_size,
    )

    source_views = [0] * global_patches
    source_patches = list(range(global_patches))
    local_rows = [-1] * global_patches
    newline_indices: list[int] = []
    cursor = global_patches
    for output_row in range(local_height):
        mosaic_row = top + output_row
        for output_column in range(local_width):
            mosaic_column = left + output_column
            tile_row, patch_row = divmod(mosaic_row, patch_side)
            tile_column, patch_column = divmod(mosaic_column, patch_side)
            view = 1 + tile_row * tile_columns + tile_column
            patch = patch_row * patch_side + patch_column
            source_views.append(view)
            source_patches.append(patch)
            local_rows.append(output_row)
            cursor += 1
        newline_indices.append(cursor)
        source_views.append(-1)
        source_patches.append(-1)
        local_rows.append(output_row)
        cursor += 1

    local_patches = local_height * local_width
    native = global_patches + local_patches + local_height
    if cursor != native:
        raise RuntimeError(
            f"LLaVA-NeXT layout cursor {cursor} does not match packed length {native}."
        )
    return LlavaNextVisualLayout(
        version=LLAVA_NEXT_LAYOUT_VERSION,
        image_height=original_size[0],
        image_width=original_size[1],
        crop_count=crop_count,
        tile_rows=tile_rows,
        tile_columns=tile_columns,
        patch_rows_per_crop=patch_side,
        patch_columns_per_crop=patch_side,
        local_top_offset=top,
        local_left_offset=left,
        local_height=local_height,
        local_width=local_width,
        native_tokens=native,
        patch_tokens=global_patches + local_patches,
        global_patch_tokens=global_patches,
        local_patch_tokens=local_patches,
        newline_tokens=local_height,
        newline_indices=tuple(newline_indices),
        source_view_by_token=tuple(source_views),
        source_patch_by_token=tuple(source_patches),
        local_row_by_token=tuple(local_rows),
    )


def _round_half_up_ratio(value: int, numerator: int, denominator: int) -> int:
    product = int(value) * int(numerator)
    denominator = int(denominator)
    return (2 * product + denominator) // (2 * denominator)


@dataclass(frozen=True)
class ResolvedHiDropSchedule:
    """One image's integer schedule and all requested/realized budget views."""

    profile_base_view_average: int
    requested_retention_ratio: Fraction
    target_content_tokens: int
    base_patch_tokens: int
    native_patch_tokens: int
    structural_tokens: int
    late_entry_layer: int
    early_exit_layer: int
    layer_list: tuple[int, ...]
    patch_keep_list: tuple[int, ...]
    patch_counts_by_layer: tuple[int, ...]
    structural_counts_by_layer: tuple[int, ...]
    total_counts_by_layer: tuple[int, ...]
    requested_scaled_patch_average: Fraction
    realized_patch_average: Fraction
    realized_structural_average: Fraction
    realized_total_average: Fraction

    def keep_patches_for_layer(self, rank_layer: int, current_patches: int) -> int:
        if rank_layer not in self.layer_list:
            raise ValueError(f"Layer {rank_layer} is not a HiDrop filtering layer.")
        keep = self.patch_keep_list[self.layer_list.index(rank_layer)]
        if not 1 <= keep <= int(current_patches):
            raise ValueError(
                f"Resolved HiDrop patch keep {keep} is invalid for {current_patches} active patches."
            )
        return keep

    @staticmethod
    def _fraction(value: Fraction) -> dict[str, int | float]:
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "value": float(value),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "budget_convention": "per_image_native_content_decoder_layer_average",
            "original_content_tokens": self.native_patch_tokens,
            "requested_retention_ratio": float(self.requested_retention_ratio),
            "target_content_tokens": self.target_content_tokens,
            "realized_content_tokens": float(self.realized_patch_average),
            "realized_retention_ratio": float(
                self.realized_patch_average / self.native_patch_tokens
            ),
            "structural_tokens": self.structural_tokens,
            "profile_base_view_patch_average": self.profile_base_view_average,
            "base_view_patch_tokens": self.base_patch_tokens,
            "native_patch_tokens": self.native_patch_tokens,
            "effective_patch_views": self._fraction(
                Fraction(self.native_patch_tokens, self.base_patch_tokens)
            ),
            "stage_patch_keeps": list(self.patch_keep_list),
            "patch_counts_by_layer": list(self.patch_counts_by_layer),
            "structural_counts_by_layer": list(self.structural_counts_by_layer),
            "total_counts_by_layer": list(self.total_counts_by_layer),
            "requested_scaled_patch_average": self._fraction(self.requested_scaled_patch_average),
            "realized_patch_average": self._fraction(self.realized_patch_average),
            "realized_structural_average": self._fraction(self.realized_structural_average),
            "realized_total_average": self._fraction(self.realized_total_average),
            "rounding": "target_and_stages_use_nearest_integer_half_up",
            "newline_budget_policy": "excluded_from_requested_patch_budget_and_reported_separately",
        }


def resolve_scaled_schedule(
    config: Any,
    layout: LlavaNextVisualLayout,
    *,
    observed_content_tokens: int | None = None,
) -> ResolvedHiDropSchedule:
    """Resolve one image's HiDrop curve from its true unpadded content count.

    HiDrop defines its budget as the decoder-layer-average content count, so
    ``K_target = max(1, round_half_up(retain_ratio * P))`` is that average target,
    not the final stage keep.  Every stage is scaled once from ``K_target`` and
    the immutable profile curve; the global ratio is never compounded.  Learned
    newline rows are structural passthrough tokens and are excluded from ``P``,
    the target, DTop-K candidates, and content retention ratios.
    """

    base_tokens = int(config.visual_tokens)
    patches = (
        int(layout.patch_tokens)
        if observed_content_tokens is None
        else int(observed_content_tokens)
    )
    if patches != int(layout.patch_tokens):
        raise ValueError(
            "LLaVA-NeXT assembled content count differs from the independently resolved "
            f"unpadded layout: observed={patches}, layout={layout.patch_tokens}."
        )
    if base_tokens < 1 or patches < 1:
        raise ValueError("HiDrop schedule scaling requires positive patch counts.")
    ratio = Fraction(str(config.retain_ratio))
    target = max(1, _round_half_up_ratio(ratio.numerator, patches, ratio.denominator))
    target = min(patches, target)
    profile_budget = int(config.average_retained_tokens)
    keeps: list[int] = []
    current = patches
    for base_keep in config.image_token_keep_list:
        keep = max(1, _round_half_up_ratio(int(base_keep), target, profile_budget))
        keep = min(current, keep)
        keeps.append(keep)
        current = keep

    keep_by_layer = dict(zip(config.layer_list, keeps, strict=True))
    patch_counts: list[int] = []
    structural_counts: list[int] = []
    total_counts: list[int] = []
    current = patches
    for rank_layer in range(1, int(config.num_decoder_layers) + 1):
        active = int(config.late_entry_layer) <= rank_layer < int(config.early_exit_layer)
        patch_count = current if active else 0
        structural_count = int(layout.newline_tokens) if active else 0
        patch_counts.append(patch_count)
        structural_counts.append(structural_count)
        total_counts.append(patch_count + structural_count)
        if active and rank_layer in keep_by_layer:
            current = keep_by_layer[rank_layer]

    depth = int(config.num_decoder_layers)
    requested = Fraction(target, 1)
    realized_patch = Fraction(sum(patch_counts), depth)
    realized_structural = Fraction(sum(structural_counts), depth)
    return ResolvedHiDropSchedule(
        profile_base_view_average=int(config.average_retained_tokens),
        requested_retention_ratio=ratio,
        target_content_tokens=target,
        base_patch_tokens=base_tokens,
        native_patch_tokens=patches,
        structural_tokens=int(layout.newline_tokens),
        late_entry_layer=int(config.late_entry_layer),
        early_exit_layer=int(config.early_exit_layer),
        layer_list=tuple(int(layer) for layer in config.layer_list),
        patch_keep_list=tuple(keeps),
        patch_counts_by_layer=tuple(patch_counts),
        structural_counts_by_layer=tuple(structural_counts),
        total_counts_by_layer=tuple(total_counts),
        requested_scaled_patch_average=requested,
        realized_patch_average=realized_patch,
        realized_structural_average=realized_structural,
        realized_total_average=realized_patch + realized_structural,
    )


def layout_from_model(
    inner_model: Any,
    image_size: torch.Tensor | Sequence[int],
    *,
    observed_crop_count: int | None = None,
) -> LlavaNextVisualLayout:
    """Resolve layout from a live or fake ``LlavaNextModel`` config."""

    config = inner_model.config
    vision = config.vision_config
    strategy = getattr(config, "vision_feature_select_strategy", "default")
    if strategy != "default":
        raise ValueError(
            "HiDrop/LLaVA-NeXT supports vision_feature_select_strategy='default' only; "
            f"got {strategy!r}."
        )
    return resolve_visual_layout(
        image_size,
        grid_pinpoints=config.image_grid_pinpoints,
        vision_image_size=int(vision.image_size),
        vision_patch_size=int(vision.patch_size),
        observed_crop_count=observed_crop_count,
    )
