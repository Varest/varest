"""Shared coordinate geometry for LLaVA-NeXT native AnyRes packing."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
    unpad_image,
)


@dataclass(frozen=True)
class LlavaNextAnyResLayout:
    """Coordinate map between crop-local patches and native packed rows."""

    crop_count: int
    patch_grid: tuple[int, int]
    tile_grid: tuple[int, int]
    local_grid: tuple[int, int]
    native_length: int
    newline_positions: tuple[int, ...]
    # [crops, patches], -1 where a local patch is removed by native unpadding.
    native_position_by_crop_patch: torch.Tensor
    # [crops, patches], used to prevent padded local patches entering AGG.
    valid_patch_mask: torch.Tensor

    @property
    def original_content_tokens(self) -> int:
        """Native visual content rows after AnyRes unpadding, excluding newlines."""

        return int(self.valid_patch_mask.sum().item())

    @property
    def structural_tokens(self) -> int:
        return len(self.newline_positions)


def build_anyres_layout(
    inner: Any,
    image_size: torch.Tensor | Sequence[int],
    crop_count: int,
    *,
    device: torch.device | str = "cpu",
) -> LlavaNextAnyResLayout:
    """Reproduce Transformers 5.5.4's crop order, stitching, and unpadding."""

    vision = inner.config.vision_config
    patch_h = int(vision.image_size) // int(vision.patch_size)
    patch_w = patch_h
    patches = patch_h * patch_w
    crops = int(crop_count)
    if crops < 1:
        raise ValueError("LLaVA-NeXT needs at least the base image view")

    positions = torch.full((crops, patches), -1, dtype=torch.long, device=device)
    valid = torch.zeros((crops, patches), dtype=torch.bool, device=device)
    positions[0] = torch.arange(patches, device=device)
    valid[0] = True

    if crops == 1:
        newline_positions = (patches,)
        return LlavaNextAnyResLayout(
            crop_count=1,
            patch_grid=(patch_h, patch_w),
            tile_grid=(0, 0),
            local_grid=(0, 0),
            native_length=patches + 1,
            newline_positions=newline_positions,
            native_position_by_crop_patch=positions,
            valid_patch_mask=valid,
        )

    tile_h, tile_w = get_anyres_image_grid_shape(
        image_size,
        inner.config.image_grid_pinpoints,
        int(vision.image_size),
    )
    tile_h, tile_w = int(tile_h), int(tile_w)
    if tile_h * tile_w != crops - 1:
        raise ValueError(
            f"AnyRes selected {tile_h}x{tile_w} local crops plus a base view, "
            f"but pixel_values carries {crops} views."
        )

    # Values encode (local tile, patch index).  Applying the exact native
    # reshape/permute/unpad path provides both the valid mask and raster order.
    local_ids = torch.arange((crops - 1) * patches, device=device).reshape(
        tile_h, tile_w, patch_h, patch_w
    )
    mosaic = (
        local_ids.permute(0, 2, 1, 3).contiguous().reshape(1, tile_h * patch_h, tile_w * patch_w)
    )
    mosaic = unpad_image(mosaic, image_size)
    local_h, local_w = int(mosaic.shape[1]), int(mosaic.shape[2])
    raster_ids = mosaic.flatten()
    local_native_positions = (
        patches
        + torch.arange(raster_ids.numel(), device=device)
        + torch.arange(raster_ids.numel(), device=device) // local_w
    )
    flat_positions = positions[1:].reshape(-1)
    flat_valid = valid[1:].reshape(-1)
    flat_positions[raster_ids] = local_native_positions
    flat_valid[raster_ids] = True
    newline_positions = tuple(patches + (row + 1) * local_w + row for row in range(local_h))
    native_length = patches + local_h * (local_w + 1)
    return LlavaNextAnyResLayout(
        crop_count=crops,
        patch_grid=(patch_h, patch_w),
        tile_grid=(tile_h, tile_w),
        local_grid=(local_h, local_w),
        native_length=native_length,
        newline_positions=newline_positions,
        native_position_by_crop_patch=positions,
        valid_patch_mask=valid,
    )
