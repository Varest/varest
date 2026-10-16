"""LLaVA-NeXT AnyRes adapter for QuietPrune.

The paper defines early, query-guided pruning inside a regular CLIP crop.  The
target NeXT checkpoint encodes one low-resolution base view followed by a
variable grid of local crops, then spatially stitches/unpads only the local
features.  This adapter therefore runs the published QuietPrune schedule on
every 24x24 crop with the same Q-CLS, retains original patch coordinates, and
reconstructs the exact native decoder order:

``base survivors, unpadded local survivors + one newline per row, aggregate``.

Nuwa's official NeXT path was used to verify the first two components and the
base-before-local order.  None of Nuwa's selection, merging, decoder pruning,
RPME, or per-view budget split is used here: those would change QuietPrune's
defining algorithm.  Newline rows are created after pruning and are protected.
The one aggregate is combined image-wide in vision space and projected once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import floor
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

from llmoe.backbone.llava_next import LLAVA_NEXT_8B_MODEL_ID, LLAVA_NEXT_8B_REVISION
from llmoe.backbone.llava_next_geometry import LlavaNextAnyResLayout, build_anyres_layout
from llmoe.compressor.prototype import VisualizationView
from llmoe.visualization import pixel_values_to_canvas
from llmoe.compressor.quietprune.quietprune import (
    TRAINING_PRUNING_RATE,
    QuietPruneAdapter,
    QuietPrunePlan,
    build_plan,
)


QUIETPRUNE_NEXT_IMPLEMENTATION_VERSION = 3
TRANSFORMERS_LAYOUT_VERSION = "5.5.4"
QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION = "flash_attention_2"
QUIETPRUNE_NEXT_FLASH_ATTN_VERSION = "2.8.3.post1"
QUIETPRUNE_PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2026/papers/"
    "Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_"
    "Vision-Language_Models_CVPR_2026_paper.pdf"
)
SOURCE_COMMITS = {
    "varest_base": "475fc2d768132d39dd8974a41e4da3df1e572ef6",
    "varest_llava_next_layout_reference": "73a55369b139c90a710023c51822df29b4f4a2e7",
    "projectneura_llava": "0a84ee262e9afdce9bdddd3b9b8b76de853ac9c7",
    "projectneura_qwen": "5a88e92f61d9c65a9d6a20c25876f4c1b13dfa16",
    "nuwa_official": "175b7d57a7890b9818dd26d82f2d91d9868b0d80",
    "conrad_launcher_reference": "1d17b954dc555faab69f2cf30dec5faedb2d9283",
}


def layer_pruning_rate_for_average(target_pruning_rate: float, stages: int) -> float:
    """Resolve paper ``R*`` to per-stage removal ``R``.

    The supplement prints *keep* ratios while labelling them as pruning rates.
    Applying those numbers as removals reverses the schedule.  The paper's
    definition is ``R* = 1 - (1 - R) ** stages``, hence this inverse.
    """

    target = float(target_pruning_rate)
    depth = int(stages)
    if not 0.0 <= target < 1.0:
        raise ValueError("target_average_pruning_rate must be in [0, 1)")
    if depth < 1:
        raise ValueError("QuietPrune needs at least one pruning stage")
    return 1.0 - (1.0 - target) ** (1.0 / depth)


def _contiguous_runs(positions: torch.Tensor) -> list[torch.Tensor]:
    if positions.numel() == 0:
        return []
    split_at = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=True)[0] + 1
    return list(torch.tensor_split(positions, split_at.tolist()))





def llava_next_visualization_views(
    model: Any,
    inputs: Mapping[str, Any],
    image_processor: Any,
) -> list[VisualizationView]:
    """Rebuild QuietPrune's base/local canvases in native AnyRes coordinates."""

    pixel_values = inputs.get("pixel_values")
    image_sizes = inputs.get("image_sizes")
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim not in (4, 5):
        raise ValueError("LLaVA-NeXT visualization needs 4D/5D pixel_values")
    if not isinstance(image_sizes, torch.Tensor) or tuple(image_sizes.shape) != (1, 2):
        raise ValueError("LLaVA-NeXT visualization supports one source image at a time")

    inner = model.model.model
    crop_count = int(
        image_size_to_num_patches(
            image_sizes[0],
            inner.config.image_grid_pinpoints,
            int(inner.config.vision_config.image_size),
        )
    )
    if pixel_values.ndim == 5:
        if pixel_values.shape[0] != 1 or pixel_values.shape[1] < crop_count:
            raise ValueError("LLaVA-NeXT visualization crop batch does not match image_sizes")
        crops = pixel_values[0, :crop_count]
    else:
        if pixel_values.shape[0] != crop_count:
            raise ValueError("LLaVA-NeXT visualization crop count does not match image_sizes")
        crops = pixel_values

    layout = build_anyres_layout(inner, image_sizes[0], crop_count, device=crops.device)
    views = [
        VisualizationView(
            name="base",
            canvas=pixel_values_to_canvas(crops[0], image_processor),
            source_columns=layout.native_position_by_crop_patch[0].detach().cpu(),
        )
    ]
    if crop_count == 1:
        return views

    tile_h, tile_w = layout.tile_grid
    channels, crop_h, crop_w = crops.shape[1:]
    local = crops[1:].reshape(tile_h, tile_w, channels, crop_h, crop_w)
    mosaic = (
        local.permute(2, 0, 3, 1, 4)
        .contiguous()
        .reshape(channels, tile_h * crop_h, tile_w * crop_w)
    )
    from transformers.models.llava_next.modeling_llava_next import unpad_image

    mosaic = unpad_image(mosaic, image_sizes[0])
    patch_size = int(inner.config.vision_config.patch_size)
    expected_pixels = (layout.local_grid[0] * patch_size, layout.local_grid[1] * patch_size)
    if tuple(mosaic.shape[-2:]) != expected_pixels:
        mosaic = F.interpolate(
            mosaic.unsqueeze(0).float(),
            expected_pixels,
            mode="bilinear",
            align_corners=False,
        )[0].to(crops.dtype)
    local_positions = layout.native_position_by_crop_patch[1:].flatten()
    local_columns = local_positions[local_positions >= 0].sort().values.detach().cpu()
    views.append(
        VisualizationView(
            name="local",
            canvas=pixel_values_to_canvas(mosaic, image_processor),
            source_columns=local_columns,
        )
    )
    return views

def target_content_tokens(original_content_tokens: int, retain_ratio: float) -> int:
    """Per-image target from the true native content count (structural rows excluded)."""

    original = int(original_content_tokens)
    ratio = float(retain_ratio)
    if original < 1:
        raise ValueError("A LLaVA-NeXT image must contain at least one visual content token")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("retain_ratio must be in (0, 1]")
    return min(original, max(1, round(ratio * original)))


def plan_for_content_retention(
    *,
    original_content_tokens: int,
    retain_ratio: float,
    grid: tuple[int, int],
    group_size: int,
    pruning_layers: Sequence[int],
    keep_border_tokens: bool,
    aggregate: bool,
    include_cls: bool = False,
) -> tuple[QuietPrunePlan, int, int]:
    """Choose the nearest uniform-rate QuietPrune schedule for one native image.

    QuietPrune's defining schedule applies one removal rate at every early cut.
    We therefore solve for that *stage rate* from a single final per-image target;
    the requested retention ratio is not reapplied independently at every stage.
    The estimate scales the image's true unpadded content count by the dense
    per-view grouped keep fraction and includes the single image-wide aggregate
    as retained visual content. The actual count is measured after selection and
    native packing because local unpadding can make grouped outcomes content
    dependent.
    """

    original = int(original_content_tokens)
    target = target_content_tokens(original, retain_ratio)
    height, width = grid
    full_height = height - height % group_size
    full_width = width - width % group_size
    groups = (full_height // group_size) * (full_width // group_size)
    if groups < 1:
        raise ValueError("QuietPrune retention planning requires at least one spatial group")
    border = height * width - groups * group_size**2 if keep_border_tokens else 0
    stages = tuple(pruning_layers)

    # Every floor transition is represented by a midpoint (k + 0.5) / g for
    # some possible active group count g. This enumerates all distinct schedules
    # without an arbitrary floating-point search resolution.
    candidate_rates = {0.0}
    for active in range(1, groups + 1):
        candidate_rates.update((dropped + 0.5) / active for dropped in range(active))

    outcomes: dict[int, float] = {}
    for rate in candidate_rates:
        active = groups
        for _ in stages:
            dropped = min(active - 1, max(0, floor(active * rate)))
            active -= dropped
        outcomes.setdefault(active, rate)
        outcomes[active] = min(outcomes[active], rate)

    best: tuple[tuple[int, int, float], float, int] | None = None
    for active, rate in outcomes.items():
        dense_patches = active * group_size**2 + border
        estimated_patches = round(original * dense_patches / (height * width))
        estimated_content = estimated_patches + int(
            aggregate and dense_patches < height * width
        ) + int(include_cls)
        # Nearest feasible estimate; on an exact tie prefer retaining more
        # context, then the gentler per-stage cut.
        key = (abs(estimated_content - target), -estimated_content, rate)
        if best is None or key < best[0]:
            best = (key, rate, estimated_content)
    if best is None:  # pragma: no cover - groups >= 1 guarantees candidates
        raise RuntimeError("QuietPrune could not construct a grouped retention schedule")

    _, rate, estimated_content = best
    plan = build_plan(
        grid,
        group_size=group_size,
        pruning_layers=stages,
        pruning_rate=rate,
        keep_border_tokens=keep_border_tokens,
        aggregate=aggregate,
        include_cls=include_cls,
    )
    return plan, target, estimated_content


@dataclass
class _Aggregate:
    token: torch.Tensor  # [C,D]
    mass: torch.Tensor  # [C]
    source_mass: torch.Tensor  # [C,P], unnormalised provenance mass


@dataclass
class CropVisionOutput:
    tokens: torch.Tensor  # [C,K,Dv]
    retained_indices: torch.Tensor  # [C,K]
    aggregate: _Aggregate | None
    stage_states: tuple[tuple[torch.Tensor, _Aggregate | None], ...]


def _aggregate_dropped(
    tokens: torch.Tensor,
    relevance: torch.Tensor,
    originals: torch.Tensor,
    validity: torch.Tensor,
    *,
    original_patches: int,
    normalize: bool,
    eps: float,
) -> _Aggregate:
    """Aggregate valid dropped patches; native-unpadded padding has zero mass."""

    weights = relevance.float().clamp_min(0.0) * validity.float()
    mass = weights.sum(dim=1)
    valid_count = validity.sum(dim=1)
    has_mass = mass > eps
    fallback = validity.float() / valid_count.clamp_min(1).unsqueeze(1)
    normalized = torch.where(
        has_mass.unsqueeze(1),
        weights / mass.clamp_min(eps).unsqueeze(1),
        fallback,
    )
    effective_mass = torch.where(
        has_mass,
        mass,
        (valid_count > 0).to(mass.dtype),
    )
    value_weights = (
        normalized if normalize else torch.where(has_mass.unsqueeze(1), weights, fallback)
    )
    token = (tokens * value_weights.to(tokens.dtype).unsqueeze(-1)).sum(dim=1)
    source_mass = torch.zeros(
        tokens.shape[0], original_patches, dtype=torch.float32, device=tokens.device
    )
    source_mass.scatter_add_(1, originals, normalized * effective_mass.unsqueeze(1))
    return _Aggregate(token=token, mass=effective_mass, source_mass=source_mass)


def _combine_aggregates(
    previous: _Aggregate | None,
    current: _Aggregate | None,
    *,
    normalize: bool,
    eps: float,
) -> _Aggregate | None:
    if previous is None:
        return current
    if current is None:
        return previous
    total = previous.mass + current.mass
    if normalize:
        token = previous.token * (previous.mass / total.clamp_min(eps)).to(
            previous.token.dtype
        ).unsqueeze(-1) + current.token * (current.mass / total.clamp_min(eps)).to(
            current.token.dtype
        ).unsqueeze(-1)
        token = torch.where((total > eps).unsqueeze(1), token, torch.zeros_like(token))
    else:
        # This is the paper's literal x_m = sum_i a_i x_i, including across
        # successive cuts.  The older Varest implementation averaged here.
        token = previous.token + current.token
    return _Aggregate(
        token=token,
        mass=total,
        source_mass=previous.source_mass + current.source_mass,
    )


def _prune_crop_batch(
    patches: torch.Tensor,
    relevance: torch.Tensor,
    originals: torch.Tensor,
    validity: torch.Tensor,
    kept_groups: torch.Tensor,
    plan: QuietPrunePlan,
    current_groups: int,
    *,
    normalize_aggregation: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _Aggregate | None]:
    """One deterministic-width paper group cut over every crop in parallel."""

    crops, _, hidden = patches.shape
    dense = torch.zeros(crops, plan.num_patches, dtype=torch.float32, device=patches.device)
    dense.scatter_(1, originals, relevance.float())
    group_scores = dense[:, plan.group_members].mean(dim=-1)
    group_scores = group_scores.masked_fill(~kept_groups, float("-inf"))
    pruned = min(current_groups - 1, max(0, floor(current_groups * plan.pruning_rate)))
    keep_groups = current_groups - pruned
    new_kept = torch.zeros_like(kept_groups)
    new_kept.scatter_(1, group_scores.topk(keep_groups, dim=1).indices, True)

    keep_lookup = torch.cat(
        [
            new_kept,
            torch.full(
                (crops, 1),
                plan.keep_border_tokens,
                dtype=torch.bool,
                device=patches.device,
            ),
        ],
        dim=1,
    )
    prune_lookup = torch.cat(
        [kept_groups & ~new_kept, torch.zeros(crops, 1, dtype=torch.bool, device=patches.device)],
        dim=1,
    )
    keep_here = keep_lookup[:, plan.group_of_patch].gather(1, originals)
    prune_here = prune_lookup[:, plan.group_of_patch].gather(1, originals)
    kept_count = int(keep_here[0].sum())
    pruned_count = int(prune_here[0].sum())
    if not bool((keep_here.sum(dim=1) == kept_count).all()):
        raise RuntimeError("QuietPrune crop schedules became ragged inside CLIP")

    keep_positions = keep_here.nonzero(as_tuple=True)[1].reshape(crops, kept_count)
    new_originals = originals.gather(1, keep_positions)
    new_patches = patches.gather(1, keep_positions.unsqueeze(-1).expand(crops, kept_count, hidden))

    aggregate = None
    if plan.aggregate and pruned_count:
        prune_positions = prune_here.nonzero(as_tuple=True)[1].reshape(crops, pruned_count)
        dropped_originals = originals.gather(1, prune_positions)
        aggregate = _aggregate_dropped(
            patches.gather(1, prune_positions.unsqueeze(-1).expand(crops, pruned_count, hidden)),
            relevance.gather(1, prune_positions),
            dropped_originals,
            validity.gather(1, dropped_originals),
            original_patches=plan.num_patches,
            normalize=normalize_aggregation,
            eps=eps,
        )
    return new_patches, new_originals, new_kept, aggregate


def quietprune_crop_vision_forward(
    vision_model: nn.Module,
    pixel_values: torch.Tensor,
    q_cls: torch.Tensor,
    *,
    plan: QuietPrunePlan,
    selected_layer_idx: int,
    valid_patch_mask: torch.Tensor,
    aggregate_mode: str,
    normalize_aggregation: bool,
    eps: float,
) -> CropVisionOutput:
    """Run Q-CLS and all three early cuts on a dense batch of NeXT views."""

    hidden_states = vision_model.embeddings(pixel_values)
    crops = int(hidden_states.shape[0])
    if hidden_states.shape[1] != plan.num_patches + 1:
        raise ValueError(
            f"QuietPrune expected CLIP CLS + {plan.num_patches} patches, "
            f"got {hidden_states.shape[1]} tokens per crop."
        )
    if tuple(valid_patch_mask.shape) != (crops, plan.num_patches):
        raise ValueError("valid_patch_mask must be [crops, patches]")
    if q_cls.shape[0] == 1:
        q_cls = q_cls.expand(crops, -1, -1)
    if tuple(q_cls.shape[:2]) != (crops, 1):
        raise ValueError(f"Q-CLS must be [1|crops,1,D], got {tuple(q_cls.shape)}")
    hidden_states = torch.cat(
        [q_cls.to(device=hidden_states.device, dtype=hidden_states.dtype), hidden_states], dim=1
    )
    hidden_states = vision_model.pre_layrnorm(hidden_states)
    patch_start = 2  # Q-CLS, then the protected native CLIP CLS.
    originals = (
        torch.arange(plan.num_patches, device=hidden_states.device)
        .expand(crops, plan.num_patches)
        .contiguous()
    )
    kept_groups = torch.ones(crops, plan.num_groups, dtype=torch.bool, device=hidden_states.device)
    current_groups = plan.num_groups
    aggregate: _Aggregate | None = None
    stage_states: list[tuple[torch.Tensor, _Aggregate | None]] = []

    for layer_idx, layer in enumerate(vision_model.encoder.layers):
        if layer_idx > selected_layer_idx:
            break
        relevance = None
        if layer_idx in plan.pruning_layers:
            normed = layer.layer_norm1(hidden_states)
            attention = layer.self_attn
            batch, length, _ = normed.shape
            heads, head_dim = attention.num_heads, attention.head_dim
            query = attention.q_proj(normed[:, :1]).view(batch, 1, heads, head_dim).transpose(1, 2)
            key = attention.k_proj(normed).view(batch, length, heads, head_dim).transpose(1, 2)
            logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * attention.scale
            relevance = logits.softmax(dim=-1)[
                :, :, 0, patch_start : patch_start + originals.shape[1]
            ].mean(dim=1)

        output = layer(hidden_states, attention_mask=None)
        hidden_states = output[0] if isinstance(output, tuple) else output
        if relevance is None:
            continue

        patch_end = patch_start + originals.shape[1]
        prefix = hidden_states[:, :patch_start]
        suffix = hidden_states[:, patch_end:]
        patches, originals, kept_groups, step_aggregate = _prune_crop_batch(
            hidden_states[:, patch_start:patch_end],
            relevance,
            originals,
            valid_patch_mask,
            kept_groups,
            plan,
            current_groups,
            normalize_aggregation=normalize_aggregation,
            eps=eps,
        )
        schedule_index = plan.pruning_layers.index(layer_idx)
        current_groups = plan.groups_after_layer[schedule_index]
        if not plan.aggregate:
            suffix = hidden_states[:, :0]
            aggregate = None
        elif aggregate_mode == "rolling":
            if aggregate is not None and suffix.shape[1]:
                aggregate = _Aggregate(
                    # A local crop can lose only rows that native AnyRes later
                    # unpads.  Such a zero-mass aggregate still traverses CLIP
                    # to keep the crop batch rectangular, but it must remain a
                    # mathematical zero when the rolling state is folded into
                    # the next cut.
                    token=(
                        suffix[:, -1].float() * (aggregate.mass > eps).to(suffix.dtype).unsqueeze(1)
                    ),
                    mass=aggregate.mass,
                    source_mass=aggregate.source_mass,
                )
            aggregate = _combine_aggregates(
                aggregate,
                step_aggregate,
                normalize=normalize_aggregation,
                eps=eps,
            )
            suffix = aggregate.token.to(hidden_states.dtype).unsqueeze(1)
        elif aggregate_mode == "final":
            aggregate = _combine_aggregates(
                aggregate,
                step_aggregate,
                normalize=normalize_aggregation,
                eps=eps,
            )
            suffix = hidden_states[:, :0]
        else:  # guarded by the compressor constructor
            raise ValueError(f"Unknown aggregate_mode={aggregate_mode!r}")
        hidden_states = torch.cat([prefix, patches, suffix], dim=1)
        stage_states.append((originals.detach().clone(), aggregate))

    patch_end = patch_start + originals.shape[1]
    tokens = hidden_states[:, patch_start:patch_end]
    if aggregate is not None and hidden_states.shape[1] > patch_end:
        aggregate = _Aggregate(
            token=(
                hidden_states[:, -1].float()
                * (aggregate.mass > eps).to(hidden_states.dtype).unsqueeze(1)
            ),
            mass=aggregate.mass,
            source_mass=aggregate.source_mass,
        )
    return CropVisionOutput(
        tokens=tokens,
        retained_indices=originals,
        aggregate=aggregate,
        stage_states=tuple(stage_states),
    )


@dataclass
class PackedQuietPruneOutput:
    tokens: torch.Tensor
    source: torch.Tensor | None
    native_positions: torch.Tensor
    patch_tokens: int
    newline_tokens: int
    aggregate_tokens: int


def _combined_aggregate(
    aggregate: _Aggregate | None,
    layout: LlavaNextAnyResLayout,
    *,
    normalize: bool,
    eps: float,
    track_source: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    if aggregate is None:
        return None
    total_mass = aggregate.mass.sum()
    if normalize:
        token = (
            aggregate.token
            * (aggregate.mass / total_mass.clamp_min(eps)).to(aggregate.token.dtype).unsqueeze(1)
        ).sum(dim=0, keepdim=True)
    else:
        token = aggregate.token.sum(dim=0, keepdim=True)

    if not track_source:
        return token, None
    source = torch.zeros(
        1, layout.native_length, dtype=torch.float32, device=aggregate.token.device
    )
    native_positions = layout.native_position_by_crop_patch
    for crop in range(layout.crop_count):
        valid = native_positions[crop] >= 0
        source[0, native_positions[crop, valid]] += aggregate.source_mass[crop, valid]
    source /= source.sum(dim=1, keepdim=True).clamp_min(eps)
    return token, source


def pack_quietprune_features(
    projected_patches: torch.Tensor,
    retained_indices: torch.Tensor,
    *,
    layout: LlavaNextAnyResLayout,
    image_newline: torch.Tensor,
    aggregate: tuple[torch.Tensor, torch.Tensor | None] | None,
    track_source: bool = True,
) -> PackedQuietPruneOutput:
    """Pack sparse survivors without changing native base/local/newline order."""

    if projected_patches.ndim != 3 or retained_indices.ndim != 2:
        raise ValueError("projected_patches/retained_indices must be [crops,K,*]")
    if tuple(projected_patches.shape[:2]) != tuple(retained_indices.shape):
        raise ValueError("retained indices do not align with projected patch tokens")
    if projected_patches.shape[0] != layout.crop_count:
        raise ValueError("projected crop count does not match AnyRes layout")

    device = projected_patches.device
    dtype = projected_patches.dtype
    hidden = projected_patches.shape[-1]
    output_tokens: list[torch.Tensor] = []
    output_positions: list[torch.Tensor] = []
    source_rows: list[torch.Tensor] = []

    def append_patches(tokens: torch.Tensor, positions: torch.Tensor) -> None:
        if not positions.numel():
            return
        output_tokens.append(tokens)
        output_positions.append(positions)
        if track_source:
            source = torch.zeros(
                positions.numel(), layout.native_length, dtype=torch.float32, device=device
            )
            source[torch.arange(positions.numel(), device=device), positions] = 1.0
            source_rows.append(source)

    base_positions = retained_indices[0].to(device=device, dtype=torch.long)
    append_patches(projected_patches[0], base_positions)
    patch_tokens = int(base_positions.numel())
    newline = image_newline.to(device=device, dtype=dtype).reshape(1, hidden)

    if layout.crop_count == 1:
        newline_position = torch.tensor(layout.newline_positions, device=device, dtype=torch.long)
        append_patches(newline, newline_position)
    else:
        local_positions = (
            layout.native_position_by_crop_patch[1:]
            .gather(1, retained_indices[1:].to(layout.native_position_by_crop_patch.device))
            .reshape(-1)
        )
        local_tokens = projected_patches[1:].reshape(-1, hidden)
        valid = local_positions >= 0
        local_positions = local_positions[valid]
        local_tokens = local_tokens[valid.to(local_tokens.device)]
        if local_positions.numel():
            order = local_positions.argsort()
            local_positions = local_positions[order]
            local_tokens = local_tokens[order.to(local_tokens.device)]
        patch_tokens += int(local_positions.numel())

        newline_boundaries = torch.tensor(
            layout.newline_positions,
            device=local_positions.device,
            dtype=local_positions.dtype,
        )
        row_ends = torch.searchsorted(local_positions, newline_boundaries).tolist()
        cursor = 0
        for newline_position, next_cursor in zip(layout.newline_positions, row_ends, strict=True):
            append_patches(local_tokens[cursor:next_cursor], local_positions[cursor:next_cursor])
            append_patches(
                newline,
                torch.tensor([newline_position], device=device, dtype=torch.long),
            )
            cursor = next_cursor
        if cursor != int(local_positions.numel()):
            raise RuntimeError("Some selected local tokens fell outside native AnyRes rows")

    aggregate_count = 0
    if aggregate is not None:
        aggregate_token, aggregate_source = aggregate
        output_tokens.append(aggregate_token.to(device=device, dtype=dtype))
        output_positions.append(torch.tensor([-1], device=device, dtype=torch.long))
        if track_source:
            if aggregate_source is None:
                raise ValueError("Aggregate provenance is required when track_source=True")
            source_rows.append(aggregate_source.to(device=device, dtype=torch.float32))
        aggregate_count = 1

    tokens = torch.cat(output_tokens, dim=0)
    positions = torch.cat(output_positions, dim=0)
    source = torch.cat(source_rows, dim=0) if track_source else None
    return PackedQuietPruneOutput(
        tokens=tokens,
        source=source,
        native_positions=positions,
        patch_tokens=patch_tokens,
        newline_tokens=len(layout.newline_positions),
        aggregate_tokens=aggregate_count,
    )


@dataclass(frozen=True)
class QuietPruneNextBudget:
    """Requested and exact realized per-image content/structural budget."""

    convention: str
    original_content_tokens: int
    requested_retention_ratio: float
    target_content_tokens: int
    realized_content_tokens: int
    realized_retention_ratio: float
    structural_tokens: int
    requested_per_stage_pruning_rate: float
    requested_final_patch_pruning_rate: float
    crop_views: int
    patches_per_view: int
    stage_patch_tokens_all_views: tuple[int, ...]
    realized_vit_patch_pruning_rate: float
    native_packed_tokens: int
    realized_patch_tokens: int
    protected_newline_tokens: int
    aggregate_tokens: int
    realized_packed_tokens: int
    realized_decoder_pruning_rate: float


def _trim_one_visual_run(
    inputs: Mapping[str, Any],
    *,
    image_token_id: int,
    native_length: int,
    target_length: int,
    pad_token_id: int,
    labels: torch.Tensor | None,
) -> dict[str, Any]:
    """Delete placeholder rows and every aligned sequence tensor together."""

    result = dict(inputs)
    input_ids = result.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("QuietPrune/LLaVA-NeXT requires [1,T] input_ids")
    if input_ids.shape[0] != 1:
        raise NotImplementedError("QuietPrune/LLaVA-NeXT supports batch_size=1 only")
    positions = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=True)[0]
    runs = _contiguous_runs(positions)
    if len(runs) != 1:
        raise NotImplementedError(
            "QuietPrune/LLaVA-NeXT supports exactly one image placeholder run per prompt"
        )
    run = runs[0]
    if int(run.numel()) != int(native_length):
        raise ValueError(
            f"Processor expanded <image> to {run.numel()} rows, but native AnyRes "
            f"geometry requires {native_length}. Check processor/model revision metadata."
        )
    if not 1 <= int(target_length) <= int(native_length):
        raise ValueError(
            f"Compressed visual length {target_length} is invalid for native length {native_length}"
        )
    keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    keep[run[target_length:]] = False
    result["input_ids"] = input_ids[:, keep]

    effective_labels = labels
    if effective_labels is None and isinstance(result.get("labels"), torch.Tensor):
        effective_labels = result["labels"]
    if effective_labels is not None:
        if effective_labels.shape != input_ids.shape:
            raise ValueError("Training labels must match the untrimmed input_ids")
        result["labels"] = effective_labels[:, keep]
    for key, value in tuple(result.items()):
        if key in {"input_ids", "labels", "position_ids"}:
            continue
        if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape == input_ids.shape:
            result[key] = value[:, keep]
    attention_mask = result.get("attention_mask")
    if "position_ids" in result:
        if isinstance(attention_mask, torch.Tensor):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        else:
            position_ids = torch.arange(
                result["input_ids"].shape[1], device=input_ids.device
            ).unsqueeze(0)
        result["position_ids"] = position_ids
    # Batch one never needs repadding, but retaining the explicit pad contract
    # documents why larger/ragged batches are rejected instead of guessed at.
    del pad_token_id
    return result


class LlavaNextQuietPruneAdapter:
    """Reversible Varest lifecycle around native ``LlavaNextModel``."""

    def __init__(self, owner: Any, vlm: Any) -> None:
        self.owner = owner
        self.vlm = vlm
        self.model = vlm.model
        self.inner = self.model.model
        self.vision_model = getattr(
            self.inner.vision_tower, "vision_model", self.inner.vision_tower
        )
        self.projector = self.inner.multi_modal_projector
        self.image_token_id = int(
            getattr(self.model.config, "image_token_id", None)
            or getattr(self.model.config, "image_token_index", -1)
        )
        self.original_get_image_features: Any | None = None
        self.original_prepare_inputs: Any | None = None
        self.original_generate: Any | None = None
        self.pending_output: BaseModelOutputWithPooling | None = None
        self.pending_image_size: torch.Tensor | None = None
        self.last_budget: QuietPruneNextBudget | None = None
        self.pruning_rate = 0.0
        self.plan: QuietPrunePlan | None = None
        self.selected_layer_idx = -1

    def install(self) -> None:
        transformers_version = __import__("transformers").__version__
        if transformers_version != TRANSFORMERS_LAYOUT_VERSION:
            raise RuntimeError(
                "QuietPrune/LLaVA-NeXT was verified against Transformers "
                f"{TRANSFORMERS_LAYOUT_VERSION}, got {transformers_version}."
            )
        config = self.model.config
        if getattr(config, "model_type", None) != "llava_next":
            raise TypeError("QuietPrune's NeXT adapter requires LlavaNextForConditionalGeneration")
        text = getattr(config, "text_config", None)
        if getattr(text, "model_type", None) != "llama":
            raise TypeError("QuietPrune/LLaVA-NeXT requires the checkpoint's Llama-3 decoder")
        if str(getattr(config, "vision_feature_select_strategy", "default")) != "default":
            raise NotImplementedError(
                "QuietPrune/LLaVA-NeXT supports vision_feature_select_strategy='default' only"
            )
        if not isinstance(getattr(config, "vision_feature_layer", None), int):
            raise NotImplementedError(
                "QuietPrune/LLaVA-NeXT does not support multi-layer vision feature concatenation"
            )
        if self.image_token_id < 0:
            raise ValueError("Could not resolve LLaVA-NeXT image_token_id")

        layers = self.vision_model.encoder.layers
        self.selected_layer_idx = self.owner._selected_encoder_layer(
            len(layers), config.vision_feature_layer
        )
        pruning_layers = tuple(
            layer
            for layer in (
                self.owner._pruning_layers or self.owner._default_pruning_layers(len(layers))
            )
            if layer <= self.selected_layer_idx
        )
        if not pruning_layers:
            raise ValueError("No QuietPrune stage runs before the selected CLIP feature layer")
        vision = config.vision_config
        patch_side = int(vision.image_size) // int(vision.patch_size)
        if self.owner._retain_ratio is not None:
            # Installation needs a concrete plan to build/validate the adapter;
            # the exact plan is re-solved from each image's native unpadded
            # content count in _compute_image_features().
            initial_plan, _, _ = plan_for_content_retention(
                original_content_tokens=patch_side * patch_side,
                retain_ratio=self.owner._retain_ratio,
                grid=(patch_side, patch_side),
                group_size=self.owner._group_size,
                pruning_layers=pruning_layers,
                keep_border_tokens=self.owner._keep_border_tokens,
                aggregate=self.owner._aggregate,
            )
            self.pruning_rate = initial_plan.pruning_rate
        elif self.owner._pruning_rate is not None:
            self.pruning_rate = float(self.owner._pruning_rate)
        else:
            target = 0.50 if self.owner._target_average is None else self.owner._target_average
            self.pruning_rate = layer_pruning_rate_for_average(target, len(pruning_layers))

        self.plan = (
            initial_plan
            if self.owner._retain_ratio is not None
            else build_plan(
                (patch_side, patch_side),
                group_size=self.owner._group_size,
                pruning_layers=pruning_layers,
                pruning_rate=self.pruning_rate,
                keep_border_tokens=self.owner._keep_border_tokens,
                aggregate=self.owner._aggregate,
                include_cls=False,
            )
        ).to(self.model.device)
        self.owner._plan = self.plan
        self.owner._backbone = {
            "vision_model": self.vision_model,
            "projector": self.projector,
        }
        self.owner._vlm = self.vlm
        self.owner._backend = "llava_next"
        dtype = self.owner._adapter_dtype or next(self.projector.parameters()).dtype
        self.owner.adapter = QuietPruneAdapter.from_projector(self.projector).to(
            device=self.model.device, dtype=dtype
        )
        self.owner._resolve_checkpoint_for_backend("llava_next")
        if self.owner._checkpoint_path is not None:
            self.owner.load_checkpoint(self.owner._checkpoint_path)

        self.original_get_image_features = self.inner.get_image_features
        self.original_prepare_inputs = self.vlm.prepare_inputs
        self.original_generate = self.vlm.generate

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            if self.owner._bypass:
                return self.original_get_image_features(*args, **kwargs)
            if self.pending_output is None:
                raise RuntimeError(
                    "QuietPrune/LLaVA-NeXT must prepare dynamic AnyRes features before "
                    "model.forward. Use the installed Varest backbone prepare_inputs(), "
                    "or compressor.prepare_training_inputs() for teacher-forced training."
                )
            image_sizes = kwargs.get("image_sizes", args[1] if len(args) > 1 else None)
            if isinstance(image_sizes, torch.Tensor) and self.pending_image_size is not None:
                if not torch.equal(
                    image_sizes.detach().to("cpu"), self.pending_image_size.detach().to("cpu")
                ):
                    raise RuntimeError(
                        "Pending QuietPrune features belong to different image_sizes"
                    )
            output = self.pending_output
            self.pending_output = None
            self.pending_image_size = None
            return output

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            inputs = self.original_prepare_inputs(*args, **kwargs)
            if self.owner._bypass:
                return inputs
            return self._prepare(inputs, labels=None, inference=True)

        def wrapped_generate(inputs: Any, *args: Any, **kwargs: Any) -> Any:
            supplied_cache = kwargs.get("past_key_values")
            if supplied_cache is None and isinstance(inputs, Mapping):
                supplied_cache = inputs.get("past_key_values")
            if supplied_cache is not None:
                raise NotImplementedError(
                    "QuietPrune/LLaVA-NeXT supports a fresh multimodal prefill followed by "
                    "cached decode, not insertion into a pre-existing KV cache."
                )
            generation_config = kwargs.get("generation_config", self.model.generation_config)
            raw_num_beams = kwargs.get("num_beams")
            if raw_num_beams is None:
                raw_num_beams = getattr(generation_config, "num_beams", None)
            num_beams = 1 if raw_num_beams is None else int(raw_num_beams)
            raw_num_return_sequences = kwargs.get("num_return_sequences")
            if raw_num_return_sequences is None:
                raw_num_return_sequences = getattr(generation_config, "num_return_sequences", None)
            num_return_sequences = (
                1 if raw_num_return_sequences is None else int(raw_num_return_sequences)
            )
            raw_use_cache = kwargs.get("use_cache")
            if raw_use_cache is None:
                raw_use_cache = getattr(generation_config, "use_cache", None)
            use_cache = True if raw_use_cache is None else bool(raw_use_cache)
            if num_beams != 1:
                raise NotImplementedError(
                    "QuietPrune/LLaVA-NeXT rejects beam search because its pending "
                    "ragged AnyRes feature batch is not beam-expanded."
                )
            if num_return_sequences != 1:
                raise NotImplementedError(
                    "QuietPrune/LLaVA-NeXT supports num_return_sequences=1 only; "
                    "dynamic image features are prepared for one decoder sequence."
                )
            if not use_cache:
                raise NotImplementedError(
                    "QuietPrune/LLaVA-NeXT generation requires use_cache=True; otherwise "
                    "Transformers resubmits pixels after the dynamic feature state is consumed."
                )
            return self.original_generate(inputs, *args, **kwargs)

        self.inner.get_image_features = wrapped_get_image_features
        self.vlm.prepare_inputs = wrapped_prepare_inputs
        self.vlm.generate = wrapped_generate
        budget_label = (
            f"retain_ratio={self.owner._retain_ratio:.6f}, per-image R is solved after unpadding"
            if self.owner._retain_ratio is not None
            else f"training R={self.pruning_rate:.6f}"
        )
        print(
            "QuietPrune/LLaVA-NeXT installed: "
            f"{budget_label}, layers={list(self.plan.pruning_layers)}; "
            "content targets exclude protected native image-newline rows.",
            flush=True,
        )
        if self.owner._checkpoint_path is None:
            print(
                "QuietPrune/LLaVA-NeXT: using the inverse-initialized, untrained adapter. "
                "A trained NeXT adapter is released -- set pretrained=true to use it; "
                "this run is the labelled ablation.",
                flush=True,
            )

    def uninstall(self) -> None:
        if self.original_get_image_features is not None:
            self.inner.get_image_features = self.original_get_image_features
        if self.original_prepare_inputs is not None:
            self.vlm.prepare_inputs = self.original_prepare_inputs
        if self.original_generate is not None:
            self.vlm.generate = self.original_generate
        self.pending_output = None
        self.pending_image_size = None
        self.original_get_image_features = None
        self.original_prepare_inputs = None
        self.original_generate = None

    def prepare_training_inputs(
        self, inputs: Mapping[str, Any], labels: torch.Tensor
    ) -> dict[str, Any]:
        """Prepare one student forward while masking answers out of Q-CLS."""

        return self._prepare(inputs, labels=labels, inference=False)

    def _prepare(
        self,
        inputs: Mapping[str, Any],
        *,
        labels: torch.Tensor | None,
        inference: bool,
    ) -> dict[str, Any]:
        if self.pending_output is not None:
            raise RuntimeError("A previous QuietPrune/LLaVA-NeXT prepared batch was not consumed")
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        image_sizes = inputs.get("image_sizes")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("QuietPrune/LLaVA-NeXT needs [1,T] input_ids")
        if input_ids.shape[0] != 1:
            raise NotImplementedError("QuietPrune/LLaVA-NeXT supports batch_size=1 only")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim not in (4, 5):
            raise NotImplementedError(
                "QuietPrune/LLaVA-NeXT accepts one image as 4D/5D pixel_values; video is unsupported"
            )
        if not isinstance(image_sizes, torch.Tensor) or tuple(image_sizes.shape) != (1, 2):
            raise NotImplementedError(
                "QuietPrune/LLaVA-NeXT supports exactly one source image per prompt"
            )
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match the untrimmed input_ids")

        attention_mask = inputs.get("attention_mask")
        query_mask = (
            attention_mask.clone()
            if isinstance(attention_mask, torch.Tensor)
            else torch.ones_like(input_ids)
        )
        query_mask *= (input_ids != self.image_token_id).to(query_mask.dtype)
        if labels is not None:
            query_mask *= (labels == -100).to(query_mask.dtype)
        context = torch.no_grad() if inference else nullcontext()
        with context:
            embeddings = self.inner.get_input_embeddings()(input_ids)
            adapter = self.owner.adapter
            if adapter is None:
                raise RuntimeError("QuietPrune adapter was not built")
            q_cls = adapter(embeddings.to(dtype=next(adapter.parameters()).dtype), query_mask)
            output, layout, budget = self._compute_image_features(
                pixel_values, image_sizes[0], q_cls
            )
        self.pending_output = BaseModelOutputWithPooling(pooler_output=[output])
        self.pending_image_size = image_sizes.detach().clone()
        self.last_budget = budget
        self.owner._last_budget_report = asdict(budget)

        tokenizer = getattr(self.vlm.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.image_token_id
        return _trim_one_visual_run(
            inputs,
            image_token_id=self.image_token_id,
            native_length=layout.native_length,
            target_length=int(output.shape[0]),
            pad_token_id=int(pad_token_id),
            labels=labels,
        )

    def _compute_image_features(
        self,
        pixel_values: torch.Tensor,
        image_size: torch.Tensor,
        q_cls: torch.Tensor,
    ) -> tuple[torch.Tensor, LlavaNextAnyResLayout, QuietPruneNextBudget]:
        if self.plan is None:
            raise RuntimeError("QuietPrune/LLaVA-NeXT is not installed")
        crop_count = image_size_to_num_patches(
            image_size,
            self.inner.config.image_grid_pinpoints,
            int(self.inner.config.vision_config.image_size),
        )
        crop_count = int(crop_count)
        if pixel_values.ndim == 5:
            if pixel_values.shape[0] != 1 or pixel_values.shape[1] != crop_count:
                raise ValueError(
                    f"Expected one crop batch with exactly {crop_count} views, "
                    f"got {tuple(pixel_values.shape)}"
                )
            crops = pixel_values[0]
        else:
            if pixel_values.shape[0] != crop_count:
                raise ValueError(
                    f"AnyRes geometry requires {crop_count} views, got {pixel_values.shape[0]}"
                )
            crops = pixel_values
        layout = build_anyres_layout(self.inner, image_size, crop_count, device=crops.device)
        original_content_tokens = layout.original_content_tokens
        structural_tokens = layout.structural_tokens
        if layout.native_length != original_content_tokens + structural_tokens:
            raise RuntimeError("Native AnyRes content and structural counts do not sum to its span")
        if self.owner._retain_ratio is not None:
            plan, target_tokens, _estimated_tokens = plan_for_content_retention(
                original_content_tokens=original_content_tokens,
                retain_ratio=self.owner._retain_ratio,
                grid=layout.patch_grid,
                group_size=self.owner._group_size,
                pruning_layers=self.plan.pruning_layers,
                keep_border_tokens=self.owner._keep_border_tokens,
                aggregate=self.owner._aggregate,
            )
            plan = plan.to(crops.device)
            requested_retention_ratio = self.owner._retain_ratio
            # Expose the latest concrete schedule without changing the public
            # ratio-valued budget property.
            self.plan = plan
            self.owner._plan = plan
            self.pruning_rate = plan.pruning_rate
        else:
            plan = self.plan
            requested_retention_ratio = (1.0 - plan.pruning_rate) ** len(
                plan.pruning_layers
            )
            target_tokens = target_content_tokens(
                original_content_tokens, requested_retention_ratio
            )
        vision = quietprune_crop_vision_forward(
            self.vision_model,
            crops,
            q_cls,
            plan=plan,
            selected_layer_idx=self.selected_layer_idx,
            valid_patch_mask=layout.valid_patch_mask,
            aggregate_mode=self.owner._aggregate_mode,
            normalize_aggregation=self.owner._normalize_aggregation,
            eps=self.owner._eps,
        )
        projected = self.projector(vision.tokens)
        aggregate = _combined_aggregate(
            vision.aggregate,
            layout,
            normalize=self.owner._normalize_aggregation,
            eps=self.owner._eps,
            track_source=self.owner._track_source,
        )
        if aggregate is not None:
            # Aggregation accumulates relevance weights in fp32 for stability.
            # Cast the resulting vision token back to the projector input dtype;
            # otherwise real bf16/fp16 checkpoints fail in the first Linear.
            aggregate_token = aggregate[0].to(
                device=vision.tokens.device,
                dtype=vision.tokens.dtype,
            )
            aggregate = (self.projector(aggregate_token), aggregate[1])
        packed = pack_quietprune_features(
            projected,
            vision.retained_indices,
            layout=layout,
            image_newline=self.inner.image_newline,
            aggregate=aggregate,
            track_source=self.owner._track_source,
        )

        if self.owner._track_source:
            history = [torch.eye(layout.native_length, dtype=torch.float32)]
            for stage_indices, stage_aggregate in vision.stage_states:
                stage_combined = _combined_aggregate(
                    stage_aggregate,
                    layout,
                    normalize=self.owner._normalize_aggregation,
                    eps=self.owner._eps,
                )
                # Source packing does not depend on feature values; one-channel
                # dummy features let the ordering routine build each stage map.
                dummy = torch.zeros(
                    stage_indices.shape[0],
                    stage_indices.shape[1],
                    1,
                    device=stage_indices.device,
                )
                stage_packed = pack_quietprune_features(
                    dummy,
                    stage_indices,
                    layout=layout,
                    image_newline=torch.zeros(1, device=stage_indices.device),
                    aggregate=(
                        (
                            torch.zeros(1, 1, device=stage_indices.device),
                            stage_combined[1],
                        )
                        if stage_combined is not None
                        else None
                    ),
                )
                if stage_packed.source is None:  # pragma: no cover - explicit above
                    raise RuntimeError("QuietPrune source tracking unexpectedly produced no map")
                history.append(stage_packed.source.detach().cpu())
            self.owner._source_history = history
        else:
            self.owner._source_history = []

        requested_final = 1.0 - requested_retention_ratio
        final_vit_patches = crop_count * plan.patches_after_layer[-1]
        realized_content_tokens = packed.patch_tokens + packed.aggregate_tokens
        if packed.newline_tokens != structural_tokens:
            raise RuntimeError("QuietPrune changed the protected AnyRes structural-token count")
        if int(packed.tokens.shape[0]) != realized_content_tokens + structural_tokens:
            raise RuntimeError("Packed QuietPrune content and structural counts are inconsistent")
        budget = QuietPruneNextBudget(
            convention="per_image_native_unpadded_content_ratio_structural_excluded",
            original_content_tokens=original_content_tokens,
            requested_retention_ratio=requested_retention_ratio,
            target_content_tokens=target_tokens,
            realized_content_tokens=realized_content_tokens,
            realized_retention_ratio=realized_content_tokens / original_content_tokens,
            structural_tokens=structural_tokens,
            requested_per_stage_pruning_rate=plan.pruning_rate,
            requested_final_patch_pruning_rate=requested_final,
            crop_views=crop_count,
            patches_per_view=plan.num_patches,
            stage_patch_tokens_all_views=tuple(
                crop_count * value for value in plan.patches_after_layer
            ),
            realized_vit_patch_pruning_rate=(
                1.0 - final_vit_patches / (crop_count * plan.num_patches)
            ),
            native_packed_tokens=layout.native_length,
            realized_patch_tokens=packed.patch_tokens,
            protected_newline_tokens=packed.newline_tokens,
            aggregate_tokens=packed.aggregate_tokens,
            realized_packed_tokens=int(packed.tokens.shape[0]),
            realized_decoder_pruning_rate=(
                1.0 - int(packed.tokens.shape[0]) / layout.native_length
            ),
        )
        return packed.tokens, layout, budget

    def checkpoint_metadata(self) -> dict[str, Any]:
        if self.plan is None:
            raise RuntimeError("QuietPrune/LLaVA-NeXT is not installed")
        return {
            "format_version": 2,
            "implementation": "quietprune_llava_next",
            "implementation_version": QUIETPRUNE_NEXT_IMPLEMENTATION_VERSION,
            "backbone": {
                "architecture": "LlavaNextForConditionalGeneration",
                "decoder": "LlamaForCausalLM",
                "model_id": LLAVA_NEXT_8B_MODEL_ID,
                "model_revision": getattr(self.vlm, "_model_revision", LLAVA_NEXT_8B_REVISION),
            },
            "transformers_version": __import__("transformers").__version__,
            "schedule": {
                "pruning_layers": list(self.plan.pruning_layers),
                "per_stage_pruning_rate": self.pruning_rate,
                "requested_final_patch_pruning_rate": (
                    1.0 - (1.0 - self.pruning_rate) ** len(self.plan.pruning_layers)
                ),
                "rounding": "drop=floor(active_2x2_groups*R), keep_at_least_one_group",
            },
            "geometry": {
                "crop_image_size": int(self.inner.config.vision_config.image_size),
                "patch_size": int(self.inner.config.vision_config.patch_size),
                "patch_grid_per_view": list(self.plan.grid),
                "image_grid_pinpoints": [
                    [int(value) for value in point]
                    for point in self.inner.config.image_grid_pinpoints
                ],
                "group_size": self.plan.group_size,
                "view_order": "base_then_processor_order_local_tiles",
                "local_order": "spatial_mosaic_then_native_unpadding",
                "newline_policy": "protected_one_per_unpadded_local_row; one_after_single_base_view",
                "aggregate_policy": "one_image_wide_token_after_native_packed_rows",
                "aggregate_pruned_tokens": self.plan.aggregate,
                "aggregate_mode": self.owner._aggregate_mode,
                "normalize_aggregation": self.owner._normalize_aggregation,
                "source_tracking_at_save": self.owner._track_source,
            },
            "budget": {
                "convention": (
                    "evaluation uses K_target=max(1,round(retain_ratio*"
                    "original_native_unpadded_content)); structural newlines excluded"
                ),
                "counts_global_and_local_together": True,
                "aggregate_counts_as_content": True,
                "stage_patch_tokens_per_view": list(self.plan.patches_after_layer),
                "realized_vit_patch_pruning_rate": (
                    1.0 - self.plan.patches_after_layer[-1] / self.plan.num_patches
                ),
                "newlines_protected": True,
                "aggregate_count": int(
                    self.plan.aggregate
                    and self.plan.patches_after_layer[-1] < self.plan.num_patches
                ),
            },
            "paper": QUIETPRUNE_PAPER_URL,
            "official_repository": None,
            "source_commits": dict(SOURCE_COMMITS),
        }

    def validate_checkpoint_metadata(self, metadata: Mapping[str, Any]) -> None:
        if int(metadata.get("format_version", -1)) != 2:
            raise ValueError(
                "LLaVA-NeXT QuietPrune requires format_version=2 provenance; "
                "LLaVA-1.5/Qwen adapter checkpoints are not interchangeable."
            )
        if metadata.get("implementation") != "quietprune_llava_next":
            raise ValueError("Checkpoint is not a QuietPrune/LLaVA-NeXT adapter")
        if (
            int(metadata.get("implementation_version", -1))
            != QUIETPRUNE_NEXT_IMPLEMENTATION_VERSION
        ):
            raise ValueError(
                "Checkpoint QuietPrune/LLaVA-NeXT implementation version is incompatible"
            )
        backbone = metadata.get("backbone")
        if not isinstance(backbone, Mapping):
            raise ValueError("Checkpoint is missing backbone provenance")
        if backbone.get("model_id") != LLAVA_NEXT_8B_MODEL_ID:
            raise ValueError(
                f"Checkpoint model_id={backbone.get('model_id')!r}; expected {LLAVA_NEXT_8B_MODEL_ID!r}"
            )
        if backbone.get("model_revision") != LLAVA_NEXT_8B_REVISION:
            raise ValueError(
                "Checkpoint model revision does not match the pinned LLaVA-NeXT-8B revision"
            )
        if backbone.get("architecture") != "LlavaNextForConditionalGeneration":
            raise ValueError("Checkpoint architecture is not LlavaNextForConditionalGeneration")
        if backbone.get("decoder") != "LlamaForCausalLM":
            raise ValueError("Checkpoint decoder is not the pinned Llama-3 architecture")
        if metadata.get("transformers_version") != TRANSFORMERS_LAYOUT_VERSION:
            raise ValueError(
                "Checkpoint was not produced with the verified Transformers layout version"
            )
        geometry = metadata.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("Checkpoint is missing LLaVA-NeXT AnyRes geometry provenance")
        if self.plan is None:
            raise RuntimeError("QuietPrune/LLaVA-NeXT is not installed")
        if geometry.get("patch_grid_per_view") != list(self.plan.grid):
            raise ValueError("Checkpoint CLIP crop geometry does not match the installed backbone")
        expected_crop_size = int(self.inner.config.vision_config.image_size)
        expected_patch_size = int(self.inner.config.vision_config.patch_size)
        if int(geometry.get("crop_image_size", -1)) != expected_crop_size:
            raise ValueError("Checkpoint CLIP crop size does not match the installed backbone")
        if int(geometry.get("patch_size", -1)) != expected_patch_size:
            raise ValueError("Checkpoint CLIP patch size does not match the installed backbone")
        expected_pinpoints = [
            [int(value) for value in point] for point in self.inner.config.image_grid_pinpoints
        ]
        if geometry.get("image_grid_pinpoints") != expected_pinpoints:
            raise ValueError("Checkpoint AnyRes grid pinpoints do not match the installed backbone")
        expected_layout = {
            "view_order": "base_then_processor_order_local_tiles",
            "local_order": "spatial_mosaic_then_native_unpadding",
            "newline_policy": ("protected_one_per_unpadded_local_row; one_after_single_base_view"),
            "aggregate_policy": "one_image_wide_token_after_native_packed_rows",
        }
        for key, expected_value in expected_layout.items():
            if geometry.get(key) != expected_value:
                raise ValueError(f"Checkpoint LLaVA-NeXT layout policy {key!r} is incompatible")
        if int(geometry.get("group_size", -1)) != self.plan.group_size:
            raise ValueError("Checkpoint QuietPrune group geometry does not match the compressor")
        if geometry.get("aggregate_pruned_tokens") is not self.plan.aggregate:
            raise ValueError("Checkpoint aggregate-token policy does not match the compressor")
        if geometry.get("aggregate_mode") != self.owner._aggregate_mode:
            raise ValueError("Checkpoint aggregate_mode does not match the compressor")
        if "normalize_aggregation" not in geometry or (
            bool(geometry["normalize_aggregation"]) != self.owner._normalize_aggregation
        ):
            raise ValueError("Checkpoint aggregation normalization does not match the compressor")
        training = metadata.get("training")
        if not isinstance(training, Mapping) or training.get("stage") not in {
            "tr",
            "ft",
            "untrained_ablation",
        }:
            raise ValueError("Checkpoint is missing an explicit QuietPrune training stage")
        if training.get("stage") in {"tr", "ft"}:
            if training.get("recipe") != "adapter_only_self_distillation_kl_plus_answer_ce":
                raise ValueError(
                    "Checkpoint training recipe is not canonical QuietPrune distillation"
                )
            expected_attention = QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION
            for key in (
                "attention_implementation",
                "effective_text_attention_implementation",
                "effective_vision_attention_implementation",
            ):
                if training.get(key) != expected_attention:
                    raise ValueError(
                        "Checkpoint was not trained with FlashAttention 2 for both "
                        f"Llama-3 and CLIP ({key}={training.get(key)!r})"
                    )
            if training.get("flash_attn_version") != QUIETPRUNE_NEXT_FLASH_ATTN_VERSION:
                raise ValueError(
                    "Checkpoint FlashAttention package version is incompatible: "
                    f"expected {QUIETPRUNE_NEXT_FLASH_ATTN_VERSION}, got "
                    f"{training.get('flash_attn_version')!r}"
                )
            schedule = metadata.get("schedule")
            trained_rate = (
                schedule.get("per_stage_pruning_rate") if isinstance(schedule, Mapping) else None
            )
            if trained_rate is None or abs(float(trained_rate) - TRAINING_PRUNING_RATE) > 1e-8:
                raise ValueError(
                    f"Checkpoint adapter was not trained at canonical R={TRAINING_PRUNING_RATE}"
                )
