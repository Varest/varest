"""Backbone-neutral Nuwa policy and decoder-attention capture.

This is the single shared implementation of Nuwa's two stages. Backbone files
only capture their native visual signals, map positions, and apply the selected
keep sets to their decoder geometry.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F


STAGE2_RATIO: float = 0.1
STAGE2_CONFIGS: dict[int, int] = {
    32: 8,
    64: 16,
    128: 32,
    192: 48,
    224: 56,
}
STAGE1_PRESERVE_CFG: dict[int, int] = {112: 16, 224: 32, 336: 48}


def build_distance_penalty_matrix(
    grid_h: int,
    grid_w: int,
    distance_threshold: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build Nuwa's spatial proximity matrix (paper Eq. 6)."""
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=dtype),
            torch.arange(grid_w, device=device, dtype=dtype),
            indexing="ij",
        ),
        dim=-1,
    )
    coords_flat = coords.reshape(-1, 2)
    distances = torch.cdist(coords_flat, coords_flat, p=2.0)
    return 1.0 - torch.clamp(
        distances / distance_threshold, max=1.0
    )


def region_candidate_selection(
    cls_attention: torch.Tensor,
    grid_h: int,
    grid_w: int,
    top_n_per_region: int,
) -> torch.Tensor:
    """Select the top candidates from each non-overlapping 2x2 region."""
    if grid_h < 2 or grid_w < 2:
        raise ValueError(
            f"Nuwa needs at least one 2x2 region; got {grid_h}x{grid_w}"
        )
    if not 1 <= top_n_per_region <= 4:
        raise ValueError(
            f"top_n_per_region must be in [1, 4], got {top_n_per_region}"
        )

    batch = cls_attention.shape[0]
    attention_grid = cls_attention.view(batch, grid_h, grid_w)
    unfolded = attention_grid.unfold(1, 2, 2).unfold(2, 2, 2)
    regions_h, regions_w = grid_h // 2, grid_w // 2
    num_regions = regions_h * regions_w
    regions = unfolded.reshape(batch, num_regions, 4)
    local_topk = regions.topk(k=top_n_per_region, dim=-1).indices

    region_ids = torch.arange(
        num_regions, device=cls_attention.device
    ).unsqueeze(0)
    region_y = region_ids // regions_w
    region_x = region_ids % regions_w
    global_y = region_y.unsqueeze(-1) * 2 + local_topk // 2
    global_x = region_x.unsqueeze(-1) * 2 + local_topk % 2
    return (global_y * grid_w + global_x).view(batch, -1)


def select_benchmark_tokens(
    cls_attention: torch.Tensor,
    candidate_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Select globally strongest candidates in original reading order."""
    scores = torch.gather(cls_attention, 1, candidate_indices)
    local = torch.topk(scores, num_tokens, dim=1).indices
    return torch.gather(candidate_indices, 1, local).sort(dim=1).values


def aggregate_tokens(
    patch_features: torch.Tensor,
    sim_features: torch.Tensor,
    cls_attention: torch.Tensor,
    benchmark_indices: torch.Tensor,
    dist_penalty: torch.Tensor,
    pillar_threshold_percentile: float = 0.55,
    *,
    self_weight_override: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate patches into Nuwa's Pillar and Collector benchmark tokens."""
    batch, n_tokens, _ = patch_features.shape
    num_benchmark = benchmark_indices.shape[1]
    sim_norm = F.normalize(sim_features.float(), p=2, dim=-1)
    similarity = torch.bmm(sim_norm, sim_norm.transpose(1, 2))
    weights = F.relu(similarity) * dist_penalty.float()

    benchmark_scores = torch.gather(
        cls_attention, 1, benchmark_indices
    ).float()
    threshold = torch.quantile(
        benchmark_scores,
        pillar_threshold_percentile,
        dim=1,
        keepdim=True,
    )
    is_pillar = benchmark_scores >= threshold
    selective = torch.ones(
        batch,
        num_benchmark,
        n_tokens,
        device=patch_features.device,
        dtype=weights.dtype,
    )
    selective[is_pillar] = 0.0
    selective.scatter_(
        dim=2, index=benchmark_indices.unsqueeze(2), value=1.0
    )

    gather_index = benchmark_indices.unsqueeze(-1).expand(
        -1, -1, n_tokens
    )
    benchmark_weights = torch.gather(
        weights, 1, gather_index
    ) * selective
    benchmark_weights = benchmark_weights / (
        benchmark_weights.sum(dim=-1, keepdim=True) + 1e-8
    )
    if self_weight_override:
        benchmark_weights.scatter_(
            dim=2, index=benchmark_indices.unsqueeze(-1), value=1.0
        )
    merged = torch.bmm(
        benchmark_weights.to(patch_features.dtype), patch_features
    )
    return merged, benchmark_weights


def nuwa_stage1(
    patch_features: torch.Tensor,
    sim_features: torch.Tensor,
    cls_attention: torch.Tensor,
    grid_h: int,
    grid_w: int,
    num_tokens: int,
    top_n_per_region: int,
    distance_threshold: float,
    pillar_threshold_percentile: float = 0.55,
    *,
    self_weight_override: bool = True,
    dist_penalty: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Nuwa Stage 1: Separation, Alignment, and Aggregation."""
    if patch_features.shape[1] != grid_h * grid_w:
        raise ValueError(
            f"Nuwa Stage 1 expects {grid_h}x{grid_w}="
            f"{grid_h * grid_w} patches, got {patch_features.shape[1]}"
        )
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
    if dist_penalty is None:
        dist_penalty = build_distance_penalty_matrix(
            grid_h,
            grid_w,
            distance_threshold,
            patch_features.device,
        )

    num_regions = (grid_h // 2) * (grid_w // 2)
    keep = min(num_tokens, num_regions * top_n_per_region)
    candidates = region_candidate_selection(
        cls_attention, grid_h, grid_w, top_n_per_region
    )
    benchmark = select_benchmark_tokens(
        cls_attention, candidates, keep
    )
    merged, weights = aggregate_tokens(
        patch_features,
        sim_features,
        cls_attention,
        benchmark,
        dist_penalty,
        pillar_threshold_percentile,
        self_weight_override=self_weight_override,
    )
    return merged, benchmark, weights


def resolve_region_budget(num_tokens: int, top_n_per_region: int,
                          grid_h: int, grid_w: int) -> tuple[int, int]:
    """Widen the regional candidate budget until it can supply Stage 1."""
    if num_tokens > top_n_per_region * grid_h * grid_w / 4:
        top_n_per_region = int(
            math.ceil((num_tokens * 4) / (grid_h * grid_w))
        )
        if top_n_per_region >= 4:
            top_n_per_region = 4
            num_tokens = grid_h * grid_w
    return top_n_per_region, num_tokens


def pool_merge_groups(values: torch.Tensor, group: int = 4) -> torch.Tensor:
    """Mean consecutive patch-merger groups, as required by Qwen."""
    if values.shape[-1] % group:
        raise ValueError(
            f"Cannot pool {values.shape[-1]} patches into groups of {group}"
        )
    return values.reshape(
        *values.shape[:-1], values.shape[-1] // group, group
    ).mean(dim=-1)


def pool_grid_blocks(values: torch.Tensor, grid_h: int, grid_w: int,
                     factor: int = 2) -> torch.Tensor:
    """Mean spatial ``factor x factor`` grid blocks, as required by InternVL."""
    if values.shape[-1] != grid_h * grid_w:
        raise ValueError(
            f"Expected {grid_h}x{grid_w}={grid_h * grid_w} values, "
            f"got {values.shape[-1]}"
        )
    if grid_h % factor or grid_w % factor:
        raise ValueError(
            f"{grid_h}x{grid_w} does not divide into {factor}x{factor} blocks"
        )
    blocks = values.reshape(
        *values.shape[:-1],
        grid_h // factor,
        factor,
        grid_w // factor,
        factor,
    )
    return blocks.mean(dim=(-3, -1)).flatten(start_dim=-2)


def tile_stitch_index(rows: int, cols: int, tile_h: int, tile_w: int,
                      device: torch.device | str = "cpu") -> torch.Tensor:
    """Map stitched-mosaic raster order back to tile-major token order."""
    grid_w = cols * tile_w
    mosaic = torch.arange(rows * tile_h * grid_w, device=device)
    mosaic_y, mosaic_x = mosaic // grid_w, mosaic % grid_w
    tile_index = (mosaic_y // tile_h) * cols + (mosaic_x // tile_w)
    within = (mosaic_y % tile_h) * tile_w + (mosaic_x % tile_w)
    return tile_index * (tile_h * tile_w) + within


def average_hidden_states(hidden_states: Sequence[torch.Tensor],
                          start: int, end: int) -> torch.Tensor:
    """Average the configured vision-tower hidden-state interval."""
    end = min(end, len(hidden_states))
    if not 0 <= start < end:
        raise ValueError(
            f"Invalid similarity-layer range [{start}, {end}) for "
            f"{len(hidden_states)} layers"
        )
    return torch.stack(list(hidden_states[start:end]), dim=0).mean(dim=0)


def get_stage2_keep_count(final_budget: int, stage1_count: int) -> int:
    """Resolve the released Stage-2 budget from the reported final budget."""
    keep = STAGE2_CONFIGS.get(final_budget)
    if keep is None:
        keep = max(
            round(final_budget / 4),
            int(stage1_count * STAGE2_RATIO),
        )
    return max(1, min(int(keep), int(stage1_count)))


def preserve_count(stage1_count: int) -> int:
    """Resolve Stage 2 from Stage-1 count, as the tiled release does."""
    stage1_count = int(stage1_count)
    keep = STAGE1_PRESERVE_CFG.get(
        stage1_count, int(stage1_count * STAGE2_RATIO)
    )
    return max(1, min(int(keep), max(1, stage1_count)))


def last_text_position(seq_len: int,
                       vision_positions: torch.Tensor) -> torch.Tensor:
    """Return the last non-visual column in each prompt row."""
    batch = vision_positions.shape[0]
    is_text = torch.ones(
        batch,
        seq_len,
        dtype=torch.bool,
        device=vision_positions.device,
    )
    is_text.scatter_(1, vision_positions, False)
    if not bool(is_text.any(dim=1).all()):
        raise ValueError(
            "Every position is visual; Nuwa Stage 2 has no text query"
        )
    reversed_first = torch.argmax(
        is_text.flip(dims=(1,)).to(torch.uint8), dim=1
    )
    return seq_len - 1 - reversed_first


def select_by_text_attention(
    attention_weights: torch.Tensor,
    vision_positions: torch.Tensor,
    k_keep: int,
    *,
    query_rows: int = 1,
    head_reduction: Literal["mean", "max"] = "mean",
) -> torch.Tensor:
    """Select visual tokens rated highest by trailing prompt attention."""
    if attention_weights.ndim != 4:
        raise ValueError(
            f"Expected [B, H, S, S] attention, got "
            f"{tuple(attention_weights.shape)}"
        )
    if head_reduction not in ("mean", "max"):
        raise ValueError(
            f"head_reduction must be 'mean' or 'max', got {head_reduction!r}"
        )
    if query_rows < 1:
        raise ValueError(f"query_rows must be >= 1, got {query_rows}")
    batch, heads, _, seq_len = attention_weights.shape
    if vision_positions.shape[0] != batch:
        raise ValueError(
            f"attention batch {batch} does not match "
            f"{vision_positions.shape[0]} visual-position rows"
        )
    num_vision = vision_positions.shape[1]
    if not 1 <= k_keep <= num_vision:
        raise ValueError(
            f"k_keep must be in [1, {num_vision}], got {k_keep}"
        )

    last = last_text_position(seq_len, vision_positions)
    offsets = torch.arange(
        query_rows - 1, -1, -1, device=last.device
    )
    rows = (last.unsqueeze(1) - offsets).clamp_min(0)
    rated = attention_weights.float().gather(
        2,
        rows[:, None, :, None].expand(
            batch, heads, query_rows, seq_len
        ),
    )
    rated = rated.mean(dim=2)
    reduced = (
        rated.mean(dim=1)
        if head_reduction == "mean"
        else rated.max(dim=1).values
    )
    scores = reduced.gather(1, vision_positions)
    return scores.topk(k_keep, dim=1).indices.sort(dim=1).values


def set_attn_implementation(model: Any, implementation: str) -> None:
    """Switch an attention backend, falling back to its config flag."""
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter(implementation)
            return
        except Exception:  # pragma: no cover - older custom implementations
            pass
    model.config._attn_implementation = implementation


def register_capture_attention(
    language_model: Any,
    capture_layer: int,
    sink: Callable[[torch.Tensor], None],
    *,
    owner: Any,
    label: str,
) -> tuple[str, str | None]:
    """Run eager attention only at Nuwa's Stage-2 capture layer."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import eager_mask
    from transformers.models.llama.modeling_llama import (
        eager_attention_forward,
    )

    try:
        from transformers.integrations.sdpa_attention import (
            sdpa_attention_forward as fallback,
        )
    except ImportError:  # pragma: no cover - keep working if the path moves
        fallback = eager_attention_forward

    def capture_attention(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple:
        if getattr(module, "layer_idx", None) != capture_layer:
            return fallback(
                module,
                query,
                key,
                value,
                attention_mask,
                *args,
                **kwargs,
            )
        output, weights = eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        )
        if weights is not None and query.shape[2] > 1:
            sink(weights.detach())
        return output, weights

    name = f"{label}_{id(owner) & 0xFFFFFF}"
    AttentionInterface.register(name, capture_attention)
    AttentionMaskInterface.register(name, eager_mask)
    previous = getattr(
        language_model.config, "_attn_implementation", None
    )
    set_attn_implementation(language_model, name)
    return name, previous


def unregister_capture_attention(name: str | None) -> None:
    """Remove a registered capture function and its compressor closure."""
    if name is None:
        return
    try:
        from transformers import AttentionInterface, AttentionMaskInterface

        AttentionInterface._global_mapping.pop(name, None)
        AttentionMaskInterface._global_mapping.pop(name, None)
    except Exception:  # pragma: no cover - a leftover entry is inert
        pass
