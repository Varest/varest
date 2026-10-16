"""Ratio-based Qwen2.5-VL adaptation of VFlowOpt.

The stock Qwen vision tower is windowed and emits a dynamic merged grid. This
adapter follows the repository's VisionZip path to replay the final full-attention
vision block, then carries that received-attention importance through VFlowOpt's
recycling stage and its two physical decoder cuts. The released VFlowOpt search
was calibrated on a fixed LLaVA-OneVision/SigLIP geometry, so those absolute
parameters and its pixel-histogram term are deliberately not reused here.
"""

from dataclasses import dataclass
from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    advance_decode_step,
    apply_prune_state,
    gather_mrope_position_ids,
    keep_columns,
    place_visual_tokens,
    pruned_attention_mask,
    ratio_keep_count,
)
from llmoe.compressor.vflowopt.vflowopt import StageLayers, llm_stage_keep


def qwen_vflow_decay(retain_ratio: float, stages: StageLayers) -> float:
    """Solve ``mean(lambda, lambda**2, lambda**3) == retain_ratio`` by stage width."""
    target = float(retain_ratio)
    if not 0.0 < target <= 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
    low, high = 0.0, 1.0
    for _ in range(80):
        decay = (low + high) / 2.0
        achieved = (
            decay * stages.l1
            + decay ** 2 * stages.l2
            + decay ** 3 * stages.l3
        ) / stages.total
        if achieved < target:
            low = decay
        else:
            high = decay
    return (low + high) / 2.0


def rectangular_cell_ids(
    grid: tuple[int, int], max_cells: int, device: torch.device
) -> torch.Tensor:
    """Assign a rectangular raster grid to at most ``max_cells`` spatial bins."""
    height, width = (int(value) for value in grid)
    if height < 1 or width < 1:
        raise ValueError(f"invalid grid {grid}")
    limit = max(1, min(int(max_cells), height * width))
    target_aspect = height / width
    best = (1, 1)
    best_key = (-1, float("inf"))
    for rows in range(1, min(height, limit) + 1):
        cols = min(width, limit // rows)
        if cols < 1:
            continue
        count = rows * cols
        aspect_error = abs(rows / cols - target_aspect)
        key = (count, -aspect_error)
        if key > best_key:
            best = rows, cols
            best_key = key

    rows, cols = best
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    y_bin = torch.div(y * rows, height, rounding_mode="floor")
    x_bin = torch.div(x * cols, width, rounding_mode="floor")
    return (y_bin[:, None] * cols + x_bin[None, :]).reshape(-1)


@dataclass(frozen=True)
class QwenVisionStageResult:
    tokens: torch.Tensor
    importance: torch.Tensor
    positions: torch.Tensor
    source: torch.Tensor


def qwen_vision_stage(
    features: torch.Tensor,
    importance: torch.Tensor,
    *,
    keep_target: int,
    grid: tuple[int, int],
    recycle_fraction: float,
) -> QwenVisionStageResult:
    """VFlowOpt Eq. 6-7 recycling over a rectangular post-merger grid."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    num_tokens = int(features.shape[0])
    if int(grid[0]) * int(grid[1]) != num_tokens:
        raise ValueError(f"grid {grid} does not match {num_tokens} tokens")
    scores = importance.reshape(-1).to(device=features.device, dtype=torch.float32)
    if scores.numel() != num_tokens:
        raise ValueError(
            f"importance {tuple(importance.shape)} does not match {num_tokens} tokens"
        )
    keep_target = max(1, min(num_tokens, int(keep_target)))
    if keep_target == num_tokens:
        return QwenVisionStageResult(
            tokens=features,
            importance=scores,
            positions=torch.arange(num_tokens, device=features.device),
            source=torch.eye(num_tokens, dtype=torch.float32, device=features.device),
        )

    max_cells = max(1, min(keep_target, int(round(keep_target * recycle_fraction))))
    cell_ids = rectangular_cell_ids(grid, max_cells, features.device)
    num_cells = int(cell_ids.max()) + 1
    pool_size = min(num_tokens, num_tokens - keep_target + num_cells)
    order = torch.argsort(scores, descending=False)
    pool = order[:pool_size]
    pool_scores = scores[pool]
    pool_cells = cell_ids[pool]
    occupied, inverse = torch.unique(pool_cells, sorted=True, return_inverse=True)
    num_groups = int(occupied.numel())

    totals = torch.zeros(num_groups, dtype=torch.float32, device=features.device)
    totals.index_add_(0, inverse, pool_scores)
    counts = torch.zeros_like(totals)
    counts.index_add_(0, inverse, torch.ones_like(pool_scores))
    degenerate = totals.abs() < 1e-8
    safe_totals = torch.where(degenerate, torch.ones_like(totals), totals)
    weights = torch.where(
        degenerate[inverse],
        1.0 / counts[inverse],
        pool_scores / safe_totals[inverse],
    )
    merged = torch.zeros(
        num_groups,
        features.shape[1],
        dtype=torch.float32,
        device=features.device,
    )
    merged.index_add_(0, inverse, weights.unsqueeze(-1) * features[pool].float())

    maxima = torch.full(
        (num_groups,), float("-inf"), dtype=torch.float32, device=features.device
    )
    maxima.scatter_reduce_(0, inverse, pool_scores, reduce="amax", include_self=True)
    slots = torch.arange(pool.numel(), device=features.device)
    representatives_at = torch.full(
        (num_groups,), pool.numel(), dtype=torch.long, device=features.device
    )
    is_max = pool_scores >= maxima[inverse]
    representatives_at.scatter_reduce_(
        0, inverse[is_max], slots[is_max], reduce="amin", include_self=True
    )
    representatives = pool[representatives_at]

    keep_mask = torch.ones(num_tokens, dtype=torch.bool, device=features.device)
    keep_mask[pool] = False
    keep_mask[representatives] = True
    fused = features.clone()
    fused[representatives] = merged.to(features.dtype)

    missing = keep_target - int(keep_mask.sum())
    if missing > 0:
        candidates = pool[~keep_mask[pool]]
        refill = candidates[scores[candidates].topk(missing).indices]
        keep_mask[refill] = True
    elif missing < 0:
        raise RuntimeError(
            f"VFlowOpt/Qwen recycling kept {int(keep_mask.sum())}, wanted {keep_target}"
        )

    positions = keep_mask.nonzero(as_tuple=True)[0]
    source = torch.zeros(
        keep_target, num_tokens, dtype=torch.float32, device=features.device
    )
    row_of = torch.full(
        (num_tokens,), -1, dtype=torch.long, device=features.device
    )
    row_of[positions] = torch.arange(keep_target, device=features.device)
    source[
        torch.arange(keep_target, device=features.device), positions
    ] = 1.0
    for group in range(num_groups):
        row = row_of[representatives[group]]
        members = pool[inverse == group]
        source[row] = 0.0
        source[row, members] = weights[inverse == group]

    return QwenVisionStageResult(
        tokens=fused[positions],
        importance=scores[positions],
        positions=positions,
        source=source,
    )


class QwenVFlowOptAdapter(QwenPruningAdapter):
    """Carry final-ViT importance through VFlowOpt's three Qwen stages."""

    method = "VFlowOpt"
    capture_vision_attention = True

    def __init__(
        self,
        *,
        retain_ratio: float,
        recycle_fraction: float,
        attention_chunk: int,
        track_source: bool,
        verbose: bool,
        enabled: bool,
    ) -> None:
        super().__init__(
            attention_chunk=attention_chunk,
            track_source=track_source,
        )
        self._retain_ratio = float(retain_ratio)
        self._recycle_fraction = float(recycle_fraction)
        self._verbose = bool(verbose)
        self._enabled = bool(enabled)
        self._stages: StageLayers | None = None
        self._decay = 1.0
        self._boundary_stage: dict[int, int] = {}
        self._image_start: int | None = None
        self._visual_count = 0
        self._stage_counts: tuple[int, int] = (0, 0)
        self._alive_importance: torch.Tensor | None = None
        self._position_ids: torch.Tensor | None = None
        self._stage_state: dict[int, dict[str, Any]] = {}
        self._composed_source: torch.Tensor | None = None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._reset_forward_state()

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        layers = getattr(self._language_model, "layers", None)
        if not layers:
            raise TypeError(
                "VFlowOpt's Qwen path needs `model.model.language_model.layers`."
            )
        if getattr(self._language_model, "has_sliding_layers", False):
            raise NotImplementedError(
                "VFlowOpt's physical stages require a full-attention Qwen decoder."
            )
        self._stages = StageLayers.even(len(layers))
        self._decay = qwen_vflow_decay(self._retain_ratio, self._stages)
        self._boundary_stage = {
            layer: stage
            for stage, layer in enumerate(self._stages.prune_layers)
        }
        self._register_common_hooks()
        for layer_idx in range(self._stages.prune_layers[0], len(layers)):
            self._handles.append(
                layers[layer_idx].register_forward_pre_hook(
                    self._make_layer_hook(layer_idx), with_kwargs=True
                )
            )
        print(
            f"[VFlowOpt/Qwen] install: retain_ratio={self._retain_ratio:g} "
            f"(layer-averaged), geometric_decay={self._decay:.6f}, "
            f"boundaries={list(self._stages.prune_layers)}, "
            f"recycle_fraction={self._recycle_fraction:g}, score @vision "
            f"block {len(self._visual.blocks) - 1}. This is the documented "
            "received-attention adaptation, not the released LLaVA calibration.",
            flush=True,
        )

    def _reset_forward_state(self) -> None:
        super()._reset_forward_state()
        self._image_start = None
        self._visual_count = 0
        self._stage_counts = (0, 0)
        self._alive_importance = None
        self._position_ids = None
        self._stage_state = {}
        self._composed_source = None

    def _prefill(
        self,
        args: tuple,
        kwargs: dict,
        inputs_embeds: torch.Tensor,
    ) -> tuple[tuple, dict] | None:
        span = self._locate_image_span(inputs_embeds)
        if span is None:
            return None
        if not self._enabled:
            return None
        if self._received is None or self._keys is None:
            raise RuntimeError(
                "VFlowOpt/Qwen found image tokens but captured no final-block "
                "vision attention."
            )
        first, num_visual = span
        importance, _ = self._merged_token_signals()
        if importance.numel() != num_visual:
            raise RuntimeError(
                f"VFlowOpt/Qwen scored {importance.numel()} tokens but the "
                f"prompt carries {num_visual}."
            )

        keep1 = ratio_keep_count(
            num_visual, self._decay, label="VFlowOpt/Qwen vision stage"
        )
        keep2 = ratio_keep_count(
            keep1, self._decay, label="VFlowOpt/Qwen decoder stage 2"
        )
        keep3 = ratio_keep_count(
            keep2, self._decay, label="VFlowOpt/Qwen decoder stage 3"
        )
        if keep3 < num_visual:
            self._require_kv_cache(kwargs)
        visual = inputs_embeds[0, first:first + num_visual]
        result = qwen_vision_stage(
            visual,
            importance.to(visual.device),
            keep_target=keep1,
            grid=self._merged_image_grid(num_visual),
            recycle_fraction=self._recycle_fraction,
        )
        changed = keep1 < num_visual
        if changed:
            modified, keep = place_visual_tokens(
                inputs_embeds,
                start=first,
                visual_count=num_visual,
                tokens=result.tokens.unsqueeze(0),
                positions=result.positions,
                label="VFlowOpt/Qwen",
            )
            self._apply_keep(kwargs, inputs_embeds, modified, keep)
            self._position_ids = kwargs["position_ids"]
        else:
            self._position_ids = self._resolve_position_ids(kwargs, inputs_embeds)

        self._image_start = first
        self._visual_count = keep1
        self._stage_counts = keep2, keep3
        self._alive_importance = result.importance
        if self._track_source:
            self._composed_source = result.source.detach().cpu()
            self._source_history = [
                torch.eye(num_visual, dtype=torch.float32),
                self._composed_source,
            ]

        widths = (self._stages.l1, self._stages.l2, self._stages.l3)
        achieved = (
            keep1 * widths[0] + keep2 * widths[1] + keep3 * widths[2]
        ) / (num_visual * self._stages.total)
        if self._verbose and self._forward_count == 1:
            new_len = (
                int(kwargs["inputs_embeds"].shape[1])
                if changed
                else inputs_embeds.shape[1]
            )
            print(
                f"[VFlowOpt/Qwen] first prefill: visual "
                f"{num_visual}->{keep1}->{keep2}->{keep3}, "
                f"realized_layer_average={achieved:.6f}, sequence "
                f"{inputs_embeds.shape[1]}->{new_len}",
                flush=True,
            )
        return (args, kwargs) if changed else None

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(
            _module: Any, args: tuple, kwargs: dict
        ) -> tuple[tuple, dict] | None:
            return self._layer_hook(args, kwargs, layer_idx)

        return hook

    def _layer_hook(
        self, args: tuple, kwargs: dict, layer_idx: int
    ) -> tuple[tuple, dict] | None:
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return None
        stage = self._boundary_stage.get(layer_idx)
        prefill = hidden.shape[1] > 1
        if stage is not None:
            if prefill:
                if self._image_start is None:
                    return None
                hidden = self._prune(hidden, kwargs, stage)
            elif stage in self._stage_state:
                advance_decode_step(
                    self._stage_state[stage], hidden, label="VFlowOpt/Qwen"
                )
        state = self._active_state(layer_idx)
        if state is None:
            return None
        apply_prune_state(kwargs, state, prefill=prefill)
        kwargs.pop("hidden_states", None)
        return (hidden,), kwargs

    def _active_state(self, layer_idx: int) -> dict[str, Any] | None:
        boundaries = self._stages.prune_layers
        for stage in range(len(boundaries) - 1, -1, -1):
            if boundaries[stage] <= layer_idx and stage in self._stage_state:
                return self._stage_state[stage]
        return None

    def _base_state(self, stage: int) -> dict[str, Any] | None:
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_state:
                return self._stage_state[earlier]
        return None

    def _prune(
        self, hidden: torch.Tensor, kwargs: dict, stage: int
    ) -> torch.Tensor:
        target = min(self._stage_counts[stage], self._visual_count)
        if target >= self._visual_count:
            return hidden
        base = self._base_state(stage)
        batch, seq_len, hidden_dim = hidden.shape
        start = int(self._image_start)
        columns = start + torch.arange(
            self._visual_count, device=hidden.device
        )
        if batch != 1 or int(columns.max()) >= seq_len:
            raise RuntimeError(
                f"VFlowOpt/Qwen visual span does not fit {tuple(hidden.shape)}."
            )
        local = llm_stage_keep(self._alive_importance, target)
        kept_columns = columns[local]
        keep_mask = torch.ones(
            batch, seq_len, dtype=torch.bool, device=hidden.device
        )
        keep_mask[:, columns] = False
        keep_mask[:, kept_columns] = True
        keep_index = keep_columns(keep_mask, label="VFlowOpt/Qwen")
        new_len = int(keep_index.shape[1])
        new_hidden = hidden.gather(
            1,
            keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim),
        )
        base_ids = (
            base["mrope_position_ids"] if base is not None else self._position_ids
        )
        base_mask = (
            base["attention_mask"]
            if base is not None
            else kwargs.get("attention_mask")
        )
        mrope_ids, text_ids = gather_mrope_position_ids(
            base_ids, keep_index, seq_len, new_len
        )
        position_embeddings = self._language_model.rotary_emb(
            new_hidden,
            mrope_ids[-3:] if mrope_ids.ndim == 3 else mrope_ids,
        )
        mask, valid = pruned_attention_mask(
            base_mask,
            keep_index,
            seq_len,
            label="VFlowOpt/Qwen",
        )
        self._stage_state[stage] = {
            "batch": batch,
            "mrope_position_ids": mrope_ids,
            "position_ids": text_ids,
            "position_embeddings": position_embeddings,
            "attention_mask": mask,
            "valid": valid,
            "cache_position": torch.arange(new_len, device=hidden.device),
            "kv_len": new_len,
        }
        self._visual_count = target
        self._alive_importance = self._alive_importance[local]
        if self._track_source and self._composed_source is not None:
            self._composed_source = self._composed_source[local.detach().cpu()]
            self._source_history.append(self._composed_source)
        if self._verbose and self._forward_count == 1:
            print(
                f"[VFlowOpt/Qwen] decoder stage {stage} "
                f"@layer{self._stages.prune_layers[stage]}: visual -> "
                f"{target}, sequence {seq_len}->{new_len}",
                flush=True,
            )
        return new_hidden
