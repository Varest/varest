"""Qwen2.5-VL integration for MustDrop.

The vision stage follows the same safe route as VisionZip: capture received
attention and rotary keys from Qwen's last vision block, fold them through the
patch merger, then run local spatial merging over the projected visual span
after Qwen has concatenated it with text. Qwen has no CLS token, so received
attention supplies the Key Retaining saliency signal.

The decoder stage keeps MustDrop's dual global/individual text-attention rule,
but ranks to an exact progressive ratio schedule. Physical cuts use the shared
M-RoPE and cache bookkeeping rather than the mask-only LLaVA compatibility
path.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import torch
from torch.nn import functional as F

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    advance_decode_step,
    apply_prune_state,
    decoder_attention_rows,
    gather_mrope_position_ids,
    keep_columns,
    place_visual_tokens,
    pruned_attention_mask,
    ratio_keep_count,
)
from llmoe.utils.qwen import (
    APPLY_MROPE,
    DEFAULT_ATTENTION_CHUNK,
    resolve_mrope_section,
)


@dataclass
class LocalSpatialMerge:
    """One exact-count local spatial merge over Qwen LLM visual tokens."""

    tokens: torch.Tensor
    metric: torch.Tensor
    importance: torch.Tensor
    positions: torch.Tensor
    source: torch.Tensor


def progressive_keep_counts(
    initial: int, final: int, stages: int
) -> tuple[int, ...]:
    """Geometrically interpolate exact survivor counts to the final budget."""
    initial, final, stages = int(initial), int(final), int(stages)
    if initial < 1 or final < 1 or final > initial:
        raise ValueError(
            f"invalid progressive counts: initial={initial}, final={final}"
        )
    if stages < 1:
        if final != initial:
            raise ValueError(
                "at least one pruning stage is required when counts differ"
            )
        return ()
    if final == initial:
        return (initial,) * stages

    decay = (final / initial) ** (1.0 / stages)
    counts: list[int] = []
    previous = initial
    for stage in range(1, stages + 1):
        count = (
            final
            if stage == stages
            else int(initial * (decay ** stage))
        )
        count = max(final, min(previous, count))
        counts.append(count)
        previous = count
    counts[-1] = final
    return tuple(counts)


def _mean_pairwise_similarity(
    metric: torch.Tensor, members: torch.Tensor
) -> torch.Tensor:
    values = F.normalize(
        metric.index_select(0, members).float(),
        dim=-1,
        eps=1e-6,
    )
    count = values.shape[0]
    if count < 2:
        return values.new_tensor(-float("inf"))
    similarities = values @ values.transpose(0, 1)
    return (
        similarities.sum() - count
    ) / float(count * (count - 1))


def local_spatial_merge(
    tokens: torch.Tensor,
    metric: torch.Tensor,
    importance: torch.Tensor,
    *,
    grid_thw: tuple[int, int, int],
    window_size: int,
    keep: int,
) -> LocalSpatialMerge:
    """Merge the most redundant local windows to exactly keep tokens.

    Full and edge windows are supported, which matters for Qwen's dynamic image
    grids. A partial final window merge supplies any remainder that whole
    window collapses cannot express. Only extremely aggressive ratios below one
    token per local window use the documented spatially ordered fallback.
    """
    if tokens.ndim != 2 or metric.ndim != 2 or importance.ndim != 1:
        raise ValueError(
            "MustDrop/Qwen local merge expects tokens [N,D], metric [N,C], "
            "and importance [N]"
        )
    num_tokens = int(tokens.shape[0])
    if metric.shape[0] != num_tokens or importance.shape[0] != num_tokens:
        raise ValueError("MustDrop/Qwen vision signals have inconsistent lengths")
    keep = max(1, min(int(keep), num_tokens))
    temporal, height, width = (int(value) for value in grid_thw)
    if temporal * height * width != num_tokens:
        raise ValueError(
            f"MustDrop/Qwen grid {grid_thw} contains "
            f"{temporal * height * width} tokens, expected {num_tokens}"
        )

    grid = torch.arange(
        num_tokens, device=tokens.device
    ).reshape(temporal, height, width)
    windows: list[torch.Tensor] = []
    step = int(window_size)
    for frame in range(temporal):
        for row in range(0, height, step):
            for column in range(0, width, step):
                members = grid[
                    frame,
                    row:min(row + step, height),
                    column:min(column + step, width),
                ].reshape(-1)
                if members.numel() > 1:
                    windows.append(members)

    remaining = num_tokens - keep
    assigned = torch.zeros(
        num_tokens, dtype=torch.bool, device=tokens.device
    )
    groups: list[torch.Tensor] = []
    if windows and remaining:
        window_scores = torch.stack([
            _mean_pairwise_similarity(metric, members)
            for members in windows
        ])
        order = window_scores.argsort(descending=True).tolist()
        for window_index in order:
            if remaining < 1:
                break
            members = windows[window_index]
            merge_size = min(
                int(members.numel()), remaining + 1
            )
            if merge_size < 2:
                continue
            if merge_size < members.numel():
                values = F.normalize(
                    metric.index_select(0, members).float(),
                    dim=-1,
                    eps=1e-6,
                )
                similarity = values @ values.transpose(0, 1)
                redundancy = (
                    similarity.sum(dim=-1) - 1.0
                ) / max(int(members.numel()) - 1, 1)
                chosen = members[
                    redundancy.topk(merge_size).indices
                ]
            else:
                chosen = members
            chosen = chosen.sort().values
            groups.append(chosen)
            assigned[chosen] = True
            remaining -= int(chosen.numel()) - 1

    for index in (~assigned).nonzero(as_tuple=True)[0].tolist():
        groups.append(
            torch.tensor(
                [index], dtype=torch.long, device=tokens.device
            )
        )
    groups.sort(key=lambda group: int(group.min()))

    if len(groups) > keep:
        # A ratio below one output per spatial window cannot be represented by
        # LSM alone. Preserve reading order while coalescing adjacent groups.
        old_groups = groups
        group_count = len(old_groups)
        groups = []
        for index in range(keep):
            first = (index * group_count) // keep
            last = ((index + 1) * group_count) // keep
            groups.append(
                torch.cat(old_groups[first:last]).sort().values
            )
    if len(groups) != keep:
        raise RuntimeError(
            f"MustDrop/Qwen local merge produced {len(groups)} groups, "
            f"expected {keep}"
        )

    merged_tokens = torch.stack([
        tokens.index_select(0, group).float().mean(dim=0)
        for group in groups
    ]).to(tokens.dtype)
    merged_metric = torch.stack([
        metric.index_select(0, group).float().mean(dim=0)
        for group in groups
    ]).to(metric.dtype)
    merged_importance = torch.stack([
        importance.index_select(0, group).float().mean()
        for group in groups
    ])
    positions = torch.tensor(
        [int(group.min()) for group in groups],
        dtype=torch.long,
        device=tokens.device,
    )
    source = torch.zeros(
        keep,
        num_tokens,
        dtype=torch.float32,
        device=tokens.device,
    )
    for row, group in enumerate(groups):
        source[row, group] = 1.0 / float(group.numel())

    return LocalSpatialMerge(
        tokens=merged_tokens,
        metric=merged_metric,
        importance=merged_importance,
        positions=positions,
        source=source,
    )


class QwenMustDropAdapter(QwenPruningAdapter):
    """VisionZip-style Qwen vision capture plus progressive MustDrop cuts."""

    method = "MustDrop"
    capture_vision_attention = True

    def __init__(
        self,
        *,
        retain_ratio: float,
        window_size: int,
        vit_retention: float,
        key_rate: float,
        prefill_stage: bool,
        pruning_layers: Sequence[int],
        global_thr: float,
        individual_thr: float,
        attention_chunk: int = DEFAULT_ATTENTION_CHUNK,
    ) -> None:
        super().__init__(
            attention_chunk=attention_chunk,
            track_source=True,
        )
        self._retain_ratio = float(retain_ratio)
        self._window_size = int(window_size)
        self._vit_retention = float(vit_retention)
        self._key_rate = float(key_rate)
        self._prefill_stage_enabled = bool(prefill_stage)
        self._requested_layers = tuple(
            sorted(set(int(layer) for layer in pruning_layers))
        )
        self._global_thr = float(global_thr)
        self._individual_thr = float(individual_thr)

        self._pruning_layers: tuple[int, ...] = ()
        self._boundary_stage: dict[int, int] = {}
        self._image_start: int | None = None
        self._visual_count = 0
        self._final_count = 0
        self._stage_counts: tuple[int, ...] = ()
        self._position_ids: torch.Tensor | None = None
        self._stage_state: dict[int, dict[str, Any]] = {}
        self._key_mask: torch.Tensor | None = None
        self._composed_source: torch.Tensor | None = None
        self._logged = False

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        layers = getattr(self._language_model, "layers", None)
        if not layers:
            raise TypeError(
                "MustDrop's Qwen path needs "
                "model.model.language_model.layers."
            )
        if self._prefill_stage_enabled:
            if APPLY_MROPE is None:
                raise RuntimeError(
                    "MustDrop cannot reproduce Qwen decoder attention "
                    "because transformers exposes no M-RoPE helper."
                )
            valid = tuple(
                layer for layer in self._requested_layers
                if 0 <= layer < len(layers)
            )
            if not valid:
                raise ValueError(
                    f"no pruning_layers fit a {len(layers)}-layer "
                    f"Qwen decoder: {self._requested_layers}"
                )
            self._pruning_layers = valid
            self._boundary_stage = {
                layer: stage
                for stage, layer in enumerate(valid)
            }

        self._register_common_hooks()
        if self._prefill_stage_enabled:
            first = self._pruning_layers[0]
            for layer_index in range(first, len(layers)):
                self._handles.append(
                    layers[layer_index].register_forward_pre_hook(
                        self._make_layer_hook(layer_index),
                        with_kwargs=True,
                    )
                )

        print(
            f"[MustDrop/Qwen] install: retain_ratio="
            f"{self._retain_ratio:g}, vit_retention="
            f"{self._vit_retention:g}, window={self._window_size}, "
            f"key_rate={self._key_rate:g}, decoder_layers="
            f"{list(self._pruning_layers)}, score @vision block "
            f"{len(self._visual.blocks) - 1}",
            flush=True,
        )

    def _reset_forward_state(self) -> None:
        super()._reset_forward_state()
        self._image_start = None
        self._visual_count = 0
        self._final_count = 0
        self._stage_counts = ()
        self._position_ids = None
        self._stage_state = {}
        self._key_mask = None
        self._composed_source = None

    def _merged_grid(self, num_visual: int) -> tuple[int, int, int]:
        grid = self._image_grid_thw
        if (
            not isinstance(grid, torch.Tensor)
            or tuple(grid.shape) != (1, 3)
        ):
            raise NotImplementedError(
                "MustDrop/Qwen supports one image grid per prompt."
            )
        temporal, height, width = (
            int(value) for value in grid[0].tolist()
        )
        merge_size = int(self._visual.spatial_merge_size)
        if height % merge_size or width % merge_size:
            raise RuntimeError(
                f"MustDrop/Qwen image grid {(temporal, height, width)} "
                f"is not divisible by spatial_merge_size={merge_size}."
            )
        merged = (
            temporal,
            height // merge_size,
            width // merge_size,
        )
        if merged[0] * merged[1] * merged[2] != num_visual:
            raise RuntimeError(
                f"MustDrop/Qwen merged grid {merged} does not match "
                f"{num_visual} visual prompt tokens."
            )
        return merged

    def _prefill(
        self,
        args: tuple,
        kwargs: dict,
        inputs_embeds: torch.Tensor,
    ) -> tuple[tuple, dict] | None:
        span = self._locate_image_span(inputs_embeds)
        if span is None:
            return None
        if (
            self._received is None
            or self._keys is None
            or self._image_grid_thw is None
        ):
            raise RuntimeError(
                "MustDrop/Qwen found image tokens but captured no last-block "
                "vision attention."
            )

        first, num_visual = span
        final_count = ratio_keep_count(
            num_visual,
            self._retain_ratio,
            label="MustDrop/Qwen",
        )
        vision_ratio = (
            max(self._retain_ratio, self._vit_retention)
            if self._prefill_stage_enabled
            else self._retain_ratio
        )
        vision_count = ratio_keep_count(
            num_visual,
            vision_ratio,
            label="MustDrop/Qwen vision stage",
        )
        vision_count = max(final_count, vision_count)
        if final_count < num_visual:
            self._require_kv_cache(kwargs)

        received, merged_keys = self._merged_token_signals()
        if (
            received.numel() != num_visual
            or merged_keys.shape[0] != num_visual
        ):
            raise RuntimeError(
                "MustDrop/Qwen vision signals do not match the prompt's "
                f"{num_visual} visual tokens."
            )
        visual = inputs_embeds[0, first:first + num_visual]
        merged = local_spatial_merge(
            visual,
            merged_keys.to(
                device=visual.device, dtype=visual.dtype
            ),
            received.to(visual.device),
            grid_thw=self._merged_grid(num_visual),
            window_size=self._window_size,
            keep=vision_count,
        )

        changed = vision_count < num_visual
        if changed:
            modified, keep = place_visual_tokens(
                inputs_embeds,
                start=first,
                visual_count=num_visual,
                tokens=merged.tokens.unsqueeze(0),
                positions=merged.positions,
                label="MustDrop/Qwen",
            )
            self._apply_keep(
                kwargs, inputs_embeds, modified, keep
            )
            self._position_ids = kwargs["position_ids"]
        else:
            self._prefill_state = None
            self._position_ids = self._resolve_position_ids(
                kwargs, inputs_embeds
            )

        key_count = min(
            final_count,
            max(1, ceil(self._key_rate * vision_count)),
        )
        key_index = merged.importance.topk(key_count).indices
        self._key_mask = torch.zeros(
            vision_count,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        self._key_mask[key_index] = True

        self._image_start = first
        self._visual_count = vision_count
        self._final_count = final_count
        self._stage_counts = (
            progressive_keep_counts(
                vision_count,
                final_count,
                len(self._pruning_layers),
            )
            if self._prefill_stage_enabled
            else ()
        )
        self._composed_source = merged.source.detach().cpu()
        self._source_history = [
            torch.eye(num_visual, dtype=torch.float32),
            self._composed_source,
        ]

        if not self._logged:
            self._logged = True
            sequence_length = int(
                kwargs["inputs_embeds"].shape[1]
                if changed
                else inputs_embeds.shape[1]
            )
            print(
                f"[MustDrop/Qwen] first prefill: visual "
                f"{num_visual} -> {vision_count} after LSM, "
                f"key_set={key_count}, decoder schedule="
                f"{list(self._stage_counts)}, final={final_count}, "
                f"sequence={sequence_length}",
                flush=True,
            )
        return (args, kwargs) if changed else None

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(
            module: Any, args: tuple, kwargs: dict
        ) -> tuple[tuple, dict] | None:
            return self._layer_hook(
                module, args, kwargs, layer_idx
            )

        return hook

    def _active_state(
        self, layer_idx: int
    ) -> dict[str, Any] | None:
        for stage in range(len(self._pruning_layers) - 1, -1, -1):
            if (
                self._pruning_layers[stage] <= layer_idx
                and stage in self._stage_state
            ):
                return self._stage_state[stage]
        return None

    def _base_state(
        self, stage: int
    ) -> dict[str, Any] | None:
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_state:
                return self._stage_state[earlier]
        return None

    def _layer_hook(
        self,
        module: Any,
        args: tuple,
        kwargs: dict,
        layer_idx: int,
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
                hidden = self._prune(
                    module, hidden, kwargs, stage
                )
            elif stage in self._stage_state:
                advance_decode_step(
                    self._stage_state[stage],
                    hidden,
                    label="MustDrop/Qwen",
                )

        state = self._active_state(layer_idx)
        if state is None:
            return None
        apply_prune_state(kwargs, state, prefill=prefill)
        kwargs.pop("hidden_states", None)
        return (hidden,), kwargs

    def _dual_importance(
        self,
        layer: Any,
        hidden: torch.Tensor,
        position_embeddings: Any,
        attention_mask: Any,
        start: int,
    ) -> torch.Tensor:
        seq_len = hidden.shape[1]
        first_text = start + self._visual_count
        raters = torch.arange(
            first_text, seq_len, device=hidden.device
        )
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.shape[-1] == seq_len
        ):
            raters = raters[
                attention_mask[0, raters].bool()
            ]
        if not raters.numel():
            raise RuntimeError(
                "MustDrop/Qwen needs at least one prompt token after the "
                "visual span for dual-attention filtering."
            )
        raters = raters.unsqueeze(0)
        section = resolve_mrope_section(layer.self_attn)
        attention = decoder_attention_rows(
            layer,
            hidden,
            position_embeddings,
            attention_mask,
            raters,
            apply_rope=lambda query, key, cos, sin: APPLY_MROPE(
                query, key, cos, sin, section
            ),
            label="MustDrop/Qwen",
        )
        visual = attention[
            :, :, start:start + self._visual_count
        ].float()
        global_score = visual.sum(dim=1)[0]
        individual_score = visual.max(dim=1).values[0]
        global_fraction = global_score / global_score.sum().clamp_min(
            1e-12
        )
        global_scaled = global_fraction / max(
            self._global_thr, 1e-12
        )
        individual_scaled = individual_score / max(
            self._individual_thr, 1e-12
        )
        return torch.maximum(global_scaled, individual_scaled)

    def _prune(
        self,
        layer: Any,
        hidden: torch.Tensor,
        kwargs: dict,
        stage: int,
    ) -> torch.Tensor:
        target = min(
            int(self._stage_counts[stage]),
            self._visual_count,
        )
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
                f"MustDrop/Qwen visual span does not fit decoder input "
                f"{tuple(hidden.shape)}."
            )
        base_ids = (
            base["mrope_position_ids"]
            if base is not None
            else self._position_ids
        )
        base_embeddings = (
            base["position_embeddings"]
            if base is not None
            else kwargs.get("position_embeddings")
        )
        base_mask = (
            base["attention_mask"]
            if base is not None
            else kwargs.get("attention_mask")
        )

        importance = self._dual_importance(
            layer,
            hidden,
            base_embeddings,
            base_mask,
            start,
        )
        key_index = (
            self._key_mask.nonzero(as_tuple=True)[0]
            if self._key_mask is not None
            else torch.zeros(
                0, dtype=torch.long, device=hidden.device
            )
        )
        if key_index.numel() > target:
            raise RuntimeError(
                f"MustDrop/Qwen key set ({key_index.numel()}) exceeds "
                f"the ratio target ({target})."
            )
        extra_count = target - int(key_index.numel())
        selectable = importance.clone()
        if key_index.numel():
            selectable[key_index] = -float("inf")
        extra = (
            selectable.topk(extra_count).indices
            if extra_count
            else key_index.new_zeros(0)
        )
        local = torch.cat([key_index, extra]).sort().values
        kept_columns = columns[local]

        keep_mask = torch.ones(
            batch,
            seq_len,
            dtype=torch.bool,
            device=hidden.device,
        )
        keep_mask[:, columns] = False
        keep_mask[:, kept_columns] = True
        keep_index = keep_columns(
            keep_mask, label="MustDrop/Qwen"
        )
        new_len = int(keep_index.shape[1])
        new_hidden = hidden.gather(
            1,
            keep_index.unsqueeze(-1).expand(
                batch, new_len, hidden_dim
            ),
        )
        mrope_ids, text_ids = gather_mrope_position_ids(
            base_ids,
            keep_index,
            seq_len,
            new_len,
        )
        rope_ids = (
            mrope_ids[-3:]
            if mrope_ids.ndim == 3
            else mrope_ids
        )
        position_embeddings = self._language_model.rotary_emb(
            new_hidden, rope_ids
        )
        mask, valid = pruned_attention_mask(
            base_mask,
            keep_index,
            seq_len,
            label="MustDrop/Qwen",
        )
        self._stage_state[stage] = {
            "batch": batch,
            "mrope_position_ids": mrope_ids,
            "position_ids": text_ids,
            "position_embeddings": position_embeddings,
            "attention_mask": mask,
            "valid": valid,
            "cache_position": torch.arange(
                new_len, device=hidden.device
            ),
            "kv_len": new_len,
        }

        if self._key_mask is not None:
            self._key_mask = self._key_mask[local]
        self._visual_count = target
        if self._composed_source is not None:
            self._composed_source = self._composed_source[
                local.detach().cpu()
            ]
            self._source_history.append(
                self._composed_source
            )
        if self._forward_count == 1:
            print(
                f"[MustDrop/Qwen] decoder stage {stage} "
                f"@layer{self._pruning_layers[stage]}: "
                f"visual -> {target}, sequence {seq_len} -> "
                f"{new_len}",
                flush=True,
            )
        return new_hidden

