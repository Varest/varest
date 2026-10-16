"""Canonical MustDrop adaptation for native Transformers LLaVA-NeXT.

The released method fixes a 3x3 local-spatial merge, roughly 440/576 vision
retention, an 8% key set, decoder boundaries 2/6/10/14, and dual global /
individual text-attention scoring. Native NeXT adds multiple AnyRes crops,
spatial unpadding, and learned newline rows. This adapter runs the released
vision policy independently on the base crop and each valid local-crop region,
protects native newline rows, and physically applies the four decoder cuts.

``retain_ratio`` is the only scale control. It denotes the visual-token count
averaged over all decoder layers; the fixed MustDrop stage shape is back-solved
to the largest integer schedule that does not exceed that ratio.
"""

import math
from typing import Any

import torch

from llmoe.compressor._pruning import (
    advance_decode_step,
    apply_prune_state,
    decoder_attention_rows,
    gather_position_ids,
    keep_columns,
    pruned_attention_mask,
)
from llmoe.compressor.internvl import image_token_id
from llmoe.compressor.llava_next import (
    LlavaNextFeatureTransformAdapter,
    balanced_group_budgets,
    crop_counts,
    decoder_visual_positions,
    patch_grid,
    split_crop_tensor,
    stitch_and_unpad_local_crops,
    validate_retain_ratio,
)
from llmoe.compressor.must_drop.must_drop import (
    _APPLY_ROPE,
    DEFAULT_KEY_RATE,
    DEFAULT_PRUNING_LAYERS,
    DEFAULT_VIT_RETENTION,
    key_set_size,
)
from llmoe.compressor.must_drop.must_drop_qwen import (
    LocalSpatialMerge,
    local_spatial_merge,
    progressive_keep_counts,
)

CANONICAL_WINDOW_SIZE = 3
CANONICAL_GLOBAL_THRESHOLD = 0.001
CANONICAL_INDIVIDUAL_THRESHOLD = 0.001


def _stage_widths(decoder_layers: int) -> tuple[int, int, int, int, int]:
    depth = int(decoder_layers)
    if depth <= DEFAULT_PRUNING_LAYERS[-1]:
        raise ValueError(
            "MustDrop's canonical layer-14 boundary needs at least 15 decoder "
            f"layers; this model has {depth}."
        )
    boundaries = (0, *DEFAULT_PRUNING_LAYERS, depth)
    return tuple(boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1))  # type: ignore[return-value]


def vision_stage_count(native_tokens: int, retain_ratio: float) -> int:
    """Return the fixed 440/576-equivalent LSM count, or identity at ratio one."""

    native = int(native_tokens)
    ratio = validate_retain_ratio(retain_ratio, label="MustDrop/LLaVA-NeXT")
    if native < 1:
        raise ValueError("MustDrop/LLaVA-NeXT cannot schedule an empty image")
    if ratio == 1.0:
        return native
    return max(1, min(native, math.floor(native * DEFAULT_VIT_RETENTION)))


def layer_average_schedule(
    native_tokens: int,
    retain_ratio: float,
    decoder_layers: int,
    *,
    key_count: int,
) -> tuple[int, tuple[int, int, int, int]]:
    """Resolve MustDrop's four progressive cuts from one layer-average ratio."""

    native = int(native_tokens)
    depth = int(decoder_layers)
    ratio = validate_retain_ratio(retain_ratio, label="MustDrop/LLaVA-NeXT")
    widths = _stage_widths(depth)
    stage1 = vision_stage_count(native, ratio)
    protected = int(key_count)
    if not 1 <= protected <= stage1:
        raise ValueError(f"MustDrop/LLaVA-NeXT key set must be in [1, {stage1}], got {protected}.")
    if ratio == 1.0:
        return native, (native, native, native, native)

    target = math.floor(native * depth * ratio)

    def cuts(final_count: int) -> tuple[int, int, int, int]:
        values = progressive_keep_counts(stage1, int(final_count), 4)
        return values  # type: ignore[return-value]

    def cost(final_count: int) -> int:
        counts = (stage1, *cuts(final_count))
        return sum(width * count for width, count in zip(widths, counts, strict=True))

    maximum = stage1 * depth
    if target > maximum:
        maximum_ratio = maximum / (native * depth)
        raise ValueError(
            f"MustDrop/LLaVA-NeXT retain_ratio={ratio:g} is above the released "
            f"vision stage's maximum {maximum_ratio:g}. Use <= {maximum_ratio:g}, "
            "or retain_ratio=1 for identity."
        )
    minimum = cost(protected)
    if target < minimum:
        minimum_ratio = minimum / (native * depth)
        raise ValueError(
            f"MustDrop/LLaVA-NeXT retain_ratio={ratio:g} is below the protected "
            f"key-set schedule's minimum {minimum_ratio:g} for this image."
        )

    low, high = protected, stage1
    while low < high:
        middle = (low + high + 1) // 2
        if cost(middle) <= target:
            low = middle
        else:
            high = middle - 1
    return stage1, cuts(low)


def _rectangular_grid(positions: torch.Tensor, width: int) -> tuple[int, int]:
    rows = torch.unique(torch.div(positions, width, rounding_mode="floor"))
    columns = torch.unique(positions % width)
    if int(rows.numel() * columns.numel()) != int(positions.numel()):
        raise RuntimeError(
            "MustDrop/LLaVA-NeXT found a non-rectangular valid region inside an AnyRes crop."
        )
    return int(rows.numel()), int(columns.numel())


class LlavaNextMustDropAdapter:
    """Crop-local LSM/KR plus four cache-safe Llama decoder cuts."""

    def __init__(
        self,
        owner: Any,
        vlm: Any,
        *,
        retain_ratio: float,
        track_source: bool,
    ) -> None:
        self._owner = owner
        self._vlm = vlm
        self._inner = vlm.model.model
        self._language_model = self._inner.language_model
        self._retain_ratio = float(retain_ratio)
        self._track_source = bool(track_source)
        self._depth = len(self._language_model.layers)
        self._image_token_id: int | None = None
        self._vision_positions: torch.Tensor | None = None
        self._visual_start: int | None = None
        self._visual_count = 0
        self._stage_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._key_mask: torch.Tensor | None = None
        self._composed_source: torch.Tensor | None = None
        self._stage_states: dict[int, dict[str, Any]] = {}
        self._boundary_stage = {layer: stage for stage, layer in enumerate(DEFAULT_PRUNING_LAYERS)}
        self._hook_handles: list[Any] = []
        self._logged = False
        self._lifecycle = LlavaNextFeatureTransformAdapter(
            label="MustDrop",
            retain_ratio=self._retain_ratio,
            transform_outputs=self._transform_outputs,
            target_for_length=self._stage1_target,
            on_prepared=self._on_prepared,
        )

    def _stage1_target(self, native_tokens: int) -> int:
        # The exact decoder schedule is resolved after packing, when the number
        # of protected native newline rows is known.
        stage1, _ = layer_average_schedule(
            native_tokens,
            self._retain_ratio,
            self._depth,
            key_count=1,
        )
        return stage1

    def install(self) -> None:
        _stage_widths(self._depth)
        if _APPLY_ROPE is None:
            raise RuntimeError(
                "MustDrop/LLaVA-NeXT cannot replay decoder attention because "
                "Transformers exposes no Llama RoPE helper."
            )
        strategy = getattr(self._vlm.model.config, "vision_feature_select_strategy", "default")
        if strategy != "default":
            raise ValueError(
                "MustDrop/LLaVA-NeXT supports the checkpoint's canonical "
                "vision_feature_select_strategy='default' only."
            )
        self._image_token_id = image_token_id(self._vlm, label="MustDrop/LLaVA-NeXT")
        vision_tower = getattr(self._inner, "vision_tower", None)
        self._owner._vit_layer = self._owner._locate_vit_attention_layer(vision_tower)
        if self._owner._vit_layer is None:
            raise RuntimeError(
                "MustDrop/LLaVA-NeXT could not locate the penultimate CLIP "
                "attention layer required for Key Retaining."
            )
        try:
            self._hook_handles.append(
                self._owner._vit_layer.register_forward_pre_hook(
                    self._owner._capture_vit_hidden, with_kwargs=True
                )
            )
            self._lifecycle.install(self._vlm)
            first = DEFAULT_PRUNING_LAYERS[0]
            for layer_index in range(first, self._depth):
                self._hook_handles.append(
                    self._language_model.layers[layer_index].register_forward_pre_hook(
                        self._make_layer_hook(layer_index), with_kwargs=True
                    )
                )
        except Exception:
            self.uninstall()
            raise

    def uninstall(self) -> None:
        self._lifecycle.uninstall()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._owner._vit_layer = None
        self._owner._captured_vit_hidden = None
        self._vision_positions = None
        self._visual_start = None
        self._visual_count = 0
        self._stage_counts = (0, 0, 0, 0)
        self._key_mask = None
        self._composed_source = None
        self._stage_states = {}

    def _on_prepared(
        self,
        inputs: Any,
        native_lengths: tuple[int, ...],
        stage1_lengths: tuple[int, ...],
    ) -> None:
        if len(native_lengths) != 1 or len(stage1_lengths) != 1:
            raise NotImplementedError(
                "MustDrop/LLaVA-NeXT follows the released batch_size=1, one-image path."
            )
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("MustDrop/LLaVA-NeXT needs input_ids")
        positions = decoder_visual_positions(
            input_ids,
            token_id=int(self._image_token_id),
            label="MustDrop",
        )
        self._owner._captured_vit_hidden = None
        self._owner._source_history = []
        self._stage_states = {}
        self._key_mask = None
        self._composed_source = None
        self._vision_positions = positions
        self._visual_start = None
        self._visual_count = 0
        if positions is None:
            return
        expected = stage1_lengths[0]
        if int(positions.numel()) != expected:
            raise RuntimeError("MustDrop's Stage-A schedule does not match the trimmed image span")
        self._visual_start = int(positions[0])
        self._visual_count = expected

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

    @torch.no_grad()
    def _transform_outputs(
        self,
        image_outputs: Any,
        targets: tuple[int, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        native_outputs = list(image_outputs.pooler_output)
        if len(native_outputs) != 1 or len(targets) != 1:
            raise NotImplementedError("MustDrop/LLaVA-NeXT supports one source image per prompt.")
        native_count = int(native_outputs[0].shape[0])
        if self._retain_ratio == 1.0:
            identity = torch.eye(native_count, dtype=torch.float32)
            self._stage_counts = (native_count,) * 4
            self._key_mask = torch.zeros(native_count, dtype=torch.bool)
            self._composed_source = identity
            if self._track_source:
                self._owner._source_history = [identity, identity.clone()]
            return image_outputs

        hidden_states = getattr(image_outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                "MustDrop/LLaVA-NeXT needs the selected CLIP hidden state for "
                "crop-local LSM before the projector."
            )
        feature_layer = self._argument(args, kwargs, "vision_feature_layer", 2)
        if feature_layer is None:
            feature_layer = getattr(self._inner.config, "vision_feature_layer", -2)
        if not isinstance(feature_layer, int):
            raise TypeError("MustDrop/LLaVA-NeXT supports one canonical vision feature layer.")
        features = hidden_states[feature_layer][:, 1:]
        metric = self._owner._captured_vit_hidden
        if not isinstance(metric, torch.Tensor) or metric.ndim != 3:
            raise RuntimeError("MustDrop/LLaVA-NeXT captured no penultimate CLIP features for LSM.")
        metric = metric[:, 1:]
        cls_attention = self._owner._compute_cls_attention()
        if cls_attention is None:
            raise RuntimeError(
                "MustDrop/LLaVA-NeXT could not compute the CLS attention required "
                "for Key Retaining."
            )

        image_sizes = self._argument(args, kwargs, "image_sizes", 1)
        if not isinstance(image_sizes, torch.Tensor):
            raise TypeError("MustDrop/LLaVA-NeXT needs processor image_sizes")
        counts = crop_counts(self._inner, image_sizes)
        feature_groups = split_crop_tensor(features, counts, label="MustDrop features")
        metric_groups = split_crop_tensor(metric, counts, label="MustDrop metrics")
        attention_groups = split_crop_tensor(cls_attention, counts, label="MustDrop CLS attention")
        packed, source, key_mask = self._compress_image(
            feature_groups[0],
            metric_groups[0],
            attention_groups[0],
            image_sizes[0],
            int(targets[0]),
            native_count,
        )
        stage1, schedule = layer_average_schedule(
            native_count,
            self._retain_ratio,
            self._depth,
            key_count=int(key_mask.sum()),
        )
        if stage1 != int(packed.shape[0]):
            raise RuntimeError(f"MustDrop packed {packed.shape[0]} rows but scheduled {stage1}.")
        self._stage_counts = schedule
        self._key_mask = key_mask
        self._composed_source = source
        if self._track_source:
            self._owner._source_history = [
                torch.eye(native_count, dtype=torch.float32),
                source,
            ]
        image_outputs.pooler_output = [packed]
        return image_outputs

    @staticmethod
    def _embed_source(
        source: torch.Tensor,
        native_columns: torch.Tensor,
        native_count: int,
    ) -> torch.Tensor:
        embedded = torch.zeros(source.shape[0], native_count, dtype=torch.float32)
        embedded[:, native_columns.detach().cpu().long()] = source.detach().cpu().float()
        return embedded

    def _compress_image(
        self,
        features: torch.Tensor,
        metric: torch.Tensor,
        importance: torch.Tensor,
        image_size: torch.Tensor,
        target: int,
        native_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crop_h, crop_w = patch_grid(self._inner)
        patches = crop_h * crop_w
        expected_shape = (features.shape[0], patches)
        if (
            features.ndim != 3
            or metric.ndim != 3
            or importance.ndim != 2
            or tuple(features.shape[:2]) != expected_shape
            or tuple(metric.shape[:2]) != expected_shape
            or tuple(importance.shape) != expected_shape
        ):
            raise ValueError("MustDrop/LLaVA-NeXT received inconsistent crop-local CLIP signals.")
        crop_count = int(features.shape[0])
        blocks = [features[0]]
        metrics = [metric[0]]
        scores = [importance[0]]
        positions = [torch.arange(patches, device=features.device)]
        grids = [(crop_h, crop_w)]
        local_h = local_w = 0
        if crop_count > 1:
            local_features, (local_h, local_w) = stitch_and_unpad_local_crops(
                self._inner, features[1:], image_size
            )
            local_metric, _ = stitch_and_unpad_local_crops(self._inner, metric[1:], image_size)
            local_scores, _ = stitch_and_unpad_local_crops(self._inner, importance[1:], image_size)
            raw_ids = torch.arange(crop_count - 1, device=features.device, dtype=torch.long)[
                :, None
            ].expand(crop_count - 1, patches)
            stitched_ids, _ = stitch_and_unpad_local_crops(self._inner, raw_ids, image_size)
            ids = stitched_ids[0, :, 0].long()
            mosaic_positions = torch.arange(local_h * local_w, device=features.device)
            for crop_index in range(crop_count - 1):
                group_positions = mosaic_positions[ids == crop_index]
                if not group_positions.numel():
                    continue
                blocks.append(local_features[0, group_positions])
                metrics.append(local_metric[0, group_positions])
                scores.append(local_scores[0, group_positions, 0])
                positions.append(group_positions)
                grids.append(_rectangular_grid(group_positions, local_w))
            newline_count = local_h
            expected_native = patches + local_h * local_w + local_h
        else:
            newline_count = 1
            expected_native = patches + 1
        if expected_native != native_count:
            raise RuntimeError(
                f"MustDrop reconstructed {expected_native} native rows, but "
                f"Transformers packed {native_count}."
            )

        patch_budget = target - newline_count
        budgets = balanced_group_budgets(
            patch_budget,
            [int(block.shape[0]) for block in blocks],
            label="MustDrop/LLaVA-NeXT crop-local LSM",
        )
        merged: list[LocalSpatialMerge] = []
        for block, block_metric, block_scores, grid, budget in zip(
            blocks, metrics, scores, grids, budgets, strict=True
        ):
            merged.append(
                local_spatial_merge(
                    block,
                    block_metric,
                    block_scores,
                    grid_thw=(1, *grid),
                    window_size=CANONICAL_WINDOW_SIZE,
                    keep=budget,
                )
            )
        projected = [self._inner.multi_modal_projector(result.tokens) for result in merged]

        base_source = self._embed_source(
            merged[0].source,
            torch.arange(patches),
            native_count,
        )
        output_parts = [projected[0]]
        score_parts = [merged[0].importance.float()]
        key_parts = [
            torch.zeros(projected[0].shape[0], dtype=torch.bool, device=projected[0].device)
        ]
        source_parts = [base_source]
        if crop_count > 1:
            local_positions: list[torch.Tensor] = []
            local_tokens: list[torch.Tensor] = []
            local_scores_out: list[torch.Tensor] = []
            local_sources: list[torch.Tensor] = []
            for group_index in range(1, len(merged)):
                result = merged[group_index]
                selected = positions[group_index].index_select(
                    0, result.positions.to(positions[group_index].device)
                )
                native_columns = (
                    patches
                    + positions[group_index]
                    + torch.div(positions[group_index], local_w, rounding_mode="floor")
                )
                local_positions.append(selected)
                local_tokens.append(projected[group_index])
                local_scores_out.append(result.importance.float())
                local_sources.append(
                    self._embed_source(result.source, native_columns, native_count)
                )
            all_positions = torch.cat(local_positions)
            order = all_positions.argsort()
            all_positions = all_positions[order]
            all_tokens = torch.cat(local_tokens)[order]
            all_scores = torch.cat(local_scores_out)[order]
            all_sources = torch.cat(local_sources)[order.detach().cpu()]
            cursor = 0
            for row in range(local_h):
                row_count = int(
                    (torch.div(all_positions, local_w, rounding_mode="floor") == row).sum()
                )
                if row_count:
                    output_parts.append(all_tokens[cursor : cursor + row_count])
                    score_parts.append(all_scores[cursor : cursor + row_count])
                    key_parts.append(
                        torch.zeros(row_count, dtype=torch.bool, device=all_tokens.device)
                    )
                    source_parts.append(all_sources[cursor : cursor + row_count])
                    cursor += row_count
                output_parts.append(self._inner.image_newline[None])
                score_parts.append(torch.full((1,), float("inf"), device=all_tokens.device))
                key_parts.append(torch.ones(1, dtype=torch.bool, device=all_tokens.device))
                newline = torch.zeros(1, native_count, dtype=torch.float32)
                newline[0, patches + (row + 1) * local_w + row] = 1.0
                source_parts.append(newline)
            if cursor != int(all_positions.numel()):
                raise RuntimeError(
                    f"MustDrop placed {cursor} of {all_positions.numel()} local rows."
                )
        else:
            output_parts.append(self._inner.image_newline[None])
            score_parts.append(torch.full((1,), float("inf"), device=projected[0].device))
            key_parts.append(torch.ones(1, dtype=torch.bool, device=projected[0].device))
            newline = torch.zeros(1, native_count, dtype=torch.float32)
            newline[0, patches] = 1.0
            source_parts.append(newline)

        output = torch.cat(output_parts)
        packed_scores = torch.cat(score_parts).to(output.device)
        key_mask = torch.cat(key_parts).to(output.device)
        source = torch.cat(source_parts)
        if int(output.shape[0]) != target:
            raise RuntimeError(
                f"MustDrop Stage A produced {output.shape[0]} rows, expected {target}."
            )
        protected = max(int(key_mask.sum()), key_set_size(target, DEFAULT_KEY_RATE))
        extra = protected - int(key_mask.sum())
        if extra:
            candidates = packed_scores.clone()
            candidates[key_mask] = -float("inf")
            key_mask[candidates.topk(extra).indices] = True
        if tuple(source.shape) != (target, native_count):
            raise RuntimeError(
                f"MustDrop Stage-A source has shape {tuple(source.shape)}, "
                f"expected {(target, native_count)}."
            )
        return output, source, key_mask

    def _make_layer_hook(self, layer_index: int) -> Any:
        def hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return None
            stage = self._boundary_stage.get(layer_index)
            prefill = hidden.shape[1] > 1
            if stage is not None:
                if prefill:
                    if self._visual_start is None:
                        return None
                    hidden = self._prune(module, hidden, kwargs, stage)
                elif stage in self._stage_states:
                    advance_decode_step(
                        self._stage_states[stage],
                        hidden,
                        label="MustDrop/LLaVA-NeXT",
                    )
            state = self._active_state(layer_index)
            if state is None:
                return None
            apply_prune_state(kwargs, state, prefill=prefill)
            kwargs.pop("hidden_states", None)
            return (hidden,), kwargs

        return hook

    def _active_state(self, layer_index: int) -> dict[str, Any] | None:
        for stage in range(len(DEFAULT_PRUNING_LAYERS) - 1, -1, -1):
            if DEFAULT_PRUNING_LAYERS[stage] <= layer_index and stage in self._stage_states:
                return self._stage_states[stage]
        return None

    def _base_state(self, stage: int) -> dict[str, Any] | None:
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_states:
                return self._stage_states[earlier]
        return None

    def _dual_importance(
        self,
        layer: Any,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
    ) -> torch.Tensor:
        start = int(self._visual_start)
        first_text = start + self._visual_count
        raters = torch.arange(first_text, hidden.shape[1], device=hidden.device)
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.shape[-1] == hidden.shape[1]
        ):
            raters = raters[attention_mask[0, raters].bool()]
        if not raters.numel():
            raise RuntimeError("MustDrop/LLaVA-NeXT needs a prompt token after the visual span.")
        attention = decoder_attention_rows(
            layer,
            hidden,
            position_embeddings,
            attention_mask,
            raters.unsqueeze(0),
            apply_rope=_APPLY_ROPE,
            label="MustDrop/LLaVA-NeXT",
        )
        visual = attention[:, :, start : start + self._visual_count].float()
        global_score = visual.sum(dim=1)[0]
        individual_score = visual.max(dim=1).values[0]
        global_fraction = global_score / global_score.sum().clamp_min(1e-12)
        return torch.maximum(
            global_fraction / CANONICAL_GLOBAL_THRESHOLD,
            individual_score / CANONICAL_INDIVIDUAL_THRESHOLD,
        )

    def _prune(
        self,
        layer: Any,
        hidden: torch.Tensor,
        kwargs: dict[str, Any],
        stage: int,
    ) -> torch.Tensor:
        target = min(self._stage_counts[stage], self._visual_count)
        if target >= self._visual_count:
            return hidden
        base = self._base_state(stage)
        batch, seq_len, hidden_dim = hidden.shape
        start = int(self._visual_start)
        columns = start + torch.arange(self._visual_count, device=hidden.device)
        if batch != 1 or int(columns.max()) >= seq_len:
            raise RuntimeError(
                f"MustDrop/LLaVA-NeXT visual span does not fit {tuple(hidden.shape)}."
            )
        base_ids = base["position_ids"] if base is not None else kwargs.get("position_ids")
        if not (
            isinstance(base_ids, torch.Tensor)
            and base_ids.ndim == 2
            and base_ids.shape[-1] == seq_len
        ):
            base_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        base_mask = base["attention_mask"] if base is not None else kwargs.get("attention_mask")
        position_embeddings = (
            base["position_embeddings"] if base is not None else kwargs.get("position_embeddings")
        )
        if not (isinstance(position_embeddings, tuple) and len(position_embeddings) == 2):
            position_embeddings = self._language_model.rotary_emb(hidden, position_ids=base_ids)
        importance = self._dual_importance(layer, hidden, position_embeddings, base_mask)
        key_index = (
            self._key_mask.nonzero(as_tuple=True)[0].to(hidden.device)
            if self._key_mask is not None
            else torch.zeros(0, dtype=torch.long, device=hidden.device)
        )
        if key_index.numel() > target:
            raise RuntimeError(
                f"MustDrop's protected set ({key_index.numel()}) exceeds "
                f"the stage target ({target})."
            )
        selectable = importance.clone()
        if key_index.numel():
            selectable[key_index] = -float("inf")
        extra_count = target - int(key_index.numel())
        extra = selectable.topk(extra_count).indices if extra_count else key_index.new_zeros(0)
        local = torch.cat([key_index, extra]).sort().values
        kept_columns = columns[local]
        keep_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden.device)
        keep_mask[:, columns] = False
        keep_mask[:, kept_columns] = True
        keep_index = keep_columns(keep_mask, label="MustDrop/LLaVA-NeXT")
        new_len = int(keep_index.shape[1])
        new_hidden = hidden.gather(
            1,
            keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim),
        )
        position_ids = gather_position_ids(base_ids, keep_index, seq_len)
        attention_mask, valid = pruned_attention_mask(
            base_mask,
            keep_index,
            seq_len,
            label="MustDrop/LLaVA-NeXT",
        )
        self._stage_states[stage] = {
            "batch": batch,
            "position_ids": position_ids,
            "position_embeddings": self._language_model.rotary_emb(
                new_hidden, position_ids=position_ids
            ),
            "attention_mask": attention_mask,
            "valid": valid,
            "cache_position": torch.arange(new_len, device=hidden.device),
            "kv_len": new_len,
        }
        self._visual_count = target
        if self._key_mask is not None:
            self._key_mask = self._key_mask[local.to(self._key_mask.device)].to(hidden.device)
        if self._track_source and self._composed_source is not None:
            self._composed_source = self._composed_source[local.detach().cpu()]
            self._owner._source_history.append(self._composed_source)
        if not self._logged:
            self._logged = True
            print(
                f"[MustDrop/LLaVA-NeXT] visual schedule "
                f"{self._stage_counts} @layers{DEFAULT_PRUNING_LAYERS}",
                flush=True,
            )
        return new_hidden
