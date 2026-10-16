"""Canonical training-free VFlowOpt on native Transformers LLaVA-NeXT.

VFlowOpt computes one ViT information-flow map, recycles low-importance spatial
cells before the LLM, then carries the same importance through two evenly
spaced decoder cuts. The released LLaVA-OneVision 25% profile fixes
t=34.75955, alpha=0, and a 15.5808-patch cell side on its 27x27 SigLIP grid.
This adapter keeps those choices, normalizes the four-cell topology over each
valid CLIP/AnyRes crop region, and derives only the three exact counts from a
decoder-layer-average retain_ratio.
"""

from dataclasses import dataclass
from typing import Any

import torch

from llmoe.compressor._pruning import (
    advance_decode_step,
    apply_prune_state,
    gather_position_ids,
    keep_columns,
    pruned_attention_mask,
    ratio_keep_count,
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
from llmoe.compressor.vflowopt.vflowopt import (
    StageLayers,
    context_relevance_safe,
    grid_cell_ids,
    grid_cells,
    head_averaged_self_attention,
    llm_stage_keep,
    vision_stage,
)
from llmoe.compressor.vflowopt.vflowopt_qwen import qwen_vflow_decay

CANONICAL_THRESHOLD = 34.759552803070484
CANONICAL_ALPHA = 0.0
CANONICAL_CELL_SIDE = 15.580797034937687
REFERENCE_GRID_SIDE = 27
CANONICAL_CELL_COUNT = grid_cells(CANONICAL_CELL_SIDE, REFERENCE_GRID_SIDE)


@dataclass(frozen=True)
class NextStrategy:
    """A searched VFlowOpt profile for LLaVA-NeXT.

    The canonical adapter fixes ``t`` and the cell side to upstream's
    LLaVA-OneVision release and ties the three stage ratios to one geometric
    decay, so ``retain_ratio`` is its only control. That is a one-parameter
    family; the paper's Algorithm 1 searches the stage ratios and the importance
    threshold per model. This carries a searched profile so a calibration for
    *this* backbone can be applied.

    ``alpha`` is absent by design: the entropy term of Eq. 4 is not implemented
    on the NeXT path (importance here is pure attention relevance), so a fitted
    value would describe nothing. Upstream's ``CANONICAL_ALPHA`` is 0 for the
    same reason.
    """

    r1: float
    r2: float
    r3: float
    t: float = CANONICAL_THRESHOLD
    a: float = CANONICAL_CELL_SIDE

    def keeps(self, native_tokens: int) -> tuple[int, int, int]:
        """The three cumulative keeps this profile produces for one image."""
        keep1 = ratio_keep_count(native_tokens, self.r1,
                                 label="VFlowOpt/LLaVA-NeXT vision stage")
        keep2 = ratio_keep_count(keep1, self.r2,
                                 label="VFlowOpt/LLaVA-NeXT decoder stage 2")
        keep3 = ratio_keep_count(keep2, self.r3,
                                 label="VFlowOpt/LLaVA-NeXT decoder stage 3")
        return keep1, keep2, keep3

    def layer_average(self, stages: StageLayers) -> float:
        """Eq. 5 in ratio space: the budget this profile actually spends."""
        return (self.r1 * stages.l1
                + self.r1 * self.r2 * stages.l2
                + self.r1 * self.r2 * self.r3 * stages.l3) / stages.total


def layer_average_schedule(
    native_tokens: int,
    retain_ratio: float,
    decoder_layers: int,
    strategy: "NextStrategy | None" = None,
) -> tuple[StageLayers, tuple[int, int, int]]:
    """Resolve VFlowOpt's three cumulative keeps.

    Without a ``strategy`` this is upstream's canonical behaviour exactly: one
    geometric scale solved from ``retain_ratio``. With one, the searched stage
    ratios are used instead and ``retain_ratio`` only validates the request.
    """

    native = int(native_tokens)
    ratio = validate_retain_ratio(retain_ratio, label="VFlowOpt/LLaVA-NeXT")
    if native < 1:
        raise ValueError("VFlowOpt/LLaVA-NeXT cannot schedule an empty image")
    stages = StageLayers.even(int(decoder_layers))
    if strategy is not None:
        return stages, strategy.keeps(native)
    decay = qwen_vflow_decay(ratio, stages)
    keep1 = ratio_keep_count(native, decay, label="VFlowOpt/LLaVA-NeXT vision stage")
    keep2 = ratio_keep_count(keep1, decay, label="VFlowOpt/LLaVA-NeXT decoder stage 2")
    keep3 = ratio_keep_count(keep2, decay, label="VFlowOpt/LLaVA-NeXT decoder stage 3")
    return stages, (keep1, keep2, keep3)


def _rectangular_grid(positions: torch.Tensor, width: int) -> tuple[int, int]:
    rows = torch.unique(torch.div(positions, width, rounding_mode="floor"))
    columns = torch.unique(positions % width)
    if int(rows.numel() * columns.numel()) != int(positions.numel()):
        raise RuntimeError(
            "VFlowOpt/LLaVA-NeXT found a non-rectangular valid region inside an AnyRes crop."
        )
    return int(rows.numel()), int(columns.numel())


class LlavaNextVFlowOptAdapter:
    """Crop-aware VFlow recycling plus two cache-safe Llama cuts."""

    def __init__(
        self,
        owner: Any,
        vlm: Any,
        *,
        retain_ratio: float,
        track_source: bool,
        verbose: bool,
        strategy: NextStrategy | None = None,
    ) -> None:
        self._owner = owner
        self._vlm = vlm
        self._inner = vlm.model.model
        self._language_model = self._inner.language_model
        self._retain_ratio = float(retain_ratio)
        self._track_source = bool(track_source)
        self._verbose = bool(verbose)
        self._depth = len(self._language_model.layers)
        # `None` keeps upstream's canonical profile; a searched one overrides the
        # threshold and cell side as well as the stage ratios.
        self._strategy = strategy
        self._threshold = CANONICAL_THRESHOLD if strategy is None else float(strategy.t)
        self._cell_side = CANONICAL_CELL_SIDE if strategy is None else float(strategy.a)
        self._cell_count = grid_cells(self._cell_side, REFERENCE_GRID_SIDE)
        self._stages, _ = layer_average_schedule(
            1, self._retain_ratio, self._depth, self._strategy)
        self._image_token_id: int | None = None
        self._visual_start: int | None = None
        self._visual_count = 0
        self._stage_counts: tuple[int, int] = (0, 0)
        self._alive_importance: torch.Tensor | None = None
        self._composed_source: torch.Tensor | None = None
        self._stage_states: dict[int, dict[str, Any]] = {}
        self._boundary_stage = {
            layer: stage for stage, layer in enumerate(self._stages.prune_layers)
        }
        self._hook_handles: list[Any] = []
        self._logged = False
        self._lifecycle = LlavaNextFeatureTransformAdapter(
            label="VFlowOpt",
            retain_ratio=self._retain_ratio,
            transform_outputs=self._transform_outputs,
            target_for_length=self._stage1_target,
            on_prepared=self._on_prepared,
        )

    def _stage1_target(self, native_tokens: int) -> int:
        _, counts = layer_average_schedule(
            native_tokens, self._retain_ratio, self._depth, self._strategy)
        return counts[0]

    def install(self) -> None:
        strategy = getattr(self._vlm.model.config, "vision_feature_select_strategy", "default")
        if strategy != "default":
            raise ValueError(
                "VFlowOpt/LLaVA-NeXT supports the checkpoint's canonical "
                "vision_feature_select_strategy='default' only."
            )
        self._image_token_id = image_token_id(self._vlm, label="VFlowOpt/LLaVA-NeXT")
        vision_tower = getattr(self._inner, "vision_tower", None)
        self._owner._vit_layer = self._owner._locate_attention_layer(
            vision_tower, self._inner.config
        )
        if self._owner._vit_layer is None:
            raise RuntimeError(
                "VFlowOpt/LLaVA-NeXT could not locate the CLIP layer feeding "
                "the multimodal projector."
            )
        try:
            self._hook_handles.append(
                self._owner._vit_layer.register_forward_pre_hook(
                    self._owner._capture_vit_hidden, with_kwargs=True
                )
            )
            self._lifecycle.install(self._vlm)
            for layer_index in range(self._stages.prune_layers[0], self._depth):
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
        self._owner._captured_hidden = None
        self._visual_start = None
        self._visual_count = 0
        self._stage_counts = (0, 0)
        self._alive_importance = None
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
                "VFlowOpt/LLaVA-NeXT follows the released batch_size=1, one-image path."
            )
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("VFlowOpt/LLaVA-NeXT needs input_ids")
        positions = decoder_visual_positions(
            input_ids,
            token_id=int(self._image_token_id),
            label="VFlowOpt",
        )
        self._owner._captured_hidden = None
        self._owner._vit_importance = []
        self._owner._source_history = []
        self._stage_states = {}
        self._alive_importance = None
        self._composed_source = None
        self._visual_start = None
        self._visual_count = 0
        if positions is None:
            return
        # Must use the same profile the lifecycle trimmed with, or this check
        # compares canonical counts against a searched span and always fails.
        _, counts = layer_average_schedule(
            native_lengths[0], self._retain_ratio, self._depth, self._strategy)
        if counts[0] != stage1_lengths[0] or int(positions.numel()) != counts[0]:
            raise RuntimeError("VFlowOpt's Stage-A schedule does not match the trimmed image span")
        self._visual_start = int(positions[0])
        self._visual_count = counts[0]
        self._stage_counts = counts[1], counts[2]

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
            raise NotImplementedError("VFlowOpt/LLaVA-NeXT supports one source image per prompt.")
        native_count = int(native_outputs[0].shape[0])
        if self._retain_ratio == 1.0:
            identity = torch.eye(native_count, dtype=torch.float32)
            self._stage_counts = (native_count, native_count)
            self._alive_importance = torch.ones(native_count, device=native_outputs[0].device)
            self._composed_source = identity
            if self._track_source:
                self._owner._source_history = [identity, identity.clone()]
            return image_outputs

        hidden_states = getattr(image_outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                "VFlowOpt/LLaVA-NeXT needs the selected CLIP hidden state for "
                "its pre-projector recycling stage."
            )
        feature_layer = self._argument(args, kwargs, "vision_feature_layer", 2)
        if feature_layer is None:
            feature_layer = getattr(self._inner.config, "vision_feature_layer", -2)
        if not isinstance(feature_layer, int):
            raise TypeError("VFlowOpt/LLaVA-NeXT supports one canonical vision feature layer.")
        features = hidden_states[feature_layer][:, 1:]
        captured = self._owner._captured_hidden
        if not isinstance(captured, torch.Tensor) or captured.ndim != 3:
            raise RuntimeError(
                "VFlowOpt/LLaVA-NeXT captured no CLIP features for attention replay."
            )
        attention = head_averaged_self_attention(self._owner._vit_layer, captured)
        context, _ = context_relevance_safe(attention, self._threshold)
        importance = context[:, 1:]
        if importance.shape[:2] != features.shape[:2]:
            raise RuntimeError(
                f"VFlowOpt scored {tuple(importance.shape)} CLIP patches but "
                f"features have shape {tuple(features.shape)}."
            )

        image_sizes = self._argument(args, kwargs, "image_sizes", 1)
        if not isinstance(image_sizes, torch.Tensor):
            raise TypeError("VFlowOpt/LLaVA-NeXT needs processor image_sizes")
        counts = crop_counts(self._inner, image_sizes)
        feature_groups = split_crop_tensor(features, counts, label="VFlowOpt features")
        importance_groups = split_crop_tensor(importance, counts, label="VFlowOpt importance")
        packed, packed_importance, source, newline_count = self._compress_image(
            feature_groups[0],
            importance_groups[0],
            image_sizes[0],
            int(targets[0]),
            native_count,
        )
        if self._stage_counts[-1] < newline_count:
            raise ValueError(
                f"VFlowOpt/LLaVA-NeXT's final stage keeps {self._stage_counts[-1]} "
                f"rows but this AnyRes image has {newline_count} protected newline "
                "rows. Use a larger retain_ratio."
            )
        self._alive_importance = packed_importance
        self._composed_source = source
        self._owner._vit_importance = [packed_importance]
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
        importance: torch.Tensor,
        image_size: torch.Tensor,
        target: int,
        native_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        crop_h, crop_w = patch_grid(self._inner)
        patches = crop_h * crop_w
        if (
            features.ndim != 3
            or importance.ndim != 2
            or tuple(features.shape[:2]) != tuple(importance.shape)
            or int(features.shape[1]) != patches
        ):
            raise ValueError("VFlowOpt/LLaVA-NeXT received inconsistent crop-local CLIP signals.")
        crop_count = int(features.shape[0])
        blocks = [features[0]]
        scores = [importance[0]]
        positions = [torch.arange(patches, device=features.device)]
        grids = [(crop_h, crop_w)]
        local_h = local_w = 0
        if crop_count > 1:
            local_features, (local_h, local_w) = stitch_and_unpad_local_crops(
                self._inner, features[1:], image_size
            )
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
                f"VFlowOpt reconstructed {expected_native} native rows, but "
                f"Transformers packed {native_count}."
            )

        patch_budget = target - newline_count
        budgets = balanced_group_budgets(
            patch_budget,
            [int(block.shape[0]) for block in blocks],
            minimum=self._cell_count,
            label="VFlowOpt/LLaVA-NeXT four-cell recycling",
        )
        results = []
        for block, block_scores, grid, budget in zip(blocks, scores, grids, budgets, strict=True):
            height, width = grid
            cells = grid_cell_ids(
                int(block.shape[0]),
                width,
                self._cell_side,
                block.device,
                grid_h=height,
                grid_w=width,
                reference_side=REFERENCE_GRID_SIDE,
            )
            n_cells = int(torch.unique(cells).numel())
            results.append(
                vision_stage(
                    block,
                    block_scores,
                    keep_target=budget,
                    a=self._cell_side,
                    side=width,
                    n_cells=n_cells,
                    grid_h=height,
                    grid_w=width,
                    reference_side=REFERENCE_GRID_SIDE,
                )
            )
        projected = [self._inner.multi_modal_projector(result.features) for result in results]

        base_source = self._embed_source(results[0].source, torch.arange(patches), native_count)
        output_parts = [projected[0]]
        importance_parts = [results[0].importance.float()]
        source_parts = [base_source]
        if crop_count > 1:
            local_positions: list[torch.Tensor] = []
            local_tokens: list[torch.Tensor] = []
            local_importance: list[torch.Tensor] = []
            local_sources: list[torch.Tensor] = []
            for group_index in range(1, len(results)):
                result = results[group_index]
                selected = positions[group_index].index_select(
                    0, result.keep_indices.to(positions[group_index].device)
                )
                native_columns = (
                    patches
                    + positions[group_index]
                    + torch.div(positions[group_index], local_w, rounding_mode="floor")
                )
                local_positions.append(selected)
                local_tokens.append(projected[group_index])
                local_importance.append(result.importance.float())
                local_sources.append(
                    self._embed_source(result.source, native_columns, native_count)
                )
            all_positions = torch.cat(local_positions)
            order = all_positions.argsort()
            all_positions = all_positions[order]
            all_tokens = torch.cat(local_tokens)[order]
            all_importance = torch.cat(local_importance)[order]
            all_sources = torch.cat(local_sources)[order.detach().cpu()]
            cursor = 0
            for row in range(local_h):
                row_count = int(
                    (torch.div(all_positions, local_w, rounding_mode="floor") == row).sum()
                )
                if row_count:
                    output_parts.append(all_tokens[cursor : cursor + row_count])
                    importance_parts.append(all_importance[cursor : cursor + row_count])
                    source_parts.append(all_sources[cursor : cursor + row_count])
                    cursor += row_count
                output_parts.append(self._inner.image_newline[None])
                importance_parts.append(torch.full((1,), float("inf"), device=all_tokens.device))
                newline = torch.zeros(1, native_count, dtype=torch.float32)
                newline[0, patches + (row + 1) * local_w + row] = 1.0
                source_parts.append(newline)
            if cursor != int(all_positions.numel()):
                raise RuntimeError(
                    f"VFlowOpt placed {cursor} of {all_positions.numel()} local rows."
                )
        else:
            output_parts.append(self._inner.image_newline[None])
            importance_parts.append(torch.full((1,), float("inf"), device=projected[0].device))
            newline = torch.zeros(1, native_count, dtype=torch.float32)
            newline[0, patches] = 1.0
            source_parts.append(newline)

        output = torch.cat(output_parts)
        packed_importance = torch.cat(importance_parts).to(output.device)
        source = torch.cat(source_parts)
        if int(output.shape[0]) != target:
            raise RuntimeError(
                f"VFlowOpt Stage A produced {output.shape[0]} rows, expected {target}."
            )
        if tuple(source.shape) != (target, native_count):
            raise RuntimeError(
                f"VFlowOpt Stage-A source has shape {tuple(source.shape)}, "
                f"expected {(target, native_count)}."
            )
        return output, packed_importance, source, newline_count

    def _make_layer_hook(self, layer_index: int) -> Any:
        def hook(_module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return None
            stage = self._boundary_stage.get(layer_index)
            prefill = hidden.shape[1] > 1
            if stage is not None:
                if prefill:
                    if self._visual_start is None:
                        return None
                    hidden = self._prune(hidden, kwargs, stage)
                elif stage in self._stage_states:
                    advance_decode_step(
                        self._stage_states[stage],
                        hidden,
                        label="VFlowOpt/LLaVA-NeXT",
                    )
            state = self._active_state(layer_index)
            if state is None:
                return None
            apply_prune_state(kwargs, state, prefill=prefill)
            kwargs.pop("hidden_states", None)
            return (hidden,), kwargs

        return hook

    def _active_state(self, layer_index: int) -> dict[str, Any] | None:
        boundaries = self._stages.prune_layers
        for stage in range(len(boundaries) - 1, -1, -1):
            if boundaries[stage] <= layer_index and stage in self._stage_states:
                return self._stage_states[stage]
        return None

    def _base_state(self, stage: int) -> dict[str, Any] | None:
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_states:
                return self._stage_states[earlier]
        return None

    def _prune(self, hidden: torch.Tensor, kwargs: dict[str, Any], stage: int) -> torch.Tensor:
        target = min(self._stage_counts[stage], self._visual_count)
        if target >= self._visual_count:
            return hidden
        importance = self._alive_importance
        if importance is None or importance.numel() != self._visual_count:
            raise RuntimeError("VFlowOpt/LLaVA-NeXT has no matching ViT importance map.")
        base = self._base_state(stage)
        batch, seq_len, hidden_dim = hidden.shape
        start = int(self._visual_start)
        columns = start + torch.arange(self._visual_count, device=hidden.device)
        if batch != 1 or int(columns.max()) >= seq_len:
            raise RuntimeError(
                f"VFlowOpt/LLaVA-NeXT visual span does not fit {tuple(hidden.shape)}."
            )
        local = llm_stage_keep(importance, target)
        kept_columns = columns[local]
        keep_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden.device)
        keep_mask[:, columns] = False
        keep_mask[:, kept_columns] = True
        keep_index = keep_columns(keep_mask, label="VFlowOpt/LLaVA-NeXT")
        new_len = int(keep_index.shape[1])
        new_hidden = hidden.gather(
            1,
            keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim),
        )
        base_ids = base["position_ids"] if base is not None else kwargs.get("position_ids")
        position_ids = gather_position_ids(base_ids, keep_index, seq_len)
        base_mask = base["attention_mask"] if base is not None else kwargs.get("attention_mask")
        attention_mask, valid = pruned_attention_mask(
            base_mask,
            keep_index,
            seq_len,
            label="VFlowOpt/LLaVA-NeXT",
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
        self._alive_importance = importance[local]
        if self._track_source and self._composed_source is not None:
            self._composed_source = self._composed_source[local.detach().cpu()]
            self._owner._source_history.append(self._composed_source)
        if self._verbose and not self._logged:
            self._logged = True
            print(
                f"[VFlowOpt/LLaVA-NeXT] visual schedule "
                f"{(self._stage_counts[0], self._stage_counts[1])} "
                f"@layers{self._stages.prune_layers}",
                flush=True,
            )
        return new_hidden
