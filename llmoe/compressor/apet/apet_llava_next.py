"""Native Transformers LLaVA-NeXT adapter for the released ApET path.

The official LLaVA-1.6 release runs approximation-error pruning independently
on every CLIP crop before the multimodal projector, packs the surviving local
patches with ``spatial_unpad`` and protected image-newline rows, then repeats
the same FPS/reconstruction/merge policy after decoder layer 16.  Its launch
scripts fix FPS, ten basis tokens, ridge ``1e-5``, and the 192 -> 64 (3:1)
stage shape.  This adapter fixes those method choices and resolves only their
scale from a per-image, decoder-layer-average ``retain_ratio``.
"""

from typing import Any

import torch

from llmoe.compressor._pruning import (
    advance_decode_step,
    apply_prune_state,
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
    two_stage_layer_average_schedule,
)

NEXT_COMPRESS_LAYER = 16
NEXT_STAGE2_RATIO = 1.0 / 3.0
NEXT_BASIS_TOKENS = 10


def layer_average_schedule(
    native_tokens: int,
    retain_ratio: float,
    decoder_layers: int,
) -> tuple[int, int]:
    """Resolve ApET's canonical 3:1 two-stage schedule for one native span."""

    return two_stage_layer_average_schedule(
        native_tokens,
        retain_ratio=retain_ratio,
        stage2_layer=NEXT_COMPRESS_LAYER,
        decoder_layers=decoder_layers,
        stage2_ratio=NEXT_STAGE2_RATIO,
        label="ApET/LLaVA-NeXT",
    )


class LlavaNextApETAdapter:
    """Crop-aware ApET Stage A plus its released layer-16 decoder cut."""

    def __init__(self, owner: Any, vlm: Any, *, retain_ratio: float, track_source: bool) -> None:
        self._owner = owner
        self._vlm = vlm
        self._inner = vlm.model.model
        self._language_model = self._inner.language_model
        self._retain_ratio = float(retain_ratio)
        self._track_source = bool(track_source)
        self._depth = len(self._language_model.layers)
        self._image_token_id: int | None = None
        self._schedules: dict[int, tuple[int, int]] = {}
        self._vision_positions: torch.Tensor | None = None
        self._stage2_keep = 0
        self._stage1_source: torch.Tensor | None = None
        self._prune_state: dict[str, Any] = {}
        self._hook_handles: list[Any] = []
        self._lifecycle = LlavaNextFeatureTransformAdapter(
            label="ApET",
            retain_ratio=self._retain_ratio,
            transform_outputs=self._transform_outputs,
            target_for_length=self._stage1_target,
            on_prepared=self._on_prepared,
        )

    def _stage1_target(self, native_tokens: int) -> int:
        schedule = layer_average_schedule(native_tokens, self._retain_ratio, self._depth)
        self._schedules[int(native_tokens)] = schedule
        return schedule[0]

    def install(self) -> None:
        if not NEXT_COMPRESS_LAYER < self._depth:
            raise ValueError(
                f"ApET's canonical layer-16 cut needs at least 17 decoder layers; "
                f"this LLaVA-NeXT model has {self._depth}."
            )
        strategy = getattr(self._vlm.model.config, "vision_feature_select_strategy", "default")
        if strategy != "default":
            raise ValueError(
                "ApET/LLaVA-NeXT supports the checkpoint's canonical "
                "vision_feature_select_strategy='default' only."
            )
        self._image_token_id = image_token_id(self._vlm, label="ApET/LLaVA-NeXT")
        try:
            self._lifecycle.install(self._vlm)
            for layer_index in range(NEXT_COMPRESS_LAYER, self._depth):
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
        self._schedules = {}
        self._vision_positions = None
        self._stage2_keep = 0
        self._stage1_source = None
        self._prune_state = {}

    def _on_prepared(
        self,
        inputs: Any,
        native_lengths: tuple[int, ...],
        stage1_lengths: tuple[int, ...],
    ) -> None:
        if len(native_lengths) != 1 or len(stage1_lengths) != 1:
            raise NotImplementedError(
                "ApET/LLaVA-NeXT follows the released batch_size=1, one-image "
                "path; evaluate this baseline with batch_size=1."
            )
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("ApET/LLaVA-NeXT needs input_ids to locate the image span")
        positions = decoder_visual_positions(
            input_ids,
            token_id=int(self._image_token_id),
            label="ApET",
        )
        self._prune_state = {}
        self._vision_positions = None
        self._stage1_source = None
        if self._track_source:
            self._owner._source_history = []
        if positions is None:
            return

        native = native_lengths[0]
        stage1, stage2 = self._schedules.get(native) or layer_average_schedule(
            native, self._retain_ratio, self._depth
        )
        if stage1 != stage1_lengths[0] or int(positions.numel()) != stage1:
            raise RuntimeError(
                "ApET's scheduled Stage-A count does not match the trimmed image span"
            )
        if stage1 != native and stage2 <= NEXT_BASIS_TOKENS:
            raise ValueError(
                f"ApET/LLaVA-NeXT's layer-16 stage would retain {stage2} tokens, "
                f"but its fixed {NEXT_BASIS_TOKENS}-basis merge needs at least "
                f"{NEXT_BASIS_TOKENS + 1}. Use a larger retain_ratio."
            )
        self._vision_positions = positions.unsqueeze(0)

        self._stage2_keep = stage2

    @staticmethod
    def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, position: int) -> Any:
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
        if self._retain_ratio == 1.0:
            if self._track_source and native_outputs:
                count = int(native_outputs[0].shape[0])
                identity = torch.eye(count, dtype=torch.float32)
                self._stage1_source = identity
                self._owner._source_history = [identity, identity.clone()]
            return image_outputs

        hidden_states = getattr(image_outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                "ApET/LLaVA-NeXT needs the selected CLIP hidden state for its "
                "released pre-projector Stage A."
            )
        feature_layer = self._argument(args, kwargs, "vision_feature_layer", 2)
        if feature_layer is None:
            feature_layer = getattr(self._inner.config, "vision_feature_layer", -2)
        if isinstance(feature_layer, int):
            features = hidden_states[feature_layer]
        else:
            features = torch.cat([hidden_states[int(index)] for index in feature_layer], dim=-1)
        features = features[:, 1:]

        image_sizes = self._argument(args, kwargs, "image_sizes", 1)
        if not isinstance(image_sizes, torch.Tensor):
            raise TypeError("ApET/LLaVA-NeXT needs image_sizes from the processor")
        counts = crop_counts(self._inner, image_sizes)
        groups = split_crop_tensor(features, counts, label="ApET features")
        if len(groups) != len(targets) or len(groups) != len(native_outputs):
            raise RuntimeError(
                f"ApET received {len(groups)} crop groups, {len(targets)} budgets, "
                f"and {len(native_outputs)} native image sequences."
            )

        outputs: list[torch.Tensor] = []
        for image_index, (crop_features, target, native) in enumerate(
            zip(groups, targets, native_outputs, strict=True)
        ):
            native_count = int(native.shape[0])
            if target == native_count:
                outputs.append(native)
                if self._track_source and image_index == 0:
                    identity = torch.eye(native_count, dtype=torch.float32)
                    self._stage1_source = identity
                    self._owner._source_history = [identity, identity.clone()]
                continue
            packed, source = self._compress_image(
                crop_features,
                image_sizes[image_index],
                int(target),
                native_count,
            )
            outputs.append(packed)
            if self._track_source and image_index == 0 and source is not None:
                self._stage1_source = source
                self._owner._source_history = [torch.eye(native_count, dtype=torch.float32), source]
        image_outputs.pooler_output = outputs
        return image_outputs

    def _reduce_group(
        self, features: torch.Tensor, keep: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        count = int(features.shape[0])
        if keep == count:
            indices = torch.arange(count, device=features.device)
            source = torch.eye(count, dtype=torch.float32) if self._track_source else None
            return features, indices, source
        if keep <= NEXT_BASIS_TOKENS:
            raise ValueError(
                f"ApET's fixed {NEXT_BASIS_TOKENS}-basis merge cannot reduce "
                f"a {count}-patch crop to {keep}; use a larger retain_ratio."
            )
        tokens, indices, source = self._owner._prune_block(
            features.float(), keep, NEXT_BASIS_TOKENS
        )
        return (
            tokens.to(features.dtype),
            indices,
            source.detach().cpu().float() if self._track_source else None,
        )

    @staticmethod
    def _embed_source(
        source: torch.Tensor,
        native_columns: torch.Tensor,
        native_count: int,
    ) -> torch.Tensor:
        embedded = torch.zeros(source.shape[0], native_count, dtype=torch.float32)
        embedded[:, native_columns.detach().cpu().long()] = source
        return embedded

    def _compress_image(
        self,
        features: torch.Tensor,
        image_size: torch.Tensor,
        target: int,
        native_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        crop_h, crop_w = patch_grid(self._inner)
        patches = crop_h * crop_w
        if features.ndim != 3 or int(features.shape[1]) != patches:
            raise ValueError(
                f"ApET expected [crops, {patches}, hidden] CLIP features, got "
                f"{tuple(features.shape)}."
            )
        crop_count = int(features.shape[0])
        local_features: torch.Tensor | None = None
        local_crop_ids: torch.Tensor | None = None
        local_h = local_w = 0
        if crop_count > 1:
            local_features, (local_h, local_w) = stitch_and_unpad_local_crops(
                self._inner, features[1:], image_size
            )
            raw_ids = torch.arange(crop_count - 1, device=features.device, dtype=torch.long)[
                :, None
            ].expand(crop_count - 1, patches)
            stitched_ids, _ = stitch_and_unpad_local_crops(self._inner, raw_ids, image_size)
            local_crop_ids = stitched_ids[0, :, 0].long()
            newline_count = local_h
            expected_native = patches + local_h * local_w + local_h
        else:
            newline_count = 1
            expected_native = patches + 1
        if expected_native != native_count:
            raise RuntimeError(
                f"ApET reconstructed {expected_native} native AnyRes rows, but "
                f"Transformers packed {native_count}."
            )

        patch_budget = target - newline_count
        if patch_budget < 1:
            raise ValueError(
                f"ApET's {newline_count} protected image-newline rows exhaust "
                f"the Stage-A budget {target}; use a larger retain_ratio."
            )
        blocks = [features[0]]
        positions = [torch.arange(patches, device=features.device)]
        capacities = [patches]
        if crop_count > 1:
            assert local_features is not None and local_crop_ids is not None
            mosaic_positions = torch.arange(local_h * local_w, device=features.device)
            for crop_index in range(crop_count - 1):
                crop_positions = mosaic_positions[local_crop_ids == crop_index]
                blocks.append(local_features[0, crop_positions])
                positions.append(crop_positions)
                capacities.append(int(crop_positions.numel()))

        budgets = balanced_group_budgets(
            patch_budget,
            capacities,
            minimum=NEXT_BASIS_TOKENS + 1,
            label=(f"ApET/LLaVA-NeXT's fixed {NEXT_BASIS_TOKENS}-basis crop merge"),
        )
        reduced = [
            self._reduce_group(block, budget) for block, budget in zip(blocks, budgets, strict=True)
        ]
        projected = [self._inner.multi_modal_projector(tokens) for tokens, _, _ in reduced]

        output_parts = [projected[0]]
        source_parts: list[torch.Tensor] = []
        if self._track_source:
            base_source = reduced[0][2]
            assert base_source is not None
            source_parts.append(
                self._embed_source(
                    base_source,
                    torch.arange(patches),
                    native_count,
                )
            )

        if crop_count > 1:
            local_positions: list[torch.Tensor] = []
            local_tokens: list[torch.Tensor] = []
            local_sources: list[torch.Tensor] = []
            for group_index in range(1, len(blocks)):
                _, selected, source = reduced[group_index]
                selected_positions = positions[group_index].index_select(
                    0, selected.to(positions[group_index].device)
                )
                local_positions.append(selected_positions)
                local_tokens.append(projected[group_index])
                if self._track_source:
                    assert source is not None
                    native_columns = (
                        patches + positions[group_index] + positions[group_index] // local_w
                    )
                    local_sources.append(self._embed_source(source, native_columns, native_count))
            all_positions = torch.cat(local_positions)
            order = all_positions.argsort()
            all_positions = all_positions[order]
            all_tokens = torch.cat(local_tokens)[order]
            all_sources = (
                torch.cat(local_sources)[order.detach().cpu()] if self._track_source else None
            )
            cursor = 0
            for row in range(local_h):
                row_count = int(((all_positions // local_w) == row).sum())
                if row_count:
                    output_parts.append(all_tokens[cursor : cursor + row_count])
                    if all_sources is not None:
                        source_parts.append(all_sources[cursor : cursor + row_count])
                    cursor += row_count
                output_parts.append(self._inner.image_newline[None])
                if self._track_source:
                    newline = torch.zeros(1, native_count, dtype=torch.float32)
                    newline_position = patches + (row + 1) * local_w + row
                    newline[0, newline_position] = 1.0
                    source_parts.append(newline)
            if cursor != int(all_positions.numel()):
                raise RuntimeError(f"ApET placed {cursor} of {all_positions.numel()} local rows")
        else:
            output_parts.append(self._inner.image_newline[None])
            if self._track_source:
                newline = torch.zeros(1, native_count, dtype=torch.float32)
                newline[0, patches] = 1.0
                source_parts.append(newline)

        output = torch.cat(output_parts, dim=0)
        source = torch.cat(source_parts, dim=0) if self._track_source else None
        if int(output.shape[0]) != target:
            raise RuntimeError(f"ApET Stage A produced {output.shape[0]} rows, expected {target}.")
        if source is not None and tuple(source.shape) != (target, native_count):
            raise RuntimeError(
                f"ApET Stage-A source has shape {tuple(source.shape)}, expected "
                f"{(target, native_count)}."
            )
        return output, source

    def _make_layer_hook(self, layer_index: int) -> Any:
        def hook(_module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                return None
            prefill = hidden_states.shape[1] > 1
            if layer_index == NEXT_COMPRESS_LAYER:
                if prefill:
                    hidden_states = self._decoder_prune(hidden_states, kwargs)
                elif self._prune_state:
                    advance_decode_step(self._prune_state, hidden_states, label="ApET")
                    position_ids = self._prune_state["cache_position"].view(1, 1)
                    self._prune_state["position_ids"] = position_ids
                    self._prune_state["position_embeddings"] = self._language_model.rotary_emb(
                        hidden_states, position_ids=position_ids
                    )
            if not self._prune_state:
                return None
            apply_prune_state(kwargs, self._prune_state, prefill=prefill)
            if not prefill:
                kwargs["position_ids"] = self._prune_state["position_ids"]
                kwargs["position_embeddings"] = self._prune_state["position_embeddings"]
            kwargs.pop("hidden_states", None)
            return (hidden_states,), kwargs

        return hook

    def _decoder_prune(self, hidden_states: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
        self._prune_state = {}
        positions = self._vision_positions
        if positions is None:
            return hidden_states
        positions = positions.to(hidden_states.device)
        batch, seq_len, hidden_dim = hidden_states.shape
        if batch != 1 or positions.shape[0] != 1:
            raise RuntimeError("ApET/LLaVA-NeXT decoder pruning requires batch_size=1")
        if int(positions.max()) >= seq_len:
            raise RuntimeError(
                f"ApET image positions {tuple(positions.shape)} do not fit "
                f"decoder input {tuple(hidden_states.shape)}."
            )
        visual_count = int(positions.shape[1])
        keep = min(int(self._stage2_keep), visual_count)
        if keep >= visual_count:
            return hidden_states
        if keep <= NEXT_BASIS_TOKENS:
            raise ValueError(
                f"ApET's decoder stage needs >{NEXT_BASIS_TOKENS} survivors, got {keep}."
            )

        block = hidden_states[0].index_select(0, positions[0]).float()
        kept, local, source = self._owner._prune_block(block, keep, NEXT_BASIS_TOKENS)
        kept = kept.to(hidden_states.dtype)
        kept_columns = positions[0].index_select(0, local).unsqueeze(0)
        keep_mask = torch.ones(1, seq_len, dtype=torch.bool, device=hidden_states.device)
        keep_mask.scatter_(1, positions, False)
        keep_mask.scatter_(1, kept_columns, True)
        keep_index = keep_columns(keep_mask, label="ApET")
        new_len = int(keep_index.shape[1])
        new_hidden = hidden_states.gather(
            1, keep_index.unsqueeze(-1).expand(1, new_len, hidden_dim)
        )
        visual_start = int(positions[0, 0])
        new_hidden[:, visual_start : visual_start + keep] = kept.unsqueeze(0)

        position_ids = torch.arange(new_len, device=hidden_states.device).unsqueeze(0)
        attention_mask, valid = pruned_attention_mask(
            kwargs.get("attention_mask"),
            keep_index,
            seq_len,
            label="ApET",
        )
        self._prune_state = {
            "batch": 1,
            "position_ids": position_ids,
            "position_embeddings": self._language_model.rotary_emb(
                new_hidden, position_ids=position_ids
            ),
            "attention_mask": attention_mask,
            "valid": valid,
            "cache_position": torch.arange(new_len, device=hidden_states.device),
            "kv_len": new_len,
        }
        if self._track_source and self._stage1_source is not None:
            composed = source.detach().cpu().float() @ self._stage1_source
            self._owner._source_history = [*self._owner._source_history[:2], composed]
        return new_hidden
