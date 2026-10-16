"""Canonical crop-local Token Merging for native Transformers LLaVA-NeXT.

The released ToMe algorithm merges progressively between attention and the MLP.
Native NeXT's stock spatial-unpad packer cannot accept shortened square crop
grids. This adapter captures the checkpoint-selected CLIP layer's head-averaged
keys, applies ToMe's balanced bipartite matching and size-weighted averaging
independently to the base crop and each valid AnyRes tile region, then projects
and repacks the reduced rows.

Merged spatial centroids restore deterministic raster positions. Learned image
newline rows are reconstructed unchanged and never enter matching. The defining
ToMe matching rule is preserved; its placement is a NeXT architecture adapter.
"""

from bisect import bisect_left
from typing import Any, NamedTuple

import torch

from llmoe.compressor.llava_next import (
    LlavaNextFeatureTransformAdapter,
    balanced_group_budgets,
    crop_counts,
    patch_grid,
    split_crop_tensor,
    stitch_and_unpad_local_crops,
)
from llmoe.compressor.tome.tome import _bipartite_soft_matching


class _ReducedGroup(NamedTuple):
    tokens: torch.Tensor
    positions: torch.Tensor
    source: torch.Tensor | None


class LlavaNextToMeAdapter:
    """Apply one retain-ratio-only ToMe policy to NeXT crop regions."""

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
        self._retain_ratio = float(retain_ratio)
        self._track_source = bool(track_source)
        self._feature_layer = -2
        self._rounds = 0
        self._keys: torch.Tensor | None = None
        self._key_hook: Any | None = None
        self._source_history: list[torch.Tensor] = []
        self._logged = False
        self._lifecycle = LlavaNextFeatureTransformAdapter(
            label="ToMe",
            retain_ratio=self._retain_ratio,
            transform_outputs=self._transform_outputs,
        )

    def install(self) -> None:
        config = self._vlm.model.config
        strategy = getattr(config, "vision_feature_select_strategy", "default")
        if strategy != "default":
            raise ValueError(
                "ToMe/LLaVA-NeXT supports the checkpoint's canonical "
                "vision_feature_select_strategy='default' only."
            )
        selected = getattr(config, "vision_feature_layer", None)
        if selected is None:
            selected = getattr(config, "vision_feature_select_layer", -2)
        if not isinstance(selected, int):
            raise TypeError("ToMe/LLaVA-NeXT supports one checkpoint-selected CLIP layer.")
        self._feature_layer = int(selected)

        encoder = self._owner._locate_clip_encoder(self._vlm)
        if encoder is None:
            raise RuntimeError("ToMe/LLaVA-NeXT could not locate the CLIP vision encoder.")
        self._rounds = self._owner._resolve_merge_layer_count(self._vlm, len(encoder.layers))
        metric_layer = encoder.layers[self._rounds - 1]
        try:
            self._key_hook = metric_layer.self_attn.register_forward_pre_hook(
                self._capture_keys,
                with_kwargs=True,
            )
            self._lifecycle.install(self._vlm)
        except Exception:
            self.uninstall()
            raise

        print(
            f"[ToMe/LLaVA-NeXT] install: retain_ratio={self._retain_ratio:g}, "
            f"metric=K@CLIP[{self._feature_layer}], rounds={self._rounds}, "
            "crop-local matching, protected newlines",
            flush=True,
        )

    def uninstall(self) -> None:
        self._lifecycle.uninstall()
        if self._key_hook is not None:
            self._key_hook.remove()
        self._key_hook = None
        self._keys = None

    def get_source_history(self) -> list[torch.Tensor]:
        return list(self._source_history)

    def _record_source_history(self, history: list[torch.Tensor]) -> None:
        self._source_history = history
        self._owner._source_history = list(history)

    def _capture_keys(
        self,
        attention: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            self._keys = None
            return
        keys = attention.k_proj(hidden_states)
        keys = keys.view(
            *hidden_states.shape[:-1],
            -1,
            int(attention.head_dim),
        )
        self._keys = keys.mean(dim=-2).detach()

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
        keys = self._keys
        self._keys = None
        self._record_source_history([])

        if self._retain_ratio == 1.0:
            if self._track_source and native_outputs:
                count = int(native_outputs[0].shape[0])
                identity = torch.eye(count, dtype=torch.float32)
                self._record_source_history([identity, identity.clone()])
            return image_outputs

        hidden_states = getattr(image_outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                "ToMe/LLaVA-NeXT needs the selected CLIP hidden state before "
                "the multimodal projector."
            )
        feature_layer = self._argument(args, kwargs, "vision_feature_layer", 2)
        if feature_layer is None:
            feature_layer = self._feature_layer
        if not isinstance(feature_layer, int) or int(feature_layer) != self._feature_layer:
            raise ValueError(
                "ToMe/LLaVA-NeXT requires the checkpoint-selected single "
                f"vision_feature_layer={self._feature_layer}."
            )
        strategy = self._argument(args, kwargs, "vision_feature_select_strategy", 3)
        if strategy is None:
            strategy = getattr(
                self._vlm.model.config,
                "vision_feature_select_strategy",
                "default",
            )
        if strategy != "default":
            raise ValueError("ToMe/LLaVA-NeXT requires vision_feature_select_strategy='default'.")

        features = hidden_states[self._feature_layer][:, 1:]
        if not isinstance(keys, torch.Tensor) or keys.ndim != 3:
            raise RuntimeError("ToMe/LLaVA-NeXT captured no keys from the selected CLIP layer.")
        metric = keys[:, 1:]
        if tuple(features.shape[:2]) != tuple(metric.shape[:2]):
            raise RuntimeError(
                "ToMe/LLaVA-NeXT selected features and key metric have "
                f"incompatible shapes {tuple(features.shape)} and "
                f"{tuple(metric.shape)}."
            )

        image_sizes = self._argument(args, kwargs, "image_sizes", 1)
        if not isinstance(image_sizes, torch.Tensor):
            raise TypeError("ToMe/LLaVA-NeXT needs processor image_sizes.")
        counts = crop_counts(self._inner, image_sizes)
        feature_groups = split_crop_tensor(features, counts, label="ToMe features")
        metric_groups = split_crop_tensor(metric, counts, label="ToMe keys")
        if (
            len(feature_groups) != len(metric_groups)
            or len(feature_groups) != len(targets)
            or len(feature_groups) != len(native_outputs)
        ):
            raise RuntimeError(
                "ToMe/LLaVA-NeXT received inconsistent crop, target, and native image group counts."
            )

        outputs: list[torch.Tensor] = []
        groups = zip(
            feature_groups,
            metric_groups,
            targets,
            native_outputs,
            strict=True,
        )
        for image_index, (crop_features, crop_metric, target, native) in enumerate(groups):
            native_count = int(native.shape[0])
            if int(target) == native_count:
                outputs.append(native)
                if self._track_source and image_index == 0:
                    identity = torch.eye(native_count, dtype=torch.float32)
                    self._record_source_history([identity, identity.clone()])
                continue
            packed, source = self._compress_image(
                crop_features,
                crop_metric,
                image_sizes[image_index],
                int(target),
                native_count,
            )
            outputs.append(packed)
            if self._track_source and image_index == 0 and source is not None:
                self._record_source_history(
                    [
                        torch.eye(native_count, dtype=torch.float32),
                        source,
                    ]
                )
            if not self._logged:
                self._logged = True
                print(
                    f"[ToMe/LLaVA-NeXT] first image: {native_count} -> "
                    f"{target} packed rows across "
                    f"{crop_features.shape[0]} crops",
                    flush=True,
                )

        image_outputs.pooler_output = outputs
        return image_outputs

    @staticmethod
    def _representative_positions(
        centroids: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        """Map ordered centroids to distinct nearby valid raster positions."""

        available = available.sort().values
        values = available.detach().cpu().long().tolist()
        centers = centroids.detach().cpu().float().tolist()
        count = len(centers)
        choices: list[int] = []
        previous = -1
        for index, center in enumerate(centers):
            lower = previous + 1
            upper = len(values) - (count - index)
            insertion = bisect_left(values, center, lo=lower, hi=upper + 1)
            candidates = {
                max(lower, min(insertion, upper)),
                max(lower, min(insertion - 1, upper)),
            }
            selected = min(
                candidates,
                key=lambda candidate: (
                    abs(float(values[candidate]) - float(center)),
                    candidate,
                ),
            )
            choices.append(selected)
            previous = selected
        choice_tensor = torch.tensor(
            choices,
            dtype=torch.long,
            device=available.device,
        )
        return available.index_select(0, choice_tensor)

    def _reduce_group(
        self,
        features: torch.Tensor,
        metric: torch.Tensor,
        keep: int,
        positions: torch.Tensor,
    ) -> _ReducedGroup:
        count = int(features.shape[0])
        if (
            features.ndim != 2
            or metric.ndim != 2
            or int(metric.shape[0]) != count
            or positions.ndim != 1
            or int(positions.numel()) != count
        ):
            raise ValueError("ToMe/LLaVA-NeXT received an inconsistent crop group.")
        if not 1 <= int(keep) <= count:
            raise ValueError(f"ToMe/LLaVA-NeXT cannot retain {keep} of {count} crop rows.")
        if int(keep) == count:
            source = torch.eye(count, dtype=torch.float32) if self._track_source else None
            return _ReducedGroup(features, positions, source)

        feature_width = int(features.shape[-1])
        metric_width = int(metric.shape[-1])
        coordinate = positions.to(
            device=features.device,
            dtype=torch.float32,
        ).view(1, count, 1)
        combined = torch.cat(
            [
                features.unsqueeze(0).float(),
                metric.unsqueeze(0).float(),
                coordinate,
            ],
            dim=-1,
        )
        metric_state = combined[:, :, feature_width : feature_width + metric_width]
        size = torch.ones(
            1,
            count,
            1,
            device=features.device,
            dtype=torch.float32,
        )
        source = (
            torch.eye(
                count,
                device=features.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            if self._track_source
            else None
        )
        schedule = self._owner._resolve_schedule(
            count,
            count - int(keep),
            self._rounds,
            protect_cls=False,
        )
        for reduction in schedule:
            if reduction < 1:
                continue
            combined, size, source = _bipartite_soft_matching(
                combined,
                metric_state,
                reduction,
                size,
                source,
                protect_cls=False,
            )
            metric_state = combined[:, :, feature_width : feature_width + metric_width]

        centroids = combined[0, :, -1]
        order = centroids.argsort()
        combined = combined[:, order]
        if source is not None:
            source = source[:, order]
        restored = self._representative_positions(
            centroids.index_select(0, order),
            positions,
        )
        tokens = combined[0, :, :feature_width].to(features.dtype)
        final_source = source[0].detach().cpu().float() if source is not None else None
        return _ReducedGroup(tokens, restored, final_source)

    @staticmethod
    def _embed_source(
        source: torch.Tensor,
        native_columns: torch.Tensor,
        native_count: int,
    ) -> torch.Tensor:
        embedded = torch.zeros(
            source.shape[0],
            native_count,
            dtype=torch.float32,
        )
        columns = native_columns.detach().cpu().long()
        embedded[:, columns] = source.detach().cpu().float()
        return embedded

    def _compress_image(
        self,
        features: torch.Tensor,
        metric: torch.Tensor,
        image_size: torch.Tensor,
        target: int,
        native_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        crop_h, crop_w = patch_grid(self._inner)
        patches = crop_h * crop_w
        if (
            features.ndim != 3
            or metric.ndim != 3
            or int(features.shape[1]) != patches
            or tuple(metric.shape[:2]) != tuple(features.shape[:2])
        ):
            raise ValueError(
                f"ToMe/LLaVA-NeXT expected [crops, {patches}, hidden] features and keys."
            )

        crop_count = int(features.shape[0])
        blocks = [features[0]]
        metrics = [metric[0]]
        positions = [torch.arange(patches, device=features.device)]
        capacities = [patches]
        local_h = local_w = 0

        if crop_count > 1:
            local_features, (local_h, local_w) = stitch_and_unpad_local_crops(
                self._inner, features[1:], image_size
            )
            local_metric, metric_shape = stitch_and_unpad_local_crops(
                self._inner, metric[1:], image_size
            )
            if metric_shape != (local_h, local_w):
                raise RuntimeError(
                    "ToMe/LLaVA-NeXT reconstructed different feature and key mosaics."
                )
            raw_ids = torch.arange(
                crop_count - 1,
                device=features.device,
                dtype=torch.long,
            )[:, None].expand(crop_count - 1, patches)
            stitched_ids, _ = stitch_and_unpad_local_crops(self._inner, raw_ids, image_size)
            local_crop_ids = stitched_ids[0, :, 0].long()
            mosaic_positions = torch.arange(
                local_h * local_w,
                device=features.device,
            )
            for crop_index in range(crop_count - 1):
                crop_positions = mosaic_positions[local_crop_ids == crop_index]
                if not int(crop_positions.numel()):
                    continue
                blocks.append(local_features[0, crop_positions])
                metrics.append(local_metric[0, crop_positions])
                positions.append(crop_positions)
                capacities.append(int(crop_positions.numel()))
            newline_count = local_h
            expected_native = patches + local_h * local_w + local_h
        else:
            newline_count = 1
            expected_native = patches + 1

        if expected_native != native_count:
            raise RuntimeError(
                f"ToMe/LLaVA-NeXT reconstructed {expected_native} native "
                f"rows, but Transformers packed {native_count}."
            )
        patch_budget = target - newline_count
        if patch_budget < len(blocks):
            raise ValueError(
                f"ToMe/LLaVA-NeXT protects {newline_count} image-newline rows "
                f"and needs one survivor in each of {len(blocks)} crop "
                f"regions, but retain_ratio budgets only {target} rows."
            )
        budgets = balanced_group_budgets(
            patch_budget,
            capacities,
            minimum=1,
            label="ToMe/LLaVA-NeXT crop-local matching",
        )
        reduced = [
            self._reduce_group(block, keys, budget, raster_positions)
            for block, keys, budget, raster_positions in zip(
                blocks,
                metrics,
                budgets,
                positions,
                strict=True,
            )
        ]
        projected = [self._inner.multi_modal_projector(group.tokens) for group in reduced]

        output_parts = [projected[0]]
        source_parts: list[torch.Tensor] = []
        if self._track_source:
            base_source = reduced[0].source
            assert base_source is not None
            source_parts.append(
                self._embed_source(
                    base_source,
                    torch.arange(patches),
                    native_count,
                )
            )

        if crop_count > 1:
            all_positions = torch.cat([group.positions for group in reduced[1:]])
            order = all_positions.argsort()
            all_positions = all_positions[order]
            all_tokens = torch.cat(projected[1:])[order]
            all_sources: torch.Tensor | None = None
            if self._track_source:
                local_sources: list[torch.Tensor] = []
                for group_index, group in enumerate(reduced[1:], start=1):
                    assert group.source is not None
                    native_columns = (
                        patches + positions[group_index] + positions[group_index] // local_w
                    )
                    local_sources.append(
                        self._embed_source(
                            group.source,
                            native_columns,
                            native_count,
                        )
                    )
                all_sources = torch.cat(local_sources)[order.detach().cpu()]

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
                    newline = torch.zeros(
                        1,
                        native_count,
                        dtype=torch.float32,
                    )
                    newline_position = patches + (row + 1) * local_w + row
                    newline[0, newline_position] = 1.0
                    source_parts.append(newline)
            if cursor != int(all_positions.numel()):
                raise RuntimeError(
                    f"ToMe/LLaVA-NeXT placed {cursor} of {all_positions.numel()} local rows."
                )
        else:
            output_parts.append(self._inner.image_newline[None])
            if self._track_source:
                newline = torch.zeros(
                    1,
                    native_count,
                    dtype=torch.float32,
                )
                newline[0, patches] = 1.0
                source_parts.append(newline)

        output = torch.cat(output_parts, dim=0)
        source = torch.cat(source_parts, dim=0) if self._track_source else None
        if int(output.shape[0]) != target:
            raise RuntimeError(
                f"ToMe/LLaVA-NeXT produced {output.shape[0]} rows, expected {target}."
            )
        if source is not None and tuple(source.shape) != (
            target,
            native_count,
        ):
            raise RuntimeError(
                f"ToMe/LLaVA-NeXT source has shape {tuple(source.shape)}, "
                f"expected {(target, native_count)}."
            )
        return output, source
