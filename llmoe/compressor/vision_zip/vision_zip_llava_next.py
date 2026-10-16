"""Native LLaVA-NeXT adapter for training-free VisionZip.

The released NeXT path reduces every CLIP crop before the multimodal projector,
then restores selected local patches in AnyRes mosaic order and appends each
crop's CLS/contextual summaries.  This adapter keeps that placement and policy,
but resolves the crop budgets from one per-image ``retain_ratio`` so variable
native placeholder spans remain directly comparable.
"""

from typing import Any, NamedTuple

import torch

from llmoe.compressor.llava_next import (
    LlavaNextFeatureTransformAdapter,
    anyres_grid_shape,
    crop_counts,
    patch_grid,
    split_crop_tensor,
)
from llmoe.compressor.vision_zip.vision_zip import contextual_merge


_CONTEXTUAL_NUMERATOR = 5
_CONTEXTUAL_DENOMINATOR = 32


class _CropReduction(NamedTuple):
    tokens: torch.Tensor
    dominant_patch_indices: torch.Tensor
    contextual_count: int
    source: torch.Tensor | None


def _balanced_budgets(total: int, groups: int, maximum: int) -> tuple[int, ...]:
    """Split an image budget evenly over its base/local CLIP crops."""

    if groups < 1 or total < 1:
        raise ValueError(f"VisionZip needs positive total/groups, got {total}/{groups}")
    quotient, remainder = divmod(int(total), int(groups))
    budgets = tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(groups)
    )
    if budgets and max(budgets) > int(maximum):
        raise ValueError(
            f"VisionZip/LLaVA-NeXT cannot distribute {total} rows over {groups} "
            f"CLIP crops with at most {maximum} pre-projector rows each. Use a "
            "lower retain_ratio (retain_ratio=1 is an identity pass-through)."
        )
    return budgets


class LlavaNextVisionZipAdapter:
    """Run VisionZip crop-locally on native Transformers LLaVA-NeXT."""

    def __init__(self, owner: Any, vlm: Any, *, retain_ratio: float,
                 track_source: bool) -> None:
        self._owner = owner
        self._vlm = vlm
        self._inner = vlm.model.model
        self._retain_ratio = float(retain_ratio)
        self._track_source = bool(track_source)
        self._attention: torch.Tensor | None = None
        self._keys: torch.Tensor | None = None
        self._hook: Any | None = None
        self._feature_layer = int(owner._resolve_feature_layer(vlm))
        self._lifecycle = LlavaNextFeatureTransformAdapter(
            label="VisionZip",
            retain_ratio=self._retain_ratio,
            transform_outputs=self._transform_outputs,
        )

    def install(self) -> None:
        strategy = getattr(
            self._vlm.model.config, "vision_feature_select_strategy", "default"
        )
        if strategy != "default":
            raise ValueError(
                "VisionZip/LLaVA-NeXT supports the checkpoint's canonical "
                "vision_feature_select_strategy='default' only."
            )
        layers = self._owner._vision_encoder_layers(self._vlm)
        target = (
            layers[self._feature_layer]
            if self._feature_layer < 0
            else layers[self._feature_layer - 1]
        )
        self._hook = target.self_attn.register_forward_pre_hook(
            self._capture_attention, with_kwargs=True
        )
        try:
            self._lifecycle.install(self._vlm)
        except Exception:
            self._hook.remove()
            self._hook = None
            raise

    def uninstall(self) -> None:
        self._lifecycle.uninstall()
        if self._hook is not None:
            self._hook.remove()
        self._hook = None
        self._attention = None
        self._keys = None

    @torch.no_grad()
    def _capture_attention(self, module: Any, args: tuple,
                           kwargs: dict) -> None:
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            return
        batch, sequence, _ = hidden.shape
        # FA2 deliberately does not materialize its full [S, S] attention map.
        # VisionZip needs only the CLS query against every key, so recompute
        # that [1, S] auxiliary slice while the native CLIP forward stays FA2.
        queries = module.q_proj(hidden[:, :1])
        keys = module.k_proj(hidden)
        q_norm = getattr(module, "q_norm", None)
        k_norm = getattr(module, "k_norm", None)
        if callable(q_norm):
            queries = q_norm(queries)
        if callable(k_norm):
            keys = k_norm(keys)
        queries = queries.view(
            batch, 1, module.num_heads, module.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch, sequence, module.num_heads, module.head_dim
        ).transpose(1, 2)
        logits = torch.matmul(
            queries[:, :, :1].float(), keys.float().transpose(-1, -2)
        ) * module.scale
        self._attention = logits.softmax(dim=-1)[:, :, 0, 1:].sum(dim=1)
        self._keys = keys.mean(dim=1)[:, 1:].contiguous()

    @staticmethod
    def _argument(args: tuple[Any, ...], kwargs: dict[str, Any],
                  name: str, position: int) -> Any:
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
        try:
            return self._transform(image_outputs, targets, args, kwargs)
        finally:
            self._attention = None
            self._keys = None

    def _transform(
        self,
        image_outputs: Any,
        targets: tuple[int, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        native = list(image_outputs.pooler_output)
        if self._retain_ratio == 1.0:
            if self._track_source and native:
                count = int(native[0].shape[0])
                identity = torch.eye(count, dtype=torch.float32)
                self._owner._source_history = [identity, identity.clone()]
            return image_outputs

        hidden_states = getattr(image_outputs, "hidden_states", None)
        attention, keys = self._attention, self._keys
        if not hidden_states or attention is None or keys is None:
            raise RuntimeError(
                "VisionZip/LLaVA-NeXT did not capture the selected CLIP hidden "
                "state, CLS attention, and key metric."
            )
        raw = hidden_states[self._feature_layer]
        image_sizes = self._argument(args, kwargs, "image_sizes", 1)
        if not isinstance(image_sizes, torch.Tensor):
            raise ValueError("VisionZip/LLaVA-NeXT needs image_sizes from the processor")
        counts = crop_counts(self._inner, image_sizes)
        raw_groups = split_crop_tensor(raw, counts, label="VisionZip")
        attention_groups = split_crop_tensor(
            attention, counts, label="VisionZip attention"
        )
        key_groups = split_crop_tensor(keys, counts, label="VisionZip keys")
        if len(targets) != len(raw_groups):
            raise RuntimeError(
                f"VisionZip has {len(raw_groups)} crop groups but {len(targets)} budgets"
            )

        grid_h, grid_w = patch_grid(self._inner)
        patches = grid_h * grid_w
        outputs: list[torch.Tensor] = []
        for image_index, (features, scores, metric, target) in enumerate(
            zip(raw_groups, attention_groups, key_groups, targets, strict=True)
        ):
            if int(features.shape[1]) != patches + 1:
                raise ValueError(
                    f"VisionZip expected CLS + {patches} patches per crop, got "
                    f"{features.shape[1]} rows."
                )
            budgets = _balanced_budgets(
                target, int(features.shape[0]), patches + 1
            )
            reductions = [
                self._reduce_crop(features[index], scores[index], metric[index], budget)
                for index, budget in enumerate(budgets)
            ]
            projected = [
                self._inner.multi_modal_projector(reduction.tokens)
                for reduction in reductions
            ]
            tile_h, tile_w = anyres_grid_shape(
                self._inner, image_sizes[image_index]
            )
            packed, row_order = self._pack_official_order(
                projected, reductions, tile_h, tile_w, grid_h, grid_w
            )
            outputs.append(packed)
            if self._track_source and image_index == 0:
                self._record_source(reductions, row_order, patches + 1)

        image_outputs.pooler_output = outputs
        return image_outputs

    def _reduce_crop(
        self,
        features: torch.Tensor,
        attention: torch.Tensor,
        keys: torch.Tensor,
        budget: int,
    ) -> _CropReduction:
        patches = int(features.shape[0]) - 1
        device = features.device
        if budget == 0:
            return _CropReduction(
                features[:0], torch.empty(0, dtype=torch.long, device=device), 0,
                torch.zeros(0, patches + 1) if self._track_source else None,
            )
        if budget == patches + 1:
            source = torch.eye(patches + 1, dtype=torch.float32) if self._track_source else None
            return _CropReduction(
                features,
                torch.arange(patches, device=device),
                0,
                source,
            )

        contextual = 0
        if budget >= 2:
            contextual = max(
                1, budget * _CONTEXTUAL_NUMERATOR // _CONTEXTUAL_DENOMINATOR
            )
            contextual = min(contextual, budget - 1)
        dominant_total = budget - contextual
        dominant_patches = max(0, dominant_total - 1)
        if dominant_patches:
            indices = attention.topk(dominant_patches).indices.sort().values
        else:
            indices = torch.empty(0, dtype=torch.long, device=device)
        keep = torch.zeros(patches, dtype=torch.bool, device=device)
        keep[indices] = True
        selected = features[1:].index_select(0, indices)
        chunks = [features[:1], selected]

        source: torch.Tensor | None = None
        if self._track_source:
            source = torch.zeros(budget, patches + 1, dtype=torch.float32)
            source[0, 0] = 1.0
            if dominant_patches:
                source[
                    torch.arange(1, 1 + dominant_patches), indices.detach().cpu() + 1
                ] = 1.0

        if contextual:
            leftover = (~keep).nonzero(as_tuple=True)[0]
            normalized = keys[leftover].float()
            normalized = normalized / normalized.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            merged = contextual_merge(
                features[1:][leftover].unsqueeze(0),
                normalized.unsqueeze(0),
                contextual,
            )
            chunks.append(merged.tokens[0].to(features.dtype))
            if source is not None:
                target_patches = leftover[merged.target_indices].cpu() + 1
                start = 1 + dominant_patches
                rows = torch.arange(start, start + contextual)
                source[rows, target_patches] = 1.0
                merged_from = leftover[merged.nontarget_mask].cpu() + 1
                if merged_from.numel():
                    source[rows.unsqueeze(1), merged_from.unsqueeze(0)] += (
                        merged.assign_one_hot[0].transpose(0, 1).float()
                        / merged.counts[0].float()
                    )

        return _CropReduction(
            torch.cat(chunks, dim=0), indices, contextual, source
        )

    @staticmethod
    def _pack_official_order(
        projected: list[torch.Tensor],
        reductions: list[_CropReduction],
        tile_h: int,
        tile_w: int,
        patch_h: int,
        patch_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match the released NeXT order: base, mosaic patches, CLS, context."""

        device = projected[0].device
        concatenated_offsets: list[int] = []
        cursor = 0
        for tensor in projected:
            concatenated_offsets.append(cursor)
            cursor += int(tensor.shape[0])

        pieces = [projected[0]]
        row_order = [
            torch.arange(projected[0].shape[0], device=device, dtype=torch.long)
        ]
        local_count = len(projected) - 1
        if local_count:
            if local_count != tile_h * tile_w:
                raise ValueError(
                    f"VisionZip has {local_count} local crops for a "
                    f"{tile_h}x{tile_w} AnyRes grid."
                )
            spatial: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            cls_rows: list[torch.Tensor] = []
            cls_order: list[torch.Tensor] = []
            context_rows: list[torch.Tensor] = []
            context_order: list[torch.Tensor] = []
            for local_index, (tensor, reduction) in enumerate(
                zip(projected[1:], reductions[1:], strict=True)
            ):
                selected = reduction.dominant_patch_indices
                selected_count = int(selected.numel())
                if selected_count:
                    patch_y, patch_x = selected // patch_w, selected % patch_w
                    tile_y, tile_x = divmod(local_index, tile_w)
                    position = (
                        (tile_y * patch_h + patch_y) * (tile_w * patch_w)
                        + tile_x * patch_w + patch_x
                    )
                    spatial.append(
                        (
                            position,
                            tensor[1 : 1 + selected_count],
                            concatenated_offsets[local_index + 1]
                            + torch.arange(1, 1 + selected_count, device=device),
                        )
                    )
                if tensor.shape[0]:
                    cls_rows.append(tensor[:1])
                    cls_order.append(
                        torch.tensor(
                            [concatenated_offsets[local_index + 1]], device=device
                        )
                    )
                if reduction.contextual_count:
                    context_rows.append(tensor[-reduction.contextual_count :])
                    context_order.append(
                        concatenated_offsets[local_index + 1]
                        + torch.arange(
                            tensor.shape[0] - reduction.contextual_count,
                            tensor.shape[0],
                            device=device,
                        )
                    )
            if spatial:
                positions = torch.cat([item[0] for item in spatial])
                rows = torch.cat([item[1] for item in spatial])
                order = torch.cat([item[2] for item in spatial])
                sorting = positions.argsort()
                pieces.append(rows[sorting])
                row_order.append(order[sorting])
            if cls_rows:
                pieces.append(torch.cat(cls_rows))
                row_order.append(torch.cat(cls_order))
            if context_rows:
                pieces.append(torch.cat(context_rows))
                row_order.append(torch.cat(context_order))

        return torch.cat(pieces), torch.cat(row_order)

    def _record_source(
        self,
        reductions: list[_CropReduction],
        row_order: torch.Tensor,
        source_width: int,
    ) -> None:
        total_input = len(reductions) * source_width
        rows: list[torch.Tensor] = []
        for crop_index, reduction in enumerate(reductions):
            if reduction.source is None:
                return
            block = torch.zeros(
                reduction.source.shape[0], total_input, dtype=torch.float32
            )
            start = crop_index * source_width
            block[:, start : start + source_width] = reduction.source
            rows.append(block)
        source = torch.cat(rows)[row_order.detach().cpu()]
        self._owner._source_history = [
            torch.eye(total_input, dtype=torch.float32), source
        ]
