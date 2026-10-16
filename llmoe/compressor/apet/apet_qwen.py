"""Qwen2.5-VL integration for ApET.

ApET owns reconstruction-error ranking and token merging. The shared Qwen
adapter owns prompt capture, span validation, M-RoPE-aware prompt cuts, decode
re-addressing, and hook lifecycle.
"""

from collections.abc import Sequence
from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    advance_decode_step,
    apply_prune_state,
    gather_mrope_position_ids,
    pruned_attention_mask,
)
from llmoe.compressor.apet.apet import (
    SelectionMethod,
    _qwen_prune_block,
)


class QwenApETAdapter(QwenPruningAdapter):
    """Run ApET's ratio-based Stage A and decoder boundaries on Qwen."""

    method = "ApET"

    def __init__(
        self,
        *,
        retain_ratio: float,
        layer_list: Sequence[int],
        image_token_ratio_list: Sequence[float],
        basis_tokens: int,
        selection_method: SelectionMethod,
        ridge: float,
        layer_ridge: float,
        track_source: bool,
    ) -> None:
        super().__init__(track_source=track_source)
        self._retain_ratio = float(retain_ratio)
        self._layer_list = tuple(int(layer) for layer in layer_list)
        self._image_token_ratio_list = tuple(
            float(ratio) for ratio in image_token_ratio_list
        )
        self._basis_tokens = int(basis_tokens)
        self._selection_method: SelectionMethod = selection_method
        self._ridge = float(ridge)
        self._layer_ridge = float(layer_ridge)
        self._boundary_stage = {
            layer: stage
            for stage, layer in enumerate(self._layer_list)
        }

        self._image_start: int | None = None
        self._visual_count = 0
        self._position_ids: torch.Tensor | None = None
        self._stage_state: dict[int, dict[str, Any]] = {}
        self._composed_source: torch.Tensor | None = None

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        layers = getattr(self._language_model, "layers", None)
        if not layers:
            raise TypeError(
                "ApET's Qwen path needs "
                "`model.model.language_model.layers`."
            )
        if self._layer_list:
            if self._layer_list[-1] >= len(layers):
                raise ValueError(
                    f"layer_list boundary {self._layer_list[-1]} is too "
                    f"large for a {len(layers)}-layer decoder."
                )
            if getattr(self._language_model, "has_sliding_layers", False):
                raise NotImplementedError(
                    "ApET's layer boundaries require a full-attention Qwen "
                    "decoder; sliding-window layers cannot share the rebuilt "
                    "causal mask."
                )

        self._register_common_hooks()
        if self._layer_list:
            for layer_idx in range(self._layer_list[0], len(layers)):
                self._handles.append(
                    layers[layer_idx].register_forward_pre_hook(
                        self._make_layer_hook(layer_idx),
                        with_kwargs=True,
                    )
                )

        print(
            f"[ApET/Qwen] install: "
            f"retain_ratio={self._retain_ratio:g}, "
            f"boundaries={list(self._layer_list)} @ratios"
            f"{list(self._image_token_ratio_list)}, "
            f"basis_tokens={self._basis_tokens} "
            f"({self._selection_method}), "
            f"ridge={self._ridge:g}/{self._layer_ridge:g} over "
            f"{len(layers)} layers, image_token_id={self._image_token_id}",
            flush=True,
        )

    def _reset_forward_state(self) -> None:
        super()._reset_forward_state()
        self._image_start = None
        self._visual_count = 0
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
        self._require_kv_cache(kwargs)
        first, num_visual = span

        keep_count = max(
            1, int(self._retain_ratio * num_visual)
        )
        result: tuple[tuple, dict] | None = None
        if keep_count < num_visual:
            block = inputs_embeds[
                0, first:first + num_visual
            ].float()
            basis_count = max(
                1, min(self._basis_tokens, keep_count)
            )
            kept_block, kept_order, source = _qwen_prune_block(
                block,
                keep_count,
                basis_count,
                selection_method=self._selection_method,
                ridge=self._ridge,
                ascending_blend=True,
            )

            keep = torch.ones(
                inputs_embeds.shape[1],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
            keep[first:first + num_visual] = False
            keep[first + kept_order] = True
            modified = inputs_embeds.clone()
            modified[:, first + kept_order] = kept_block.to(
                modified.dtype
            )
            self._apply_keep(
                kwargs, inputs_embeds, modified, keep
            )

            if self._track_source:
                self._composed_source = source.detach().cpu().float()
                self._source_history = [
                    torch.eye(num_visual, dtype=torch.float32),
                    self._composed_source,
                ]
            result = args, kwargs

        self._image_start = first
        self._visual_count = (
            keep_count if result is not None else num_visual
        )
        if self._track_source and result is None:
            self._composed_source = torch.eye(
                num_visual, dtype=torch.float32
            )
            self._source_history = [self._composed_source]
        stored = kwargs.get("position_ids")
        if isinstance(stored, torch.Tensor):
            self._position_ids = stored
        if self._forward_count == 1:
            new_length = (
                int(kwargs["inputs_embeds"].shape[1])
                if result is not None
                else inputs_embeds.shape[1]
            )
            print(
                f"[ApET/Qwen] first prefill: seq "
                f"{inputs_embeds.shape[1]} -> {new_length} "
                f"(visual {num_visual} -> {self._visual_count}), "
                f"boundaries={list(self._layer_list)}",
                flush=True,
            )
        return result

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(
            _module: Any, args: tuple, kwargs: dict
        ) -> tuple[tuple, dict] | None:
            return self._layer_hook(args, kwargs, layer_idx)

        return hook

    def _layer_hook(
        self,
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
                if self._image_start is not None:
                    hidden = self._boundary_prune(
                        hidden, kwargs, stage
                    )
            elif stage in self._stage_state:
                advance_decode_step(
                    self._stage_state[stage], hidden, label="ApET/Qwen"
                )

        state = self._active_state(layer_idx)
        if state is None:
            return None
        apply_prune_state(kwargs, state, prefill=prefill)
        kwargs.pop("hidden_states", None)
        return (hidden,), kwargs

    def _active_state(
        self, layer_idx: int
    ) -> dict[str, Any] | None:
        for stage in range(len(self._layer_list) - 1, -1, -1):
            if (
                self._layer_list[stage] <= layer_idx
                and stage in self._stage_state
            ):
                return self._stage_state[stage]
        return None

    def _base_state(self, stage: int) -> dict[str, Any] | None:
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_state:
                return self._stage_state[earlier]
        return None

    def _boundary_prune(
        self,
        hidden: torch.Tensor,
        kwargs: dict,
        stage: int,
    ) -> torch.Tensor:
        base = self._base_state(stage)
        batch, seq_len, _ = hidden.shape
        current = self._visual_count
        target = max(
            1,
            int(self._image_token_ratio_list[stage] * current),
        )
        if current <= 0 or target >= current:
            return hidden

        start = self._image_start
        end = start + current
        if batch != 1 or end > seq_len:
            self._warn_once(
                "geometry_changed",
                f"the visual span [{start}, {end}) does not fit the "
                f"layer-{self._layer_list[stage]} input "
                f"{tuple(hidden.shape)}; the boundary is SKIPPED for "
                "this batch.",
            )
            return hidden

        block = hidden[0, start:end].float()
        basis_count = max(1, min(self._basis_tokens, target))
        kept_block, local, source = _qwen_prune_block(
            block,
            target,
            basis_count,
            selection_method=self._selection_method,
            ridge=self._layer_ridge,
            ascending_blend=False,
        )
        new_hidden = torch.cat(
            [
                hidden[:, :start],
                kept_block.to(hidden.dtype).unsqueeze(0),
                hidden[:, end:],
            ],
            dim=1,
        )
        keep_index = torch.cat(
            [
                torch.arange(start, device=hidden.device),
                start + local,
                torch.arange(end, seq_len, device=hidden.device),
            ]
        ).unsqueeze(0)
        new_len = int(keep_index.shape[1])

        base_ids = (
            base["mrope_position_ids"]
            if base is not None
            else self._position_ids
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
            base_mask, keep_index, seq_len, label="ApET/Qwen"
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
        if self._forward_count == 1:
            print(
                f"[ApET/Qwen] boundary {stage} "
                f"@layer{self._layer_list[stage]}: seq {seq_len} -> "
                f"{new_len} (visual {current} -> {target})",
                flush=True,
            )
        self._visual_count = target
        if self._track_source and self._composed_source is not None:
            self._composed_source = (
                source.detach().cpu().float() @ self._composed_source
            )
            self._source_history.append(self._composed_source)
        return new_hidden
