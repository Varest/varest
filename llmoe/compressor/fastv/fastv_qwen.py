"""Qwen2.5-VL integration for FastV's shared ranking policy.

This module contains only the Qwen-specific hook placement and M-RoPE geometry.
Budgeting, attention reconstruction, selection, keep-set construction, and
source tracking remain in ``fastv.py``.
"""

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
from llmoe.compressor.fastv.fastv import (
    last_query_attention,
    normalize_attention_scores,
    resolve_keep,
    select_visual_keep,
    source_history,
)
from llmoe.utils.qwen import APPLY_MROPE, resolve_mrope_section


class QwenFastVAdapter(QwenPruningAdapter):
    """Apply FastV's decoder cut to Qwen2.5-VL's M-RoPE sequence."""

    method = "FastV"

    def __init__(self, *, retain_ratio: float, k: int,
                 track_source: bool) -> None:
        super().__init__(track_source=track_source)
        self._retain_ratio = float(retain_ratio)
        self._k = int(k)
        self._mrope_section: list[int] | None = None

        self._image_positions: torch.Tensor | None = None
        self._position_ids: torch.Tensor | None = None
        self._scores: torch.Tensor | None = None
        self._prune_state: dict[str, Any] = {}

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        if APPLY_MROPE is None:
            raise RuntimeError(
                "FastV cannot reproduce Qwen's layer k-1 attention because "
                "transformers exposes no multimodal rotary helper."
            )
        layers = self._language_model.layers
        if not 1 <= self._k < len(layers):
            raise ValueError(
                f"k={self._k} does not fit a {len(layers)}-layer decoder"
            )
        self._mrope_section = resolve_mrope_section(self._language_model)

        self._register_common_hooks()
        self._handles.append(
            layers[self._k - 1].self_attn.register_forward_pre_hook(
                self._capture_scores, with_kwargs=True
            )
        )
        for layer_idx in range(self._k, len(layers)):
            self._handles.append(
                layers[layer_idx].register_forward_pre_hook(
                    self._make_layer_hook(layer_idx), with_kwargs=True
                )
            )

        print(
            f"[FastV/Qwen] install: retain_ratio={self._retain_ratio:g}, "
            f"k={self._k} (score @layer{self._k - 1}, "
            f"prune @layer{self._k} of {len(layers)}), "
            f"image_token_id={self._image_token_id}",
            flush=True,
        )

    def _reset_forward_state(self) -> None:
        super()._reset_forward_state()
        self._image_positions = None
        self._position_ids = None
        self._scores = None
        self._prune_state = {}

    def _prefill(self, args: tuple, kwargs: dict,
                 inputs_embeds: torch.Tensor) -> tuple[tuple, dict] | None:
        span = self._locate_image_span(inputs_embeds)
        if span is None:
            return None
        first, n_visual = span
        self._require_kv_cache(kwargs)
        self._image_positions = (
            first + torch.arange(n_visual, device=inputs_embeds.device)
        ).unsqueeze(0)
        self._position_ids = self._resolve_position_ids(kwargs, inputs_embeds)
        if self._forward_count == 1:
            keep = resolve_keep(
                n_visual,
                max_tokens=n_visual,
                retain_ratio=self._retain_ratio,
            )
            print(
                f"[FastV/Qwen] first prefill: seq_len={inputs_embeds.shape[1]}, "
                f"image_tokens={n_visual}, keep={keep}",
                flush=True,
            )
        return None

    def _capture_scores(self, module: Any, args: tuple,
                        kwargs: dict) -> None:
        if self._image_positions is None:
            return None
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or hidden_states.shape[1] <= 1
        ):
            return None
        self._scores = last_query_attention(
            module,
            hidden_states,
            kwargs.get("position_embeddings"),
            kwargs.get("attention_mask"),
            apply_rope=APPLY_MROPE,
            rope_args=(self._mrope_section,),
        )
        return None

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(_module: Any, args: tuple,
                 kwargs: dict) -> tuple[tuple, dict] | None:
            return self._layer_hook(args, kwargs, layer_idx)

        return hook

    def _layer_hook(self, args: tuple, kwargs: dict,
                    layer_idx: int) -> tuple[tuple, dict] | None:
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return None

        prefill = hidden_states.shape[1] > 1
        if layer_idx == self._k:
            if prefill:
                hidden_states = self._prune(hidden_states, kwargs)
            elif self._prune_state:
                advance_decode_step(
                    self._prune_state,
                    hidden_states,
                    label="FastV/Qwen",
                )
        if not self._prune_state:
            return None
        apply_prune_state(kwargs, self._prune_state, prefill=prefill)
        kwargs.pop("hidden_states", None)
        return (hidden_states,), kwargs

    def _prune(self, hidden_states: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """Apply FastV's shared keep policy to Qwen's M-RoPE geometry."""
        self._prune_state = {}
        scores, self._scores = self._scores, None
        positions = self._image_positions
        if positions is None:
            return hidden_states
        batch, seq_len, hidden_dim = hidden_states.shape
        if scores is None:
            raise RuntimeError(
                f"FastV captured no attention at layer {self._k - 1}; the "
                "decoder must expose q_proj and k_proj modules."
            )

        positions = positions.to(hidden_states.device)
        if positions.shape[0] != batch or int(positions.max()) >= seq_len:
            self._warn_once(
                "geometry_changed",
                f"the image span {tuple(positions.shape)} does not fit decoder "
                f"input {tuple(hidden_states.shape)}; pruning is skipped.",
            )
            return hidden_states

        n_visual = int(positions.shape[1])
        keep = resolve_keep(
            n_visual,
            max_tokens=n_visual,
            retain_ratio=self._retain_ratio,
        )
        if keep >= n_visual:
            self._record_source(torch.arange(n_visual), n_visual)
            return hidden_states

        scores = normalize_attention_scores(
            scores.to(hidden_states.device), batch, seq_len
        )
        local, keep_index = select_visual_keep(
            scores,
            positions,
            keep,
            seq_len,
            label="FastV/Qwen",
        )
        new_len = int(keep_index.shape[1])
        new_hidden = hidden_states.gather(
            1, keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim)
        )

        position_ids, text_ids = gather_mrope_position_ids(
            self._position_ids, keep_index, seq_len, new_len
        )
        rope_ids = position_ids[-3:] if position_ids.ndim == 3 else position_ids
        position_embeddings = self._language_model.rotary_emb(
            new_hidden, rope_ids
        )
        attention_mask, valid = pruned_attention_mask(
            kwargs.get("attention_mask"),
            keep_index,
            seq_len,
            label="FastV/Qwen",
        )
        self._prune_state = {
            "batch": batch,
            "position_ids": text_ids,
            "position_embeddings": position_embeddings,
            "attention_mask": attention_mask,
            "valid": valid,
            "cache_position": torch.arange(
                new_len, device=hidden_states.device
            ),
            "kv_len": new_len,
        }
        if self._forward_count == 1:
            print(
                f"[FastV/Qwen] prune @layer{self._k}: "
                f"seq {seq_len} -> {new_len} "
                f"(image {n_visual} -> {keep})",
                flush=True,
            )
        self._record_source(local[0], n_visual)
        return new_hidden

    def _record_source(self, indices: torch.Tensor, n_visual: int) -> None:
        if self._track_source:
            self._source_history = source_history(indices, n_visual)
