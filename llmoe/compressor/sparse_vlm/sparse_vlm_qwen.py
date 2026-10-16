"""Qwen2.5-VL integration for SparseVLM's shared sparsification policy."""

from collections.abc import Sequence
from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    advance_decode_step,
    apply_prune_state,
    gather_mrope_position_ids,
)
from llmoe.compressor.sparse_vlm.sparse_vlm import (
    SparseVLMPolicy,
    append_source_history,
    causal_mask,
    rater_attention,
    select_raters,
)
from llmoe.utils.qwen import APPLY_MROPE, resolve_mrope_section


class QwenSparseVLMAdapter(QwenPruningAdapter):
    """Run SparseVLM's progressive decoder stages under Qwen's M-RoPE."""

    method = "SparseVLM"

    def __init__(self, *, schedule_ratios: Sequence[float],
                 pruning_layers: Sequence[int],
                 policy: SparseVLMPolicy) -> None:
        super().__init__(track_source=policy.track_source)
        self._schedule_ratios = tuple(float(ratio) for ratio in schedule_ratios)
        self._pruning_layers = tuple(int(layer) for layer in pruning_layers)
        self._policy = policy
        self._mrope_section: list[int] | None = None
        self._trigger_layers = {
            layer + 1: stage
            for stage, layer in enumerate(self._pruning_layers)
        }

        self._image_start = 0
        self._original_count = 0
        self._visual_count = 0
        self._rater_offsets: torch.Tensor | None = None
        self._position_ids: torch.Tensor | None = None
        self._scores: dict[int, torch.Tensor] = {}
        self._stage_state: dict[int, dict[str, Any]] = {}
        self._composed_source: torch.Tensor | None = None

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        if APPLY_MROPE is None:
            raise RuntimeError(
                "SparseVLM cannot reproduce Qwen rater attention because "
                "transformers exposes no multimodal rotary helper."
            )
        layers = self._language_model.layers
        last_pruning_layer = self._pruning_layers[-1]
        if last_pruning_layer + 1 >= len(layers):
            raise ValueError(
                f"pruning layer {last_pruning_layer} leaves no layer to "
                f"prune into in a {len(layers)}-layer decoder"
            )
        if getattr(self._language_model, "has_sliding_layers", False):
            raise NotImplementedError(
                "SparseVLM requires a full-attention Qwen configuration"
            )
        self._mrope_section = resolve_mrope_section(self._language_model)

        self._register_common_hooks()
        for layer in self._pruning_layers:
            self._handles.append(
                layers[layer].self_attn.register_forward_pre_hook(
                    self._make_score_hook(layer), with_kwargs=True
                )
            )
        for layer_idx in range(self._pruning_layers[0] + 1, len(layers)):
            self._handles.append(
                layers[layer_idx].register_forward_pre_hook(
                    self._make_layer_hook(layer_idx), with_kwargs=True
                )
            )
        print(
            f"[SparseVLM/Qwen] install: "
            f"ratios={list(self._schedule_ratios)} "
            f"@layers{list(self._pruning_layers)} of {len(layers)}, "
            f"recycling={self._policy.recycling}, "
            f"image_token_id={self._image_token_id}",
            flush=True,
        )

    def _reset_forward_state(self) -> None:
        super()._reset_forward_state()
        self._image_start = 0
        self._original_count = 0
        self._visual_count = 0
        self._rater_offsets = None
        self._position_ids = None
        self._scores = {}
        self._stage_state = {}
        self._composed_source = None
        self._source_history = []

    def _prefill(self, args: tuple, kwargs: dict,
                 inputs_embeds: torch.Tensor) -> tuple[tuple, dict] | None:
        span = self._locate_image_span(inputs_embeds)
        if span is None:
            return None
        image_start, visual_count = span
        self._require_kv_cache(kwargs)
        self._policy.reseed()

        self._image_start = image_start
        self._original_count = visual_count
        self._visual_count = visual_count
        self._position_ids = self._resolve_position_ids(
            kwargs, inputs_embeds
        )
        self._rater_offsets = select_raters(
            inputs_embeds, image_start, visual_count
        )
        if self._forward_count == 1:
            print(
                f"[SparseVLM/Qwen] first prefill: "
                f"seq_len={inputs_embeds.shape[1]}, "
                f"image_tokens={visual_count} at "
                f"[{image_start}, {image_start + visual_count}), "
                f"raters={self._rater_offsets.numel()}",
                flush=True,
            )
        return None

    def _make_score_hook(self, layer: int) -> Any:
        def hook(module: Any, args: tuple, kwargs: dict) -> None:
            return self._score_hook(module, args, kwargs, layer)

        return hook

    def _score_hook(self, module: Any, args: tuple,
                    kwargs: dict, layer: int) -> None:
        if self._rater_offsets is None:
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
        start, count = self._image_start, self._visual_count
        raters = (
            self._rater_offsets.to(hidden_states.device) + start + count
        )
        if int(raters.max()) >= hidden_states.shape[1]:
            raise RuntimeError(
                f"SparseVLM rater index {int(raters.max())} exceeds "
                f"layer-{layer} length {hidden_states.shape[1]}"
            )
        attention = rater_attention(
            module,
            hidden_states,
            kwargs.get("position_embeddings"),
            raters,
            apply_rope=APPLY_MROPE,
            rope_args=(self._mrope_section,),
        )
        self._scores[layer] = attention[
            :, start:start + count
        ].mean(dim=0)
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
        stage = self._trigger_layers.get(layer_idx)
        if stage is not None:
            if prefill:
                if self._rater_offsets is not None:
                    hidden_states = self._prune_stage(
                        hidden_states, kwargs, stage
                    )
            elif stage in self._stage_state:
                advance_decode_step(
                    self._stage_state[stage],
                    hidden_states,
                    label="SparseVLM/Qwen",
                )

        state = self._active_state(layer_idx)
        if state is None:
            return None
        apply_prune_state(kwargs, state, prefill=prefill)
        kwargs.pop("hidden_states", None)
        return (hidden_states,), kwargs

    def _active_state(self, layer_idx: int) -> dict[str, Any] | None:
        for stage in range(len(self._pruning_layers) - 1, -1, -1):
            if (
                self._pruning_layers[stage] < layer_idx
                and stage in self._stage_state
            ):
                return self._stage_state[stage]
        return None

    def _base_state(self, stage: int) -> dict[str, Any] | None:
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_state:
                return self._stage_state[earlier]
        return None

    def _prune_stage(self, hidden_states: torch.Tensor,
                     kwargs: dict, stage: int) -> torch.Tensor:
        layer = self._pruning_layers[stage]
        scores = self._scores.pop(layer, None)
        if scores is None:
            raise RuntimeError(
                f"SparseVLM captured no attention at pruning layer {layer}"
            )
        batch, seq_len, _ = hidden_states.shape
        start, count = self._image_start, self._visual_count
        end = start + count
        if count <= 1 or end > seq_len or scores.shape[0] != count:
            self._warn_once(
                "geometry_changed",
                f"visual span [{start}, {end}) does not fit layer-{layer} "
                f"input {tuple(hidden_states.shape)}; stage {stage} is skipped.",
            )
            return hidden_states

        keep_k = min(
            max(
                1,
                int(
                    self._schedule_ratios[stage]
                    * self._original_count
                ),
            ),
            count - 1,
        )
        visual, donor, source = self._policy.sparsify(
            hidden_states[:, start:end, :], scores, keep_k
        )
        new_hidden = torch.cat(
            [
                hidden_states[:, :start, :],
                visual,
                hidden_states[:, end:, :],
            ],
            dim=1,
        )

        device = hidden_states.device
        donor_index = torch.cat(
            [
                torch.arange(start, device=device),
                start + donor.to(device),
                torch.arange(end, seq_len, device=device),
            ]
        ).unsqueeze(0)
        new_len = int(donor_index.shape[1])
        base = self._base_state(stage)
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
        mrope_ids, position_ids = gather_mrope_position_ids(
            base_ids, donor_index, seq_len, new_len
        )
        rope_ids = mrope_ids[-3:] if mrope_ids.ndim == 3 else mrope_ids
        self._stage_state[stage] = {
            "batch": batch,
            "mrope_position_ids": mrope_ids,
            "position_ids": position_ids,
            "position_embeddings": self._language_model.rotary_emb(
                new_hidden, rope_ids
            ),
            "attention_mask": causal_mask(base_mask, new_len, device),
            "valid": torch.ones(
                batch, new_len, dtype=torch.bool, device=device
            ),
            "cache_position": torch.arange(new_len, device=device),
            "kv_len": new_len,
        }
        self._visual_count = int(visual.shape[1])
        if self._forward_count == 1:
            print(
                f"[SparseVLM/Qwen] stage {stage} @layer{layer}: "
                f"keep_k={keep_k}, visual {count} -> {self._visual_count}, "
                f"seq {seq_len} -> {new_len}",
                flush=True,
            )
        if self._track_source:
            self._composed_source = append_source_history(
                self._source_history,
                self._composed_source,
                source,
            )
        return new_hidden
