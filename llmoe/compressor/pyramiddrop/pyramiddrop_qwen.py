"""M-RoPE-aware Qwen2.5-VL integration for PyramidDrop."""

from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    advance_decode_step,
    apply_prune_state,
    decoder_attention_rows,
    gather_mrope_position_ids,
    keep_columns,
    pruned_attention_mask,
    ratio_keep_count,
)
from llmoe.compressor.pyramiddrop.pyramiddrop import qwen_pyramid_schedule
from llmoe.utils.qwen import APPLY_MROPE, resolve_mrope_section


class QwenPyramidDropAdapter(QwenPruningAdapter):
    """Apply PyramidDrop boundaries to Qwen's concatenated prompt."""

    method = "PyramidDrop"

    def __init__(self, *, retain_ratio: float, track_source: bool) -> None:
        super().__init__(track_source=track_source)
        self._retain_ratio = float(retain_ratio)
        self._layer_list: tuple[int, ...] = ()
        self._stage_ratios: tuple[float, ...] = ()
        self._boundary_stage: dict[int, int] = {}
        self._image_start: torch.Tensor | None = None
        self._query_offset: torch.Tensor | None = None
        self._visual_count = 0
        self._stage_counts: tuple[int, ...] = ()
        self._position_ids: torch.Tensor | None = None
        self._stage_state: dict[int, dict[str, Any]] = {}
        self._composed_source: torch.Tensor | None = None

    @property
    def layer_list(self) -> tuple[int, ...]:
        return self._layer_list

    @property
    def stage_ratios(self) -> tuple[float, ...]:
        return self._stage_ratios

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        layers = getattr(self._language_model, "layers", None)
        if not layers:
            raise TypeError(
                "PyramidDrop's Qwen path needs "
                "`model.model.language_model.layers`."
            )
        if APPLY_MROPE is None and self._retain_ratio < 1.0:
            raise RuntimeError(
                "Could not import `apply_multimodal_rotary_pos_emb` from "
                "transformers; PyramidDrop cannot reproduce its boundary rank."
            )
        if getattr(self._language_model, "has_sliding_layers", False):
            raise NotImplementedError(
                "PyramidDrop's physical boundaries require a full-attention Qwen "
                "decoder; sliding-window layers cannot share the rebuilt mask."
            )

        self._layer_list, self._stage_ratios = qwen_pyramid_schedule(
            self._retain_ratio,
            len(layers),
        )
        self._boundary_stage = {
            layer: stage for stage, layer in enumerate(self._layer_list)
        }
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
            f"[PyramidDrop/Qwen] install: retain_ratio={self._retain_ratio:g} "
            f"(layer-averaged), boundaries={list(self._layer_list)}, "
            f"cumulative_stage_ratios={[round(r, 6) for r in self._stage_ratios]} "
            f"over {len(layers)} layers, image_token_id={self._image_token_id}",
            flush=True,
        )

    def _reset_forward_state(self) -> None:
        super()._reset_forward_state()
        self._image_start = None
        self._query_offset = None
        self._visual_count = 0
        self._stage_counts = ()
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
        first, num_visual = span
        if not self._layer_list:
            if self._track_source:
                identity = torch.eye(num_visual, dtype=torch.float32)
                self._source_history = [identity, identity.clone()]
            return None

        self._require_kv_cache(kwargs)
        attention_mask = kwargs.get("attention_mask")
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.shape[-1] == inputs_embeds.shape[1]
        ):
            valid = attention_mask[0].bool()
            positions = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )
            last_valid = int(positions.masked_fill(~valid, -1).max())
        else:
            last_valid = inputs_embeds.shape[1] - 1
        if last_valid < first + num_visual:
            self._warn_once(
                "no_rater",
                "no prompt token follows the visual span; pruning is SKIPPED.",
            )
            return None

        self._image_start = torch.tensor(
            [first], dtype=torch.long, device=inputs_embeds.device
        )
        self._query_offset = torch.tensor(
            [last_valid - (first + num_visual)],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        self._visual_count = num_visual
        self._stage_counts = tuple(
            ratio_keep_count(
                num_visual,
                ratio,
                label="PyramidDrop/Qwen stage",
            )
            for ratio in self._stage_ratios
        )
        self._position_ids = self._resolve_position_ids(kwargs, inputs_embeds)
        if self._track_source:
            identity = torch.eye(num_visual, dtype=torch.float32)
            self._composed_source = identity
            self._source_history = [identity]
        if self._forward_count == 1:
            print(
                f"[PyramidDrop/Qwen] first prefill: seq_len={inputs_embeds.shape[1]}, "
                f"visual={num_visual} at [{first}, {first + num_visual}), "
                f"stage_counts={list(self._stage_counts)}",
                flush=True,
            )
        return None

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(
            module: Any, args: tuple, kwargs: dict
        ) -> tuple[tuple, dict] | None:
            return self._layer_hook(module, args, kwargs, layer_idx)

        return hook

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
                hidden = self._prune(module, hidden, kwargs, stage)
            elif stage in self._stage_state:
                advance_decode_step(
                    self._stage_state[stage],
                    hidden,
                    label="PyramidDrop/Qwen",
                )

        state = self._active_state(layer_idx)
        if state is None:
            return None
        apply_prune_state(kwargs, state, prefill=prefill)
        kwargs.pop("hidden_states", None)
        return (hidden,), kwargs

    def _active_state(self, layer_idx: int) -> dict[str, Any] | None:
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

    def _query_columns(self, device: torch.device) -> torch.Tensor:
        start = self._image_start.to(device)
        return self._query_offset.to(device) + start + self._visual_count

    def _rank_scores(
        self,
        layer: Any,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: Any,
    ) -> torch.Tensor:
        section = resolve_mrope_section(layer.self_attn)
        attention = decoder_attention_rows(
            layer,
            hidden,
            position_embeddings,
            attention_mask,
            self._query_columns(hidden.device).unsqueeze(1),
            apply_rope=lambda query, key, cos, sin: APPLY_MROPE(
                query, key, cos, sin, section
            ),
            label="PyramidDrop/Qwen",
        )
        start = int(self._image_start[0])
        return attention[:, 0, start:start + self._visual_count]

    def _prune(
        self,
        layer: Any,
        hidden: torch.Tensor,
        kwargs: dict,
        stage: int,
    ) -> torch.Tensor:
        replay = self._stage_state.get(stage)
        if replay is not None and "keep_index" in replay:
            index = replay["keep_index"]
            return hidden.gather(
                1,
                index.unsqueeze(-1).expand(
                    *index.shape, hidden.shape[-1]
                ),
            )

        target = min(self._stage_counts[stage], self._visual_count)
        if target >= self._visual_count:
            return hidden
        base = self._base_state(stage)
        batch, seq_len, hidden_dim = hidden.shape
        start = self._image_start.to(hidden.device)
        columns = start.unsqueeze(1) + torch.arange(
            self._visual_count, device=hidden.device
        )
        if batch != 1 or int(columns.max()) >= seq_len:
            raise RuntimeError(
                f"PyramidDrop/Qwen visual span does not fit decoder input "
                f"{tuple(hidden.shape)}."
            )

        base_ids = (
            base["mrope_position_ids"] if base is not None else self._position_ids
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
        scores = self._rank_scores(
            layer, hidden, base_embeddings, base_mask
        )
        local = scores.topk(target, dim=1).indices.sort(dim=1).values
        kept_columns = columns.gather(1, local)
        keep_mask = torch.ones(
            batch, seq_len, dtype=torch.bool, device=hidden.device
        )
        keep_mask.scatter_(1, columns, False)
        keep_mask.scatter_(1, kept_columns, True)
        keep_index = keep_columns(
            keep_mask, label="PyramidDrop/Qwen"
        )
        new_len = int(keep_index.shape[1])
        new_hidden = hidden.gather(
            1,
            keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim),
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
            label="PyramidDrop/Qwen",
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
            "keep_index": keep_index,
        }
        self._visual_count = target
        if self._track_source and self._composed_source is not None:
            self._composed_source = self._composed_source[
                local[0].detach().cpu()
            ]
            self._source_history.append(self._composed_source)
        if self._forward_count == 1:
            print(
                f"[PyramidDrop/Qwen] stage {stage} "
                f"@layer{self._layer_list[stage]}: visual -> {target}, "
                f"sequence {seq_len} -> {new_len}",
                flush=True,
            )
        return new_hidden
