"""Qwen2.5-VL integration for DUET.

DUET owns its two-stage selection policy. The shared Qwen adapter owns prompt
capture, vision-attention replay, span validation, the first M-RoPE-aware cut,
decode re-addressing, and hook lifecycle.
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
    keep_columns,
    pruned_attention_mask,
)
from llmoe.compressor.duet.duet import boundary_attention_scores
from llmoe.compressor.duet.stage1 import duet_stage1_plan
from llmoe.compressor.duet.stage2 import (
    reduce_rater_scores,
    select_text_raters,
    stage_keep_counts,
)
from llmoe.utils.qwen import APPLY_MROPE, resolve_mrope_section


class QwenDuetAdapter(QwenPruningAdapter):
    """Run DUET Stage 1 and its PyramidDrop boundaries on Qwen."""

    method = "DUET"

    def __init__(
        self,
        *,
        dominant_ratio: float,
        contextual_ratio: float,
        cluster_width: int,
        pdrop_layers: Sequence[int],
        pdrop_keep_ratios: Sequence[float],
        text_rater: str,
        score_reduction: str,
        use_stage1: bool,
        use_stage2: bool,
        track_source: bool,
        attention_chunk: int,
        retain_ratio: float | None = None,
    ) -> None:
        super().__init__(
            attention_chunk=attention_chunk,
            track_source=track_source,
        )
        self._dominant_ratio = float(dominant_ratio)
        self._contextual_ratio = float(contextual_ratio)
        # Reported budget (layer-averaged), kept for the install log only; the
        # Stage-1 ratios above are what actually drive selection.
        self._retain_ratio = None if retain_ratio is None else float(retain_ratio)
        self._cluster_width = int(cluster_width)
        self._pdrop_layers = tuple(
            int(layer) for layer in pdrop_layers
        )
        self._pdrop_keep_ratios = tuple(
            float(ratio) for ratio in pdrop_keep_ratios
        )
        self._text_rater = str(text_rater)
        self._score_reduction = str(score_reduction)
        self._use_stage1 = bool(use_stage1)
        self._use_stage2 = bool(use_stage2)
        self.capture_vision_attention = self._use_stage1
        self._boundary_stage = {
            layer: stage
            for stage, layer in enumerate(self._pdrop_layers)
        }

        self._tokenizer: Any | None = None
        self._image_start: torch.Tensor | None = None
        self._visual_count = 0
        self._keep_counts: tuple[int, ...] = ()
        self._rater_index: torch.Tensor | None = None
        self._rater_valid: torch.Tensor | None = None
        self._position_ids: torch.Tensor | None = None
        self._stage_state: dict[int, dict[str, Any]] = {}
        self._composed_source: torch.Tensor | None = None

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        layers = getattr(self._language_model, "layers", None)
        if not layers:
            raise TypeError(
                "DUET's Qwen path needs "
                "`model.model.language_model.layers`."
            )
        if self._use_stage2 and APPLY_MROPE is None:
            raise RuntimeError(
                "Could not import `apply_multimodal_rotary_pos_emb` from "
                "transformers; DUET cannot reproduce boundary attention."
            )
        if (
            self._use_stage2
            and self._pdrop_layers[-1] >= len(layers)
        ):
            raise ValueError(
                f"pdrop layer {self._pdrop_layers[-1]} is too large for a "
                f"{len(layers)}-layer decoder."
            )

        self._tokenizer = getattr(vlm.processor, "tokenizer", None)
        self._register_common_hooks()
        if self._use_stage2:
            for layer_idx in range(self._pdrop_layers[0], len(layers)):
                self._handles.append(
                    layers[layer_idx].register_forward_pre_hook(
                        self._make_layer_hook(layer_idx),
                        with_kwargs=True,
                    )
                )

        stage1_retain = self._dominant_ratio + self._contextual_ratio
        budget = (
            "" if self._retain_ratio is None
            else f" -> layer-average {self._retain_ratio:g}"
        )
        stage1 = (
            f"dominant={self._dominant_ratio:g} + "
            f"contextual={self._contextual_ratio:g} "
            f"(ratios, stage-1 retain={stage1_retain:g}{budget})"
        )
        print(
            f"[DUET/Qwen] install: stage1={stage1}, "
            f"cluster_width={self._cluster_width}, score @vision block "
            f"{len(self._visual.blocks) - 1}, "
            f"pdrop={list(self._pdrop_layers)} @ratios"
            f"{list(self._pdrop_keep_ratios)}, "
            f"text_rater={self._text_rater}, "
            f"stages={int(self._use_stage1)}{int(self._use_stage2)} "
            f"over {len(layers)} layers",
            flush=True,
        )

    def uninstall(self) -> None:
        super().uninstall()
        self._tokenizer = None

    def _reset_forward_state(self) -> None:
        super()._reset_forward_state()
        self._image_start = None
        self._visual_count = 0
        self._keep_counts = ()
        self._rater_index = None
        self._rater_valid = None
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
        if self._use_stage1 or self._use_stage2:
            self._require_kv_cache(kwargs)
        first, num_visual = span
        last = first + num_visual

        keep = self._stage1(
            kwargs, inputs_embeds, first, num_visual
        )
        visual_count = (
            num_visual
            if keep is None
            else int(keep[first:last].sum())
        )
        pruned_ids = (
            self._input_ids
            if keep is None
            else self._input_ids[:, keep]
        )
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        if isinstance(position_ids, torch.Tensor):
            self._position_ids = position_ids

        if not self._use_stage2:
            return (args, kwargs) if keep is not None else None

        starts = torch.tensor(
            [first], dtype=torch.long, device=inputs_embeds.device
        )
        index, valid = select_text_raters(
            pruned_ids,
            attention_mask
            if isinstance(attention_mask, torch.Tensor)
            else None,
            starts,
            visual_count,
            mode=self._text_rater,
            tokenizer=self._tokenizer,
        )
        if not bool(valid.any()):
            self._warn_once(
                "no_raters",
                "no prompt token is available to rate the visual span; "
                "Stage 2 is SKIPPED.",
            )
            return (args, kwargs) if keep is not None else None

        self._image_start = starts
        self._visual_count = visual_count
        self._keep_counts = stage_keep_counts(
            visual_count, self._pdrop_keep_ratios
        )
        self._rater_index = index - (
            starts.unsqueeze(1) + visual_count
        )
        self._rater_valid = valid
        if self._forward_count == 1:
            print(
                f"[DUET/Qwen] first prefill: "
                f"seq_len={pruned_ids.shape[1]}, visual={visual_count} "
                f"at [{first}, {first + visual_count}), "
                f"raters={int(valid[0].sum())}, "
                f"schedule={list(self._keep_counts)}",
                flush=True,
            )
        return (args, kwargs) if keep is not None else None

    def _stage1(
        self,
        kwargs: dict,
        inputs_embeds: torch.Tensor,
        first: int,
        num_visual: int,
    ) -> torch.Tensor | None:
        if (
            not self._use_stage1
            or self._received is None
            or self._keys is None
            or self._image_grid_thw is None
        ):
            return None

        scores, merged_keys = self._merged_token_signals()
        if scores.numel() != num_visual:
            raise RuntimeError(
                f"DUET scored {scores.numel()} visual tokens but the prompt "
                f"carries {num_visual} image placeholders."
            )

        dominant_num = max(0, int(self._dominant_ratio * num_visual))
        contextual_num = max(1, int(self._contextual_ratio * num_visual))
        if dominant_num + contextual_num >= num_visual:
            return None

        image_embeds = inputs_embeds[
            :, first:first + num_visual
        ]
        plan = duet_stage1_plan(
            scores.unsqueeze(0),
            merged_keys.unsqueeze(0),
            dominant=dominant_num,
            contextual=contextual_num,
            cluster_width=self._cluster_width,
            dtype=inputs_embeds.dtype,
        )
        anchors = plan.anchor_index
        channels = image_embeds.shape[-1]
        contextual_tokens = image_embeds.gather(
            1,
            anchors.unsqueeze(-1).expand(
                1, anchors.shape[1], channels
            ),
        )
        if plan.assign_one_hot is not None and plan.counts is not None:
            merge_tokens = image_embeds.gather(
                1,
                plan.merge_index.unsqueeze(-1).expand(
                    1, plan.merge_index.shape[1], channels
                ),
            )
            contextual_tokens = contextual_tokens + torch.bmm(
                plan.assign_one_hot.transpose(1, 2), merge_tokens
            ) / plan.counts

        keep_visual = torch.zeros(
            num_visual,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        keep_visual[plan.dominant_index[0]] = True
        keep_visual[anchors[0]] = True
        keep = torch.ones(
            inputs_embeds.shape[1],
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        keep[first:first + num_visual] = keep_visual

        modified = inputs_embeds.clone()
        modified[:, first + anchors[0]] = contextual_tokens.to(
            modified.dtype
        )
        state = self._apply_keep(
            kwargs, inputs_embeds, modified, keep
        )
        if self._track_source:
            self._record_stage1_source(
                num_visual, keep_visual, plan
            )
        if self._forward_count == 1:
            print(
                f"[DUET/Qwen] stage 1: visual {num_visual} -> "
                f"{int(keep_visual.sum())} ({dominant_num} dominant + "
                f"{contextual_num} contextual), sequence "
                f"{inputs_embeds.shape[1]} -> {state['prefill_len']}",
                flush=True,
            )
        return keep

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(
            module: Any, args: tuple, kwargs: dict
        ) -> tuple[tuple, dict] | None:
            return self._layer_hook(
                module, args, kwargs, layer_idx
            )

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
                hidden = self._prune(
                    module, hidden, kwargs, stage
                )
            elif stage in self._stage_state:
                advance_decode_step(
                    self._stage_state[stage], hidden, label="DUET/Qwen"
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
        for stage in range(len(self._pdrop_layers) - 1, -1, -1):
            if (
                self._pdrop_layers[stage] <= layer_idx
                and stage in self._stage_state
            ):
                return self._stage_state[stage]
        return None

    def _base_state(self, stage: int) -> dict[str, Any] | None:
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_state:
                return self._stage_state[earlier]
        return None

    def _prune(
        self,
        layer: Any,
        hidden: torch.Tensor,
        kwargs: dict,
        stage: int,
    ) -> torch.Tensor:
        base = self._base_state(stage)
        batch, seq_len, hidden_dim = hidden.shape
        target = max(
            0,
            min(int(self._keep_counts[stage]), self._visual_count),
        )
        if target == self._visual_count:
            return hidden

        start = self._image_start.to(hidden.device)
        columns = start.unsqueeze(1) + torch.arange(
            self._visual_count, device=hidden.device
        )
        if start.shape[0] != batch or int(columns.max()) >= seq_len:
            self._warn_once(
                "geometry_changed",
                f"the visual span {tuple(columns.shape)} does not fit "
                f"the layer-{self._pdrop_layers[stage]} input "
                f"{tuple(hidden.shape)}; Stage 2 is SKIPPED for this batch.",
            )
            return hidden

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

        if target:
            scores = self._boundary_scores(
                layer,
                hidden,
                base_embeddings,
                base_mask,
                start,
            )
            local = scores.topk(target, dim=1).indices.sort(
                dim=1
            ).values
            kept_columns = columns.gather(1, local)
        else:
            local = torch.zeros(
                batch, 0, dtype=torch.long, device=hidden.device
            )
            kept_columns = local

        keep_mask = torch.ones(
            batch, seq_len, dtype=torch.bool, device=hidden.device
        )
        keep_mask.scatter_(1, columns, False)
        if target:
            keep_mask.scatter_(1, kept_columns, True)
        keep_index = keep_columns(keep_mask, label="DUET/Qwen")
        new_len = keep_index.shape[1]
        new_hidden = hidden.gather(
            1,
            keep_index.unsqueeze(-1).expand(
                batch, new_len, hidden_dim
            ),
        )
        mrope_ids, text_ids = gather_mrope_position_ids(
            base_ids, keep_index, seq_len, new_len
        )
        position_embeddings = self._language_model.rotary_emb(
            new_hidden,
            mrope_ids[-3:] if mrope_ids.ndim == 3 else mrope_ids,
        )
        mask, valid = pruned_attention_mask(
            base_mask, keep_index, seq_len, label="DUET/Qwen"
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
                f"[DUET/Qwen] stage {stage} "
                f"@layer{self._pdrop_layers[stage]}: seq {seq_len} -> "
                f"{new_len} (visual {self._visual_count} -> {target})",
                flush=True,
            )
        self._visual_count = target
        self._record_cut(local)
        return new_hidden

    def _boundary_scores(
        self,
        layer: Any,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: Any,
        start: torch.Tensor,
    ) -> torch.Tensor:
        raters = self._rater_columns(hidden.device)
        mrope_section = resolve_mrope_section(layer.self_attn)
        attention = boundary_attention_scores(
            layer,
            hidden,
            position_embeddings,
            attention_mask,
            raters,
            apply_rope=lambda query, key, cos, sin: APPLY_MROPE(
                query, key, cos, sin, mrope_section
            ),
        )
        return reduce_rater_scores(
            attention,
            self._rater_valid,
            start,
            self._visual_count,
            reduction=self._score_reduction,
        )

    def _rater_columns(self, device: torch.device) -> torch.Tensor:
        start = self._image_start.to(device)
        return self._rater_index.to(device) + (
            start.unsqueeze(1) + self._visual_count
        )

    def _record_stage1_source(
        self,
        num_visual: int,
        keep_visual: torch.Tensor,
        plan: Any,
    ) -> None:
        kept = keep_visual.nonzero(as_tuple=True)[0]
        num_kept = int(kept.numel())
        source = torch.zeros(
            num_kept, num_visual, dtype=torch.float32
        )
        kept_cpu = kept.cpu()
        source[torch.arange(num_kept), kept_cpu] = 1.0

        if plan.assign_one_hot is not None and plan.counts is not None:
            row_of = torch.full(
                (num_visual,), -1, dtype=torch.long
            )
            row_of[kept_cpu] = torch.arange(num_kept)
            anchor_rows = row_of[plan.anchor_index[0].cpu()]
            merged_from = plan.merge_index[0].cpu()
            weights = (
                plan.assign_one_hot[0].transpose(0, 1)
                / plan.counts[0]
            ).float().cpu()
            source[
                anchor_rows.unsqueeze(1), merged_from.unsqueeze(0)
            ] += weights

        self._composed_source = source
        self._source_history = [
            torch.eye(num_visual, dtype=torch.float32),
            source,
        ]

    def _record_cut(self, local: torch.Tensor) -> None:
        if (
            not self._track_source
            or self._composed_source is None
            or local.shape[1] == 0
        ):
            return
        self._composed_source = self._composed_source[
            local[0].detach().cpu()
        ]
        self._source_history.append(self._composed_source)
