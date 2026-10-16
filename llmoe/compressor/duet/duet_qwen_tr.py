"""Batched compression-aware DUET training for Qwen2.5-VL.

The inference adapter targets generation and owns single-prompt KV-cache
bookkeeping. Training uses a separate decoder: Qwen images have different NaViT
token counts, so every DUET cut is applied row by row and the results are
right-padded back to a rectangle, following HiDrop's Qwen training design.

There is no published Qwen DUET training recipe. The runner adapts the released
LLaVA recipe by training visual.merger in stage 1 and the merger plus language
model in stage 2 while the rest of the vision tower remains frozen.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast

from llmoe.compressor._pruning import decoder_attention_rows
from llmoe.compressor.duet.stage1 import duet_stage1_plan
from llmoe.compressor.duet.stage2 import select_text_raters
from llmoe.utils.qwen import (
    APPLY_MROPE,
    resolve_mrope_section,
)
from llmoe.utils.qwen_training import (
    QwenVisionSignalCapture,
    QwenVisualSpan as VisualSpan,
    pad_qwen_training_rows as _pad_rows,
    qwen_training_row_views as _row_views,
    qwen_visual_spans as _visual_spans,
)


IGNORE_INDEX = -100


@dataclass
class DuetQwenModelOutputWithPast(BaseModelOutputWithPast):
    """Decoder output carrying the labels that survived DUET's cuts."""

    updated_labels: torch.LongTensor | None = None


class DuetQwenTrainingDecoder(nn.Module):
    """Qwen text decoder with batched, row-wise PyramidDrop boundaries."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        image_token_id: int,
        pdrop_layers: Sequence[int],
        pdrop_keep_ratios: Sequence[float],
        text_rater: str,
        score_reduction: str,
        tokenizer: Any,
        use_stage2: bool,
    ) -> None:
        super().__init__()
        self.config = base_model.config
        self.embed_tokens = base_model.embed_tokens
        self.layers = base_model.layers
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb
        self.gradient_checkpointing = getattr(base_model, "gradient_checkpointing", False)
        self.image_token_id = int(image_token_id)
        self.pdrop_layers = tuple(int(layer) for layer in pdrop_layers)
        self.pdrop_keep_ratios = tuple(float(ratio) for ratio in pdrop_keep_ratios)
        self.boundary_stage = {
            layer: stage for stage, layer in enumerate(self.pdrop_layers)
        }
        self.text_rater = str(text_rater)
        self.score_reduction = str(score_reduction)
        self.tokenizer = tokenizer
        self.use_stage2 = bool(use_stage2)
        if len(self.pdrop_layers) != len(self.pdrop_keep_ratios):
            raise ValueError(
                "DUET Qwen training needs one keep ratio per pdrop layer; "
                f"got {len(self.pdrop_layers)} layers and "
                f"{len(self.pdrop_keep_ratios)} ratios."
            )
        if (
            self.use_stage2
            and self.pdrop_layers
            and self.pdrop_layers[-1] >= len(self.layers)
        ):
            raise ValueError(
                f"DUET pdrop layer {self.pdrop_layers[-1]} does not fit "
                f"the Qwen decoder's {len(self.layers)} layers."
            )
        if self.use_stage2 and APPLY_MROPE is None:
            raise RuntimeError(
                "DUET Qwen training could not import "
                "apply_multimodal_rotary_pos_emb from Transformers."
            )
        self.mrope_section = resolve_mrope_section(self.config)
        self.train(base_model.training)
        object.__setattr__(self, "_base_model", base_model)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    @staticmethod
    def _split_position_ids(
        position_ids: torch.Tensor | None, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        batch, length = inputs_embeds.shape[:2]
        if position_ids is None:
            flat = torch.arange(length, device=inputs_embeds.device)
            return flat.view(1, 1, -1).expand(3, batch, -1)
        if position_ids.ndim == 2:
            return position_ids[None].expand(3, batch, -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            return position_ids[1:]
        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            return position_ids
        raise ValueError(
            "DUET expected Qwen position ids shaped [3, B, L] or [4, B, L], got "
            f"{tuple(position_ids.shape)}."
        )

    def _select_raters(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        spans: tuple[VisualSpan | None, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        for row, span in enumerate(spans):
            if span is None:
                selected.append(
                    torch.zeros(1, dtype=torch.long, device=input_ids.device)
                )
                valid_rows.append(
                    torch.zeros(1, dtype=torch.bool, device=input_ids.device)
                )
                continue
            row_mask = attention_mask[row:row + 1] if attention_mask is not None else None
            index, valid = select_text_raters(
                input_ids[row:row + 1],
                row_mask,
                torch.tensor(
                    [span.start], dtype=torch.long, device=input_ids.device
                ),
                span.count,
                mode=self.text_rater,
                tokenizer=self.tokenizer,
            )
            selected.append(index[0])
            valid_rows.append(valid[0])
        width = max(int(row.numel()) for row in selected)
        index_batch = torch.zeros(
            len(selected), width, dtype=torch.long, device=input_ids.device
        )
        valid_batch = torch.zeros(
            len(selected), width, dtype=torch.bool, device=input_ids.device
        )
        for row, (index, valid) in enumerate(
            zip(selected, valid_rows, strict=True)
        ):
            count = int(index.numel())
            index_batch[row, :count] = index
            valid_batch[row, :count] = valid
            if count < width:
                index_batch[row, count:] = index[-1]
        return index_batch, valid_batch

    def _prune(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        mrope_ids: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        stage1_counts: list[int],
        ratio: float,
        *,
        pad_token_id: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        spans = _visual_spans(input_ids, self.image_token_id, attention_mask)
        if not any(span is not None for span in spans):
            # ZeRO-3 gathers a module's parameters when it is called. Another
            # data-parallel rank can have image rows at this boundary, so a
            # text-only rank must touch the same selector projections in the
            # same order. Match HiDrop's Qwen path while limiting the otherwise
            # useless selector work to one token per row.
            sync_hidden = hidden_states[:, :1]
            sync_positions = self.rotary_emb(
                sync_hidden, mrope_ids[..., :1]
            )
            sync_mask = (
                attention_mask[:, :1]
                if attention_mask is not None else None
            )
            decoder_attention_rows(
                layer,
                sync_hidden,
                sync_positions,
                sync_mask,
                torch.zeros(
                    hidden_states.shape[0],
                    1,
                    dtype=torch.long,
                    device=hidden_states.device,
                ),
                apply_rope=lambda query, key, cos, sin: APPLY_MROPE(
                    query, key, cos, sin, self.mrope_section
                ),
                label="DUET/Qwen training selector synchronization",
            )
            return hidden_states, mrope_ids, input_ids, attention_mask, labels
        raters, valid_raters = self._select_raters(
            input_ids, attention_mask, spans
        )
        position_embeddings = self.rotary_emb(hidden_states, mrope_ids)
        attention = decoder_attention_rows(
            layer,
            hidden_states,
            position_embeddings,
            attention_mask,
            raters,
            apply_rope=lambda query, key, cos, sin: APPLY_MROPE(
                query, key, cos, sin, self.mrope_section
            ),
            label="DUET/Qwen training",
        )

        hidden_rows: list[torch.Tensor] = []
        mrope_rows: list[torch.Tensor] = []
        id_rows: list[torch.Tensor] = []
        mask_rows = [] if attention_mask is not None else None
        label_rows = [] if labels is not None else None
        for row, span in enumerate(spans):
            hidden_row, mrope_row, id_row, mask_row, label_row = _row_views(
                hidden_states,
                mrope_ids,
                input_ids,
                attention_mask,
                labels,
                row,
            )
            if span is not None and bool(valid_raters[row].any()):
                target = max(
                    0,
                    min(
                        int(int(stage1_counts[row]) * float(ratio)),
                        span.count,
                    ),
                )
                if target < span.count:
                    scores = attention[
                        row, valid_raters[row], span.start:span.end
                    ]
                    if self.score_reduction == "mean":
                        scores = scores.mean(dim=0)
                    elif self.score_reduction == "max":
                        scores = scores.max(dim=0).values
                    else:
                        raise ValueError(
                            "Unknown DUET score reduction "
                            f"{self.score_reduction!r}."
                        )
                    keep_local = (
                        scores.topk(target).indices.sort().values
                        if target
                        else torch.zeros(
                            0, dtype=torch.long, device=scores.device
                        )
                    )
                    policy = torch.ones(
                        hidden_row.shape[0],
                        dtype=torch.bool,
                        device=hidden_row.device,
                    )
                    policy[span.start:span.end] = False
                    if target:
                        policy[span.start + keep_local] = True
                    selected_columns = torch.where(policy)[0]
                    hidden_row = hidden_row.index_select(0, selected_columns)
                    mrope_row = mrope_row.index_select(-1, selected_columns)
                    id_row = id_row.index_select(0, selected_columns)
                    if mask_row is not None:
                        mask_row = mask_row.index_select(0, selected_columns)
                    if label_row is not None:
                        label_row = label_row.index_select(0, selected_columns)
            hidden_rows.append(hidden_row)
            mrope_rows.append(mrope_row)
            id_rows.append(id_row)
            if mask_rows is not None:
                mask_rows.append(mask_row)
            if label_rows is not None:
                label_rows.append(label_row)
        return _pad_rows(
            hidden_rows,
            mrope_rows,
            id_rows,
            mask_rows,
            label_rows,
            pad_token_id=pad_token_id,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        stage1_counts: Sequence[int] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> DuetQwenModelOutputWithPast:
        if input_ids is None:
            raise ValueError(
                "DUET Qwen training needs input_ids to locate visual spans."
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[:2] != input_ids.shape:
            raise ValueError(
                "DUET Qwen input_ids and inputs_embeds must have matching "
                f"batch/sequence dimensions, got {tuple(input_ids.shape)} and "
                f"{tuple(inputs_embeds.shape[:2])}."
            )
        if (
            attention_mask is not None
            and attention_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "DUET Qwen attention_mask must be 2-D and match input_ids; "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}."
            )
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError(
                "DUET Qwen labels must match input_ids before pruning; "
                f"got {tuple(labels.shape)} and {tuple(input_ids.shape)}."
            )
        if use_cache:
            raise ValueError("DUET Qwen training runs with use_cache=False.")
        if output_attentions:
            raise ValueError(
                "DUET Qwen training does not return decoder attention maps."
            )
        kwargs.pop("cache_position", None)
        kwargs.pop("past_key_values", None)
        output_hidden_states = bool(output_hidden_states)
        mrope_ids = self._split_position_ids(position_ids, inputs_embeds)
        padding_mask = (
            attention_mask.to(torch.bool)
            if attention_mask is not None else None
        )
        current_ids = input_ids
        current_labels = labels
        if stage1_counts is None:
            stage1_counts = [
                span.count if span is not None else 0
                for span in _visual_spans(
                    current_ids, self.image_token_id, padding_mask
                )
            ]
        stage1_counts = [int(value) for value in stage1_counts]
        if len(stage1_counts) != int(inputs_embeds.shape[0]):
            raise ValueError(
                f"DUET got {len(stage1_counts)} Stage-1 counts for batch "
                f"{inputs_embeds.shape[0]}."
            )

        hidden_states = inputs_embeds
        all_hidden_states: list[torch.Tensor] | None = (
            [] if output_hidden_states else None
        )
        pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        for layer_idx, decoder_layer in enumerate(
            self.layers[:self.config.num_hidden_layers]
        ):
            stage = self.boundary_stage.get(layer_idx)
            if self.use_stage2 and stage is not None:
                (
                    hidden_states,
                    mrope_ids,
                    current_ids,
                    padding_mask,
                    current_labels,
                ) = self._prune(
                    decoder_layer,
                    hidden_states,
                    mrope_ids,
                    current_ids,
                    padding_mask,
                    current_labels,
                    stage1_counts,
                    self.pdrop_keep_ratios[stage],
                    pad_token_id=pad_token_id,
                )
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            position_embeddings = self.rotary_emb(hidden_states, mrope_ids)
            if self.training and self.gradient_checkpointing:
                layer_kwargs = dict(kwargs)

                def checkpointed_layer_forward(
                    layer_hidden_states: torch.Tensor,
                    *,
                    checkpointed_layer: nn.Module = decoder_layer,
                    checkpointed_mask: torch.Tensor | None = padding_mask,
                    checkpointed_positions: tuple[
                        torch.Tensor, torch.Tensor
                    ] = position_embeddings,
                    checkpointed_kwargs: dict[str, Any] = layer_kwargs,
                ) -> torch.Tensor:
                    output = checkpointed_layer(
                        layer_hidden_states,
                        attention_mask=checkpointed_mask,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=checkpointed_positions,
                        **checkpointed_kwargs,
                    )
                    return output[0] if isinstance(output, tuple) else output

                hidden_states = checkpoint(
                    checkpointed_layer_forward,
                    hidden_states,
                    use_reentrant=True,
                )
            else:
                output = decoder_layer(
                    hidden_states,
                    attention_mask=padding_mask,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = output[0] if isinstance(output, tuple) else output

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        return DuetQwenModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=(
                tuple(all_hidden_states)
                if all_hidden_states is not None else None
            ),
            attentions=None,
            updated_labels=current_labels,
        )


class DuetTrainingQwenForConditionalGeneration(nn.Module):
    """Batched Qwen VLM wrapper used by DUET's tr and ft commands."""

    def __init__(self, base_model: nn.Module, compressor: Any, processor: Any) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = base_model.generation_config
        self.compressor = compressor
        self.processor = processor
        self.image_token_id = int(base_model.config.image_token_id)
        self.pad_token_id = int(
            getattr(processor.tokenizer, "pad_token_id", 0) or 0
        )
        visual = base_model.model.visual
        self._vision_capture = QwenVisionSignalCapture(
            visual,
            attention_chunk=int(compressor._attention_chunk),
            label="DUET/Qwen training",
        )
        original_language_model = base_model.model.language_model
        object.__setattr__(
            self, "_original_language_model", original_language_model
        )
        # `retain_ratio` is layer-averaged, so Stage 1's share is it divided by
        # what Stage 2 averages over this decoder. Resolved once here, through the
        # same method the inference adapter uses, so a trained checkpoint sees the
        # widths it will be evaluated at.
        self._stage1_ratios: tuple[float, float] = compressor.qwen_stage1_ratios(
            len(original_language_model.layers)
        )
        base_model.model.language_model = DuetQwenTrainingDecoder(
            original_language_model,
            image_token_id=self.image_token_id,
            pdrop_layers=compressor._pdrop_layers,
            pdrop_keep_ratios=compressor._pdrop_keep_ratios,
            text_rater=compressor._text_rater,
            score_reduction=compressor._score_reduction,
            tokenizer=processor.tokenizer,
            use_stage2=compressor._use_stage2,
        )

    @property
    def model(self) -> nn.Module:
        return self.base_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.base_model.lm_head

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        if gradient_checkpointing_kwargs is None:
            self.base_model.gradient_checkpointing_enable()
        else:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        self.base_model.model.language_model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.base_model.gradient_checkpointing_disable()
        self.base_model.model.language_model.gradient_checkpointing = False

    def remove_training_hooks(self) -> None:
        self._vision_capture.remove()

    def _position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if position_ids is not None:
            if position_ids.ndim == 2 and position_ids.shape == input_ids.shape:
                return position_ids[None].expand(3, *position_ids.shape)
            if (
                position_ids.ndim == 3
                and position_ids.shape[0] in (3, 4)
                and position_ids.shape[1:] == input_ids.shape
            ):
                return position_ids
            raise ValueError(
                "DUET Qwen position_ids must have shape [B, L], [3, B, L], "
                f"or [4, B, L]; got {tuple(position_ids.shape)}."
            )
        image_rows = (input_ids == self.image_token_id).any(dim=1)
        if image_grid_thw.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "DUET's Qwen collator must provide one real or dummy image grid per row; "
                f"got {image_grid_thw.shape[0]} grids for batch "
                f"{input_ids.shape[0]}."
            )
        real_grids = image_grid_thw[image_rows]
        token_types = (input_ids == self.image_token_id).to(torch.int)
        mrope_ids, _ = self.base_model.model.get_rope_index(
            input_ids,
            token_types,
            image_grid_thw=real_grids,
            attention_mask=attention_mask,
        )
        return mrope_ids

    def _stage1_counts(self, visual_count: int) -> tuple[int, int]:
        """Stage-1 widths for one image, from the back-solved layer-averaged budget."""
        dominant_ratio, contextual_ratio = self._stage1_ratios
        dominant = int(dominant_ratio * visual_count)
        contextual = max(int(contextual_ratio * visual_count), 1)
        return max(0, dominant), max(0, contextual)

    def _apply_stage1(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        mrope_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        feature_rows: list[torch.Tensor],
        score_rows: list[torch.Tensor],
        metric_rows: list[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[int],
    ]:
        spans = _visual_spans(input_ids, self.image_token_id, attention_mask)
        hidden_rows: list[torch.Tensor] = []
        mrope_rows: list[torch.Tensor] = []
        id_rows: list[torch.Tensor] = []
        mask_rows = [] if attention_mask is not None else None
        label_rows = [] if labels is not None else None
        stage1_counts: list[int] = []
        for row, span in enumerate(spans):
            hidden_row, mrope_row, id_row, mask_row, label_row = _row_views(
                inputs_embeds,
                mrope_ids,
                input_ids,
                attention_mask,
                labels,
                row,
            )
            features = feature_rows[row].to(
                hidden_row.device, hidden_row.dtype
            )
            if span is None:
                hidden_row = hidden_row + features[0:0].sum()
                stage1_counts.append(0)
            else:
                if span.count != int(features.shape[0]):
                    raise ValueError(
                        "DUET Qwen image features and placeholders disagree for "
                        f"row {row}: tokens={span.count}, "
                        f"features={features.shape[0]}."
                    )
                selected_features = features
                keep_local = torch.arange(
                    span.count, device=features.device
                )
                if self.compressor._use_stage1:
                    dominant, contextual = self._stage1_counts(span.count)
                    if dominant + contextual < span.count:
                        plan = duet_stage1_plan(
                            score_rows[row].unsqueeze(0),
                            metric_rows[row].unsqueeze(0),
                            dominant=dominant,
                            contextual=contextual,
                            cluster_width=int(
                                self.compressor._cluster_width
                            ),
                            dtype=features.dtype,
                        )
                        anchors = plan.anchor_index[0]
                        contextual_tokens = features.index_select(0, anchors)
                        if (
                            plan.assign_one_hot is not None
                            and plan.counts is not None
                        ):
                            merge_tokens = features.index_select(
                                0, plan.merge_index[0]
                            )
                            contextual_tokens = (
                                contextual_tokens
                                + torch.matmul(
                                    plan.assign_one_hot[0].transpose(0, 1),
                                    merge_tokens,
                                )
                                / plan.counts[0]
                            )
                        modified = features.index_copy(
                            0, anchors, contextual_tokens
                        )
                        keep_mask = torch.zeros(
                            span.count,
                            dtype=torch.bool,
                            device=features.device,
                        )
                        keep_mask[plan.dominant_index[0]] = True
                        keep_mask[anchors] = True
                        keep_local = torch.where(keep_mask)[0]
                        selected_features = modified.index_select(
                            0, keep_local
                        )
                stage1_counts.append(int(selected_features.shape[0]))
                hidden_row = torch.cat(
                    (
                        hidden_row[:span.start],
                        selected_features,
                        hidden_row[span.end:],
                    ),
                    dim=0,
                )
                selected_columns = torch.cat(
                    (
                        torch.arange(span.start, device=id_row.device),
                        span.start + keep_local.to(id_row.device),
                        torch.arange(
                            span.end,
                            id_row.shape[0],
                            device=id_row.device,
                        ),
                    )
                )
                mrope_row = mrope_row.index_select(-1, selected_columns)
                id_row = id_row.index_select(0, selected_columns)
                if mask_row is not None:
                    mask_row = mask_row.index_select(0, selected_columns)
                if label_row is not None:
                    label_row = label_row.index_select(
                        0, selected_columns
                    )
            hidden_rows.append(hidden_row)
            mrope_rows.append(mrope_row)
            id_rows.append(id_row)
            if mask_rows is not None:
                mask_rows.append(mask_row)
            if label_rows is not None:
                label_rows.append(label_row)
        hidden, mrope, ids, mask, updated_labels = _pad_rows(
            hidden_rows,
            mrope_rows,
            id_rows,
            mask_rows,
            label_rows,
            pad_token_id=self.pad_token_id,
        )
        return hidden, mrope, ids, mask, updated_labels, stage1_counts

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> Any:
        if input_ids is None or inputs_embeds is not None:
            raise ValueError(
                "DUET Qwen training requires input_ids and builds its own embeddings."
            )
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(
                "DUET Qwen training requires the collator's pixel_values and "
                "image_grid_thw."
            )
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("use_cache", None)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        position_ids = self._position_ids(
            input_ids, attention_mask, image_grid_thw, position_ids
        )
        image_output = self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        feature_rows = list(image_output.pooler_output)
        if len(feature_rows) != int(input_ids.shape[0]):
            raise ValueError(
                "DUET Qwen training requires one real or dummy image per row; "
                f"got {len(feature_rows)} features for batch "
                f"{input_ids.shape[0]}."
            )
        score_rows, metric_rows = self._vision_capture.consume(
            image_grid_thw
        )
        (
            inputs_embeds,
            position_ids,
            input_ids,
            attention_mask,
            labels,
            stage1_counts,
        ) = self._apply_stage1(
            inputs_embeds,
            input_ids,
            position_ids,
            attention_mask,
            labels,
            feature_rows,
            score_rows,
            metric_rows,
        )
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            stage1_counts=stage1_counts,
            use_cache=False,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        updated_labels = outputs.updated_labels
        if labels is not None:
            logits = self.lm_head(hidden_states).float()
        else:
            indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, indices, :])
        loss = None
        if updated_labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = updated_labels[..., 1:].contiguous().to(
                shift_logits.device
            )
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLCausalLMOutputWithPast,
        )

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
            rope_deltas=None,
        )

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        """Save stock Qwen weights without the training-only decoder wrapper."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state_dict = kwargs.pop("state_dict", None)
        base_state_dict = None
        if state_dict is not None:
            base_state_dict = {
                key.removeprefix("base_model."): value
                for key, value in state_dict.items()
                if key.startswith("base_model.")
            }
        wrapped = self.base_model.model.language_model
        self.base_model.model.language_model = (
            self._original_language_model
        )
        try:
            self.base_model.save_pretrained(
                output, state_dict=base_state_dict, **kwargs
            )
        finally:
            self.base_model.model.language_model = wrapped
        metadata = {
            "backbone": "qwen",
            "released_training_recipe": False,
            "retain_ratio": self.compressor._retain_ratio,
            "contextual_ratio": self.compressor._contextual_ratio,
            # Stage-1 shares this checkpoint was trained at, back-solved from the
            # layer-averaged retain_ratio above.
            "stage1_dominant_ratio": self._stage1_ratios[0],
            "stage1_contextual_ratio": self._stage1_ratios[1],
            "pdrop_layers": list(self.compressor._pdrop_layers),
            "pdrop_keep_ratios": list(
                self.compressor._pdrop_keep_ratios
            ),
        }
        (output / "duet_config.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
