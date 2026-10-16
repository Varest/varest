"""The HiDrop decoder: a drop-in replacement for LLaVA's `language_model`.

`HiDropCompressor.install()` swaps `vlm.model.model.language_model` for a
`HiDropLlamaModel` wrapping the original LLaMA stack. The wrapper reuses the
original embeddings, decoder layers, norm, and rotary embedding -- it only changes
*which tokens are in the sequence* at each layer:

1. **Late injection (§3.1).** The visual span is removed before layer 1, so layers
   1..`late_entry_layer - 1` run text-only. At `late_entry_layer` the *original*
   projected visual embeddings are spliced back in at their original positions.
2. **Concave pyramid pruning (§3.2).** After each layer in `layer_list`, the
   *next* decoder block's q/k projections score the visual tokens, and DTop-K
   keeps `image_token_keep_list[i]` of them.
3. **Early exit (§3.1).** Every remaining visual token is removed before
   `early_exit_layer`.

Two invariants make this work with a KV cache: position IDs travel with their
tokens and are never renumbered after a removal (so the surviving text keeps its
original RoPE phase), and pruning only ever runs on a prefill (`seq_len > 1`), so
decoding just extends each layer's already-pruned cache.
"""

from dataclasses import dataclass
from typing import Any
import inspect
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from llmoe.compressor.hidrop.config import DifferentiableTopK
from llmoe.compressor.hidrop.hidrop_llava_next import (
    LlavaNextVisualLayout,
    ResolvedHiDropSchedule,
)


IGNORE_INDEX = -100


@dataclass
class HiDropModelOutputWithPast(BaseModelOutputWithPast):
    # Pruning removes label positions along with their tokens, so a training
    # wrapper has to read the surviving labels back out of the decoder.
    updated_labels: torch.LongTensor | None = None


@dataclass
class VisualTokenSpan:
    start: int
    end: int
    input_seq_len: int

    @property
    def count(self) -> int:
        return self.end - self.start


@dataclass
class HiDropForwardContext:
    span: VisualTokenSpan
    spans: tuple[VisualTokenSpan | None, ...]
    original_hidden_states: torch.Tensor
    original_attention_mask: torch.Tensor | None
    original_position_ids: torch.Tensor
    original_labels: torch.Tensor | None
    active_visual_count: int
    active_patch_count: int
    llava_next_layout: LlavaNextVisualLayout | None
    resolved_schedule: ResolvedHiDropSchedule | None
    preserved_visual_rows: set[int]
    pruning_events: list[dict[str, Any]]



class HiDropLlamaModel(nn.Module):
    def __init__(self, base_model: nn.Module, compressor: Any) -> None:
        super().__init__()
        self.config = base_model.config
        self.embed_tokens = base_model.embed_tokens
        self.layers = base_model.layers
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb
        self.gradient_checkpointing = getattr(base_model, "gradient_checkpointing", False)
        # Kept off the module registry: the wrapper already re-registers every
        # submodule above, and registering the original too would double-count
        # its parameters.
        object.__setattr__(self, "_base_model", base_model)
        object.__setattr__(self, "_compressor", compressor)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> HiDropModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        compressor = self._compressor
        attn_implementation = getattr(self.config, "_attn_implementation", None)
        if attn_implementation not in {"eager", "flash_attention_2", "sdpa"}:
            raise ValueError(
                f"HiDrop does not support attention implementation {attn_implementation!r}."
            )
        output_attentions = bool(
            getattr(self.config, "output_attentions", False)
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = bool(
            getattr(self.config, "output_hidden_states", False)
            if output_hidden_states is None
            else output_hidden_states
        )
        # Accepted for signature compatibility with `LlamaModel.forward`; this
        # wrapper always returns a `HiDropModelOutputWithPast`, and reading
        # `config.use_return_dict` only raises a Transformers 5 deprecation.
        _ = return_dict
        if output_attentions:
            raise ValueError(
                "HiDrop does not support output_attentions=True; token scoring uses the separate q/k selector."
            )
        use_cache = bool(
            getattr(self.config, "use_cache", False) if use_cache is None else use_cache
        )
        if self.training or labels is not None:
            use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        batch_size, seq_len = inputs_embeds.shape[:2]
        if batch_size != 1 and not compressor.config.batch_size_supported:
            raise ValueError("HiDrop currently supports batch_size=1 only.")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                f"HiDrop expected attention_mask [B, L] or None, got {tuple(attention_mask.shape)}."
            )
        attention_mask_all_valid = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
            attention_mask_all_valid = bool(attention_mask.all().item())
        if (
            attention_mask is not None
            and not attention_mask_all_valid
            and not compressor.config.batch_size_supported
        ):
            raise ValueError(
                "HiDrop padded batches are not implemented yet; use batch_size=1 without padding."
            )

        expected_decode_position: int | None = None
        if compressor.config.backbone_family == "llava_next" and past_key_values is not None:
            if not isinstance(past_key_values, DynamicCache):
                raise NotImplementedError(
                    "HiDrop/LLaVA-NeXT decode supports Transformers DynamicCache only."
                )
        if (
            compressor.config.backbone_family == "llava_next"
            and past_key_values is not None
            and int(past_key_values.get_seq_length(0)) > 0
        ):
            if seq_len != 1:
                raise NotImplementedError(
                    "HiDrop/LLaVA-NeXT supports one-token greedy decode steps only after the image prefill."
                )
            prefill_context = compressor.last_context
            if prefill_context is None or prefill_context.llava_next_layout is None:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT received a cache that was not produced by the current image prefill."
                )
            if len(compressor.last_visual_token_counts_by_layer) != int(
                self.config.num_hidden_layers
            ):
                raise RuntimeError(
                    "HiDrop/LLaVA-NeXT prefill did not record per-layer cache geometry."
                )
            text_prefill_tokens = int(
                prefill_context.span.input_seq_len - prefill_context.span.count
            )
            shallow_length = int(past_key_values.get_seq_length(0))
            decode_steps = shallow_length - text_prefill_tokens
            if decode_steps < 0:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT cache is shorter than the text-only shallow prefill."
                )
            for cache_layer, visual_tokens in enumerate(
                compressor.last_visual_token_counts_by_layer
            ):
                expected_length = text_prefill_tokens + int(visual_tokens) + decode_steps
                actual_length = int(past_key_values.get_seq_length(cache_layer))
                if actual_length != expected_length:
                    raise ValueError(
                        "HiDrop/LLaVA-NeXT per-layer cache length mismatch: "
                        f"layer={cache_layer + 1}, actual={actual_length}, expected={expected_length}."
                    )
            expected_decode_position = int(prefill_context.span.input_seq_len + decode_steps)

        if position_ids is None:
            if cache_position is not None:
                cache_position = (
                    cache_position[-seq_len:]
                    if cache_position.ndim == 1
                    else cache_position[..., -seq_len:]
                )
                position_ids = (
                    cache_position.unsqueeze(0) if cache_position.ndim == 1 else cache_position
                )
            elif attention_mask is not None and attention_mask.shape[-1] >= seq_len:
                start = int(attention_mask.shape[-1] - seq_len)
                position_ids = torch.arange(
                    start, start + seq_len, device=inputs_embeds.device, dtype=torch.long
                ).unsqueeze(0)
            else:
                past_seen_tokens = (
                    past_key_values.get_seq_length() if past_key_values is not None else 0
                )
                position_ids = torch.arange(
                    seq_len, device=inputs_embeds.device, dtype=torch.long
                ).unsqueeze(0) + int(past_seen_tokens)
        elif position_ids.shape[-1] != seq_len:
            position_ids = position_ids[..., -seq_len:]
        if position_ids.shape[0] == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1).clone()
        if expected_decode_position is not None:
            expected_positions = torch.arange(
                expected_decode_position,
                expected_decode_position + seq_len,
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).unsqueeze(0)
            if position_ids.shape != expected_positions.shape or not torch.equal(
                position_ids, expected_positions
            ):
                raise ValueError(
                    "HiDrop/LLaVA-NeXT decode position does not continue the persistent prefill positions: "
                    f"got {position_ids.tolist()}, expected {expected_positions.tolist()}. "
                    "Pass the original full attention_mask or explicit position_ids."
                )

        hidden_states = inputs_embeds
        # Only a prefill carries a visual span; a decode step (seq_len == 1)
        # extends the caches the prefill already pruned.
        context = (
            compressor._build_hidrop_context(
                hidden_states, input_ids, attention_mask, position_ids, labels
            )
            if hidden_states.shape[1] > 1
            else None
        )
        if attention_mask is not None and attention_mask_all_valid:
            attention_mask = None
        updated_labels = labels
        if context is not None and compressor.config.enabled:
            hidden_states, attention_mask, position_ids, updated_labels = self._remove_visual_span(
                hidden_states,
                attention_mask,
                position_ids,
                labels,
                context.spans,
                context.span.count,
            )

        all_hidden_states: list[torch.Tensor] | None = [] if output_hidden_states else None
        all_self_attns: list[torch.Tensor] | None = [] if output_attentions else None
        fired_layers: list[int] = []
        visual_counts: list[int] = []
        patch_counts: list[int] = []

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            rank_layer = int(layer_idx + 1)
            if (
                context is not None
                and compressor.config.enabled
                and rank_layer == compressor.config.late_entry_layer
            ):
                hidden_states, attention_mask, position_ids, updated_labels = (
                    self._late_inject_visual_span(
                        hidden_states, attention_mask, position_ids, updated_labels, context
                    )
                )
                context.active_visual_count = context.span.count
                context.active_patch_count = (
                    context.llava_next_layout.patch_tokens
                    if context.llava_next_layout is not None
                    else context.span.count
                )
            if (
                context is not None
                and compressor.config.enabled
                and rank_layer == compressor.config.early_exit_layer
                and context.active_visual_count > 0
            ):
                removal_counts = tuple(
                    context.span.count
                    if batch_idx in context.preserved_visual_rows
                    else context.active_visual_count
                    for batch_idx in range(batch_size)
                )
                hidden_states, attention_mask, position_ids, updated_labels = (
                    self._remove_visual_span(
                        hidden_states,
                        attention_mask,
                        position_ids,
                        updated_labels,
                        context.spans,
                        removal_counts,
                    )
                )
                context.active_visual_count = 0
                context.active_patch_count = 0

            visual_counts.append(
                int(
                    context.active_visual_count
                    if context is not None and compressor.config.enabled
                    else 0
                )
            )
            patch_counts.append(
                int(
                    context.active_patch_count
                    if context is not None and compressor.config.enabled
                    else 0
                )
            )
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            if attention_mask is not None and attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"HiDrop expected a current 2D mask matching hidden states "
                    f"{tuple(hidden_states.shape[:2])}, got {tuple(attention_mask.shape)}."
                )
            layer_mask = (
                self._eager_causal_mask(
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    layer_idx,
                )
                if attn_implementation == "eager"
                else attention_mask
            )
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
            if self.training and self.gradient_checkpointing:
                layer_kwargs = dict(kwargs)

                def checkpointed_layer_forward(
                    layer_hidden_states: torch.Tensor,
                    *,
                    checkpointed_decoder_layer: nn.Module = decoder_layer,
                    checkpointed_attention_mask: torch.Tensor | None = layer_mask,
                    checkpointed_position_ids: torch.Tensor = position_ids,
                    checkpointed_position_embeddings: tuple[
                        torch.Tensor, torch.Tensor
                    ] = position_embeddings,
                    checkpointed_cache_position: torch.Tensor | None = cache_position,
                    checkpointed_kwargs: dict[str, Any] = layer_kwargs,
                ) -> torch.Tensor:
                    return self._run_decoder_layer(
                        checkpointed_decoder_layer,
                        layer_hidden_states,
                        attention_mask=checkpointed_attention_mask,
                        position_ids=checkpointed_position_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=checkpointed_position_embeddings,
                        cache_position=checkpointed_cache_position,
                        **checkpointed_kwargs,
                    )

                hidden_states = checkpoint(
                    checkpointed_layer_forward,
                    hidden_states,
                    # Match the released LLaVA/HiDrop checkpoint path.  The
                    # non-reentrant engine performs saved-tensor metadata
                    # checks that are incompatible with ZeRO-3 parameter
                    # partitioning during recomputation.
                    use_reentrant=True,
                )
            else:
                hidden_states = self._run_decoder_layer(
                    decoder_layer,
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    **kwargs,
                )

            if all_self_attns is not None:
                all_self_attns.append(torch.empty(0, device=hidden_states.device))
            if (
                self.training
                and compressor.config.enabled
                and context is None
                and hidden_states.shape[1] > 1
                and rank_layer in compressor.config.layer_list
            ):
                # ZeRO-3 gathers parameters when a module is called, so every
                # distributed rank must touch the selector's next-layer norm
                # and q/k projections in the same order.  A text-only rank has
                # no visual context to score, while another rank in the same
                # step can contain images.  Touch a single token to preserve
                # the collective order without paying for a full-sequence
                # selector that cannot produce a visual-token decision.
                if rank_layer >= len(self.layers):
                    raise ValueError(
                        f"HiDrop selector synchronization after layer {rank_layer} requires decoder layer index "
                        f"{rank_layer}, but that next layer does not exist."
                    )
                self._project_qk(
                    self.layers[rank_layer],
                    hidden_states[:, :1],
                    position_ids[:, :1],
                )
            if (
                context is not None
                and compressor.config.enabled
                and hidden_states.shape[1] > 1
                and rank_layer in compressor.config.layer_list
            ):
                if rank_layer >= len(self.layers):
                    raise ValueError(
                        f"HiDrop pruning after layer {rank_layer} requires decoder layer index {rank_layer} "
                        "for upstream-compatible q/k scoring, but that next layer does not exist."
                    )
                hidden_states, attention_mask, position_ids, updated_labels = (
                    self._prune_visual_tokens(
                        self.layers[rank_layer],
                        hidden_states,
                        attention_mask,
                        position_ids,
                        updated_labels,
                        context,
                        rank_layer,
                    )
                )
                fired_layers.append(rank_layer)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        if context is not None and compressor.config.enabled:
            missing = [layer for layer in compressor.config.layer_list if layer not in fired_layers]
            if missing and hidden_states.shape[1] > 1:
                raise RuntimeError(
                    f"HiDrop prefill completed without firing all configured pruning layers: "
                    f"missing={missing}, fired={fired_layers}."
                )
            compressor.last_context = context
            compressor.last_visual_token_counts_by_layer = visual_counts
            compressor.last_patch_token_counts_by_layer = patch_counts
            compressor.pruning_history.extend(context.pruning_events)
            compressor.debugger.log(f"visual_token_counts_by_layer={visual_counts}")

        return HiDropModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_self_attns is not None else None,
            updated_labels=updated_labels,
        )

    @staticmethod
    def _eager_causal_mask(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None,
        layer_idx: int,
    ) -> torch.Tensor:
        """Build the additive 4-D mask required by eager Llama attention.

        HiDrop can leave a different number of cached visual tokens at each
        decoder layer.  Querying the cache for the current layer is therefore
        necessary; a single model-wide past length is not correct after token
        pruning.
        """
        batch_size, query_length = hidden_states.shape[:2]
        past_length = (
            int(past_key_values.get_seq_length(layer_idx)) if past_key_values is not None else 0
        )
        key_length = past_length + query_length
        query_positions = torch.arange(
            past_length,
            key_length,
            device=hidden_states.device,
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=hidden_states.device).unsqueeze(0)
        blocked = key_positions > query_positions
        min_value = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.zeros(
            (query_length, key_length),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).masked_fill(blocked, min_value)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, query_length):
                raise ValueError(
                    "HiDrop eager attention expected the current 2-D mask to match "
                    f"hidden states {(batch_size, query_length)}, got {tuple(attention_mask.shape)}."
                )
            current_key_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            if past_length:
                current_key_mask = torch.cat(
                    (
                        torch.ones(
                            (batch_size, past_length),
                            dtype=torch.bool,
                            device=hidden_states.device,
                        ),
                        current_key_mask,
                    ),
                    dim=-1,
                )
            causal_mask = causal_mask.masked_fill(
                ~current_key_mask[:, None, None, :],
                min_value,
            )
        return causal_mask

    def _run_decoder_layer(
        self,
        decoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values: Any | None,
        use_cache: bool,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        parameters = inspect.signature(decoder_layer.forward).parameters
        layer_kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "position_embeddings": position_embeddings,
            **kwargs,
        }
        # Transformers 5.5.4 treats non-contiguous position_ids under FA2 as
        # packed-sequence metadata. HiDrop keeps persistent RoPE positions, so
        # FA2 receives only the already-computed position_embeddings.
        if getattr(self.config, "_attn_implementation", None) != "flash_attention_2":
            layer_kwargs["position_ids"] = position_ids
        if "past_key_values" in parameters:
            layer_kwargs["past_key_values"] = past_key_values
        elif "past_key_value" in parameters:
            layer_kwargs["past_key_value"] = past_key_values
        if cache_position is not None and "cache_position" in parameters:
            layer_kwargs["cache_position"] = cache_position
        output = decoder_layer(hidden_states, **layer_kwargs)
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _valid_row_length(row: torch.Tensor, attention_mask_row: torch.Tensor | None) -> int:
        if attention_mask_row is None:
            return int(row.shape[0])
        return int(attention_mask_row.to(torch.long).sum().item())

    @staticmethod
    def _stack_variable_rows(
        hidden_rows: list[torch.Tensor],
        position_rows: list[torch.Tensor],
        mask_rows: list[torch.Tensor] | None,
        label_rows: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Re-pad rows to a rectangle after each row lost a different token count."""
        max_length = max(int(row.shape[0]) for row in hidden_rows)

        def pad_hidden(row: torch.Tensor) -> torch.Tensor:
            pad = max_length - int(row.shape[0])
            if pad == 0:
                return row
            return torch.cat((row, row.new_zeros((pad, row.shape[-1]))), dim=0)

        def pad_vector(row: torch.Tensor, value: int | bool) -> torch.Tensor:
            pad = max_length - int(row.shape[0])
            if pad == 0:
                return row
            return torch.cat((row, row.new_full((pad,), value)), dim=0)

        hidden_states = torch.stack([pad_hidden(row) for row in hidden_rows], dim=0)
        position_ids = torch.stack([pad_vector(row, 0) for row in position_rows], dim=0)
        if mask_rows is None and any(int(row.shape[0]) != max_length for row in hidden_rows):
            mask_rows = [
                torch.ones(row.shape[0], dtype=torch.bool, device=row.device) for row in hidden_rows
            ]
        attention_mask = (
            torch.stack([pad_vector(row.to(torch.bool), False) for row in mask_rows], dim=0)
            if mask_rows is not None
            else None
        )
        labels = (
            torch.stack([pad_vector(row, IGNORE_INDEX) for row in label_rows], dim=0)
            if label_rows is not None
            else None
        )
        return hidden_states, attention_mask, position_ids, labels

    def _remove_visual_span(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        labels: torch.Tensor | None,
        spans: tuple[VisualTokenSpan | None, ...],
        visual_count: int | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Cut `visual_count` tokens out of each row's visual span (late entry, early exit)."""
        visual_counts = (
            tuple(int(count) for count in visual_count)
            if isinstance(visual_count, tuple)
            else tuple(int(visual_count) for _ in spans)
        )
        if len(visual_counts) != len(spans):
            raise ValueError("HiDrop per-row visual counts must match the batch size.")
        if not any(count > 0 for count in visual_counts):
            return hidden_states, attention_mask, position_ids, labels
        hidden_rows = []
        position_rows = []
        mask_rows = [] if attention_mask is not None else None
        label_rows = [] if labels is not None else None
        for batch_idx, span in enumerate(spans):
            current_mask = attention_mask[batch_idx] if attention_mask is not None else None
            valid_length = self._valid_row_length(hidden_states[batch_idx], current_mask)
            hidden_row = hidden_states[batch_idx, :valid_length]
            position_row = position_ids[batch_idx, :valid_length]
            mask_row = current_mask[:valid_length] if current_mask is not None else None
            label_row = labels[batch_idx, :valid_length] if labels is not None else None
            if span is not None:
                visual_start = int(span.start)
                visual_end = int(visual_start + visual_counts[batch_idx])
                hidden_row = torch.cat((hidden_row[:visual_start], hidden_row[visual_end:]), dim=0)
                position_row = torch.cat(
                    (position_row[:visual_start], position_row[visual_end:]), dim=0
                )
                if mask_row is not None:
                    mask_row = torch.cat((mask_row[:visual_start], mask_row[visual_end:]), dim=0)
                if label_row is not None:
                    label_row = torch.cat((label_row[:visual_start], label_row[visual_end:]), dim=0)
            hidden_rows.append(hidden_row)
            position_rows.append(position_row)
            if mask_rows is not None:
                mask_rows.append(mask_row)
            if label_rows is not None:
                label_rows.append(label_row)
        return self._stack_variable_rows(hidden_rows, position_rows, mask_rows, label_rows)

    def _late_inject_visual_span(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        labels: torch.Tensor | None,
        context: HiDropForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Splice the *original* projected visual embeddings back in at `late_entry_layer`.

        The visual tokens enter unprocessed -- they never went through the shallow
        layers -- while the text around them carries the state those layers built.
        """
        hidden_rows = []
        position_rows = []
        mask_rows = [] if attention_mask is not None else None
        label_rows = [] if labels is not None else None
        for batch_idx, span in enumerate(context.spans):
            current_mask = attention_mask[batch_idx] if attention_mask is not None else None
            valid_length = self._valid_row_length(hidden_states[batch_idx], current_mask)
            hidden_row = hidden_states[batch_idx, :valid_length]
            position_row = position_ids[batch_idx, :valid_length]
            mask_row = current_mask[:valid_length] if current_mask is not None else None
            label_row = labels[batch_idx, :valid_length] if labels is not None else None
            if span is not None:
                start, end = int(span.start), int(span.end)
                hidden_row = torch.cat(
                    (
                        hidden_row[:start],
                        context.original_hidden_states[batch_idx, start:end],
                        hidden_row[start:],
                    ),
                    dim=0,
                )
                position_row = torch.cat(
                    (
                        position_row[:start],
                        context.original_position_ids[batch_idx, start:end],
                        position_row[start:],
                    ),
                    dim=0,
                )
                if mask_row is not None:
                    if context.original_attention_mask is None:
                        raise ValueError(
                            "HiDrop cannot restore an attention mask that was not saved at prefill."
                        )
                    mask_row = torch.cat(
                        (
                            mask_row[:start],
                            context.original_attention_mask[batch_idx, start:end],
                            mask_row[start:],
                        ),
                        dim=0,
                    )
                if label_row is not None:
                    if context.original_labels is None:
                        raise ValueError(
                            "HiDrop cannot restore labels that were not saved at prefill."
                        )
                    label_row = torch.cat(
                        (
                            label_row[:start],
                            context.original_labels[batch_idx, start:end],
                            label_row[start:],
                        ),
                        dim=0,
                    )
            hidden_rows.append(hidden_row)
            position_rows.append(position_row)
            if mask_rows is not None:
                mask_rows.append(mask_row)
            if label_rows is not None:
                label_rows.append(label_row)
        return self._stack_variable_rows(hidden_rows, position_rows, mask_rows, label_rows)

    def _query_indices_for_scoring(
        self,
        labels: torch.Tensor | None,
        hidden_states: torch.Tensor,
        context: HiDropForwardContext,
    ) -> list[torch.Tensor | None]:
        """Which query rows score the visual tokens.

        At inference that is the last prompt token; during training it is each
        answer-start boundary, so the selector is supervised by the tokens that
        actually consume the image.
        """
        device = hidden_states.device
        if labels is None:
            return [
                torch.tensor([hidden_states.shape[1] - 1], dtype=torch.long, device=device)
                if span is not None
                else None
                for span in context.spans
            ]
        query_indices_by_sample = []
        for batch_idx, span in enumerate(context.spans):
            if span is None:
                query_indices_by_sample.append(None)
                continue
            answer_positions = torch.where(labels[batch_idx] != IGNORE_INDEX)[0]
            if answer_positions.numel() == 0:
                # Released HiDrop preserves the complete sample when upstream
                # Vicuna preprocessing masks it after a tokenization mismatch.
                query_indices_by_sample.append(None)
                continue
            starts = []
            for idx in answer_positions.tolist():
                if idx > 0 and int(labels[batch_idx, idx - 1].item()) == IGNORE_INDEX:
                    starts.append(idx - 1)
            if not starts:
                # Match the released selector: a row without an answer-start
                # boundary is passed through without rank-and-drop.
                query_indices_by_sample.append(None)
                continue
            query_indices = torch.tensor(starts, dtype=torch.long, device=device)
            # LLaVA plain pretraining has no textual instruction boundary: the
            # token immediately before the caption is the final active visual
            # token. Vicuna-v1 finetuning queries occur after the visual span.
            earliest_query = span.start + max(0, context.active_visual_count - 1)
            if int(query_indices.min().item()) < earliest_query:
                if context.llava_next_layout is None:
                    raise ValueError(
                        "HiDrop query tokens must be at or after the final active visual token."
                    )
                # NeXT instruction rows can contain earlier dialogue before the
                # user supplies the image. Those causal assistant turns cannot
                # score a future image; retain their labels for LM loss but use
                # only post-image answer boundaries for visual selection.
                query_indices = query_indices[query_indices >= earliest_query]
                if query_indices.numel() == 0:
                    raise ValueError(
                        "HiDrop/LLaVA-NeXT training requires an assistant response after the image turn."
                    )
            query_indices_by_sample.append(query_indices)
        return query_indices_by_sample

    def _project_qk(
        self, decoder_layer: nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        attn = decoder_layer.self_attn
        if not all(hasattr(attn, name) for name in ("q_proj", "k_proj")):
            raise TypeError(
                "HiDrop q/k scoring requires LLaMA-style self attention with q_proj and k_proj."
            )
        # Upstream HiDrop ranks with the *next* decoder block and explicitly
        # detaches the selector path from the language-model graph.
        normed = decoder_layer.input_layernorm(hidden_states.detach().clone())
        input_shape = normed.shape[:-1]
        query_states = attn.q_proj(normed).view(*input_shape, -1, attn.head_dim).transpose(1, 2)
        key_states = attn.k_proj(normed).view(*input_shape, -1, attn.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(normed, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        scale = float(getattr(attn, "scaling", 1.0 / math.sqrt(float(attn.head_dim))))
        return query_states, key_states, scale

    def _visual_scores_from_qk(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        scale: float,
        *,
        batch_idx: int,
        query_indices: torch.Tensor,
        visual_start: int,
        visual_count: int,
        attention_mask: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        visual_end = int(visual_start + visual_count)
        q = query_states[batch_idx : batch_idx + 1, :, query_indices, :]
        k = key_states[batch_idx : batch_idx + 1]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        key_positions = torch.arange(key_states.shape[-2], device=query_indices.device)
        allowed = key_positions.unsqueeze(0) <= query_indices.unsqueeze(1)
        if attention_mask is not None:
            allowed = allowed & attention_mask[batch_idx].to(
                device=allowed.device, dtype=torch.bool
            ).unsqueeze(0)
        scores = scores.float().masked_fill(
            ~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min
        )
        weights = torch.softmax(scores, dim=-1).to(dtype)
        # Match upstream: normalize over the complete causal sequence first,
        # then slice the visual span and average heads and query tokens.
        return weights[..., visual_start:visual_end].mean(dim=(0, 1, 2))

    def _prune_visual_tokens(
        self,
        decoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        labels: torch.Tensor | None,
        context: HiDropForwardContext,
        rank_layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        visual_start = context.span.start
        visual_count = int(context.active_visual_count)
        layout = context.llava_next_layout
        if layout is None:
            keep_patch_k = self._compressor.config.keep_k_for_layer(rank_layer, visual_count)
            keep_total = keep_patch_k
        else:
            if context.resolved_schedule is None:
                raise RuntimeError("LLaVA-NeXT HiDrop context is missing its resolved schedule.")
            keep_patch_k = context.resolved_schedule.keep_patches_for_layer(
                rank_layer,
                context.active_patch_count,
            )
            keep_total = keep_patch_k + int(layout.newline_tokens)
        query_indices_by_sample = self._query_indices_for_scoring(labels, hidden_states, context)
        query_states, key_states, scale = self._project_qk(
            decoder_layer, hidden_states, position_ids
        )
        selector = DifferentiableTopK(
            lambda_=self._compressor.config.dtopk_lambda,
            normalize=self._compressor.config.dtopk_normalize,
            mask_threshold=self._compressor.config.dtopk_mask_threshold,
            debugger=self._compressor.debugger,
        )
        hidden_rows = []
        position_rows = []
        mask_rows = [] if attention_mask is not None else None
        label_rows = [] if labels is not None else None
        sources = []
        for batch_idx, span in enumerate(context.spans):
            current_mask = attention_mask[batch_idx] if attention_mask is not None else None
            valid_length = self._valid_row_length(hidden_states[batch_idx], current_mask)
            if span is None:
                hidden_rows.append(hidden_states[batch_idx, :valid_length])
                position_rows.append(position_ids[batch_idx, :valid_length])
                if mask_rows is not None:
                    mask_rows.append(current_mask[:valid_length])
                if label_rows is not None:
                    label_rows.append(labels[batch_idx, :valid_length])
                sources.append(
                    torch.zeros(
                        (keep_total, context.span.count),
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                )
                continue
            visual_start = int(span.start)
            query_indices = query_indices_by_sample[batch_idx]
            if query_indices is None:
                context.preserved_visual_rows.add(batch_idx)
                hidden_rows.append(hidden_states[batch_idx, :valid_length])
                position_rows.append(position_ids[batch_idx, :valid_length])
                if mask_rows is not None:
                    mask_rows.append(current_mask[:valid_length])
                if label_rows is not None:
                    label_rows.append(labels[batch_idx, :valid_length])
                sources.append(
                    torch.zeros(
                        (keep_total, context.span.count),
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                )
                continue
            scores = self._visual_scores_from_qk(
                query_states,
                key_states,
                scale,
                batch_idx=batch_idx,
                query_indices=query_indices,
                visual_start=visual_start,
                visual_count=visual_count,
                attention_mask=attention_mask,
                dtype=hidden_states.dtype,
            )
            if layout is None:
                keep_rel = selector.apply_k(scores, keep_patch_k).indices
            else:
                original_rel = (
                    position_ids[batch_idx, visual_start : visual_start + visual_count]
                    - int(span.start)
                ).to(torch.long)
                newline_original = torch.tensor(
                    layout.newline_indices,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                structural_mask = torch.isin(original_rel, newline_original)
                structural_rel = torch.where(structural_mask)[0]
                patch_rel = torch.where(~structural_mask)[0]
                if int(structural_rel.numel()) != int(layout.newline_tokens):
                    raise RuntimeError(
                        "HiDrop/LLaVA-NeXT lost or duplicated a structural newline before pruning: "
                        f"found {structural_rel.numel()}, expected {layout.newline_tokens}."
                    )
                if int(patch_rel.numel()) != int(context.active_patch_count):
                    raise RuntimeError(
                        "HiDrop/LLaVA-NeXT active patch bookkeeping differs from persistent positions: "
                        f"found {patch_rel.numel()}, recorded {context.active_patch_count}."
                    )
                patch_scores = scores.index_select(0, patch_rel)
                selected_patch_rel = selector.apply_k(patch_scores, keep_patch_k).indices
                keep_rel = torch.sort(
                    torch.cat((patch_rel.index_select(0, selected_patch_rel), structural_rel))
                ).values
                if int(keep_rel.numel()) != keep_total:
                    raise RuntimeError(
                        f"HiDrop/LLaVA-NeXT selected {keep_rel.numel()} total visual rows, "
                        f"expected {keep_total}."
                    )
            keep_abs = keep_rel + visual_start
            policy = torch.ones(valid_length, dtype=torch.bool, device=hidden_states.device)
            policy[visual_start : visual_start + visual_count] = False
            policy[keep_abs] = True
            selected_abs = torch.where(policy)[0]
            hidden_rows.append(
                hidden_states[batch_idx, :valid_length].index_select(0, selected_abs)
            )
            position_rows.append(
                position_ids[batch_idx, :valid_length].index_select(0, selected_abs)
            )
            if mask_rows is not None:
                mask_rows.append(current_mask[:valid_length].index_select(0, selected_abs))
            if label_rows is not None:
                label_rows.append(labels[batch_idx, :valid_length].index_select(0, selected_abs))
            # Persistent positions make the surviving tokens self-describing: a
            # kept token's original visual index is its position minus the span
            # start, however many stages it has already survived.
            source = torch.zeros(
                (keep_total, context.span.count),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            source_indices = (
                position_rows[-1][visual_start : visual_start + keep_total] - span.start
            ).to(torch.long)
            valid = (source_indices >= 0) & (source_indices < context.span.count)
            if bool(valid.any().item()):
                source[
                    torch.arange(keep_total, device=hidden_states.device)[valid],
                    source_indices[valid],
                ] = 1
            sources.append(source)

        hidden_states, attention_mask, position_ids, labels = self._stack_variable_rows(
            hidden_rows,
            position_rows,
            mask_rows,
            label_rows,
        )
        context.pruning_events.append(
            {
                "layer": int(rank_layer),
                "visual_tokens_before": int(visual_count),
                "patch_tokens_before": int(context.active_patch_count),
                "structural_tokens_before": int(visual_count - context.active_patch_count),
                "keep_k": int(keep_patch_k),
                "patch_keep_k": int(keep_patch_k),
                "structural_tokens_after": int(layout.newline_tokens if layout is not None else 0),
                "visual_tokens_after": int(keep_total),
                "batch_size": int(hidden_states.shape[0]),
                "image_rows": int(sum(span is not None for span in context.spans)),
            }
        )
        context.active_visual_count = int(keep_total)
        context.active_patch_count = int(keep_patch_k)
        # `llmoe.viz` renders one `[T_cur, T_orig]` matrix per layer for a single
        # image (see `viz.make_visualization`), where the released sidecar kept
        # the whole stacked batch. Inference is batch-size-1, so row 0 is the
        # image being visualized.
        self._compressor._source_history.append(sources[0].detach().cpu())
        self._compressor.debugger.log(
            f"layer={rank_layer} visual_before={visual_count} patch_keep={keep_patch_k} "
            f"visual_after={keep_total}"
        )
        return hidden_states, attention_mask, position_ids, labels
