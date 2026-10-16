"""The HiDrop decoder for Qwen2.5-VL: a drop-in replacement for `language_model`.

`HiDropCompressor.install()` (and the training wrapper in `hidrop_tr.py`) swaps
`vlm.model.model.language_model` for a `HiDropQwenModel` wrapping the original
`Qwen2_5_VLTextModel`. The wrapper reuses the original embeddings, decoder layers,
norm, and rotary embedding -- it only changes *which tokens are in the sequence* at
each layer, exactly as `hidrop_llama.py` does on LLaVA-1.5:

1. **Late injection (3.1).** The visual span is removed before layer 1, so layers
   1..`late_entry_layer - 1` run text-only. At `late_entry_layer` the *original*
   merged visual embeddings are spliced back in at their original positions.
2. **Concave pyramid pruning (3.2).** After each layer in `layer_list`, the *next*
   decoder block's q/k projections score the visual tokens, and DTop-K keeps
   `keep_k_for_layer(...)` of them.
3. **Early exit (3.1).** Every remaining visual token is removed before
   `early_exit_layer`.

Two invariants make this work with a KV cache: position IDs travel with their tokens
and are never renumbered after a removal, and pruning only ever runs on a prefill
(`seq_len > 1`), so decoding just extends each layer's already-pruned cache.

What Qwen2.5-VL changes
-----------------------
* **M-RoPE, not 1-D RoPE.** Position ids arrive as ``[3, B, L]`` -- the temporal,
  height and width grid coordinates -- or as ``[4, B, L]`` on transformers 5.x, which
  prepends a plain running row used for packed-sequence detection rather than for
  RoPE. Rows 1-3 travel with their tokens and are never renumbered, which is HiDrop's
  persistent position encoding (3.3) expressed in Qwen's coordinates; the leading
  bookkeeping row *is* recompacted, because ``find_packed_sequence_indices`` reads any
  gap > 1 in it as a sequence boundary and would build a block-diagonal mask over the
  prompt. `vision_zip.py` and `nuwa/` hit the identical issue and resolve it the same
  way. Rotary tables are rebuilt from the surviving coordinates after every mutation,
  and the selector's q/k scoring goes through ``apply_multimodal_rotary_pos_emb`` with
  the config's ``mrope_section`` instead of LLaMA's ``apply_rotary_pos_emb``.
* **Per-row budgets.** The NaViT encoder emits a different visual-token count per
  image, so each stage's budget is a *fraction* of that row's own span (see
  `config.py`). Where `hidrop_llama.py` cuts every row in a batch to the same
  integer `keep_k`, this wrapper resolves `keep_k` per row, so a batch of differently
  sized images prunes each to its own count and the rows are re-padded afterwards.
* **Depth-relative layers.** The released layer indices are written against LLaVA's
  32-layer decoder and are rescaled to Qwen's own depth (28 on the 7B).
* **Attribution is tracked, not inferred.** `hidrop_llama.py` recovers a surviving
  token's original visual index from ``position - span.start``, which works only
  because LLaVA positions are a linear counter. Qwen positions are grid coordinates,
  so this wrapper carries an explicit `visual_origin` index vector through every
  stage instead.

Attention masking
-----------------
HiDrop leaves each decoder layer holding a *different* number of cached visual tokens,
so no single model-wide past length describes the stack and the mask
`Qwen2_5_VLTextModel` builds once up front is wrong below the first cut. This wrapper
therefore builds the mask itself, per layer and per backend:

* ``flash_attention_2`` takes the 2-D padding mask and applies causality itself. This
  is the training path (the released recipe is FA2-only), where right-padded batches
  make a padding mask mandatory.
* ``sdpa`` takes ``None`` when nothing is padded -- batch-size-1 inference -- because
  the backend's own ``is_causal`` is then exactly the causal mask over the current
  sequence, and at decode the single query attends to that layer's whole cache. With
  padding present it gets the explicit 4-D mask.
* ``eager`` applies no causality of its own, so it always gets the 4-D mask, built
  from the layer's *own* ``past_key_values.get_seq_length(layer_idx)``.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from llmoe.compressor.hidrop.config import DifferentiableTopK

try:  # The M-RoPE application and GQA expansion the selector's q/k scoring needs.
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb as _APPLY_MROPE,
        repeat_kv as _REPEAT_KV,
    )
except ImportError:  # pragma: no cover - degrade loudly if the path moves
    _APPLY_MROPE = None
    _REPEAT_KV = None


IGNORE_INDEX = -100
SUPPORTED_ATTN_IMPLEMENTATIONS = ("sdpa", "flash_attention_2", "eager")


@dataclass
class QwenHiDropModelOutputWithPast(BaseModelOutputWithPast):
    # Pruning removes label positions along with their tokens, so a training wrapper
    # has to read the surviving labels back out of the decoder.
    updated_labels: torch.LongTensor | None = None


@dataclass
class QwenVisualTokenSpan:
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start


@dataclass
class HiDropQwenForwardContext:
    """What one prefill needs to remember across the layer loop."""

    span: QwenVisualTokenSpan
    spans: tuple[QwenVisualTokenSpan | None, ...]
    original_hidden_states: torch.Tensor
    original_mrope_ids: torch.Tensor
    original_attention_mask: torch.Tensor | None
    original_labels: torch.Tensor | None
    # Per row: how many visual tokens are live, and which original visual index each
    # live row came from. Qwen positions are grid coordinates, so attribution cannot be
    # recovered from position arithmetic the way it can on LLaVA.
    active_visual_counts: list[int] = field(default_factory=list)
    visual_origin: list[torch.Tensor | None] = field(default_factory=list)
    preserved_visual_rows: set[int] = field(default_factory=set)
    pruning_events: list[dict[str, int]] = field(default_factory=list)

    @property
    def active_visual_count(self) -> int:
        """The live count, for diagnostics and the single-image visualization path."""
        return max(self.active_visual_counts) if self.active_visual_counts else 0


def resolve_mrope_section(config: Any) -> list[int]:
    """The channel split M-RoPE applies per grid axis, across transformers releases."""
    for attribute in ("rope_parameters", "rope_scaling"):
        parameters = getattr(config, attribute, None)
        if isinstance(parameters, dict) and parameters.get("mrope_section") is not None:
            return list(parameters["mrope_section"])
    raise ValueError(
        "HiDrop could not read `mrope_section` from the Qwen2.5-VL text config "
        "(looked at `rope_parameters` and `rope_scaling`); the selector cannot apply "
        "M-RoPE without it."
    )


class HiDropQwenModel(nn.Module):
    def __init__(self, base_model: nn.Module, compressor: Any) -> None:
        super().__init__()
        self.config = base_model.config
        self.embed_tokens = base_model.embed_tokens
        self.layers = base_model.layers
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb
        self.gradient_checkpointing = getattr(base_model, "gradient_checkpointing", False)
        self._mrope_section: list[int] = resolve_mrope_section(self.config)
        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is not None and any(kind != "full_attention" for kind in layer_types):
            raise NotImplementedError(
                "HiDrop on Qwen2.5-VL builds its own per-layer causal mask and does not apply a "
                f"sliding window; this checkpoint declares layer types {sorted(set(layer_types))}. "
                "Run a full-attention Qwen2.5-VL config."
            )
        # A freshly constructed nn.Module defaults to training=True, so without this the
        # wrapper would report training mode inside an `.eval()` backbone -- enabling
        # dropout in the layers it re-registers. The swap must not change the mode.
        self.train(base_model.training)
        # Kept off the module registry: the wrapper already re-registers every submodule
        # above, and registering the original too would double-count its parameters.
        object.__setattr__(self, "_base_model", base_model)
        object.__setattr__(self, "_compressor", compressor)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    # --------------------------------------------------------------------- forward

    def forward(self, input_ids: torch.LongTensor | None = None,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None, past_key_values: Any | None = None,
                inputs_embeds: torch.FloatTensor | None = None, use_cache: bool | None = None,
                output_attentions: bool | None = None, output_hidden_states: bool | None = None,
                return_dict: bool | None = None, cache_position: torch.LongTensor | None = None,
                labels: torch.LongTensor | None = None,
                **kwargs: Any) -> QwenHiDropModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        compressor = self._compressor
        attn_implementation = getattr(self.config, "_attn_implementation", None)
        if attn_implementation not in SUPPORTED_ATTN_IMPLEMENTATIONS:
            raise ValueError(
                f"HiDrop does not support attention implementation {attn_implementation!r}; "
                f"supported: {SUPPORTED_ATTN_IMPLEMENTATIONS}."
            )
        if output_attentions:
            raise ValueError(
                "HiDrop does not support output_attentions=True; token scoring uses the separate "
                "q/k selector."
            )
        # `cache_position` describes the unpruned sequence and every layer here holds a
        # different number of cached tokens, so no single vector is correct below the
        # first cut. `DynamicCache.update` concatenates and does not read it, so it is
        # dropped rather than recomputed; a static cache cannot represent a ragged stack
        # in the first place.
        del cache_position
        _ = return_dict  # accepted for signature compatibility; the output is a dataclass
        output_hidden_states = bool(
            getattr(self.config, "output_hidden_states", False)
            if output_hidden_states is None else output_hidden_states
        )
        use_cache = bool(getattr(self.config, "use_cache", True) if use_cache is None else use_cache)
        if self.training or labels is not None:
            use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        batch_size, seq_len = inputs_embeds.shape[:2]
        if batch_size != 1 and not compressor.config.batch_size_supported:
            raise NotImplementedError(
                f"HiDrop on Qwen2.5-VL supports batch_size=1 at inference, got {batch_size}. The "
                f"NaViT encoder gives each image a different visual-token count, so a padded "
                f"batch cannot share one keep-set at generation time."
            )
        padding_mask: torch.Tensor | None = None
        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.ndim != 2:
                raise NotImplementedError(
                    f"HiDrop on Qwen2.5-VL expects the 2-D padding mask or None, got "
                    f"{tuple(attention_mask.shape)}."
                )
            padding_mask = attention_mask.to(torch.bool)
            if bool(padding_mask.all().item()):
                padding_mask = None  # nothing padded: the causal default is already right

        mrope_ids, text_ids = self._split_position_ids(
            position_ids, inputs_embeds, past_key_values, batch_size, seq_len
        )

        hidden_states = inputs_embeds
        # Only a prefill carries a visual span; a decode step (seq_len == 1) extends the
        # caches the prefill already pruned.
        context = (
            self._build_context(hidden_states, input_ids, mrope_ids, padding_mask, labels)
            if seq_len > 1 else None
        )
        active = context is not None and compressor.config.enabled
        updated_labels = labels
        if active:
            hidden_states, mrope_ids, text_ids, padding_mask, updated_labels = self._remove_visual_span(
                hidden_states, mrope_ids, text_ids, padding_mask, labels, context.spans,
                [span.count if span is not None else 0 for span in context.spans],
            )

        all_hidden_states: list[torch.Tensor] | None = [] if output_hidden_states else None
        fired_layers: list[int] = []
        visual_counts: list[int] = []

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            rank_layer = int(layer_idx + 1)
            if active and rank_layer == compressor.config.late_entry_layer:
                hidden_states, mrope_ids, text_ids, padding_mask, updated_labels = \
                    self._late_inject_visual_span(
                        hidden_states, mrope_ids, text_ids, padding_mask, updated_labels, context
                    )
                context.active_visual_counts = [
                    span.count if span is not None else 0 for span in context.spans
                ]
                context.visual_origin = [
                    torch.arange(span.count, device=hidden_states.device, dtype=torch.long)
                    if span is not None else None
                    for span in context.spans
                ]
            if active and rank_layer == compressor.config.early_exit_layer \
                    and context.active_visual_count > 0:
                removal_counts = [
                    span.count if (span is not None and index in context.preserved_visual_rows)
                    else context.active_visual_counts[index]
                    for index, span in enumerate(context.spans)
                ]
                hidden_states, mrope_ids, text_ids, padding_mask, updated_labels = \
                    self._remove_visual_span(
                        hidden_states, mrope_ids, text_ids, padding_mask, updated_labels,
                        context.spans, removal_counts,
                    )
                context.active_visual_counts = [0] * len(context.spans)
                context.visual_origin = [None] * len(context.spans)

            visual_counts.append(int(context.active_visual_count if active else 0))
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            layer_mask = self._layer_attention_mask(
                hidden_states, padding_mask, past_key_values, layer_idx, attn_implementation
            )
            position_embeddings = self.rotary_emb(hidden_states, mrope_ids)
            if self.training and self.gradient_checkpointing:
                layer_kwargs = dict(kwargs)

                def checkpointed_layer_forward(
                    layer_hidden_states: torch.Tensor,
                    *,
                    checkpointed_decoder_layer: nn.Module = decoder_layer,
                    checkpointed_attention_mask: torch.Tensor | None = layer_mask,
                    checkpointed_text_ids: torch.Tensor | None = text_ids,
                    checkpointed_position_embeddings: tuple[torch.Tensor, torch.Tensor] = position_embeddings,
                    checkpointed_kwargs: dict[str, Any] = layer_kwargs,
                ) -> torch.Tensor:
                    return self._run_decoder_layer(
                        checkpointed_decoder_layer,
                        layer_hidden_states,
                        attention_mask=checkpointed_attention_mask,
                        text_ids=checkpointed_text_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=checkpointed_position_embeddings,
                        **checkpointed_kwargs,
                    )

                hidden_states = checkpoint(
                    checkpointed_layer_forward,
                    hidden_states,
                    # Match the released LLaVA/HiDrop checkpoint path. The non-reentrant
                    # engine performs saved-tensor metadata checks that are incompatible
                    # with ZeRO-3 parameter partitioning during recomputation.
                    use_reentrant=True,
                )
            else:
                hidden_states = self._run_decoder_layer(
                    decoder_layer,
                    hidden_states,
                    attention_mask=layer_mask,
                    text_ids=text_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            if (
                self.training
                and compressor.config.enabled
                and context is None
                and hidden_states.shape[1] > 1
                and rank_layer in compressor.config.layer_list
            ):
                # ZeRO-3 gathers parameters when a module is called, so every distributed
                # rank must touch the selector's next-layer norm and q/k projections in
                # the same order. A text-only rank has no visual context to score, while
                # another rank in the same step can contain images. Touch a single token
                # to preserve the collective order without paying for a full-sequence
                # selector that cannot produce a visual-token decision.
                if rank_layer >= len(self.layers):
                    raise ValueError(
                        f"HiDrop selector synchronization after layer {rank_layer} requires "
                        f"decoder layer index {rank_layer}, but that next layer does not exist."
                    )
                self._project_qk(self.layers[rank_layer], hidden_states[:, :1], mrope_ids[..., :1])
            if active and context.active_visual_count > 0 \
                    and rank_layer in compressor.config.layer_list:
                if rank_layer >= len(self.layers):
                    raise ValueError(
                        f"HiDrop pruning after layer {rank_layer} requires decoder layer index "
                        f"{rank_layer} for upstream-compatible q/k scoring, but that next layer "
                        f"does not exist."
                    )
                hidden_states, mrope_ids, text_ids, padding_mask, updated_labels = \
                    self._prune_visual_tokens(
                        self.layers[rank_layer], hidden_states, mrope_ids, text_ids, padding_mask,
                        updated_labels, context, rank_layer,
                    )
                fired_layers.append(rank_layer)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        if active:
            missing = [layer for layer in compressor.config.layer_list if layer not in fired_layers]
            if missing:
                raise RuntimeError(
                    f"HiDrop prefill completed without firing all configured pruning layers: "
                    f"missing={missing}, fired={fired_layers}."
                )
            compressor.last_context = context
            compressor.last_visual_token_counts_by_layer = visual_counts
            compressor.pruning_history.extend(context.pruning_events)
            compressor.debugger.log(f"visual_token_counts_by_layer={visual_counts}")

        return QwenHiDropModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=None,
            updated_labels=updated_labels,
        )

    def _run_decoder_layer(self, decoder_layer: nn.Module, hidden_states: torch.Tensor, *,
                           attention_mask: torch.Tensor | None, text_ids: torch.Tensor | None,
                           past_key_values: Any | None, use_cache: bool,
                           position_embeddings: tuple[torch.Tensor, torch.Tensor],
                           **kwargs: Any) -> torch.Tensor:
        output = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return output[0] if isinstance(output, tuple) else output

    # ------------------------------------------------------------------- positions

    @staticmethod
    def _split_position_ids(position_ids: Any, inputs_embeds: torch.Tensor,
                            past_key_values: Any | None, batch_size: int, seq_len: int
                            ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Separate M-RoPE grid coordinates from the packed-sequence bookkeeping row.

        Mirrors `Qwen2_5_VLTextModel.forward`: a ``[4, B, L]`` tensor carries the plain
        running row first and the ``(t, h, w)`` grid rows after it, while a ``[3, B, L]``
        tensor is grid rows only and signals that no packing metadata was supplied.

        :return: ``(mrope_ids [3, B, L], text_ids [B, L] or None)``.
        """
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            flat = torch.arange(seq_len, device=inputs_embeds.device) + int(past_seen)
            return flat.view(1, 1, -1).expand(3, batch_size, -1), None
        if position_ids.ndim == 2:
            return position_ids[None, ...].expand(3, position_ids.shape[0], -1), None
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            return position_ids[1:], position_ids[0]
        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            return position_ids, None
        raise ValueError(
            f"HiDrop expected Qwen position ids shaped [3, B, L] or [4, B, L], got "
            f"{tuple(position_ids.shape)}."
        )

    @staticmethod
    def _recompact_text_ids(text_ids: torch.Tensor | None, batch: int,
                            length: int) -> torch.Tensor | None:
        """Renumber the packing row over the surviving tokens.

        Only this row is renumbered. It is not a RoPE input -- the grid rows are, and
        they keep their original coordinates -- but `find_packed_sequence_indices` reads
        a gap greater than one in it as a sequence boundary, which after a removal would
        split the prompt into a block-diagonal mask.
        """
        if text_ids is None:
            return None
        return torch.arange(length, device=text_ids.device, dtype=text_ids.dtype).expand(batch, -1)

    def _layer_attention_mask(self, hidden_states: torch.Tensor,
                              padding_mask: torch.Tensor | None, past_key_values: Any | None,
                              layer_idx: int, attn_implementation: str) -> torch.Tensor | None:
        """The mask this layer needs, sized to the cache *this layer* actually holds.

        See the module docstring: FA2 takes the 2-D padding mask and applies causality
        itself, SDPA can use its own ``is_causal`` when nothing is padded, and eager
        always needs the explicit 4-D mask.
        """
        if attn_implementation == "flash_attention_2":
            return padding_mask
        if attn_implementation == "sdpa" and padding_mask is None:
            return None
        batch_size, query_length = hidden_states.shape[:2]
        past_length = (
            int(past_key_values.get_seq_length(layer_idx)) if past_key_values is not None else 0
        )
        key_length = past_length + query_length
        query_positions = torch.arange(
            past_length, key_length, device=hidden_states.device
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=hidden_states.device).unsqueeze(0)
        minimum = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.zeros(
            (query_length, key_length), dtype=hidden_states.dtype, device=hidden_states.device
        ).masked_fill(key_positions > query_positions, minimum)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        if padding_mask is not None:
            key_mask = padding_mask.to(device=hidden_states.device, dtype=torch.bool)
            if past_length:
                key_mask = torch.cat(
                    (
                        torch.ones((batch_size, past_length), dtype=torch.bool,
                                   device=hidden_states.device),
                        key_mask,
                    ),
                    dim=-1,
                )
            causal_mask = causal_mask.masked_fill(~key_mask[:, None, None, :], minimum)
        return causal_mask

    # ---------------------------------------------------------------- span surgery

    def _build_context(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None,
                       mrope_ids: torch.Tensor, padding_mask: torch.Tensor | None,
                       labels: torch.Tensor | None) -> HiDropQwenForwardContext | None:
        """Locate this prefill's visual spans and snapshot what late injection restores."""
        compressor = self._compressor
        ids = input_ids if input_ids is not None else compressor._latest_input_ids
        if ids is None:
            return None
        if ids.ndim != 2 or ids.shape[0] != hidden_states.shape[0] \
                or ids.shape[1] != hidden_states.shape[1]:
            return None
        spans: list[QwenVisualTokenSpan | None] = []
        image_rows = 0
        for batch_idx in range(int(ids.shape[0])):
            positions = torch.where(ids[batch_idx] == int(compressor._image_token_id))[0]
            if positions.numel() == 0:
                spans.append(None)
                continue
            image_rows += 1
            start = int(positions[0].item())
            end = int(positions[-1].item()) + 1
            if int(positions.numel()) != end - start:
                raise NotImplementedError(
                    f"HiDrop on Qwen2.5-VL expects one contiguous visual-token span per prompt; "
                    f"row {batch_idx} carries {int(positions.numel())} image tokens spread from "
                    f"{start} to {end}."
                )
            spans.append(QwenVisualTokenSpan(start=start, end=end))
        if image_rows == 0:
            return None  # text-only batch: nothing to schedule
        if labels is not None and labels.shape != ids.shape:
            raise ValueError(
                f"HiDrop labels must match input_ids before pruning, got labels "
                f"{tuple(labels.shape)} and input_ids {tuple(ids.shape)}."
            )
        span = next(item for item in spans if item is not None)
        compressor.last_context = None
        compressor.last_visual_token_counts_by_layer = []
        compressor._source_history.clear()
        compressor.debugger.log(
            f"visual_spans={[(item.start, item.end) if item is not None else None for item in spans]}"
        )
        return HiDropQwenForwardContext(
            span=span,
            spans=tuple(spans),
            original_hidden_states=hidden_states.clone(),
            original_mrope_ids=mrope_ids.clone(),
            original_attention_mask=padding_mask.clone() if padding_mask is not None else None,
            original_labels=labels.clone() if labels is not None else None,
            active_visual_counts=[0] * len(spans),
            visual_origin=[None] * len(spans),
        )

    @staticmethod
    def _valid_row_length(row_length: int, mask_row: torch.Tensor | None) -> int:
        if mask_row is None:
            return int(row_length)
        return int(mask_row.to(torch.long).sum().item())

    @staticmethod
    def _stack_variable_rows(hidden_rows: list[torch.Tensor], mrope_rows: list[torch.Tensor],
                             mask_rows: list[torch.Tensor] | None,
                             label_rows: list[torch.Tensor] | None, text_ids: torch.Tensor | None
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
                                        torch.Tensor | None, torch.Tensor | None]:
        """Re-pad rows to a rectangle after each row lost a different token count.

        Qwen's per-image token counts make this the common case rather than the
        exception: two images in one batch prune to two different keep counts.
        """
        max_length = max(int(row.shape[0]) for row in hidden_rows)

        def pad_hidden(row: torch.Tensor) -> torch.Tensor:
            pad = max_length - int(row.shape[0])
            if pad == 0:
                return row
            return torch.cat((row, row.new_zeros((pad, row.shape[-1]))), dim=0)

        def pad_mrope(row: torch.Tensor) -> torch.Tensor:  # [3, L_i]
            pad = max_length - int(row.shape[-1])
            if pad == 0:
                return row
            return torch.cat((row, row.new_zeros((row.shape[0], pad))), dim=-1)

        def pad_vector(row: torch.Tensor, value: int | bool) -> torch.Tensor:
            pad = max_length - int(row.shape[0])
            if pad == 0:
                return row
            return torch.cat((row, row.new_full((pad,), value)), dim=0)

        hidden_states = torch.stack([pad_hidden(row) for row in hidden_rows], dim=0)
        mrope_ids = torch.stack([pad_mrope(row) for row in mrope_rows], dim=1)
        ragged = any(int(row.shape[0]) != max_length for row in hidden_rows)
        if mask_rows is None and ragged:
            mask_rows = [
                torch.ones(row.shape[0], dtype=torch.bool, device=row.device) for row in hidden_rows
            ]
        attention_mask = (
            torch.stack([pad_vector(row.to(torch.bool), False) for row in mask_rows], dim=0)
            if mask_rows is not None else None
        )
        labels = (
            torch.stack([pad_vector(row, IGNORE_INDEX) for row in label_rows], dim=0)
            if label_rows is not None else None
        )
        new_text_ids = HiDropQwenModel._recompact_text_ids(
            text_ids, len(hidden_rows), max_length
        )
        return hidden_states, mrope_ids, new_text_ids, attention_mask, labels

    def _row_views(self, hidden_states: torch.Tensor, mrope_ids: torch.Tensor,
                   padding_mask: torch.Tensor | None, labels: torch.Tensor | None, batch_idx: int
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """This row's live (unpadded) slice of each tensor."""
        mask_row = padding_mask[batch_idx] if padding_mask is not None else None
        valid = self._valid_row_length(int(hidden_states.shape[1]), mask_row)
        return (
            hidden_states[batch_idx, :valid],
            mrope_ids[:, batch_idx, :valid],
            mask_row[:valid] if mask_row is not None else None,
            labels[batch_idx, :valid] if labels is not None else None,
        )

    def _remove_visual_span(self, hidden_states: torch.Tensor, mrope_ids: torch.Tensor,
                            text_ids: torch.Tensor | None, padding_mask: torch.Tensor | None,
                            labels: torch.Tensor | None,
                            spans: tuple[QwenVisualTokenSpan | None, ...], counts: list[int]
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
                                       torch.Tensor | None, torch.Tensor | None]:
        """Cut `counts[row]` tokens out of each row's visual span (late entry, early exit)."""
        if not any(count > 0 for count in counts):
            return hidden_states, mrope_ids, text_ids, padding_mask, labels
        hidden_rows, mrope_rows = [], []
        mask_rows = [] if padding_mask is not None else None
        label_rows = [] if labels is not None else None
        for batch_idx, span in enumerate(spans):
            hidden_row, mrope_row, mask_row, label_row = self._row_views(
                hidden_states, mrope_ids, padding_mask, labels, batch_idx
            )
            if span is not None and counts[batch_idx] > 0:
                start = int(span.start)
                end = start + int(counts[batch_idx])
                hidden_row = torch.cat((hidden_row[:start], hidden_row[end:]), dim=0)
                mrope_row = torch.cat((mrope_row[:, :start], mrope_row[:, end:]), dim=-1)
                if mask_row is not None:
                    mask_row = torch.cat((mask_row[:start], mask_row[end:]), dim=0)
                if label_row is not None:
                    label_row = torch.cat((label_row[:start], label_row[end:]), dim=0)
            hidden_rows.append(hidden_row)
            mrope_rows.append(mrope_row)
            if mask_rows is not None:
                mask_rows.append(mask_row)
            if label_rows is not None:
                label_rows.append(label_row)
        return self._stack_variable_rows(hidden_rows, mrope_rows, mask_rows, label_rows, text_ids)

    def _late_inject_visual_span(self, hidden_states: torch.Tensor, mrope_ids: torch.Tensor,
                                 text_ids: torch.Tensor | None,
                                 padding_mask: torch.Tensor | None, labels: torch.Tensor | None,
                                 context: HiDropQwenForwardContext
                                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
                                            torch.Tensor | None, torch.Tensor | None]:
        """Splice the *original* merged visual embeddings back in at `late_entry_layer`.

        The visual tokens enter unprocessed -- they never went through the shallow layers
        -- while the text around them carries the state those layers built. Their M-RoPE
        coordinates are restored verbatim, so the reinserted patches land back on their
        true grid positions and the surrounding text keeps the phase it started with.
        """
        hidden_rows, mrope_rows = [], []
        mask_rows = [] if padding_mask is not None else None
        label_rows = [] if labels is not None else None
        for batch_idx, span in enumerate(context.spans):
            hidden_row, mrope_row, mask_row, label_row = self._row_views(
                hidden_states, mrope_ids, padding_mask, labels, batch_idx
            )
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
                mrope_row = torch.cat(
                    (
                        mrope_row[:, :start],
                        context.original_mrope_ids[:, batch_idx, start:end],
                        mrope_row[:, start:],
                    ),
                    dim=-1,
                )
                if mask_row is not None:
                    if context.original_attention_mask is None:
                        # The visual span was unpadded when it was captured, so it comes
                        # back as attendable; only rows that lost tokens are padded.
                        restored = torch.ones(end - start, dtype=mask_row.dtype,
                                              device=mask_row.device)
                    else:
                        restored = context.original_attention_mask[batch_idx, start:end]
                    mask_row = torch.cat((mask_row[:start], restored, mask_row[start:]), dim=0)
                if label_row is not None:
                    if context.original_labels is None:
                        raise ValueError(
                            "HiDrop cannot restore labels that were not saved at prefill."
                        )
                    mask_labels = context.original_labels[batch_idx, start:end]
                    label_row = torch.cat((label_row[:start], mask_labels, label_row[start:]), dim=0)
            hidden_rows.append(hidden_row)
            mrope_rows.append(mrope_row)
            if mask_rows is not None:
                mask_rows.append(mask_row)
            if label_rows is not None:
                label_rows.append(label_row)
        return self._stack_variable_rows(hidden_rows, mrope_rows, mask_rows, label_rows, text_ids)

    # -------------------------------------------------------------------- selector

    def _query_indices_for_scoring(self, labels: torch.Tensor | None, hidden_states: torch.Tensor,
                                   padding_mask: torch.Tensor | None,
                                   context: HiDropQwenForwardContext) -> list[torch.Tensor | None]:
        """Which query rows score the visual tokens.

        At inference that is the last real token of the row; during training it is each
        answer-start boundary, so the selector is supervised by the tokens that actually
        consume the image. Mirrors `hidrop_llama._query_indices_for_scoring`.
        """
        device = hidden_states.device
        if labels is None:
            indices: list[torch.Tensor | None] = []
            for batch_idx, span in enumerate(context.spans):
                if span is None:
                    indices.append(None)
                    continue
                mask_row = padding_mask[batch_idx] if padding_mask is not None else None
                last = self._valid_row_length(int(hidden_states.shape[1]), mask_row) - 1
                indices.append(torch.tensor([last], dtype=torch.long, device=device))
            return indices
        query_indices_by_sample: list[torch.Tensor | None] = []
        for batch_idx, span in enumerate(context.spans):
            if span is None:
                query_indices_by_sample.append(None)
                continue
            answer_positions = torch.where(labels[batch_idx] != IGNORE_INDEX)[0]
            if answer_positions.numel() == 0:
                # Released HiDrop preserves the complete sample when upstream
                # preprocessing masks it after a tokenization mismatch.
                query_indices_by_sample.append(None)
                continue
            starts = [
                idx - 1 for idx in answer_positions.tolist()
                if idx > 0 and int(labels[batch_idx, idx - 1].item()) == IGNORE_INDEX
            ]
            if not starts:
                # Match the released selector: a row without an answer-start boundary is
                # passed through without rank-and-drop.
                query_indices_by_sample.append(None)
                continue
            query_indices = torch.tensor(starts, dtype=torch.long, device=device)
            earliest_query = span.start + max(0, context.active_visual_counts[batch_idx] - 1)
            if int(query_indices.min().item()) < earliest_query:
                raise ValueError(
                    "HiDrop query tokens must be at or after the final active visual token."
                )
            query_indices_by_sample.append(query_indices)
        return query_indices_by_sample

    def _project_qk(self, decoder_layer: nn.Module, hidden_states: torch.Tensor,
                    mrope_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Score q/k with the *next* decoder block, detached from the LM graph."""
        if _APPLY_MROPE is None or _REPEAT_KV is None:
            raise RuntimeError(
                "Could not import `apply_multimodal_rotary_pos_emb`/`repeat_kv` from "
                "transformers; HiDrop cannot score Qwen visual tokens without them."
            )
        attn = decoder_layer.self_attn
        if not all(hasattr(attn, name) for name in ("q_proj", "k_proj")):
            raise TypeError(
                "HiDrop q/k scoring requires Qwen2.5-VL self attention with q_proj and k_proj."
            )
        # Upstream HiDrop ranks with the next decoder block and explicitly detaches the
        # selector path from the language-model graph.
        normed = decoder_layer.input_layernorm(hidden_states.detach().clone())
        input_shape = normed.shape[:-1]
        query_states = attn.q_proj(normed).view(*input_shape, -1, attn.head_dim).transpose(1, 2)
        key_states = attn.k_proj(normed).view(*input_shape, -1, attn.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(normed, mrope_ids)
        query_states, key_states = _APPLY_MROPE(
            query_states, key_states, cos, sin, self._mrope_section
        )
        key_states = _REPEAT_KV(key_states, attn.num_key_value_groups)
        scale = float(getattr(attn, "scaling", float(attn.head_dim) ** -0.5))
        return query_states, key_states, scale

    @staticmethod
    def _visual_scores_from_qk(query_states: torch.Tensor, key_states: torch.Tensor, scale: float,
                               *, batch_idx: int, query_indices: torch.Tensor, visual_start: int,
                               visual_count: int, padding_mask: torch.Tensor | None,
                               dtype: torch.dtype) -> torch.Tensor:
        """Per-visual-token attention received, normalised over the whole causal row."""
        visual_end = int(visual_start) + int(visual_count)
        q = query_states[batch_idx: batch_idx + 1, :, query_indices, :]
        k = key_states[batch_idx: batch_idx + 1]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        key_positions = torch.arange(key_states.shape[-2], device=query_indices.device)
        allowed = key_positions.unsqueeze(0) <= query_indices.unsqueeze(1)
        if padding_mask is not None:
            allowed = allowed & padding_mask[batch_idx].to(
                device=allowed.device, dtype=torch.bool
            ).unsqueeze(0)
        scores = scores.float().masked_fill(
            ~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min
        )
        weights = torch.softmax(scores, dim=-1).to(dtype)
        # Match upstream: normalize over the complete causal sequence first, then slice
        # the visual span and average heads and query tokens.
        return weights[..., visual_start:visual_end].mean(dim=(0, 1, 2))

    def _prune_visual_tokens(self, decoder_layer: nn.Module, hidden_states: torch.Tensor,
                             mrope_ids: torch.Tensor, text_ids: torch.Tensor | None,
                             padding_mask: torch.Tensor | None, labels: torch.Tensor | None,
                             context: HiDropQwenForwardContext, rank_layer: int
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
                                        torch.Tensor | None, torch.Tensor | None]:
        compressor = self._compressor
        query_indices_by_sample = self._query_indices_for_scoring(
            labels, hidden_states, padding_mask, context
        )
        query_states, key_states, scale = self._project_qk(decoder_layer, hidden_states, mrope_ids)
        selector = DifferentiableTopK(
            lambda_=compressor.config.dtopk_lambda,
            normalize=compressor.config.dtopk_normalize,
            mask_threshold=compressor.config.dtopk_mask_threshold,
            debugger=compressor.debugger,
        )
        hidden_rows, mrope_rows = [], []
        mask_rows = [] if padding_mask is not None else None
        label_rows = [] if labels is not None else None
        sources: list[torch.Tensor | None] = []
        keep_counts: list[int] = []
        before_counts = list(context.active_visual_counts)
        before_counts_max = max(before_counts) if before_counts else 0

        for batch_idx, span in enumerate(context.spans):
            hidden_row, mrope_row, mask_row, label_row = self._row_views(
                hidden_states, mrope_ids, padding_mask, labels, batch_idx
            )
            visual_count = int(context.active_visual_counts[batch_idx])
            query_indices = query_indices_by_sample[batch_idx]
            if span is None or visual_count == 0 or query_indices is None:
                if span is not None and query_indices is None:
                    context.preserved_visual_rows.add(batch_idx)
                hidden_rows.append(hidden_row)
                mrope_rows.append(mrope_row)
                if mask_rows is not None:
                    mask_rows.append(mask_row)
                if label_rows is not None:
                    label_rows.append(label_row)
                sources.append(None)
                keep_counts.append(visual_count)
                continue

            visual_start = int(span.start)
            # Per row, because the budget is a fraction of *this* image's span.
            keep_k = compressor.config.keep_k_for_layer(rank_layer, visual_count, int(span.count))
            if keep_k >= visual_count:
                # The stage's share of the original span rounds to at least what is still
                # active, so it keeps everything. Legitimate on the flat tail of a
                # high-budget curve, and on small spans where two neighbouring ratios
                # round together.
                hidden_rows.append(hidden_row)
                mrope_rows.append(mrope_row)
                if mask_rows is not None:
                    mask_rows.append(mask_row)
                if label_rows is not None:
                    label_rows.append(label_row)
                sources.append(self._source_matrix(context, batch_idx, visual_count))
                keep_counts.append(visual_count)
                continue

            scores = self._visual_scores_from_qk(
                query_states, key_states, scale,
                batch_idx=batch_idx, query_indices=query_indices, visual_start=visual_start,
                visual_count=visual_count, padding_mask=padding_mask, dtype=hidden_states.dtype,
            )
            keep_rel = selector.apply_k(scores, keep_k).indices  # ascending
            policy = torch.ones(int(hidden_row.shape[0]), dtype=torch.bool,
                                device=hidden_states.device)
            policy[visual_start:visual_start + visual_count] = False
            policy[keep_rel + visual_start] = True
            selected = torch.where(policy)[0]

            hidden_rows.append(hidden_row.index_select(0, selected))
            mrope_rows.append(mrope_row.index_select(-1, selected))
            if mask_rows is not None:
                mask_rows.append(mask_row.index_select(0, selected))
            if label_rows is not None:
                label_rows.append(label_row.index_select(0, selected))

            origin = context.visual_origin[batch_idx]
            if origin is None:
                raise RuntimeError("HiDrop lost the visual attribution index before pruning.")
            context.visual_origin[batch_idx] = origin.index_select(0, keep_rel)
            sources.append(self._source_matrix(context, batch_idx, keep_k))
            keep_counts.append(keep_k)

        for batch_idx, keep_k in enumerate(keep_counts):
            context.active_visual_counts[batch_idx] = int(keep_k)
        # The scalar keys mirror `hidrop_llama.py`'s event shape so `pruning_history`
        # reads the same on both backbones; `keep_counts` carries the per-row detail that
        # only exists here, because each row's budget is a fraction of its own span.
        context.pruning_events.append(
            {
                "layer": int(rank_layer),
                "visual_tokens_before": int(before_counts_max),
                "keep_k": int(max(keep_counts) if keep_counts else 0),
                "visual_tokens_after": int(max(keep_counts) if keep_counts else 0),
                "batch_size": int(len(context.spans)),
                "image_rows": int(sum(span is not None for span in context.spans)),
                "keep_counts": list(keep_counts),
                "visual_tokens_before_by_row": list(before_counts),
            }
        )
        # `llmoe.viz` renders one `[T_cur, T_orig]` matrix per layer for a single image,
        # so row 0 is the image being visualized. Rows are one-hot: HiDrop prunes, it
        # never merges.
        if sources and sources[0] is not None:
            compressor._source_history.append(sources[0])
        compressor.debugger.log(
            f"layer={rank_layer} keep_counts={keep_counts} "
            f"spans={[span.count if span is not None else 0 for span in context.spans]}"
        )
        return self._stack_variable_rows(hidden_rows, mrope_rows, mask_rows, label_rows, text_ids)

    @staticmethod
    def _source_matrix(context: HiDropQwenForwardContext, batch_idx: int,
                       keep_k: int) -> torch.Tensor | None:
        """``[keep_k, original_span]`` one-hot attribution for one row."""
        origin = context.visual_origin[batch_idx]
        span = context.spans[batch_idx]
        if origin is None or span is None:
            return None
        source = torch.zeros((int(keep_k), int(span.count)), dtype=torch.float32)
        rows = min(int(keep_k), int(origin.numel()))
        source[torch.arange(rows), origin[:rows].detach().cpu()] = 1.0
        return source
