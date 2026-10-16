"""Shared decoder-pruning adapters and tensor bookkeeping.

Baseline modules own scoring and selection. This private module owns the model
plumbing required to apply a keep-set safely, so compressors never import one
another's private implementations.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle

from llmoe.backbone.prototype import VLM
from llmoe.utils.qwen import (
    APPLY_VISION_ROPE,
    DEFAULT_ATTENTION_CHUNK,
    chunked_received_attention,
    narrow_mrope_position_ids,
    window_reverse_indices,
)

# ---------------------------------------------------------------------------
# Decoder-layer cuts shared across baselines and backbones
# ---------------------------------------------------------------------------


def ratio_keep_count(num_tokens: int, retain_ratio: float, *,
                     label: str = "token pruning") -> int:
    """Resolve a dynamic visual-token budget with the shared floor convention.

    Qwen's NaViT encoder emits a different visual-token count for each image, so
    its compressors express budgets as fractions of the current image. Every
    ratio-based adapter uses this helper to guarantee the same validation,
    flooring, and one-token minimum.
    """
    num_tokens = int(num_tokens)
    retain_ratio = float(retain_ratio)
    if num_tokens < 1:
        raise ValueError(f"{label} cannot prune an empty token sequence")
    if not 0.0 < retain_ratio <= 1.0:
        raise ValueError(
            f"{label} retain_ratio must be in (0, 1], got {retain_ratio}"
        )
    return max(1, min(num_tokens, int(num_tokens * retain_ratio)))


def place_visual_tokens(inputs_embeds: torch.Tensor, *, start: int,
                        visual_count: int, tokens: torch.Tensor,
                        positions: torch.Tensor,
                        label: str = "visual-token pruning",
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Write reduced visual tokens into survivor slots of a concatenated prompt.

    Qwen must first scatter the full visual block so placeholder validation and
    M-RoPE grid construction see the original geometry. Static reducers then
    write their output into representative positions before the shared adapter
    physically gathers the keep mask.
    """
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError(
            f"{label} expected inputs_embeds [1, L, D], got "
            f"{tuple(inputs_embeds.shape)}"
        )
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(
            f"{label} expected reduced tokens [1, K, D], got {tuple(tokens.shape)}"
        )
    positions = positions.to(
        device=inputs_embeds.device, dtype=torch.long
    ).reshape(-1)
    keep_count = int(tokens.shape[1])
    if positions.numel() != keep_count:
        raise ValueError(
            f"{label} got {positions.numel()} survivor positions for "
            f"{keep_count} tokens"
        )
    if keep_count < 1 or keep_count > int(visual_count):
        raise ValueError(
            f"{label} cannot place {keep_count} tokens in a span of "
            f"{visual_count}"
        )
    if int(start) < 0 or int(start) + int(visual_count) > inputs_embeds.shape[1]:
        raise ValueError(
            f"{label} visual span [{start}, {start + visual_count}) does not fit "
            f"sequence length {inputs_embeds.shape[1]}"
        )
    if int(positions.min()) < 0 or int(positions.max()) >= int(visual_count):
        raise ValueError(
            f"{label} survivor positions must be inside [0, {visual_count})"
        )
    if positions.numel() > 1 and not bool(
        (positions[1:] > positions[:-1]).all()
    ):
        raise ValueError(
            f"{label} survivor positions must be unique and ascending"
        )
    if tokens.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError(
            f"{label} reduced width {tokens.shape[-1]} does not match prompt "
            f"width {inputs_embeds.shape[-1]}"
        )

    modified = inputs_embeds.clone()
    modified[:, int(start) + positions] = tokens.to(
        device=modified.device, dtype=modified.dtype
    )
    keep = torch.ones(
        inputs_embeds.shape[1],
        dtype=torch.bool,
        device=inputs_embeds.device,
    )
    keep[int(start):int(start) + int(visual_count)] = False
    keep[int(start) + positions] = True
    return modified, keep


def keep_columns(keep_mask: torch.Tensor, *, label: str = "decoder pruning") -> torch.Tensor:
    """Return ascending ``[B, L']`` indices for a boolean ``[B, L]`` keep mask."""
    if keep_mask.ndim != 2 or keep_mask.dtype != torch.bool:
        raise ValueError(
            f"{label} expected a boolean [B, L] keep mask, got "
            f"{tuple(keep_mask.shape)} with dtype {keep_mask.dtype}."
        )
    batch = keep_mask.shape[0]
    counts = keep_mask.sum(dim=1)
    if int(counts.min()) != int(counts.max()):
        raise RuntimeError(
            f"{label} needs the same number of survivors in every row, got "
            f"{counts.tolist()}."
        )
    return keep_mask.nonzero(as_tuple=True)[1].view(batch, -1)


def gather_position_ids(position_ids: Any, keep_index: torch.Tensor,
                        seq_len: int) -> torch.Tensor:
    """Select survivors' ``[B, L']`` position ids without renumbering them."""
    batch = keep_index.shape[0]
    if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 2 \
            and position_ids.shape[-1] == seq_len:
        return position_ids.expand(batch, seq_len).gather(1, keep_index)
    return keep_index


def gather_mrope_position_ids(position_ids: Any, keep_index: torch.Tensor, seq_len: int,
                              new_len: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Narrow Qwen ``[3 or 4, B, L]`` M-RoPE ids to surviving columns.

    Grid-coordinate rows remain absolute. Transformers 5.x prepends a packing
    row which must be compacted after a cut, otherwise a gap is interpreted as a
    packed-sequence boundary.
    """
    batch = keep_index.shape[0]
    if not isinstance(position_ids, torch.Tensor) or position_ids.shape[-1] != seq_len:
        return keep_index.unsqueeze(0).expand(3, batch, new_len), None
    narrowed = position_ids[..., keep_index[0]]
    if narrowed.ndim == 2:
        text = torch.arange(
            new_len, device=keep_index.device, dtype=narrowed.dtype
        ).expand(narrowed.shape[0], -1)
        return text, text
    if narrowed.ndim == 3 and narrowed.shape[0] == 4:
        narrowed = narrowed.clone()
        narrowed[0] = torch.arange(
            new_len, device=keep_index.device, dtype=narrowed.dtype
        ).expand(narrowed.shape[1], -1)
        return narrowed, narrowed[0]
    return narrowed, None


def select_rope_tables(position_embeddings: Any, keep_index: torch.Tensor,
                       seq_len: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select survivors from precomputed 1-D or M-RoPE ``(cos, sin)`` tables."""
    batch, new_len = keep_index.shape
    if not (
        isinstance(position_embeddings, tuple)
        and len(position_embeddings) == 2
        and all(
            isinstance(table, torch.Tensor)
            and table.ndim in (3, 4)
            and table.shape[-2] == seq_len
            for table in position_embeddings
        )
    ):
        return None
    selected: list[torch.Tensor] = []
    for table in position_embeddings:
        if table.ndim == 3:
            index = keep_index.unsqueeze(-1).expand(batch, new_len, table.shape[-1])
            selected.append(table.expand(batch, seq_len, -1).gather(1, index))
        else:
            axes = table.shape[0]
            index = keep_index[None, :, :, None].expand(
                axes, batch, new_len, table.shape[-1]
            )
            selected.append(table.expand(axes, batch, seq_len, -1).gather(2, index))
    return selected[0], selected[1]


def pruned_attention_mask(attention_mask: Any, keep_index: torch.Tensor,
                          seq_len: int, *,
                          label: str = "decoder pruning") -> tuple[torch.Tensor | None,
                                                                    torch.Tensor]:
    """Select kept rows/columns and return ``(mask, valid_keys)``."""
    batch, new_len = keep_index.shape
    ones = torch.ones(batch, new_len, dtype=torch.bool, device=keep_index.device)

    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
        if attention_mask.shape[-1] != seq_len:
            raise RuntimeError(
                f"{label} expected a [B, H, {seq_len}, {seq_len}] prefill mask, got "
                f"{tuple(attention_mask.shape)}."
            )
        mask = attention_mask.expand(batch, -1, -1, -1)
        heads = mask.shape[1]
        mask = mask.gather(
            2, keep_index[:, None, :, None].expand(batch, heads, new_len, seq_len)
        )
        mask = mask.gather(
            3, keep_index[:, None, None, :].expand(batch, heads, new_len, new_len)
        )
        if mask.dtype == torch.bool:
            return mask, mask[:, 0, -1, :]
        return mask, mask[:, 0, -1, :] > (torch.finfo(mask.dtype).min / 2)

    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 \
            and attention_mask.shape[-1] == seq_len:
        padding = attention_mask.expand(batch, seq_len).gather(1, keep_index)
        return padding, padding.bool()

    return None, ones


def decode_attention_mask(attention_mask: Any,
                          valid: torch.Tensor) -> torch.Tensor | None:
    """Rebuild a pruned layer's key-width mask for one decode step."""
    if not isinstance(attention_mask, torch.Tensor):
        return attention_mask
    if attention_mask.shape[-1] == valid.shape[-1]:
        return attention_mask
    if attention_mask.ndim == 2:
        return valid.to(dtype=attention_mask.dtype)
    if attention_mask.dtype == torch.bool:
        return valid[:, None, None, :]
    minimum = torch.finfo(attention_mask.dtype).min
    return torch.where(
        valid[:, None, None, :],
        torch.zeros((), dtype=attention_mask.dtype, device=valid.device),
        torch.full((), minimum, dtype=attention_mask.dtype, device=valid.device),
    )


def advance_decode_step(state: dict[str, Any], hidden_states: torch.Tensor, *,
                        label: str = "decoder pruning") -> None:
    """Extend a pruned key space by the token generated this step."""
    if hidden_states.shape[0] != state["batch"]:
        raise RuntimeError(
            f"{label} cut the cache for batch {state['batch']} but decoding now runs batch "
            f"{hidden_states.shape[0]}. Beam search expands the batch after prefill and is "
            f"not supported; use greedy or sampling decoding."
        )
    column = torch.ones(
        state["batch"], 1, dtype=torch.bool, device=state["valid"].device
    )
    state["valid"] = torch.cat([state["valid"], column], dim=1)
    state["kv_len"] += 1
    state["cache_position"] = torch.tensor(
        [state["kv_len"] - 1], device=hidden_states.device, dtype=torch.long
    )


def apply_prune_state(kwargs: dict, state: dict[str, Any], *, prefill: bool) -> None:
    """Apply a shortened sequence's geometry to a decoder layer call."""
    if prefill:
        kwargs["position_ids"] = state["position_ids"]
        kwargs["position_embeddings"] = state["position_embeddings"]
        kwargs["attention_mask"] = state["attention_mask"]
    else:
        kwargs["attention_mask"] = decode_attention_mask(
            kwargs.get("attention_mask"), state["valid"]
        )
    if "cache_position" in kwargs:
        kwargs["cache_position"] = state["cache_position"]


def decoder_attention_rows(
    layer: Any,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    attention_mask: Any,
    raters: torch.Tensor,
    *,
    apply_rope: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None,
    label: str = "decoder pruning",
) -> torch.Tensor:
    """Recompute selected decoder attention rows as [B, R, L].

    This is the backend-independent Q/K path shared by decoder-guided methods.
    It avoids materialising the full quadratic attention matrix and works with
    plain RoPE or Qwen M-RoPE through the injected apply_rope callable.
    """
    attn = getattr(layer, "self_attn", None)
    layernorm = getattr(layer, "input_layernorm", None)
    if (
        attn is None
        or layernorm is None
        or not hasattr(attn, "q_proj")
        or not hasattr(attn, "k_proj")
    ):
        raise RuntimeError(
            f"{label} needs decoder layers exposing input_layernorm and "
            "self_attn.q_proj/k_proj."
        )
    if hidden.ndim != 3 or raters.ndim != 2:
        raise ValueError(
            f"{label} expected hidden [B, L, D] and raters [B, R], got "
            f"{tuple(hidden.shape)} and {tuple(raters.shape)}"
        )
    batch, seq_len, _ = hidden.shape
    if raters.shape[0] != batch or raters.shape[1] < 1:
        raise ValueError(
            f"{label} rater geometry {tuple(raters.shape)} does not fit "
            f"batch {batch}"
        )
    if int(raters.min()) < 0 or int(raters.max()) >= seq_len:
        raise RuntimeError(
            f"{label} rater columns [{int(raters.min())}, "
            f"{int(raters.max())}] do not fit sequence length {seq_len}."
        )

    config = attn.config
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(
        getattr(config, "num_key_value_heads", num_heads)
    )
    head_dim = int(
        getattr(attn, "head_dim", hidden.shape[-1] // num_heads)
    )
    normed = layernorm(hidden)
    query = attn.q_proj(normed).view(
        batch, seq_len, num_heads, head_dim
    )
    key = attn.k_proj(normed).view(
        batch, seq_len, num_kv_heads, head_dim
    )
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    if callable(q_norm):
        query = q_norm(query)
    if callable(k_norm):
        key = k_norm(key)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    if position_embeddings is not None and apply_rope is not None:
        cos, sin = position_embeddings
        query, key = apply_rope(query, key, cos, sin)
    if num_kv_heads != num_heads:
        key = key.repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )

    rater_count = raters.shape[1]
    query = query.gather(
        2,
        raters[:, None, :, None].expand(
            batch, num_heads, rater_count, head_dim
        ),
    )
    scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
    logits = (
        query.float() @ key.float().transpose(-1, -2)
    ) * scaling
    minimum = torch.finfo(logits.dtype).min
    columns = torch.arange(seq_len, device=hidden.device)
    logits = logits.masked_fill(
        columns[None, None, None, :]
        > raters[:, None, :, None],
        minimum,
    )

    if isinstance(attention_mask, torch.Tensor):
        if (
            attention_mask.ndim == 4
            and attention_mask.shape[-2:] == (seq_len, seq_len)
        ):
            rows = attention_mask[:, :1].expand(
                batch, 1, seq_len, seq_len
            )
            rows = rows.gather(
                2,
                raters[:, None, :, None].expand(
                    batch, 1, rater_count, seq_len
                ),
            )
            if rows.dtype == torch.bool:
                logits = logits.masked_fill(~rows, minimum)
            else:
                logits = logits + rows.float()
        elif (
            attention_mask.ndim == 2
            and attention_mask.shape[-1] == seq_len
        ):
            valid = attention_mask.expand(batch, seq_len).bool()
            logits = logits.masked_fill(
                ~valid[:, None, None, :], minimum
            )
    return logits.softmax(dim=-1).mean(dim=1)


# ---------------------------------------------------------------------------
# Qwen2.5-VL adapter
# ---------------------------------------------------------------------------


class QwenPruningAdapter:
    """Backbone plumbing shared by decoder-pruning methods on Qwen2.5-VL.

    A method-specific subclass supplies scoring and selection in :meth:`_prefill`.
    This adapter owns hook lifecycle, prompt and vision capture, image-span
    validation, M-RoPE narrowing, the first sequence cut, and decode bookkeeping.

    The supported contract is batch size one, one contiguous image span, no
    video, no beam expansion, and ``use_cache=True``.
    """

    method: str = "Compressor"
    capture_vision_attention: bool = False

    def __init__(self, *, attention_chunk: int = DEFAULT_ATTENTION_CHUNK,
                 track_source: bool = False) -> None:
        self._attention_chunk = int(attention_chunk)
        if self._attention_chunk < 1:
            raise ValueError("attention_chunk must be >= 1")
        self._track_source = bool(track_source)

        self._handles: list[RemovableHandle] = []
        self._inner: Any | None = None
        self._visual: Any | None = None
        self._language_model: Any | None = None
        self._image_token_id: int | None = None

        self._input_ids: torch.Tensor | None = None
        self._image_grid_thw: torch.Tensor | None = None
        self._received: torch.Tensor | None = None
        self._keys: torch.Tensor | None = None

        self._prefill_state: dict[str, Any] | None = None
        self._source_history: list[torch.Tensor] = []
        self._forward_count = 0
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[{self.method}/Qwen] WARNING: {message}", flush=True)

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        self._register_common_hooks()

    def _bind(self, vlm: VLM, *, image_token_id: int) -> None:
        inner = vlm.model.model
        visual = getattr(inner, "visual", None)
        blocks = getattr(visual, "blocks", None)
        language_model = getattr(inner, "language_model", None)
        if visual is None or not blocks or language_model is None:
            raise TypeError(
                f"{self.method}'s Qwen path needs `model.model.visual.blocks` and "
                "`model.model.language_model`; this backbone provides neither."
            )
        if self.capture_vision_attention and APPLY_VISION_ROPE is None:
            raise RuntimeError(
                "Could not import `apply_rotary_pos_emb_vision` from transformers; "
                f"{self.method} cannot reproduce the last vision block's attention without it."
            )
        self._inner = inner
        self._visual = visual
        self._language_model = language_model
        self._image_token_id = int(image_token_id)

    def _register_common_hooks(self) -> None:
        self._handles.append(
            self._inner.register_forward_pre_hook(
                self._capture_model_inputs, with_kwargs=True
            )
        )
        if self.capture_vision_attention:
            self._handles.append(
                self._visual.blocks[-1].attn.register_forward_pre_hook(
                    self._capture_attention, with_kwargs=True
                )
            )
        self._handles.append(
            self._language_model.register_forward_pre_hook(
                self._language_model_pre_hook, with_kwargs=True
            )
        )

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._inner = None
        self._visual = None
        self._language_model = None
        self._image_token_id = None
        self._reset_forward_state()
        self._source_history = []

    def _reset_forward_state(self) -> None:
        self._input_ids = None
        self._image_grid_thw = None
        self._received = None
        self._keys = None
        self._prefill_state = None
        self._source_history = []

    @property
    def last_keep_mask(self) -> torch.Tensor | None:
        """Boolean keep mask over the latest prompt, if it was pruned."""
        return None if self._prefill_state is None else self._prefill_state.get("keep")

    def get_source_history(self) -> list[torch.Tensor]:
        return list(self._source_history)

    def _capture_model_inputs(self, _module: Any, args: tuple, kwargs: dict) -> None:
        if kwargs.get("pixel_values_videos") is not None:
            raise NotImplementedError(
                f"{self.method} on Qwen2.5-VL does not support video inputs."
            )

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 \
                or input_ids.shape[1] <= 1:
            return None

        self._reset_forward_state()
        self._forward_count += 1
        self._input_ids = input_ids
        self._image_grid_thw = kwargs.get("image_grid_thw")
        return None

    def _capture_attention(self, module: Any, args: tuple, kwargs: dict) -> None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        cu_seqlens = kwargs.get("cu_seqlens")
        position_embeddings = kwargs.get("position_embeddings")
        if not isinstance(hidden_states, torch.Tensor) or cu_seqlens is None \
                or position_embeddings is None:
            return None
        self._received, self._keys = chunked_received_attention(
            module,
            hidden_states,
            cu_seqlens,
            position_embeddings,
            attention_chunk=self._attention_chunk,
        )
        return None

    def _window_reverse_indices(self) -> tuple[int, torch.Tensor]:
        return window_reverse_indices(
            self._visual,
            self._image_grid_thw,
            int(self._received.numel()),
            label=self.method,
            device=self._received.device,
        )

    def _merged_token_signals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold captured patch signals into merged LLM tokens in raster order."""
        merge_unit, reverse_indices = self._window_reverse_indices()
        received, keys = self._received, self._keys
        scores = received.view(-1, merge_unit).mean(dim=-1)[reverse_indices]
        num_heads, _, head_dim = keys.shape
        merged_keys = keys.view(num_heads, -1, merge_unit, head_dim).mean(dim=2)
        merged_keys = merged_keys[:, reverse_indices, :].mean(dim=0)
        return scores, merged_keys

    def _merged_image_grid(self, num_visual: int) -> tuple[int, int]:
        """Resolve the post-merger ``(height, width)`` for one Qwen image.

        ``image_grid_thw`` describes pre-merger patches. Compressors operate on
        the visual block after ``visual.merger``, so both spatial sides are
        divided by the model's own merge size and checked against the prompt's
        observed placeholder count.
        """
        grid = self._image_grid_thw
        if not isinstance(grid, torch.Tensor) or tuple(grid.shape) != (1, 3):
            raise NotImplementedError(
                f"{self.method} on Qwen2.5-VL supports one image grid per prompt."
            )
        temporal, height, width = (int(value) for value in grid[0].tolist())
        if temporal != 1:
            raise NotImplementedError(
                f"{self.method} on Qwen2.5-VL supports still images only; "
                f"the grid has temporal extent {temporal}."
            )
        merge_size = int(getattr(self._visual, "spatial_merge_size", 2))
        if height % merge_size or width % merge_size:
            raise RuntimeError(
                f"{self.method}/Qwen image grid {(temporal, height, width)} "
                f"is not divisible by spatial_merge_size={merge_size}."
            )
        merged = height // merge_size, width // merge_size
        if merged[0] * merged[1] != int(num_visual):
            raise RuntimeError(
                f"{self.method}/Qwen merged grid {merged} does not match "
                f"the prompt's {num_visual} visual tokens."
            )
        return merged

    def _language_model_pre_hook(self, _module: Any, args: tuple,
                                 kwargs: dict) -> tuple[tuple, dict] | None:
        inputs_embeds = kwargs.get("inputs_embeds")
        if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3:
            return None
        if inputs_embeds.shape[1] == 1:
            return self._decode(args, kwargs, inputs_embeds)
        return self._prefill(args, kwargs, inputs_embeds)

    def _prefill(self, args: tuple, kwargs: dict,
                 inputs_embeds: torch.Tensor) -> tuple[tuple, dict] | None:
        """Select and apply a method-specific keep-set during prefill."""
        raise NotImplementedError

    def _locate_image_span(self, inputs_embeds: torch.Tensor) -> tuple[int, int] | None:
        batch = inputs_embeds.shape[0]
        if batch != 1:
            raise NotImplementedError(
                f"{self.method} on Qwen2.5-VL supports batch_size=1 only, got {batch}. "
                "Qwen's NaViT encoder gives each image a different token count, so a padded "
                "batch cannot share one keep-set."
            )
        if self._input_ids is None or self._input_ids.shape[1] != inputs_embeds.shape[1]:
            raise NotImplementedError(
                f"{self.method} on Qwen2.5-VL locates the image span in `input_ids`, so it "
                "cannot prune a forward driven by `inputs_embeds` alone."
            )
        image_positions = (
            self._input_ids[0] == self._image_token_id
        ).nonzero(as_tuple=True)[0]
        num_visual = int(image_positions.numel())
        if num_visual == 0:
            return None
        first, last = int(image_positions[0]), int(image_positions[-1])
        if num_visual != last - first + 1:
            raise NotImplementedError(
                f"{self.method} on Qwen2.5-VL expects one contiguous image span per prompt; "
                "this prompt carries several."
            )
        return first, num_visual

    def _require_kv_cache(self, kwargs: dict) -> None:
        if kwargs.get("use_cache") is False:
            raise NotImplementedError(
                f"{self.method} on Qwen2.5-VL prunes the KV cache, so `use_cache=False` would "
                "run the unpruned sequence on every step. Run with use_cache=True."
            )

    def _resolve_position_ids(self, kwargs: dict,
                              inputs_embeds: torch.Tensor) -> torch.Tensor:
        position_ids = kwargs.get("position_ids")
        if isinstance(position_ids, torch.Tensor):
            if position_ids.shape[-1] != inputs_embeds.shape[1]:
                raise RuntimeError(
                    f"{self.method} expected position ids covering {inputs_embeds.shape[1]} "
                    f"tokens, got {tuple(position_ids.shape)}."
                )
            return position_ids
        if self._inner is None or self._input_ids is None:
            raise RuntimeError(
                f"{self.method} needs the prompt's M-RoPE position ids to prune."
            )
        mm_token_type_ids = (self._input_ids == self._image_token_id).to(torch.int)
        position_ids, _ = self._inner.get_rope_index(
            self._input_ids,
            mm_token_type_ids,
            image_grid_thw=self._image_grid_thw,
            attention_mask=kwargs.get("attention_mask"),
        )
        return position_ids

    def _apply_keep(self, kwargs: dict, inputs_embeds: torch.Tensor,
                    modified_embeds: torch.Tensor,
                    keep: torch.Tensor) -> dict[str, Any]:
        """Apply a prompt keep-set and record state needed during generation."""
        device = inputs_embeds.device
        new_embeds = modified_embeds[:, keep]
        new_length = int(new_embeds.shape[1])
        position_ids = self._resolve_position_ids(kwargs, inputs_embeds)
        new_position_ids = narrow_mrope_position_ids(position_ids, keep)

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            new_attention_mask = None
        elif isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            new_attention_mask = attention_mask[:, keep]
        else:
            mask_shape = (
                tuple(attention_mask.shape)
                if isinstance(attention_mask, torch.Tensor) else ""
            )
            raise NotImplementedError(
                f"{self.method} on Qwen2.5-VL needs the 2-D padding mask to prune; got "
                f"{type(attention_mask).__name__} "
                f"{mask_shape}."
            )

        kwargs["inputs_embeds"] = new_embeds
        kwargs["position_ids"] = new_position_ids
        if new_attention_mask is not None:
            kwargs["attention_mask"] = new_attention_mask
        if "cache_position" in kwargs:
            kwargs["cache_position"] = torch.arange(new_length, device=device)

        self._prefill_state = {
            "batch": int(inputs_embeds.shape[0]),
            "prefill_len": new_length,
            "kv_len": new_length,
            "keep": keep,
            "position_ids": new_position_ids,
            "base_position": new_position_ids[..., -1:].clone(),
        }
        return self._prefill_state

    def _decode(self, args: tuple, kwargs: dict,
                inputs_embeds: torch.Tensor) -> tuple[tuple, dict] | None:
        state = self._prefill_state
        if state is None:
            return None
        if inputs_embeds.shape[0] != state["batch"]:
            raise RuntimeError(
                f"{self.method} pruned the cache for batch {state['batch']} but decoding now "
                f"runs batch {inputs_embeds.shape[0]}. Beam search expands the batch after "
                "prefill and is not supported; use greedy or sampling decoding."
            )

        state["kv_len"] += 1
        kv_len = int(state["kv_len"])
        step = kv_len - int(state["prefill_len"])
        kwargs["position_ids"] = state["base_position"] + step
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            kwargs["attention_mask"] = attention_mask.new_ones(
                (attention_mask.shape[0], kv_len)
            )
        if "cache_position" in kwargs:
            kwargs["cache_position"] = torch.arange(
                kv_len - 1, kv_len, device=inputs_embeds.device
            )
        return args, kwargs
