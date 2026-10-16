"""Shared Qwen2.5-VL mechanics used by the token compressors.

Every Qwen port needs the same backbone plumbing regardless of how it scores or
selects tokens: resolving the ``<|image_pad|>`` placeholder id, replaying a
vision block's attention without materialising the full map, un-permuting the
windowed ViT's token order, and narrowing M-RoPE position ids after a cut.
Those pieces live here; the pruning flow that strings them together is
:class:`llmoe.compressor._pruning.QwenPruningAdapter`, and anything
method-specific stays with its compressor.
"""

from typing import Any

import torch

try:  # RoPE helpers whose module path is a transformers implementation detail.
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision as APPLY_VISION_ROPE,
    )
except Exception:  # pragma: no cover - degrade loudly if the path moved
    APPLY_VISION_ROPE = None

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb as APPLY_MROPE,
    )
except Exception:  # pragma: no cover - degrade loudly if the path moved
    APPLY_MROPE = None

# Query rows scored per matmul when replaying a vision block's attention. The
# full map would be [heads, L, L]; chunking bounds it to [heads, chunk, L].
DEFAULT_ATTENTION_CHUNK: int = 512


def resolve_image_token_id(config: Any, processor: Any, *, label: str = "the compressor") -> int:
    """Resolve the image placeholder id, mirroring LLaVA's own lookup order.

    Works for every supported backbone (Qwen's ``<|image_pad|>`` included) — no
    token id is ever hardcoded; ``label`` only names the caller in the error.
    """
    token_id = getattr(config, "image_token_id", None)
    if token_id is None:
        token_id = getattr(config, "image_token_index", None)
    if token_id is None:
        image_token = getattr(processor, "image_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if image_token is not None and tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids(str(image_token))
    if token_id is None:
        raise RuntimeError(f"Could not resolve the image placeholder token id for {label}.")
    return int(token_id)


def resolve_mrope_section(source: Any) -> list[int]:
    """The M-RoPE channel split, wherever this transformers version keeps it.

    Probes ``rope_parameters`` (5.x) then ``rope_scaling`` (4.x) on ``source``
    itself and on ``source.config``, so both a config object and a module that
    carries one can be passed.
    """
    for holder in (source, getattr(source, "config", None)):
        if holder is None:
            continue
        for field in ("rope_parameters", "rope_scaling"):
            parameters = getattr(holder, field, None)
            if isinstance(parameters, dict) and parameters.get("mrope_section"):
                return list(parameters["mrope_section"])
    raise RuntimeError("Could not find `mrope_section` on the given object or its config.")


def narrow_mrope_position_ids(position_ids: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Gather ``[3 or 4, B, L]`` M-RoPE position ids down to the kept columns.

    ``keep`` may be a boolean mask or an index tensor over the last dimension.
    Survivors keep their true grid coordinates. On transformers 5.x the tensor
    carries a fourth leading row of plain running indices, used for
    packed-sequence detection rather than for RoPE; a plain gather would leave a
    gap where the dropped tokens were, and ``find_packed_sequence_indices``
    reads any gap > 1 as a sequence boundary — which would build a
    block-diagonal mask over the prompt. Only that bookkeeping row is
    recompacted to ``arange``. Upstream sits on transformers 4.x, where the
    tensor has no such row and a plain gather is the whole story.
    """
    narrowed = position_ids[..., keep]
    if narrowed.ndim == 3 and narrowed.shape[0] == 4:
        narrowed = narrowed.clone()
        new_length = int(narrowed.shape[-1])
        narrowed[0] = torch.arange(
            new_length, device=narrowed.device, dtype=narrowed.dtype
        ).expand(narrowed.shape[1], -1)
    return narrowed


def chunked_received_attention(module: Any, hidden_states: torch.Tensor,
                               cu_seqlens: torch.Tensor,
                               position_embeddings: tuple[torch.Tensor, torch.Tensor], *,
                               attention_chunk: int = DEFAULT_ATTENTION_CHUNK,
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention received per patch in one vision block, plus its RoPE'd keys.

    Reproduces upstream's ``logits.mean(0).sum(0)`` from the block's own fused
    ``qkv`` without materialising the ``[heads, L, L]`` map: the reduction over
    query rows is a sum, so it accumulates exactly over chunks of queries. Runs
    under ``no_grad`` and is attention-backend independent — callers hook the
    block instead of forcing eager attention.

    :return: ``(received [L] float32, keys [heads, L, head_dim])``, both in the
        tower's window order.
    """
    if APPLY_VISION_ROPE is None:
        raise RuntimeError(
            "Could not import `apply_rotary_pos_emb_vision` from transformers; the vision "
            "block's attention cannot be replayed without it."
        )
    with torch.no_grad():
        seq_length, _ = hidden_states.shape
        num_heads = int(module.num_heads)
        queries, keys, _ = (
            module.qkv(hidden_states)
            .reshape(seq_length, 3, num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        queries, keys = APPLY_VISION_ROPE(queries, keys, cos, sin)
        queries = queries.transpose(0, 1)  # [heads, L, head_dim]
        keys = keys.transpose(0, 1)
        scaling = float(getattr(module, "scaling", queries.shape[-1] ** -0.5))

        received = torch.zeros(seq_length, dtype=torch.float32, device=hidden_states.device)
        bounds = cu_seqlens.tolist()
        for start, end in zip(bounds[:-1], bounds[1:]):
            start, end = int(start), int(end)
            if end <= start:
                continue
            # The block's own attention span (one whole image when the block is a
            # full-attention one); no token attends across it.
            block_keys = keys[:, start:end]
            for query_start in range(start, end, attention_chunk):
                query_end = min(query_start + attention_chunk, end)
                weights = torch.matmul(
                    queries[:, query_start:query_end], block_keys.transpose(-1, -2)
                ) * scaling
                weights = weights.softmax(dim=-1, dtype=torch.float32)
                received[start:end] += weights.mean(dim=0).sum(dim=0)
        return received, keys


def window_reverse_indices(visual: Any, image_grid_thw: torch.Tensor, num_patches: int, *,
                           label: str = "the compressor",
                           device: torch.device | None = None) -> tuple[int, torch.Tensor]:
    """``(spatial_merge_unit, argsort(window_index))`` for folding patches onto LLM tokens.

    The windowed ViT permutes patches in groups of ``spatial_merge_unit`` and the
    merger collapses each group into one LLM token. Group-pool a per-patch signal
    and index it with the returned permutation to get per-token values in raster
    order — exactly what upstream does at the end of the tower's forward.
    """
    merge_unit = int(visual.spatial_merge_unit)
    if num_patches % merge_unit:
        raise RuntimeError(
            f"{label} expected the patch count ({num_patches}) to be divisible by the "
            f"merger unit ({merge_unit})."
        )
    window_index, _ = visual.get_window_index(image_grid_thw)
    if device is not None:
        window_index = window_index.to(device)
    reverse_indices = torch.argsort(window_index)
    if reverse_indices.numel() != num_patches // merge_unit:
        raise RuntimeError(
            f"{label} window index ({reverse_indices.numel()}) does not match the merged "
            f"token count ({num_patches // merge_unit})."
        )
    return merge_unit, reverse_indices
