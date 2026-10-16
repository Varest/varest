"""DUET Stage 2 -- PyramidDrop-style text-to-vision pruning in the decoder.

The decoder is cut into stages by ``pdrop_layers``. At each boundary layer the
surviving visual tokens are ranked by how much the prompt's *text* attends to
them -- read from that boundary layer's own attention -- and only
``floor(stage1_tokens * keep_ratio)`` of them continue. The default schedule
``layers (16, 24)`` with ratios ``(0.5, 0.0)`` means layers 0-15 see all
``S = stage1_tokens`` visual tokens, layers 16-23 see ``floor(0.5 S)``, and layers
24-31 see none at all.

Because the count changes per layer, the reported budget is the **layer-weighted
average**, not a final token count -- exactly the convention ``sparse_vlm.py`` and
``must_drop.py`` use here. ``average_image_tokens`` computes it, and the published
DUET-VLM profiles below are the (dominant, contextual) splits that land on
224/192/128/64/32.

Everything in this module is a pure tensor / string op so it is testable without a
model; the hook wiring lives in ``duet.py``.
"""

import re
from math import floor
from typing import Any, Sequence

import torch

from llmoe.utils.depth import REFERENCE_DEPTH, LayerSchedule


# Published DUET/LLaVA-1.5 profiles as ``budget -> (dominant patches, contextual)``.
# Upstream counts ``[CLS]`` inside the dominant branch; here ``[CLS]`` is added by
# ``include_cls_token``, so every dominant value is the paper's minus one.
DUET_PROFILES: dict[int, tuple[int, int]] = {
    224: (349, 8),   # extrapolated
    192: (299, 7),   # paper
    128: (169, 35),  # paper
    64: (71, 30),    # paper
    32: (31, 19),    # extrapolated
}

DUET_PAPER_BUDGETS: tuple[int, ...] = (192, 128, 64)
DEFAULT_PDROP_LAYERS: tuple[int, ...] = (16, 24)
DEFAULT_PDROP_KEEP_RATIOS: tuple[float, ...] = (0.5, 0.0)

# Function words carry no visual grounding, so `text_rater="salient"` drops them.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for", "from", "had",
    "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "ours", "she", "so", "that", "the", "their", "theirs",
    "them", "there", "these", "they", "this", "those", "to", "was", "we", "were", "with", "you",
    "your", "yours", "assistant", "user", "system",
})
_QUESTION_WORDS: frozenset[str] = frozenset({
    "who", "what", "where", "when", "why", "how", "which", "whom", "whose",
})
_IMPORTANT_PUNCT: frozenset[str] = frozenset({"?", "!", ":"})

TEXT_RATERS: tuple[str, ...] = ("last", "all", "salient")
SCORE_REDUCTIONS: tuple[str, ...] = ("mean", "max")


def stage_keep_counts(stage1_tokens: int, keep_ratios: Sequence[float]) -> tuple[int, ...]:
    """Visual tokens surviving each boundary, ``floor(S * ratio)`` as in the reference."""
    return tuple(max(0, floor(int(stage1_tokens) * float(ratio))) for ratio in keep_ratios)


def average_image_tokens(stage1_tokens: int, pdrop_layers: Sequence[int],
                         keep_ratios: Sequence[float], num_layers: int = REFERENCE_DEPTH) -> float:
    """The reported budget: visual tokens averaged over the decoder's layers.

    :param stage1_tokens: tokens entering the decoder, ``[CLS]`` included.
    :param pdrop_layers: boundary layers; layer ``L`` is the first to run reduced.
    :param keep_ratios: fraction of ``stage1_tokens`` surviving each boundary.
    :param num_layers: decoder depth (32 for LLaVA-1.5-7B, 40 for the 13B).
    """
    return LayerSchedule(
        tuple(int(layer) for layer in pdrop_layers),
        stage_keep_counts(stage1_tokens, keep_ratios),
        int(num_layers),
        int(stage1_tokens),
    ).layer_average


def pdrop_average_factor(pdrop_layers: Sequence[int], keep_ratios: Sequence[float],
                         num_layers: int) -> float:
    """The layer-averaged budget as a fraction of the Stage-1 count.

    The ratio-mode counterpart of :func:`average_image_tokens`. Qwen's NaViT
    encoder emits a different token count per image, so a budget there can only
    be a fraction; this is what Stage 2 multiplies that fraction by, and so what
    a layer-averaged target has to be divided by to recover Stage 1's share.

    :param pdrop_layers: boundary layers; layer ``L`` is the first to run reduced.
    :param keep_ratios: fraction of the Stage-1 count surviving each boundary.
    :param num_layers: decoder depth (28 for Qwen2.5-VL-7B).
    """
    return LayerSchedule(
        tuple(int(layer) for layer in pdrop_layers),
        tuple(float(ratio) for ratio in keep_ratios),
        int(num_layers),
        n_visual=1,
    ).layer_average


def _normalise_token_string(token: str) -> str:
    """Strip the tokenizer's word-boundary markers and surrounding punctuation."""
    text = token.strip()
    for prefix in ("▁", "Ġ", "##"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace("</w>", "")
    if text in _IMPORTANT_PUNCT:
        return text
    return re.sub(r"^[^A-Za-z0-9?!.:]+|[^A-Za-z0-9?!.:]+$", "", text).lower()


def token_is_salient(token: str) -> bool:
    """Whether a prompt token is content-bearing enough to rate the image with."""
    text = _normalise_token_string(token)
    if not text:
        return False
    if text in _QUESTION_WORDS or text in _IMPORTANT_PUNCT:
        return True
    if text in _STOPWORDS:
        return False
    if any(character.isdigit() for character in text):
        return True
    return any(character.isalpha() for character in text) and len(text) > 1


def select_text_raters(input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                       image_start: torch.Tensor, image_len: int, *, mode: str,
                       tokenizer: Any | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the prompt tokens whose attention ranks the visual span.

    Only real, non-visual tokens *after* the image qualify (the reference falls
    back to the whole prompt when nothing follows the image). ``"last"`` uses the
    final one, ``"all"`` uses every one, and ``"salient"`` keeps the
    content-bearing ones plus the final token. A trailing end-of-sequence token is
    dropped whenever it is not the only rater.

    :param input_ids: ``[B, L]`` prefill ids.
    :param attention_mask: ``[B, L]`` padding mask, or ``None``.
    :param image_start: ``[B]`` first visual column of each row.
    :param image_len: visual tokens per row.
    :param mode: one of ``TEXT_RATERS``.
    :param tokenizer: needed by ``"salient"`` to read the token strings.
    :return: ``(index, valid)``, both ``[B, R]``; padded entries are ``False`` in
        ``valid`` so reductions can ignore them. A row with no usable rater is
        all-``False``, which the caller checks before pruning at all.
    """
    if mode not in TEXT_RATERS:
        raise ValueError(f"text_rater must be one of {TEXT_RATERS}, got {mode!r}")
    if input_ids.ndim != 2:
        raise ValueError(f"Expected [B, L] input_ids, got {tuple(input_ids.shape)}")
    batch, seq_len = input_ids.shape
    pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    selected_rows: list[list[int]] = []
    for row in range(batch):
        # Pulled to CPU once: the reference scans this row token by token, and
        # doing that against device tensors would sync per position.
        ids = input_ids[row].tolist()
        mask = attention_mask[row].tolist() if isinstance(attention_mask, torch.Tensor) else None
        start = int(image_start[row])
        end = start + int(image_len)
        valid = [
            pos for pos in range(seq_len)
            if not (start <= pos < end)
            and not (mask is not None and not mask[pos])
            and not (pad_id is not None and ids[pos] == int(pad_id))
        ]
        query = [pos for pos in valid if pos >= end] or valid
        if not query:
            selected_rows.append([])
            continue
        if mode == "last":
            selected = [query[-1]]
        elif mode == "all":
            selected = list(query)
        else:
            selected = _salient_positions(ids, query, tokenizer)
        if eos_id is not None and len(selected) > 1 and ids[selected[-1]] == int(eos_id):
            selected = selected[:-1]
        selected_rows.append(sorted(set(selected)))

    width = max(1, max(len(row) for row in selected_rows))
    index = torch.zeros(batch, width, dtype=torch.long, device=input_ids.device)
    valid = torch.zeros(batch, width, dtype=torch.bool, device=input_ids.device)
    for row, positions in enumerate(selected_rows):
        if not positions:
            continue  # left as index 0 / invalid; the caller skips an all-invalid batch
        index[row, :len(positions)] = torch.tensor(positions, dtype=torch.long,
                                                   device=input_ids.device)
        # Pad by repeating the last real rater so the gather stays in range.
        index[row, len(positions):] = positions[-1]
        valid[row, :len(positions)] = True
    return index, valid


def _salient_positions(ids: list[int], query: list[int], tokenizer: Any | None) -> list[int]:
    """Content-bearing raters, always including the final prompt token."""
    if tokenizer is None or not hasattr(tokenizer, "convert_ids_to_tokens"):
        return list(query)
    tokens = tokenizer.convert_ids_to_tokens([ids[pos] for pos in query])
    selected = [pos for pos, token in zip(query, tokens) if token_is_salient(str(token))]
    if query[-1] not in selected:
        selected.append(query[-1])
    return selected


def reduce_rater_scores(attention: torch.Tensor, valid: torch.Tensor, image_start: torch.Tensor,
                        image_len: int, *, reduction: str) -> torch.Tensor:
    """Collapse ``[B, R, L]`` rater attention to ``[B, image_len]`` visual scores.

    :param attention: head-averaged attention rows of the raters.
    :param valid: ``[B, R]`` mask of real (non-padded) raters.
    :param image_start: ``[B]`` first visual column of each row.
    :param image_len: visual tokens per row.
    :param reduction: ``"mean"`` or ``"max"`` over the raters.
    """
    if reduction not in SCORE_REDUCTIONS:
        raise ValueError(f"score_reduction must be one of {SCORE_REDUCTIONS}, got {reduction!r}")
    if attention.ndim != 3:
        raise ValueError(f"Expected [B, R, L] rater attention, got {tuple(attention.shape)}")
    batch, raters, seq_len = attention.shape
    if image_len < 1:
        return attention.new_zeros(batch, 0)
    columns = image_start.to(attention.device).unsqueeze(1) + torch.arange(
        image_len, device=attention.device).unsqueeze(0)  # [B, V]
    if int(columns.max()) >= seq_len:
        raise ValueError(
            f"visual span ends at {int(columns.max()) + 1} but the attention is only "
            f"{seq_len} wide"
        )
    rows = attention.gather(2, columns.unsqueeze(1).expand(batch, raters, image_len)).float()
    keep = valid.to(attention.device).unsqueeze(-1)
    if reduction == "mean":
        return (rows * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1)
    return rows.masked_fill(~keep, float("-inf")).max(dim=1).values
