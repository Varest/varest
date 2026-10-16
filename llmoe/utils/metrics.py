"""Shared scoring primitives used by more than one benchmark.

Benchmarks should reach for helpers here instead of inlining their own
versions or — worse — importing across benchmark modules.
"""

import string
from typing import Any, Iterable

import numpy as np


_DEFAULT_ARTICLES: frozenset[str] = frozenset({"a", "an", "the"})
_DEFAULT_PUNCT: frozenset[str] = frozenset(string.punctuation)


def safe_div(num: float, den: float) -> float:
    """Return num/den as float, or 0.0 when den is zero."""
    return float(num / den) if den else 0.0


def pair_accuracy(correct: np.ndarray, group_idx: np.ndarray) -> np.ndarray:
    """Return a length-`n_groups` bool array: True when every member of a group is correct."""
    correct = np.asarray(correct, dtype=bool)
    group_idx = np.asarray(group_idx, dtype=np.int64)
    sums = np.bincount(group_idx, weights=correct.astype(np.int64))
    counts = np.bincount(group_idx)
    return sums == counts


def binary_label_metrics(
    golds: Iterable[Any],
    preds: Iterable[Any],
    positive: Any,
) -> dict[str, float | int]:
    """Precision/recall/F1 for one positive label vs all others.

    Returns: {precision, recall, f1, tp, fp, fn, tn, accuracy, support_pos, n}.
    `accuracy` includes true negatives (everything not equal to `positive` and
    not predicted as `positive` counts as correct).
    """
    golds_list = list(golds)
    preds_list = list(preds)
    tp = sum(g == positive and p == positive for g, p in zip(golds_list, preds_list))
    fp = sum(g != positive and p == positive for g, p in zip(golds_list, preds_list))
    fn = sum(g == positive and p != positive for g, p in zip(golds_list, preds_list))
    tn = sum(g != positive and p != positive for g, p in zip(golds_list, preds_list))
    n = len(golds_list)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_score": f1,  # legacy alias used by MME's classification dict
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": safe_div(tp + tn, n),
        "support_pos": int(sum(g == positive for g in golds_list)),
        "n": int(n),
    }


def normalize_yes_no(text: str, fallback: str | None = "other") -> str | None:
    """Map a model's free-text answer to 'yes' | 'no' | fallback.

    - exact 'yes'/'no' (after lowercase/strip/period removal) → that label
    - 1-char 'y'/'n' → 'yes'/'no'
    - prefix 'yes' or 'no' in the first 4 chars → that label
    - otherwise → `fallback` (use None to caller-handle the miss)
    """
    s = str(text).lower().strip().replace(".", "")
    if s in ("yes", "no"):
        return s
    if len(s) == 1:
        return {"y": "yes", "n": "no"}.get(s, fallback)
    head = s[:4]
    if "yes" in head:
        return "yes"
    if "no" in head:
        return "no"
    return fallback


def strip_punct_and_articles(
    text: str,
    articles: Iterable[str] = _DEFAULT_ARTICLES,
    punct: Iterable[str] | None = None,
    punct_replacement: str = "",
) -> str:
    """Lowercase, remove punctuation, drop articles, collapse whitespace.

    `punct_replacement` is the string substituted for each punctuation char
    ("" by default — matches MMBench's strip-and-rejoin; pass " " for
    EvalAI-style splitting on punct boundaries).
    """
    article_set = frozenset(articles)
    punct_set = frozenset(punct) if punct is not None else _DEFAULT_PUNCT
    text = str(text).lower().strip()
    text = "".join(punct_replacement if ch in punct_set else ch for ch in text)
    return " ".join(token for token in text.split() if token not in article_set)


def round_numeric(value: Any, digits: int = 4) -> Any:
    """Recursively round floats inside a JSON-like structure."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_numeric(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_numeric(item, digits) for item in value]
    return value
