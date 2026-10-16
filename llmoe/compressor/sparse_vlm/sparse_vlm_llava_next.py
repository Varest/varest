"""Canonical dynamic schedule for SparseVLM on native LLaVA-NeXT.

SparseVLM's released default (v1/v1.5, ``RETAIN_TOKN=192``) uses text-guided
top-k stages ``[300, 200, 110]`` after decoder layers ``[2, 6, 15]`` and
recycles the best 30% of dropped tokens into roughly one cluster per ten rows.
NeXT's AnyRes span is image-dependent, so this module scales that one published
profile from a single layer-average ``retain_ratio`` while leaving its relative
stage shape and recycling constants fixed.
"""

import math

from llmoe.compressor.llava_next import validate_retain_ratio

CANONICAL_PROFILE: tuple[int, int, int] = (300, 200, 110)
CANONICAL_PRUNING_LAYERS: tuple[int, int, int] = (2, 6, 15)
RECYCLE_FRACTION = 0.3
CLUSTER_DIVISOR = 10


def recycled_output_count(input_count: int, keep: int) -> int:
    """Mirror the released keep + top-30%-cluster output length exactly."""

    incoming = int(input_count)
    kept = max(0, min(int(keep), incoming))
    dropped = incoming - kept
    if dropped <= 0:
        return incoming
    pool = min(dropped, int(dropped * RECYCLE_FRACTION) + 1)
    clusters = pool // CLUSTER_DIVISOR + 1
    return kept + clusters


def _schedule_from_primary(
    native_tokens: int,
    primary_keep: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Scale the official 300:200:110 profile and simulate recycling."""

    native = int(native_tokens)
    primary = int(primary_keep)
    raw = tuple(
        max(1, math.floor(primary * value / CANONICAL_PROFILE[0])) for value in CANONICAL_PROFILE
    )
    keeps: list[int] = []
    counts: list[int] = []
    incoming = native
    for requested in raw:
        keep = min(requested, incoming - 1)
        keep = max(1, keep)
        incoming = recycled_output_count(incoming, keep)
        keeps.append(keep)
        counts.append(incoming)
    return tuple(keeps), tuple(counts)  # type: ignore[return-value]


def layer_average_schedule(
    native_tokens: int,
    retain_ratio: float,
    decoder_layers: int,
    pruning_layers: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return ``(topk_schedule, actual_counts_after_recycling)``.

    The cost uses the physical hook boundaries: layers 0--2 see the native
    span, then 4, 9, and ``depth - 16`` layers see the three recycled outputs.
    As with the other NeXT ports, the largest integer schedule not exceeding
    the requested layer-average token cost is selected.
    """

    native = int(native_tokens)
    depth = int(decoder_layers)
    ratio = validate_retain_ratio(retain_ratio, label="SparseVLM/LLaVA-NeXT")
    # `widths[0]` layers run at the native span before the first cut, so the
    # reachable layer-average floor is about that count over the depth. The
    # canonical (2, 6, 15) gives 3/32 = 0.094, which cannot express the
    # sweep's lowest point. SparseVLM's own LLaVA-1.5 table already solves
    # this for budget 32 by moving the first cut a layer earlier -- see
    # SPARSEVLM_PROFILES, where 32 uses (1, 6, 15) and every other budget
    # uses (2, 6, 15). This lets a caller port that same adaptation.
    layers = tuple(CANONICAL_PRUNING_LAYERS if pruning_layers is None else pruning_layers)
    first_reduced_layer = tuple(layer + 1 for layer in layers)
    if native < 2:
        raise ValueError("SparseVLM/LLaVA-NeXT needs at least two visual tokens")
    if depth <= first_reduced_layer[-1]:
        raise ValueError(
            f"SparseVLM's canonical layer-15 stage needs at least 17 decoder "
            f"layers; this model has {depth}."
        )
    if ratio == 1.0:
        identity = (native, native, native)
        return identity, identity

    widths = (
        first_reduced_layer[0],
        first_reduced_layer[1] - first_reduced_layer[0],
        first_reduced_layer[2] - first_reduced_layer[1],
        depth - first_reduced_layer[2],
    )

    def resolve(primary: int) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
        keeps, counts = _schedule_from_primary(native, primary)
        cost = (
            widths[0] * native
            + widths[1] * counts[0]
            + widths[2] * counts[1]
            + widths[3] * counts[2]
        )
        return cost, keeps, counts

    target = math.floor(native * depth * ratio)
    minimum_cost, _, _ = resolve(1)
    maximum_cost, _, _ = resolve(native - 1)
    if target < minimum_cost:
        minimum_ratio = minimum_cost / (native * depth)
        raise ValueError(
            f"SparseVLM/LLaVA-NeXT retain_ratio={ratio:g} is below the "
            f"canonical profile's minimum {minimum_ratio:g} for this image."
        )
    if target > maximum_cost:
        maximum_ratio = maximum_cost / (native * depth)
        raise ValueError(
            f"SparseVLM/LLaVA-NeXT retain_ratio={ratio:g} is above the "
            f"canonical 300:200:110 profile's maximum {maximum_ratio:g}. Use "
            f"<= {maximum_ratio:g}, or retain_ratio=1 for an identity pass-through."
        )

    low, high = 1, native - 1
    while low < high:
        middle = (low + high + 1) // 2
        cost, _, _ = resolve(middle)
        if cost <= target:
            low = middle
        else:
            high = middle - 1
    _, keeps, counts = resolve(low)
    return keeps, counts
