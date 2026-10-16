"""Agglomerative partitioning of a cell grid by boundary strength (average linkage, weakest boundary first).

Two entry points share one merge rule:

* :func:`partition_grid` merges a full ``n_r x n_c`` grid, given the edge field ``[2, n_r, n_c]``; with ``trace`` it
  records the merge-cost curve that :func:`varest.tokenizer.kmin_from_curve` reads its knee from.
* :func:`agglo_edges` merges an arbitrary cell set over an explicit edge list; the tokenizer uses it to cut the text
  area and the object area into an exact number of regions each.

Average linkage, not single: single linkage leaks two regions together through one weak pair.
"""
from __future__ import annotations

import heapq
from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import torch

# Last-resort bridge weight between disconnected pieces: above any real boundary (fields peak near 0.64, plus the text
# gradient's <= 0.25), so every genuine boundary is dissolved first and islands are fused only when the target demands.
BRIDGE = 1e3


def partition_grid(boundary: np.ndarray | torch.Tensor, k: int, trace: list[tuple[int, float]] | None = None
                   ) -> np.ndarray:
    """``[2, n_r, n_c]`` boundary strengths -> ``[n_r, n_c]`` int labels with EXACTLY ``k`` connected regions.

    Channel 0 is the edge to the right neighbour, channel 1 to the one below. With ``trace``, every accepted merge
    appends ``(regions_remaining, cost)`` where cost is the mean boundary strength dissolved (non-decreasing)."""
    a = boundary.detach().float().cpu().numpy() if torch.is_tensor(boundary) else np.asarray(boundary)
    a = -a
    nr, nc = a.shape[1], a.shape[2]
    n = nr * nc
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sums: dict[tuple[int, int], float] = {}
    cnts: dict[tuple[int, int], int] = {}
    for r in range(nr):
        for c in range(nc):
            i = r * nc + c
            if c + 1 < nc:
                sums[(i, i + 1)] = float(a[0, r, c])
                cnts[(i, i + 1)] = 1
            if r + 1 < nr:
                sums[(i, i + nc)] = float(a[1, r, c])
                cnts[(i, i + nc)] = 1
    heap = [(-sums[e] / cnts[e], e[0], e[1]) for e in sums]
    heapq.heapify(heap)
    nreg = n
    while nreg > k and heap:
        negavg, x, y = heapq.heappop(heap)
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        key = (min(rx, ry), max(rx, ry))
        if key not in sums:
            continue
        if abs(-negavg - sums[key] / cnts[key]) > 1e-9:
            heapq.heappush(heap, (-sums[key] / cnts[key], rx, ry))
            continue
        parent[ry] = rx
        nreg -= 1
        if trace is not None:
            trace.append((nreg, negavg))       # `a` is negated, so negavg is the POSITIVE strength dissolved
        for (p, q) in [e for e in list(sums) if ry in e]:
            sv, cv = sums.pop((p, q)), cnts.pop((p, q))
            o = q if p == ry else p
            if find(o) == rx:
                continue
            nk = (min(rx, o), max(rx, o))
            sums[nk] = sums.get(nk, 0.0) + sv
            cnts[nk] = cnts.get(nk, 0) + cv
            heapq.heappush(heap, (-sums[nk] / cnts[nk], rx, o))
        sums.pop(key, None)
        cnts.pop(key, None)
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab.reshape(nr, nc)


def _find(parent: dict, x: Hashable) -> Hashable:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def agglo_edges(nodes: Sequence[int], edges: Iterable[tuple[int, int, float]], n_target: int) -> dict[int, int]:
    """Average-linkage union-find over an explicit edge list ``[(a, b, strength)]`` -> ``{node: component}``.

    Merges the pair whose shared boundary has the LOWEST mean strength first, until ``n_target`` components remain.
    It can only merge pairs it has an edge for (see :func:`bridge_components`)."""
    edges = list(edges)
    parent = {n: n for n in nodes}
    ncomp = len(nodes)
    if ncomp <= n_target or not edges:
        lab: dict = {}
        for n in nodes:
            lab.setdefault(_find(parent, n), len(lab))
        return {n: lab[_find(parent, n)] for n in nodes}
    sums: dict[tuple[int, int], float] = {}
    cnts: dict[tuple[int, int], int] = {}
    for a, b, w in edges:
        key = (min(a, b), max(a, b))
        sums[key] = sums.get(key, 0.0) + float(w)
        cnts[key] = cnts.get(key, 0) + 1
    heap = [(sums[e] / cnts[e], e[0], e[1]) for e in sums]
    heapq.heapify(heap)
    while ncomp > n_target and heap:
        w, x, y = heapq.heappop(heap)
        rx, ry = _find(parent, x), _find(parent, y)
        if rx == ry:
            continue
        key = (min(rx, ry), max(rx, ry))
        if key not in sums:
            continue
        if abs(w - sums[key] / cnts[key]) > 1e-9:
            heapq.heappush(heap, (sums[key] / cnts[key], rx, ry))
            continue
        parent[ry] = rx
        ncomp -= 1
        for (p, q) in [e for e in list(sums) if ry in e]:
            sv, cv = sums.pop((p, q)), cnts.pop((p, q))
            o = q if p == ry else p
            if _find(parent, o) == rx:
                continue
            nk = (min(rx, o), max(rx, o))
            sums[nk] = sums.get(nk, 0.0) + sv
            cnts[nk] = cnts.get(nk, 0) + cv
            heapq.heappush(heap, (sums[nk] / cnts[nk], rx, o))
        sums.pop(key, None)
        cnts.pop(key, None)
    lab = {}
    out = {}
    for n in nodes:
        r = _find(parent, n)
        if r not in lab:
            lab[r] = len(lab)
        out[n] = lab[r]
    return out


def cell_edges(cells: Iterable[int], boundary: np.ndarray, n_cols: int) -> list[tuple[int, int, float]]:
    """4-neighbour edges restricted to ``cells`` (flat indices on an ``n_cols``-wide grid), weighted by the
    INTER-CELL boundary strength (the field is stored per edge: ``[0, r, c]`` is (r,c)-(r,c+1), ``[1, r, c]`` is
    (r,c)-(r+1,c))."""
    cs = set(int(c) for c in cells)
    edges = []
    for c in cs:
        r, cc = divmod(int(c), n_cols)
        if cc + 1 < n_cols and (c + 1) in cs:
            edges.append((int(c), int(c + 1), float(boundary[0, r, cc])))
        if (c + n_cols) in cs:
            edges.append((int(c), int(c + n_cols), float(boundary[1, r, cc])))
    return edges


def bridge_components(cells: Iterable[int], edges: list[tuple[int, int, float]], n_cols: int
                      ) -> list[tuple[int, int, float]]:
    """Add last-resort edges (weight ``BRIDGE`` + centroid distance) so a disconnected cell set can still be merged
    to any target count; nearest islands are fused first. No-op when the set is already connected."""
    cs = sorted(int(c) for c in cells)
    if len(cs) < 2:
        return edges
    parent = {c: c for c in cs}
    for a, bb, _ in edges:
        ra, rb = _find(parent, a), _find(parent, bb)
        if ra != rb:
            parent[rb] = ra
    comp: dict[int, list[int]] = {}
    for c in cs:
        comp.setdefault(_find(parent, c), []).append(c)
    if len(comp) < 2:
        return edges
    cen = {r: (np.mean([c // n_cols for c in m]), np.mean([c % n_cols for c in m])) for r, m in comp.items()}
    roots = list(comp)
    used = {roots[0]}
    rest = set(roots[1:])
    out = list(edges)
    while rest:
        u, v = min(((u, v) for u in used for v in rest),
                   key=lambda uv: (cen[uv[0]][0] - cen[uv[1]][0]) ** 2 + (cen[uv[0]][1] - cen[uv[1]][1]) ** 2)
        d = ((cen[u][0] - cen[v][0]) ** 2 + (cen[u][1] - cen[v][1]) ** 2) ** 0.5
        out.append((comp[u][0], comp[v][0], BRIDGE + float(d)))
        used.add(v)
        rest.discard(v)
    return out
