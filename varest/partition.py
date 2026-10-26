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

from varest import _partition_jit as _jit

# Last-resort bridge weight between disconnected pieces: above any real boundary (fields peak near 0.64, plus the text
# gradient's <= 0.25), so every genuine boundary is dissolved first and islands are fused only when the target demands.
BRIDGE = 1e3


def partition_grid(boundary: np.ndarray | torch.Tensor, k: int, trace: list[tuple[int, float]] | None = None
                   ) -> np.ndarray:
    """``[2, n_r, n_c]`` boundary strengths -> ``[n_r, n_c]`` int labels with EXACTLY ``k`` connected regions.

    Channel 0 is the edge to the right neighbour, channel 1 to the one below. With ``trace``, every accepted merge
    appends ``(regions_remaining, cost)`` where cost is the mean boundary strength dissolved (non-decreasing).

    Each region keeps an adjacency set, so a merge visits only the absorbed region's neighbours instead of scanning
    every pair. The arithmetic and the merge order are those of the full scan: every pair key is a pair of current
    roots, each merge adds ONE term (``existing + absorbed``) to each re-keyed pair whatever order the neighbours are
    visited in, and the heap pops by its ``(mean, x, y)`` tuples alone -- the labels are bit-identical
    (``varest/tests/test_partition_fast.py`` checks this against the full-scan reference)."""
    a = boundary.detach().float().cpu().numpy() if torch.is_tensor(boundary) else np.asarray(boundary)
    a = -a
    nr, nc = a.shape[1], a.shape[2]
    n = nr * nc
    if _jit.AVAILABLE and np.isfinite(a).all():
        roots, tr = _jit.grid_roots(np.asarray(a, np.float64), k, trace is not None)
        if trace is not None:
            trace.extend(tr)
        _, lab = np.unique(roots, return_inverse=True)
        return lab.reshape(nr, nc)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sums: dict[tuple[int, int], float] = {}
    cnts: dict[tuple[int, int], int] = {}
    adj: list[set[int]] = [set() for _ in range(n)]
    for r in range(nr):
        for c in range(nc):
            i = r * nc + c
            if c + 1 < nc:
                sums[(i, i + 1)] = float(a[0, r, c])
                cnts[(i, i + 1)] = 1
                adj[i].add(i + 1)
                adj[i + 1].add(i)
            if r + 1 < nr:
                sums[(i, i + nc)] = float(a[1, r, c])
                cnts[(i, i + nc)] = 1
                adj[i].add(i + nc)
                adj[i + nc].add(i)
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
        for o in adj[ry]:
            pk = (min(ry, o), max(ry, o))
            sv, cv = sums.pop(pk), cnts.pop(pk)
            adj[o].discard(ry)
            if find(o) == rx:
                continue
            nk = (min(rx, o), max(rx, o))
            sums[nk] = sums.get(nk, 0.0) + sv
            cnts[nk] = cnts.get(nk, 0) + cv
            adj[rx].add(o)
            adj[o].add(rx)
            heapq.heappush(heap, (-sums[nk] / cnts[nk], rx, o))
        adj[ry] = set()
        adj[rx].discard(ry)
        sums.pop(key, None)
        cnts.pop(key, None)
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab.reshape(nr, nc)


class EdgeList(list):
    """A list of ``(a, b, strength)`` edges that also carries the same edges, in the same order, as arrays
    ``(a int64, b int64, strength float64)``, so the compiled merge can take them without a Python round trip."""
    __slots__ = ("arrays",)


def _find(parent: dict, x: Hashable) -> Hashable:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def agglo_edges(nodes: Sequence[int], edges: Iterable[tuple[int, int, float]], n_target: int) -> dict[int, int]:
    """Average-linkage union-find over an explicit edge list ``[(a, b, strength)]`` -> ``{node: component}``.

    Merges the pair whose shared boundary has the LOWEST mean strength first, until ``n_target`` components remain.
    It can only merge pairs it has an edge for (see :func:`bridge_components`). Adjacency sets as in
    :func:`partition_grid`, with the same exactness argument."""
    arrays = getattr(edges, "arrays", None)
    edges = list(edges)
    parent = {n: n for n in nodes}
    ncomp = len(nodes)
    if ncomp <= n_target or not edges:
        lab: dict = {}
        for n in nodes:
            lab.setdefault(_find(parent, n), len(lab))
        return {n: lab[_find(parent, n)] for n in nodes}
    weights = arrays[2] if arrays is not None else np.fromiter((w for _, _, w in edges), np.float64, len(edges))
    if _jit.AVAILABLE and np.isfinite(weights).all():
        roots = _jit.edge_roots(nodes, edges, n_target, arrays)
        lab = {}
        out = {}
        for n, r in zip(nodes, roots):
            if r not in lab:
                lab[r] = len(lab)
            out[n] = lab[r]
        return out
    sums: dict[tuple[int, int], float] = {}
    cnts: dict[tuple[int, int], int] = {}
    adj: dict[int, set[int]] = {}
    for a, b, w in edges:
        key = (min(a, b), max(a, b))
        sums[key] = sums.get(key, 0.0) + float(w)
        cnts[key] = cnts.get(key, 0) + 1
        if key[0] != key[1]:                    # a self-loop can never be merged; keep it out of the adjacency
            adj.setdefault(key[0], set()).add(key[1])
            adj.setdefault(key[1], set()).add(key[0])
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
        for o in adj.get(ry, ()):
            pk = (min(ry, o), max(ry, o))
            sv, cv = sums.pop(pk), cnts.pop(pk)
            adj[o].discard(ry)
            if _find(parent, o) == rx:
                continue
            nk = (min(rx, o), max(rx, o))
            sums[nk] = sums.get(nk, 0.0) + sv
            cnts[nk] = cnts.get(nk, 0) + cv
            adj.setdefault(rx, set()).add(o)
            adj[o].add(rx)
            heapq.heappush(heap, (sums[nk] / cnts[nk], rx, o))
        adj[ry] = set()
        if rx in adj:
            adj[rx].discard(ry)
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
    (r,c)-(r+1,c)).

    The edges come out in the order the cell SET iterates, right edge before down edge for each cell -- the order of
    the original per-cell loop, which matters downstream (:func:`bridge_components` breaks ties by component root, and
    roots follow the union order); only the membership tests and lookups are vectorised."""
    cs = set(int(c) for c in cells)
    if not cs:
        return []
    order = np.fromiter(cs, dtype=np.int64, count=len(cs))      # the set's own iteration order
    hi = int(order.max()) + n_cols + 2
    member = np.zeros(hi, bool)
    member[order] = True
    r, cc = np.divmod(order, n_cols)
    b = np.asarray(boundary)
    right = (cc + 1 < n_cols) & member[order + 1]
    down = member[order + n_cols]
    # interleave per cell: [right_0, down_0, right_1, down_1, ...], keeping only the edges that exist
    ok = np.stack([right, down], 1).ravel()
    dst = np.stack([order + 1, order + n_cols], 1).ravel()
    rr = np.minimum(r, b.shape[1] - 1)
    w = np.stack([b[0, rr, cc], b[1, rr, cc]], 1).ravel().astype(np.float64)   # float(float32) == float64 cast
    src = np.repeat(order, 2)
    a, b_, w = src[ok], dst[ok], w[ok]
    out = EdgeList(zip(a.tolist(), b_.tolist(), w.tolist()))
    out.arrays = (a, b_, w)
    return out


def bridge_components(cells: Iterable[int], edges: list[tuple[int, int, float]], n_cols: int
                      ) -> list[tuple[int, int, float]]:
    """Add last-resort edges (weight ``BRIDGE`` + centroid distance) so a disconnected cell set can still be merged
    to any target count; nearest islands are fused first. No-op when the set is already connected."""
    cs = sorted(int(c) for c in cells)
    if len(cs) < 2:
        return edges
    arrays = getattr(edges, "arrays", None)
    if arrays is not None and _n_components(np.asarray(cs, np.int64), arrays[0], arrays[1]) == 1:
        return edges                          # connected: nothing to bridge (the union-find below would agree)
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


def _n_components(cells: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    """Connected components of the graph on the sorted ``cells`` with edges ``a[i]-b[i]`` (endpoints are cells)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(cells)
    ia, ib = np.searchsorted(cells, a), np.searchsorted(cells, b)
    g = coo_matrix((np.ones(len(ia), np.int8), (ia, ib)), shape=(n, n))
    return int(connected_components(g, directed=False)[0])


def agglo_edge_labels(nodes: np.ndarray, edges: Iterable[tuple[int, int, float]], n_target: int) -> np.ndarray:
    """:func:`agglo_edges` as an array: the component of every node, in ``nodes`` order, numbered by first appearance
    along ``nodes`` (exactly the dict :func:`agglo_edges` returns)."""
    nodes = np.asarray(nodes, np.int64)
    arrays = getattr(edges, "arrays", None)
    if (_jit.AVAILABLE and arrays is not None and len(arrays[0]) and len(nodes) > n_target
            and np.isfinite(arrays[2]).all()):
        roots = np.asarray(_jit.edge_roots(nodes, edges, n_target, arrays))
        _, first, inv = np.unique(roots, return_index=True, return_inverse=True)
        rank = np.empty(len(first), np.int64)
        rank[np.argsort(first)] = np.arange(len(first))
        return rank[inv]
    d = agglo_edges([int(n) for n in nodes], edges, n_target)
    return np.array([d[int(n)] for n in nodes], np.int64)
