"""The adjacency-set partition (and its numba twin) must equal the ORIGINAL full-scan implementation bit for bit.

The reference functions below are verbatim copies of ``varest/partition.py`` before the rewrite (each merge rescanned
every pair). Fields are float16-quantised or deliberately tie-heavy (constant, two-level), because ties are where a
reordering would show; grids cover LLaVA-1.5 (24x24) and LLaVA-NeXT-sized windows."""
import heapq
import importlib
import os

import numpy as np
import pytest
import torch

import varest.partition as P
import varest.tokenizer as T

BRIDGE = P.BRIDGE


def _ref_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def partition_grid_ref(boundary: object, k: int, trace: object = None
                   ):
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


def agglo_edges_ref(nodes, edges, n_target):
    """Average-linkage union-find over an explicit edge list ``[(a, b, strength)]`` -> ``{node: component}``.

    Merges the pair whose shared boundary has the LOWEST mean strength first, until ``n_target`` components remain.
    It can only merge pairs it has an edge for (see :func:`bridge_components`)."""
    edges = list(edges)
    parent = {n: n for n in nodes}
    ncomp = len(nodes)
    if ncomp <= n_target or not edges:
        lab: dict = {}
        for n in nodes:
            lab.setdefault(_ref_find(parent, n), len(lab))
        return {n: lab[_ref_find(parent, n)] for n in nodes}
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
        rx, ry = _ref_find(parent, x), _ref_find(parent, y)
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
            if _ref_find(parent, o) == rx:
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
        r = _ref_find(parent, n)
        if r not in lab:
            lab[r] = len(lab)
        out[n] = lab[r]
    return out




def cell_edges_ref(cells, boundary, n_cols):
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


def _fields(rng, shape, kind):
    nr, nc = shape
    if kind == "f16":
        b = rng.random((2, nr, nc)).astype(np.float16).astype(np.float32)
    elif kind == "const":
        b = np.full((2, nr, nc), 0.25, np.float32)
    else:
        b = rng.choice([0.1, 0.6], (2, nr, nc)).astype(np.float32)
    b[0, :, -1] = 0.0
    b[1, -1, :] = 0.0
    t = rng.choice([0.0, 0.3, 0.99], (nr, nc)).astype(np.float32) if kind != "f16" else rng.random((nr, nc)).astype(np.float32)
    return b, t


CASES = [(s, k) for s in [(24, 24), (17, 29), (32, 40)] for k in ("f16", "const", "twolevel")]


@pytest.mark.parametrize("shape,kind", CASES)
def test_partition_grid_matches_reference(shape, kind):
    rng = np.random.default_rng(hash((shape, kind)) % 2**32)
    for rep in range(3):
        b, _ = _fields(rng, shape, kind)
        for k in (2, 13, 51, 102, shape[0] * shape[1] // 3):
            tr_a, tr_b = [], []
            assert np.array_equal(partition_grid_ref(torch.from_numpy(b), k, trace=tr_a),
                                  P.partition_grid(torch.from_numpy(b), k, trace=tr_b))
            assert tr_a == tr_b


@pytest.mark.parametrize("shape,kind", CASES)
def test_agglo_edges_matches_reference(shape, kind):
    rng = np.random.default_rng(1 + hash((shape, kind)) % 2**32)
    nr, nc = shape
    for rep in range(3):
        b, _ = _fields(rng, shape, kind)
        cells = [int(c) for c in np.nonzero(rng.random(nr * nc) < 0.6)[0]]
        ed = P.bridge_components(cells, P.cell_edges(cells, b, nc), nc)
        for want in (1, 5, 17, max(1, len(cells) // 2)):
            assert agglo_edges_ref(cells, ed, want) == P.agglo_edges(cells, ed, want)


def test_python_path_matches_reference(monkeypatch):
    """The pure-Python fallback (no numba / VAREST_NO_JIT) is exact too."""
    monkeypatch.setattr(P._jit, "AVAILABLE", False)
    test_partition_grid_matches_reference((24, 24), "twolevel")
    test_agglo_edges_matches_reference((24, 24), "const")


def test_build_partition_unchanged_on_letterbox():
    rng = np.random.default_rng(7)
    b, t = _fields(rng, (24, 24), "f16")
    inimg = np.ones((24, 24), bool)
    inimg[:, :4] = False
    inimg[:, 20:] = False
    b[0, :, 3] = 9.0
    b[0, :, 19] = 9.0
    for k in (32, 51, 102, 205, 358):
        lab, info = T.build_partition(b, t, k, inimg=inimg)
        assert info["tokens"] == len(np.unique(lab))
        # renumbering: region ids appear in raster order of their first cell
        firsts = [int(np.argmax(lab.reshape(-1) == j)) for j in range(info["tokens"])]
        assert firsts == sorted(firsts)


@pytest.mark.parametrize("shape,kind", CASES)
def test_cell_edges_matches_reference(shape, kind):
    rng = np.random.default_rng(3 + hash((shape, kind)) % 2**32)
    nr, nc = shape
    for rep in range(4):
        b, _ = _fields(rng, shape, kind)
        cells = [int(c) for c in np.nonzero(rng.random(nr * nc) < rng.uniform(0.2, 0.9))[0]]
        assert cell_edges_ref(cells, b, nc) == P.cell_edges(cells, b, nc)
