"""Compiled (numba) twins of :func:`varest.partition.partition_grid` and :func:`varest.partition.agglo_edges`.

Same algorithm, same arithmetic, same merge order -- the outputs are bit-identical to the Python implementations
(``varest/tests/test_partition_fast.py`` checks every case against the original full-scan code):

* every quantity is an IEEE double computed with the same expressions in the same order (``(-s) / c`` for the grid's
  heap key, ``s / c`` for the edge list's, ``existing + absorbed`` for a merged pair, the ``1e-9`` staleness test);
  numba compiles them without ``fastmath``, so no reassociation or fused contraction is allowed;
* the heap is an explicit binary min-heap over ``(key, x, y)`` compared exactly as Python compares those tuples, and
  the pop sequence of a min-heap is determined by the multiset of its entries alone;
* union-find uses the same path halving.

Non-finite fields fall back to Python (Python's ordering of NaN tuples is not a total order and is not reproduced here).
``VAREST_NO_JIT=1`` disables the compiled path; without numba it is simply unavailable.
"""
from __future__ import annotations

import os

import numpy as np

try:
    if os.environ.get("VAREST_NO_JIT"):
        raise ImportError("VAREST_NO_JIT is set")
    from numba import njit
    AVAILABLE = True
except ImportError:          # pragma: no cover - exercised only without numba
    AVAILABLE = False

if AVAILABLE:
    @njit(cache=True, nogil=True)
    def _less(ak, ax, ay, bk, bx, by):
        # Python tuple order for (float, int, int): the first unequal element decides
        if ak != bk:
            return ak < bk
        if ax != bx:
            return ax < bx
        return ay < by

    @njit(cache=True, nogil=True)
    def _push(hk, hx, hy, size, k, x, y):
        if size == hk.shape[0]:
            nk = np.empty(2 * size + 16, np.float64); nk[:size] = hk[:size]
            nx = np.empty(2 * size + 16, np.int64); nx[:size] = hx[:size]
            ny = np.empty(2 * size + 16, np.int64); ny[:size] = hy[:size]
            hk, hx, hy = nk, nx, ny
        i = size
        hk[i] = k; hx[i] = x; hy[i] = y
        while i > 0:
            p = (i - 1) // 2
            if _less(hk[i], hx[i], hy[i], hk[p], hx[p], hy[p]):
                hk[i], hk[p] = hk[p], hk[i]; hx[i], hx[p] = hx[p], hx[i]; hy[i], hy[p] = hy[p], hy[i]
                i = p
            else:
                break
        return hk, hx, hy, size + 1

    @njit(cache=True, nogil=True)
    def _pop(hk, hx, hy, size):
        k, x, y = hk[0], hx[0], hy[0]
        size -= 1
        hk[0] = hk[size]; hx[0] = hx[size]; hy[0] = hy[size]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= size:
                break
            c = l
            r = l + 1
            if r < size and _less(hk[r], hx[r], hy[r], hk[l], hx[l], hy[l]):
                c = r
            if _less(hk[c], hx[c], hy[c], hk[i], hx[i], hy[i]):
                hk[i], hk[c] = hk[c], hk[i]; hx[i], hx[c] = hx[c], hx[i]; hy[i], hy[c] = hy[c], hy[i]
                i = c
            else:
                break
        return k, x, y, size

    @njit(cache=True, nogil=True)
    def _find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @njit(cache=True, nogil=True)
    def _slot(st, ln, pv, u, v):
        for j in range(st[u], st[u] + ln[u]):
            if pv[j] == v:
                return j
        return -1

    @njit(cache=True, nogil=True)
    def _append(st, ln, cp, pv, ps, pc, top, u, v, sv, cv):
        if ln[u] == cp[u]:                              # move u's list to the end of the pool with double capacity
            nc = max(4, 2 * cp[u])
            if top + nc > pv.shape[0]:
                m = max(2 * pv.shape[0], top + nc)
                a = np.empty(m, np.int64); a[:top] = pv[:top]; pv = a
                b = np.empty(m, np.float64); b[:top] = ps[:top]; ps = b
                c = np.empty(m, np.int64); c[:top] = pc[:top]; pc = c
            pv[top:top + ln[u]] = pv[st[u]:st[u] + ln[u]]
            ps[top:top + ln[u]] = ps[st[u]:st[u] + ln[u]]
            pc[top:top + ln[u]] = pc[st[u]:st[u] + ln[u]]
            st[u] = top
            cp[u] = nc
            top += nc
        j = st[u] + ln[u]
        pv[j] = v; ps[j] = sv; pc[j] = cv
        ln[u] += 1
        return pv, ps, pc, top

    @njit(cache=True, nogil=True)
    def _remove(st, ln, pv, ps, pc, u, v):
        j = _slot(st, ln, pv, u, v)
        last = st[u] + ln[u] - 1
        pv[j] = pv[last]; ps[j] = ps[last]; pc[j] = pc[last]
        ln[u] -= 1

    @njit(cache=True, nogil=True)
    def _merge(parent, n_ids, ea, eb, ew, n_live, target, negate, trace_n, trace_c):
        """Shared average-linkage loop over per-node neighbour lists (each pair's sum and count stored on both sides).
        ``negate``: the grid -- heap key ``(-s) / c`` (its sums hold NEGATED strengths) and a first insertion that
        ASSIGNS the edge value, where the edge list accumulates ``0.0 + w``; kept apart so a zero boundary keeps the
        sign of zero each original gives it. Returns the number of trace rows written."""
        deg = np.zeros(n_ids, np.int64)
        for e in range(ea.shape[0]):
            if ea[e] != eb[e]:
                deg[ea[e]] += 1
                deg[eb[e]] += 1
        cp = np.maximum(deg, 2)
        st = np.zeros(n_ids, np.int64)
        top = 0
        for u in range(n_ids):
            st[u] = top
            top += cp[u]
        size0 = max(2 * top, 64)
        pv = np.empty(size0, np.int64); ps = np.empty(size0, np.float64); pc = np.empty(size0, np.int64)
        ln = np.zeros(n_ids, np.int64)
        for e in range(ea.shape[0]):
            a, b = ea[e], eb[e]
            if a == b:                                  # a self-loop can never be merged (the dict versions skip it too)
                continue
            lo, hi = min(a, b), max(a, b)
            j = _slot(st, ln, pv, lo, hi)
            if j >= 0:
                k = _slot(st, ln, pv, hi, lo)
                ps[j] = ps[j] + ew[e]; pc[j] = pc[j] + 1
                ps[k] = ps[j]; pc[k] = pc[j]
            else:
                v0 = ew[e] if negate else 0.0 + ew[e]
                pv, ps, pc, top = _append(st, ln, cp, pv, ps, pc, top, lo, hi, v0, 1)
                pv, ps, pc, top = _append(st, ln, cp, pv, ps, pc, top, hi, lo, v0, 1)
        m = 0
        for u in range(n_ids):
            m += ln[u]
        hk = np.empty(m + 16, np.float64)
        hx = np.empty(m + 16, np.int64)
        hy = np.empty(m + 16, np.int64)
        size = 0
        for u in range(n_ids):
            for j in range(st[u], st[u] + ln[u]):
                v = pv[j]
                if u < v:
                    key = ((-ps[j]) / pc[j]) if negate else (ps[j] / pc[j])
                    hk, hx, hy, size = _push(hk, hx, hy, size, key, u, v)
        nt = 0
        nb = np.empty(16, np.int64); nbs = np.empty(16, np.float64); nbc = np.empty(16, np.int64)
        while n_live > target and size > 0:
            w, x, y, size = _pop(hk, hx, hy, size)
            rx, ry = _find(parent, x), _find(parent, y)
            if rx == ry:
                continue
            lo, hi = min(rx, ry), max(rx, ry)
            j = _slot(st, ln, pv, lo, hi)
            if j < 0:
                continue
            s_, c_ = ps[j], pc[j]
            cur = ((-s_) / c_) if negate else (s_ / c_)
            stale = (abs(-w - s_ / c_) > 1e-9) if negate else (abs(w - s_ / c_) > 1e-9)
            if stale:
                hk, hx, hy, size = _push(hk, hx, hy, size, cur, rx, ry)
                continue
            parent[ry] = rx
            n_live -= 1
            if trace_n.shape[0] > 0:
                trace_n[nt] = n_live
                trace_c[nt] = w
                nt += 1
            d = ln[ry]
            if d > nb.shape[0]:
                nb = np.empty(2 * d, np.int64); nbs = np.empty(2 * d, np.float64); nbc = np.empty(2 * d, np.int64)
            for q in range(d):
                nb[q] = pv[st[ry] + q]; nbs[q] = ps[st[ry] + q]; nbc[q] = pc[st[ry] + q]
            ln[ry] = 0
            for q in range(d):
                o, sv, cv = nb[q], nbs[q], nbc[q]
                _remove(st, ln, pv, ps, pc, o, ry)
                if _find(parent, o) == rx:
                    continue
                k1 = _slot(st, ln, pv, rx, o)
                if k1 >= 0:
                    k2 = _slot(st, ln, pv, o, rx)
                    ps[k1] = ps[k1] + sv; pc[k1] = pc[k1] + cv
                    ps[k2] = ps[k1]; pc[k2] = pc[k1]
                    sn, cn = ps[k1], pc[k1]
                else:
                    sn, cn = 0.0 + sv, cv
                    pv, ps, pc, top = _append(st, ln, cp, pv, ps, pc, top, rx, o, sn, cn)
                    pv, ps, pc, top = _append(st, ln, cp, pv, ps, pc, top, o, rx, sn, cn)
                key = ((-sn) / cn) if negate else (sn / cn)
                hk, hx, hy, size = _push(hk, hx, hy, size, key, rx, o)
        return nt


if AVAILABLE:
    @njit(cache=True, nogil=True)
    def _roots(parent, ids):
        out = np.empty(ids.shape[0], np.int64)
        for i in range(ids.shape[0]):
            out[i] = _find(parent, ids[i])
        return out


def grid_roots(neg: np.ndarray, k: int, want_trace: bool):
    """``neg`` = the NEGATED ``[2, n_r, n_c]`` field as float64 -> (root of every cell ``[n]``, trace rows)."""
    nr, nc = neg.shape[1], neg.shape[2]
    n = nr * nc
    idx = np.arange(n, dtype=np.int64).reshape(nr, nc)
    ea = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    eb = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    ew = np.concatenate([neg[0, :, :-1].ravel(), neg[1, :-1, :].ravel()]).astype(np.float64)
    parent = np.arange(n, dtype=np.int64)
    tn = np.empty(n if want_trace else 0, np.int64)
    tc = np.empty(n if want_trace else 0, np.float64)
    nt = _merge(parent, n, ea, eb, ew, n, int(k), True, tn, tc)
    roots = _roots(parent, np.arange(n, dtype=np.int64))
    return roots, [(int(tn[i]), float(tc[i])) for i in range(nt)]


def edge_roots(nodes, edges, n_target: int, arrays=None):
    """-> the root of every node (in ``nodes`` order) after merging to ``n_target`` components. ``arrays``: the same
    edges as ``(a int64, b int64, w float64)`` arrays (:class:`varest.partition.EdgeList`), skipping the list walk."""
    ids = np.asarray(nodes, np.int64)
    if arrays is not None:
        ea, eb, ew = (np.ascontiguousarray(arrays[0], np.int64), np.ascontiguousarray(arrays[1], np.int64),
                      np.ascontiguousarray(arrays[2], np.float64))
        n_ids = int(max(int(ids.max()), int(ea.max()), int(eb.max()))) + 1
    else:
        n_ids = int(max(int(ids.max()), max(max(a, b) for a, b, _ in edges))) + 1
        ea = np.asarray([a for a, _, _ in edges], np.int64)
        eb = np.asarray([b for _, b, _ in edges], np.int64)
        ew = np.asarray([float(w) for _, _, w in edges], np.float64)
    parent = np.arange(n_ids, dtype=np.int64)
    empty_i, empty_f = np.empty(0, np.int64), np.empty(0, np.float64)
    _merge(parent, n_ids, ea, eb, ew, len(nodes), int(n_target), False, empty_i, empty_f)
    return _roots(parent, ids).tolist()
