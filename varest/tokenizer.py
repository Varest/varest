"""The W5 tokenizer: a text-first partition of a cell grid into EXACTLY K regions, one visual token each.

Given a budget K, the boundary field of the OBJECT head and the textness of the TEXT head on the same cells:

1. The text field's own gradient is folded into the boundary map (elementwise max, weight ``TEXT_EDGE_W``), so region
   borders also fall on text borders -- otherwise a caption on a uniform wall would be one object region.
2. Text is decided FIRST. A cell is text when its probability exceeds the budget-dependent pin threshold
   ``tau(K)``, linear in K from 1.0 at K=0 to 0.5 at K=576 (so a small budget pins only confident text).
3. ``k_min`` -- the knee of the agglomerative merge-cost curve, relative to the image's own boundary distribution and
   capped at K/2 -- is the fewest object regions that still respect the image's structure.
4. If the remaining budget covers ``k_min``, every text cell is PINNED as its own token (native resolution) and the
   rest of the budget becomes that many object regions. Otherwise (documents) ``k_min`` object regions are kept and
   the text area is partitioned into the remaining tokens.
5. Letterbox cells (7B padded square) form ONE extra token.
6. Regions are renumbered in raster order of their first cell, so the emitted sequence is a top-to-bottom,
   left-to-right sweep -- the traversal LLaVA makes over its native patches.

Each region is emitted as the MEAN of its cells' features plus a position tag ``(cx, cy, size)`` in the grid's
normalised coordinates (see :func:`pool_regions`). "W5" is this rule with the v5n text head; the same builder serves a
LLaVA-1.5 24x24 grid and each view of LLaVA-NeXT.
"""
from __future__ import annotations

import numpy as np
import torch

from varest.partition import agglo_edge_labels, bridge_components, cell_edges, partition_grid

GRID = 24
TEXT_EDGE_W = 0.25          # weight of the text field's gradient in the boundary map
KMIN_Q = 0.75               # merge-cost knee, as a quantile of the image's own boundary distribution
KMIN_FRAC = 0.5             # k_min <= K * KMIN_FRAC, so text keeps at least half the budget
TAU_K0, TAU_KMAX, TAU_SPAN = 1.0, 0.5, 576   # pin threshold: 1.0 at K=0 -> 0.5 at K=576, linear in K


def pin_threshold(k: float) -> float:
    """The W5 text pin threshold ``tau(K)``: 1.0 at K=0 -> 0.5 at K=576, linear in K, clamped.
    0.972 / 0.944 / 0.889 / 0.806 at K = 32 / 64 / 128 / 224."""
    a = min(1.0, max(0.0, float(k) / TAU_SPAN))
    return TAU_K0 + (TAU_KMAX - TAU_K0) * a


def _shape(grid: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(grid, (tuple, list)):
        return int(grid[0]), int(grid[1])
    return int(grid), int(grid)


def text_cut(boundary: np.ndarray, text: np.ndarray, w: float = TEXT_EDGE_W) -> np.ndarray:
    """Fold the text field's gradient into the boundary map: ``max(boundary, w * |Delta text|)`` per edge."""
    b = np.array(boundary, np.float32)
    t = np.asarray(text, np.float32).reshape(b.shape[1], b.shape[2])
    if w <= 0:
        return b
    b[0, :, :-1] = np.maximum(b[0, :, :-1], w * np.abs(t[:, :-1] - t[:, 1:]))
    b[1, :-1, :] = np.maximum(b[1, :-1, :], w * np.abs(t[:-1, :] - t[1:, :]))
    return b


def kmin_from_curve(boundary: np.ndarray, q: float = KMIN_Q, cap: int | None = None) -> int:
    """Structural floor on the region count: the count at which the merge cost (weakest-boundary-first) first exceeds
    the ``q``-quantile of the image's own boundary strengths. Pass the field the partition uses (post
    :func:`text_cut`). ``cap`` clamps the result (the builder uses ``K * KMIN_FRAC``)."""
    b = np.asarray(boundary, np.float32)
    trace: list[tuple[int, float]] = []
    partition_grid(torch.from_numpy(b), 2, trace=trace)
    if not trace:
        return 1
    nreg = np.array([x[0] for x in trace])
    cost = np.array([x[1] for x in trace])
    thr = float(np.quantile(np.maximum(b[0], b[1]).ravel(), q))
    k = int(nreg[np.argmax(cost > thr)]) if (cost > thr).any() else 1
    return max(1, min(k, cap) if cap else k)


def _build(b: np.ndarray, t: np.ndarray, k: int, inimg: np.ndarray, grid: tuple[int, int], kmin: int | None,
           text_thr: float) -> tuple[np.ndarray, dict]:
    nr, nc = grid
    inimg = np.asarray(inimg, bool)
    t2 = np.asarray(t, np.float32).reshape(nr, nc)
    npad = int((~inimg).sum())
    kb = int(k) - (1 if npad else 0)
    if kb < 2:
        raise ValueError(f"the W tokenizer needs K >= 3 with letterbox padding; got K={k}")
    txt = (t2 > float(text_thr)) & inimg
    n_text = int(txt.sum())
    cap = max(1, int(round(k * KMIN_FRAC)))
    k_min = min(int(kmin), cap) if kmin is not None else kmin_from_curve(b, cap=cap)
    k_min = max(1, k_min)
    obj_cells = np.nonzero((inimg & ~txt).reshape(-1))[0]
    txt_cells = np.nonzero(txt.reshape(-1))[0]
    n_obj_cells = len(obj_cells)
    if kb - n_text >= k_min and n_text > 0:
        pinned, k_obj_n, regime = True, kb - n_text, "pin"
    else:
        pinned, k_obj_n, regime = False, min(k_min, max(kb - 1, 1)), "kmin"
    k_obj_n = min(k_obj_n, n_obj_cells) if n_obj_cells else 0
    k_txt_n = kb - k_obj_n
    if n_text == 0:
        k_obj_n = min(kb, n_obj_cells)
        k_txt_n = 0
    k_txt_n = min(k_txt_n, len(txt_cells))
    slack = kb - k_obj_n - k_txt_n
    if slack > 0:
        take = min(slack, n_obj_cells - k_obj_n)
        k_obj_n += take
        slack -= take
        k_txt_n += min(slack, len(txt_cells) - k_txt_n)
    out = np.full((nr, nc), -1, np.int64)
    nxt = 0
    for cells, want in ((txt_cells, k_txt_n), (obj_cells, k_obj_n)):
        if len(cells) == 0 or want <= 0:
            continue
        if want >= len(cells):
            out.reshape(-1)[cells] = nxt + np.arange(len(cells))
            nxt += len(cells)
            continue
        ed = bridge_components(cells, cell_edges(cells, b, nc), nc)
        sub = agglo_edge_labels(cells, ed, int(want))          # component of each cell, first-appearance numbering
        out.reshape(-1)[cells] = nxt + sub
        nxt += int(sub.max()) + 1 if len(sub) else 0
    if npad:
        out[~inimg] = nxt
        nxt += 1
    flat = out.reshape(-1)
    # raster order of each region's first cell: a stable sort of the first indices (a label with no cell sorts as
    # index 0, as np.argmax of an all-False mask did) -- the same order as sorting by np.argmax(flat == j)
    first = np.zeros(nxt, np.int64)
    u, fi = np.unique(flat, return_index=True)
    ok = (u >= 0) & (u < nxt)
    first[u[ok]] = fi[ok]
    order = np.argsort(first, kind="stable")
    remap = np.empty(nxt, np.int64)
    remap[order] = np.arange(nxt, dtype=np.int64)
    out = remap[flat].reshape(nr, nc)
    info = {"tokens": int(nxt), "n_text": n_text, "k_min": int(k_min), "k_obj": int(k_obj_n), "k_txt": int(k_txt_n),
            "regime": regime, "pinned": bool(pinned), "text_thr": float(text_thr)}
    return out, info


def build_partition(boundary: np.ndarray, text: np.ndarray, k: int, *, grid: int | tuple[int, int] = GRID,
                    inimg: np.ndarray | None = None, kmin: int | None = None, tau_k: float | None = None,
                    text_edge_w: float = TEXT_EDGE_W) -> tuple[np.ndarray, dict]:
    """-> ``(labels [n_r, n_c], info)`` with EXACTLY ``k`` regions (fewer only if the grid has fewer cells).

    boundary  ``[2, n_r, n_c]`` edge strengths (right, down); letterbox edges sealed by the caller.
    text      textness PROBABILITIES on the same cells (``[n_r * n_c]`` or ``[n_r, n_c]``).
    inimg     cells on real pixels (default: all); the rest become one letterbox token.
    kmin      cached raw merge-cost knee of ``text_cut(boundary, text)``; ``None`` computes it.
    tau_k     the budget that sets the pin threshold, in LLaVA-1.5 units; defaults to ``k``. LLaVA-NeXT views pass
              ``576 * retain_ratio`` so a view compressed to the same fraction as a LLaVA-1.5 image gets its threshold.
    """
    nr, nc = _shape(grid)
    t = np.asarray(text, np.float32).reshape(nr, nc)
    b = text_cut(np.asarray(boundary, np.float32), t, float(text_edge_w))
    if inimg is None:
        inimg = np.ones((nr, nc), bool)
    thr = pin_threshold(k if tau_k is None else tau_k)
    lab, info = _build(b, t, int(k), inimg, (nr, nc), kmin, thr)
    info["tau_k"] = float(k if tau_k is None else tau_k)
    return lab, info


def pool_regions(features: torch.Tensor, labels: np.ndarray, grid: int | tuple[int, int] = GRID
                 ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Mean-pool per-cell features into one token per region.

    features  ``[n_r * n_c, D]`` in raster order (any dtype; accumulated in float32).
    labels    ``[n_r, n_c]`` region ids ``0..n-1`` (raster-ordered by :func:`build_partition`).
    grid      an int for LLaVA-1.5's square grid, a ``(n_r, n_c)`` pair for a LLaVA-NeXT view (the two were trained
              with slightly different float arithmetic for the size term, reproduced exactly here).
    Returns ``(tokens float32 [n, D], positions float32 [n, 3], cells-per-region [n])`` where a position is
    ``((cx + 0.5) / n_c, (cy + 0.5) / n_r, sqrt(cells / (n_r * n_c)))``; on the 24x24 grid the size term is
    ``sqrt(cells) / 24``."""
    nr, nc = _shape(grid)
    lf = np.ascontiguousarray(np.asarray(labels).reshape(-1))
    n = int(lf.max()) + 1
    lt = torch.from_numpy(lf).to(features.device, non_blocking=True).long()
    d = features.shape[-1]
    acc = torch.zeros(n, d, device=features.device, dtype=torch.float32).index_add_(0, lt, features.float())
    cnt = torch.bincount(lt, minlength=n).unsqueeze(-1).float()
    tokens = acc / cnt.clamp(min=1)
    ys, xs = np.divmod(np.arange(nr * nc), nc)
    if isinstance(grid, int):
        # LLaVA-1.5's square grid: float32 counts, size = sqrt(cells) / grid (the trained arithmetic, bit for bit)
        c = np.bincount(lf, minlength=n).astype(np.float32)
        size = np.sqrt(np.maximum(c, 1)) / nr
    else:
        # a LLaVA-NeXT view: float64 counts, size = sqrt(cells / view cells)
        c = np.bincount(lf, minlength=n).astype(np.float64)
        size = np.sqrt(np.maximum(c, 1) / float(nr * nc))
    cx = np.bincount(lf, weights=xs, minlength=n) / np.maximum(c, 1)
    cy = np.bincount(lf, weights=ys, minlength=n) / np.maximum(c, 1)
    pos = np.stack([(cx + .5) / nc, (cy + .5) / nr, size], 1).astype(np.float32)
    return tokens, pos, c
