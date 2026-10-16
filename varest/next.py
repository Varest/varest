"""W5 on LLaVA-NeXT's AnyRes input: both views, harness retain-ratio budgets.

NeXT encodes an image twice: a BASE view (the whole image squashed to 336x336, 24x24 CLIP cells) and a HIGH-RES view
(the image resized into the best-fitting pinpoint canvas, cut into 336 tiles, stitched, and the letterbox removed by
``unpad_image``). Both views are tokenized by the unchanged W5 builder and emitted base-first, each in raster order,
with NO newline tokens -- every token carries its ``(cx, cy, size)`` tag in image coordinates instead.

BUDGET. A budget is a RETAIN RATIO applied per image to NeXT's native visual length
``N_native = 576 + n_r * n_c + n_r`` (base + hi-res cells + one newline per hi-res row): ``kept = floor(N_native * r)``.
The standard ratios are b/576 for b = 32/64/128/224 (0.0556/0.1111/0.2222/0.3889). Because no newline tokens are
emitted, every kept token is content.

SPLIT. The K tokens go to the two views in proportion to their cells, so both are compressed to the same fraction.
The pin threshold uses ``tau_k = 576 * K / N_native`` -- the LLaVA-1.5 budget at the same retain ratio -- so a view
gets the threshold LLaVA-1.5 would use at that compression, independent of how many cells the canvas has.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from varest.tokenizer import GRID, TEXT_EDGE_W, build_partition, pool_regions

PINPOINTS = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]   # llama3-llava-next-8b-hf
TILE = 336
NB = GRID * GRID
STANDARD_RATIOS = {32: 0.0556, 64: 0.1111, 128: 0.2222, 224: 0.3889}


@dataclass(frozen=True)
class Layout:
    """AnyRes layout of one image: ``nh x nw`` tiles, stitched grid ``gh x gw``, unpadded window rows
    ``r0:r0+nr`` and columns ``c0:c0+nc``."""
    nh: int
    nw: int
    gh: int
    gw: int
    r0: int
    nr: int
    c0: int
    nc: int


def layout(width: int, height: int, pinpoints: list[list[int]] = PINPOINTS) -> Layout:
    """Raw size -> :class:`Layout`, computed by calling the model's own ``select_best_resolution`` and
    ``unpad_image`` on an index map, so the window is by construction what the model itself keeps."""
    from transformers.image_processing_utils import select_best_resolution
    from transformers.models.llava_next.modeling_llava_next import unpad_image
    bh, bw = select_best_resolution((int(height), int(width)), pinpoints)
    nh, nw = bh // TILE, bw // TILE
    gh, gw = GRID * nh, GRID * nw
    u = unpad_image(torch.arange(gh * gw).reshape(1, gh, gw), (int(height), int(width)))
    first = int(u[0, 0, 0])
    return Layout(nh, nw, gh, gw, first // gw, int(u.shape[1]), first % gw, int(u.shape[2]))


def stitch(tile_features: torch.Tensor, lay: Layout) -> torch.Tensor:
    """``[nh*nw, 576, D]`` per-tile features (row-major tiles) -> ``[nr*nc, D]`` window cells in raster order."""
    d = tile_features.shape[-1]
    g = tile_features.reshape(lay.nh, lay.nw, GRID, GRID, d).permute(0, 2, 1, 3, 4).reshape(lay.gh, lay.gw, d)
    return g[lay.r0:lay.r0 + lay.nr, lay.c0:lay.c0 + lay.nc].reshape(lay.nr * lay.nc, d)


def n_native(nr: int, nc: int) -> int:
    """NeXT's packed visual length: base + unpadded hi-res + one newline per hi-res row."""
    return NB + nr * nc + nr


def kept(nr: int, nc: int, ratio: float) -> int:
    """The harness's per-image budget ``max(1, floor(N_native * ratio))``."""
    return max(1, int(math.floor(n_native(nr, nc) * float(ratio))))


def split(k: int, nr: int, nc: int, kmin_view: int = 3) -> tuple[int, int, int, float]:
    """K tokens -> ``(K, K_base, K_hi, tau_k)``. K is clamped to the content capacity ``576 + nr*nc``; each view keeps
    at least ``kmin_view`` tokens."""
    nh = nr * nc
    k = int(min(max(int(k), 2 * kmin_view), NB + nh))
    kb = int(round(k * NB / (NB + nh)))
    kb = min(max(kb, kmin_view), NB, k - kmin_view)
    kh = k - kb
    if kh > nh:
        kh = nh
        kb = k - kh
    return k, kb, kh, NB * k / n_native(nr, nc)


@dataclass
class NextPlan:
    """Both views' partitions for one image at one budget (the CPU half of :func:`tokenize`)."""
    k: int
    k_base: int
    k_hi: int
    tau_k: float
    base_labels: np.ndarray
    hi_labels: np.ndarray


def plan(base_fields: tuple[np.ndarray, np.ndarray, int | None], hi_fields: tuple[np.ndarray, np.ndarray, int | None],
         lay: Layout, k: int, text_edge_w: float = TEXT_EDGE_W) -> NextPlan:
    """Partition both views. ``*_fields = (boundary [2,r,c], text [r,c], kmin or None)`` for base (24x24) and hi-res
    window (nr x nc)."""
    k, kb, kh, tau = split(k, lay.nr, lay.nc)
    bb, bt, bk = base_fields
    hb, ht, hk = hi_fields
    lb, _ = build_partition(bb, bt, kb, grid=(GRID, GRID), kmin=bk, tau_k=tau, text_edge_w=text_edge_w)
    lh, _ = build_partition(hb, ht, kh, grid=(lay.nr, lay.nc), kmin=hk, tau_k=tau, text_edge_w=text_edge_w)
    return NextPlan(k, kb, kh, tau, lb, lh)


def tokenize(base_features: torch.Tensor, window_features: torch.Tensor, p: NextPlan, lay: Layout
             ) -> tuple[torch.Tensor, np.ndarray]:
    """Pool both views: base ``[576, D]`` and window ``[nr*nc, D]`` projected features -> ``(tokens float32 [K, D],
    positions float32 [K, 3])``, base view first."""
    tb, pb, _ = pool_regions(base_features, p.base_labels, (GRID, GRID))
    th, ph, _ = pool_regions(window_features, p.hi_labels, (lay.nr, lay.nc))
    if tb.shape[0] != p.k_base or th.shape[0] != p.k_hi:
        raise RuntimeError(f"W5 emitted {tb.shape[0]}+{th.shape[0]} tokens, budget {p.k_base}+{p.k_hi}")
    return torch.cat([tb, th], 0), np.concatenate([pb, ph], 0)
