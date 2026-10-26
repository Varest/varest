"""OBJECT head: boundary strength between neighbouring cells, distilled from DINOv3 (released weights: w96h288).

The head regresses, per pair of adjacent cells, how strongly they belong APART -- the cosine distance between their
DINOv3 features (``varest.train.object_targets``). It sees RGB (object boundaries are often colour boundaries at
equal luminance), downsamples by 8 with three stride-2 stages, pools each output cell's 2x2 block by mean and
per-image z-scored variance, and applies dilated residual blocks (1, 2, 4, 8) for context. Output channel 0 is the
boundary to the RIGHT neighbour, channel 1 to the neighbour BELOW.

It predicts on its own 16-px cell grid at the image's aspect ratio (long side 768 by default). The tokenizer's cells are
coarser, so :func:`pool_edges` reduces the field by taking the MAX over the fine edges that cross each coarse
boundary: a boundary that is sharp anywhere along its length is a boundary, and averaging would dilute thin ridges.
"""
from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np
import torch
from torch import nn

CELL_PX = 16
WIDTH = 96
HID = 288
DILS = (1, 2, 4, 8)
LONG_PX = 768


def raw_grid(width: int, height: int, long_px: int = LONG_PX, patch: int = CELL_PX) -> tuple[int, int]:
    """Aspect-preserving cell grid for the raw image: long side -> ``long_px``, both divided by ``patch``.
    Returns ``(n_rows, n_cols)``, at least 2 each."""
    s = long_px / max(width, height)
    nc = max(2, int(round(width * s / patch)))
    nr = max(2, int(round(height * s / patch)))
    return nr, nc


class _ResBlock(nn.Module):
    def __init__(self, h: int, dil: int = 1) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(h, h, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv2d(h, h, 3, padding=dil, dilation=dil)
        self.n1 = nn.GroupNorm(8, h)
        self.n2 = nn.GroupNorm(8, h)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.c1(z)))
        return z + self.n2(self.c2(y))


class ObjectHead(nn.Module):
    """``[B, 3, n_r * cell_px, n_c * cell_px]`` RGB in [0, 1] -> ``[B, 2, n_r, n_c]`` boundary strength."""

    def __init__(self, width: int = WIDTH, hid: int = HID, dils: tuple[int, ...] = DILS, cell_px: int = CELL_PX,
                 rgb: bool = True, use_norm: bool = False) -> None:
        super().__init__()
        self.cell_px = int(cell_px)
        self.rgb = bool(rgb)
        self.width, self.hid, self.dils, self.use_norm = int(width), int(hid), tuple(int(d) for d in dils), bool(use_norm)
        w = self.width
        cin = 3 if self.rgb else 1
        self.stem = nn.Sequential(
            nn.Conv2d(cin, w, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, w), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 2 * w), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 4 * w), nn.GELU(),
        )
        self.norm = nn.GroupNorm(1, 8 * w) if self.use_norm else nn.Identity()
        self.proj = nn.Conv2d(8 * w, self.hid, 1)
        self.b = nn.ModuleList([_ResBlock(self.hid, dil=d) for d in self.dils])
        self.out = nn.Conv2d(self.hid, 2, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.cell_px // 8
        y = self.stem(x)
        c = y.shape[1]
        nr, nc = y.shape[2] // t, y.shape[3] // t
        y = y[..., :nr * t, :nc * t]
        cells = y.reshape(y.shape[0], c, nr, t, nc, t).permute(0, 1, 2, 4, 3, 5).reshape(y.shape[0], c, nr, nc, t * t)
        var = cells.var(-1, unbiased=False)
        m = var.mean(dim=(2, 3), keepdim=True)
        sd = var.std(dim=(2, 3), keepdim=True).clamp_min(1e-9)
        z = torch.cat([cells.mean(-1), (var - m) / sd], 1)
        z = self.act(self.proj(self.norm(z)))
        for blk in self.b:
            z = blk(z)
        return self.out(z)

    def config(self) -> dict[str, Any]:
        return {"width": self.width, "hid": self.hid, "dils": self.dils, "cell_px": self.cell_px, "rgb": self.rgb,
                "use_norm": self.use_norm}


def pool_edges(e: np.ndarray | torch.Tensor, nr: int, nc: int) -> torch.Tensor:
    """``[2, NR, NC]`` fine boundary strengths -> ``[2, nr, nc]`` on a coarser grid, by MAX over crossing edges.

    The coarse grid is mapped linearly onto the fine one, so this reduces onto ANY cell grid (a letterboxed window,
    a squashed view) without resampling the image. The right edge of coarse cell (i, j) is the max of fine column
    ``cb[j+1]-1`` over fine rows ``[rb[i], max(rb[i+1], rb[i]+1))``; the down edge likewise with rows and columns
    swapped; the last column / row has no right / down edge (0). Vectorised with ``np.maximum.reduceat``, which over
    ``[rb[i], rb[i+1])`` is exactly that max and, for an empty range, the single element ``rb[i]`` -- the same values as
    the per-cell loop it replaces (max is exact)."""
    e = np.asarray(torch.as_tensor(np.asarray(e), dtype=torch.float32))
    big_r, big_c = e.shape[1], e.shape[2]
    out = np.zeros((2, nr, nc), np.float32)
    rb = np.linspace(0, big_r, nr + 1).round().astype(int)
    cb = np.linspace(0, big_c, nc + 1).round().astype(int)
    if nc > 1:
        cols = np.maximum(cb[1:nc] - 1, 0)                     # the fine column each right edge crosses
        out[0, :, :nc - 1] = _segment_max(e[0][:, cols], rb[:nr], big_r, axis=0)
    if nr > 1:
        rows = np.maximum(rb[1:nr] - 1, 0)
        out[1, :nr - 1, :] = _segment_max(e[1][rows, :], cb[:nc], big_c, axis=1)
    return torch.from_numpy(out)


def _segment_max(a: np.ndarray, starts: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Max of ``a`` over ``[starts[k], starts[k+1])`` along ``axis`` (the last range runs to ``n``); an empty range
    gives the single element at its start, and a start at ``n`` itself (past the end) gives 0."""
    ok = starts < n
    r = np.maximum.reduceat(a, np.minimum(starts, n - 1), axis=axis)
    if not ok.all():
        idx = [slice(None)] * a.ndim
        idx[axis] = ~ok
        r[tuple(idx)] = 0.0
    return r


def object_head_from_state(blob: dict[str, Any]) -> ObjectHead:
    """Build an :class:`ObjectHead` from ``{"sd": state_dict, "width": ..., ...}`` or a bare state dict."""
    sd = blob["sd"] if isinstance(blob, dict) and "sd" in blob else blob
    meta = blob if isinstance(blob, dict) and "sd" in blob else {}
    kw = {k: meta[k] for k in ("width", "hid", "dils", "cell_px", "rgb", "use_norm") if k in meta}
    kw.setdefault("use_norm", any(str(k).startswith("norm.") for k in sd))
    head = ObjectHead(**kw)
    head.load_state_dict(sd)
    return head.eval()


def load_object_head(path: str | PathLike[str], device: str | torch.device = "cpu") -> ObjectHead:
    return object_head_from_state(torch.load(path, map_location="cpu", weights_only=False)).to(device)
