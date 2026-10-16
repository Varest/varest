"""TEXT head: per-cell textness from raw grayscale pixels (v5 architecture; the released weights are "v5n").

The head runs on the image at its own aspect ratio -- no letterbox, no CLIP features. The image is resized so that one
cell is ``cell_px`` pixels (28 by default), two stride-2 convolutions take each cell to ``cell_px / 4`` pixels, the
remaining intra-cell pixels are summarised by their MEAN and (per-image z-scored) VARIANCE, dilated residual blocks
supply context between cells, and a 1x1 convolution emits one logit per cell. Text is defined by relative
high-frequency energy, which is why the variance channel is standardised per image.

"v5n" is this architecture trained on OCR-positive images WITH text-free negatives (``varest.train.text_head``);
the negatives stop the head firing on fabric patterns, faces and bottle ridges, which the W tokenizers would
otherwise pin at native resolution.
"""
from __future__ import annotations

from os import PathLike
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

CELL_PX = 28
WIDTH = 32
HID = 128
DILS = (1, 2, 4)


def cell_shape(width: int, height: int, grid: int = 24) -> tuple[int, int, int, int]:
    """Raw image size -> ``(r0, n_rows, c0, n_cols)``: the cells of LLaVA's padded ``grid x grid`` square that lie on
    real pixels. A cell counts when its CENTRE is on the image. One rule for training and inference."""
    s = max(width, height)
    oy = (s - height) / 2
    ox = (s - width) / 2
    c = (torch.arange(grid).float() + 0.5) * s / grid
    rows = torch.nonzero((c >= oy) & (c < oy + height)).flatten()
    cols = torch.nonzero((c >= ox) & (c < ox + width)).flatten()
    return int(rows[0]), len(rows), int(cols[0]), len(cols)


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


class TextHead(nn.Module):
    """``[B, 1, n_r * cell_px, n_c * cell_px]`` grayscale in [0, 1] -> ``[B, n_r, n_c]`` textness LOGITS."""

    def __init__(self, width: int = WIDTH, hid: int = HID, dils: tuple[int, ...] = DILS, cell_px: int = CELL_PX,
                 use_norm: bool = False) -> None:
        super().__init__()
        self.cell_px = int(cell_px)
        self.width, self.hid, self.dils, self.use_norm = int(width), int(hid), tuple(int(d) for d in dils), bool(use_norm)
        w = self.width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, w), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 2 * w), nn.GELU(),
        )
        self.norm = nn.GroupNorm(1, 4 * w) if self.use_norm else nn.Identity()
        self.proj = nn.Conv2d(4 * w, self.hid, 1)
        self.b = nn.ModuleList([_ResBlock(self.hid, dil=d) for d in self.dils])
        self.out = nn.Conv2d(self.hid, 1, 1)
        self.act = nn.GELU()

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        t = self.cell_px // 4
        y = self.stem(gray)
        nr, nc = y.shape[2] // t, y.shape[3] // t
        y = y[..., :nr * t, :nc * t]
        # Per-cell moments by average pooling (Var = E[x^2] - E[x]^2). Every field the released W5 checkpoints were
        # trained and evaluated on was computed this way; it equals the reshape form to ~1e-5 at the logit, which is
        # below float16 resolution but not bit-identical, so the cached and the live fields agree only with this form.
        mu = F.avg_pool2d(y, t)
        var = (F.avg_pool2d(y * y, t) - mu * mu).clamp_min(0)
        m = var.mean(dim=(2, 3), keepdim=True)
        sd = var.std(dim=(2, 3), keepdim=True).clamp_min(1e-9)
        z = torch.cat([mu, (var - m) / sd], 1)
        x = self.act(self.proj(self.norm(z)))
        for blk in self.b:
            x = blk(x)
        return self.out(x)[:, 0]

    def config(self) -> dict[str, Any]:
        return {"width": self.width, "hid": self.hid, "dils": self.dils, "cell_px": self.cell_px,
                "use_norm": self.use_norm}


def text_head_from_state(blob: dict[str, Any]) -> TextHead:
    """Build a :class:`TextHead` from a saved blob ``{"sd": state_dict, "width": ..., ...}`` or a bare state dict.

    ``use_norm`` is inferred from the weights when the blob does not record it (every released head predates the
    knob), so the module built always matches the tensors it is given."""
    sd = blob["sd"] if isinstance(blob, dict) and "sd" in blob else blob
    if any(k.startswith("stems.0.") for k in sd):             # the single-scale v6 layout of the same network
        sd = {("stem." + k[len("stems.0."):]) if k.startswith("stems.0.") else k: v for k, v in sd.items()}
    meta = blob if isinstance(blob, dict) and "sd" in blob else {}
    kw = {k: meta[k] for k in ("width", "hid", "dils", "cell_px", "use_norm") if k in meta}
    kw.setdefault("use_norm", any(str(k).startswith("norm.") for k in sd))
    head = TextHead(**kw)
    head.load_state_dict(sd)
    return head.eval()


def load_text_head(path: str | PathLike[str], device: str | torch.device = "cpu") -> TextHead:
    return text_head_from_state(torch.load(path, map_location="cpu", weights_only=False)).to(device)
