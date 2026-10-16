"""The two per-image fields the W5 tokenizer consumes, computed live from pixels by the TEXT and OBJECT heads.

LLaVA-1.5 (7B/13B) works on the 24x24 PADDED square its image processor produces (``do_pad=True``):

  text      the text head runs on the grayscale image at its own aspect (28 px per cell of the image's in-image
            cell shape); probabilities are written into the padded grid, 0 on letterbox cells.
  boundary  the object head runs on the RGB image at long side 768 (16-px cells), its edges are max-pooled onto the
            in-image cells, written into the padded grid, and the image is SEALED against the letterbox with a large
            edge (``PAD_EDGE``) so no region straddles image and padding; edges inside the padding are 0.

LLaVA-NeXT works on its two AnyRes views (see :mod:`varest.next`): the unpadded hi-res window and the 24x24 base
view (the whole image squashed to 336). Both carry a text and a boundary field; neither has letterbox cells.

Both fields are stored in float16, exactly as the training caches were; the builders read them back as float32.

DECODING. The released checkpoints were trained and evaluated on fields whose boundary branch decoded large JPEGs with
PIL's ``draft`` (DCT-domain downscaling to at least the head's input size). Passing a file path reproduces that decode;
passing a decoded image gives the plain full-resolution decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import numpy as np
import torch
from PIL import Image

from varest.heads.object import ObjectHead, pool_edges, raw_grid
from varest.heads.text import TextHead, cell_shape
from varest.tokenizer import GRID, TEXT_EDGE_W, kmin_from_curve, text_cut

PAD_EDGE = 9.0          # boundary between image and letterbox: above any real boundary, so regions never straddle it
OBJ_LONG = 768          # the object head's trained long side (48 cells of 16 px)

ImageLike = Image.Image | str | PathLike[str]


@dataclass
class LlavaFields:
    """Fields on LLaVA-1.5's 24x24 padded grid."""
    text: np.ndarray            # float16 [24, 24] probabilities, 0 on letterbox cells
    boundary: np.ndarray        # float16 [2, 24, 24] (right, down), letterbox sealed
    inimg: np.ndarray           # bool [24, 24], cells on real pixels
    size: tuple[int, int]       # (width, height) of the raw image


@dataclass
class ViewFields:
    """Fields on one LLaVA-NeXT view (no letterbox)."""
    text: np.ndarray            # float16 [n_r, n_c]
    boundary: np.ndarray        # float16 [2, n_r, n_c]

    def kmin(self, text_edge_w: float = TEXT_EDGE_W) -> int:
        """Raw merge-cost knee of this view (uncapped)."""
        return int(kmin_from_curve(text_cut(np.asarray(self.boundary, np.float32),
                                            np.asarray(self.text, np.float32), text_edge_w)))


def _device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


@torch.no_grad()
def text_probabilities(head: TextHead, image: Image.Image, n_rows: int, n_cols: int) -> np.ndarray:
    """Text head on ``image`` resized to ``n_cols x n_rows`` cells -> float32 ``[n_rows, n_cols]`` probabilities."""
    g = image.convert("L").resize((n_cols * head.cell_px, n_rows * head.cell_px), Image.BICUBIC)
    x = torch.from_numpy(np.array(g, dtype=np.uint8)).to(_device(head)).float()[None, None] / 255.0
    return torch.sigmoid(head(x)[0]).float().cpu().numpy().reshape(n_rows, n_cols)


@torch.no_grad()
def object_edges(head: ObjectHead, image: Image.Image, n_rows: int, n_cols: int) -> torch.Tensor:
    """Object head on ``image`` resized to ``n_cols x n_rows`` of its own cells -> float32 ``[2, n_rows, n_cols]``."""
    x = image.resize((n_cols * head.cell_px, n_rows * head.cell_px), Image.BICUBIC)
    a = np.asarray(x if head.rgb else x.convert("L"), np.uint8)
    t = torch.from_numpy(a.copy())
    t = (t.permute(2, 0, 1) if head.rgb else t[None]).float()[None].to(_device(head)) / 255.0
    return head(t)[0].float().cpu()


def _open(image: ImageLike, draft_to: tuple[int, int] | None, mode: str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert(mode)
    with Image.open(image) as im:
        if draft_to is not None:
            im.draft(mode, draft_to)
        return im.convert(mode)


def _size(image: ImageLike) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    with Image.open(image) as im:
        return im.size


def llava_fields(image: ImageLike, text_head: TextHead, object_head: ObjectHead) -> LlavaFields:
    """Both fields of one image on LLaVA-1.5's padded 24x24 grid. ``image`` is a PIL image or a file path."""
    w, h = _size(image)
    r0, nr, c0, nc = cell_shape(w, h)
    inimg = np.zeros((GRID, GRID), bool)
    inimg[r0:r0 + nr, c0:c0 + nc] = True
    # text: the head on the in-image cells; 0 (no textness) on the letterbox
    if isinstance(image, Image.Image):
        gray = image
    else:
        with Image.open(image) as im:
            gray = im.convert("L")
    text = np.zeros((GRID, GRID), np.float32)
    text[r0:r0 + nr, c0:c0 + nc] = text_probabilities(text_head, gray, nr, nc)
    # boundary: the head at its own resolution, max-pooled onto the in-image cells, sealed against the letterbox
    big_r, big_c = raw_grid(w, h, OBJ_LONG, object_head.cell_px)
    rgb = _open(image, (big_c * object_head.cell_px, big_r * object_head.cell_px), "RGB")
    p = pool_edges(object_edges(object_head, rgb, big_r, big_c), nr, nc).numpy()
    bnd = np.zeros((2, GRID, GRID), np.float32)
    bnd[:, r0:r0 + nr, c0:c0 + nc] = p
    if c0 > 0:
        bnd[0, r0:r0 + nr, c0 - 1] = PAD_EDGE
    if c0 + nc < GRID:
        bnd[0, r0:r0 + nr, c0 + nc - 1] = PAD_EDGE
    if r0 > 0:
        bnd[1, r0 - 1, c0:c0 + nc] = PAD_EDGE
    if r0 + nr < GRID:
        bnd[1, r0 + nr - 1, c0:c0 + nc] = PAD_EDGE
    bnd[0, :, -1] = 0.0
    bnd[1, -1, :] = 0.0
    return LlavaFields(text=text.astype(np.float16), boundary=bnd.astype(np.float16), inimg=inimg, size=(w, h))


def next_fields(image: ImageLike, text_head: TextHead, object_head: ObjectHead, window: tuple[int, int],
                base: int = GRID) -> tuple[ViewFields, ViewFields]:
    """Fields of both LLaVA-NeXT views: ``(base view [base x base], hi-res window [n_r x n_c])``.

    The object head runs ONCE on the aspect-preserved image (long side 768, or 16 px per window cell when the window
    is longer than 48 cells) and is max-pooled onto both grids; the text head runs at its own cell size on each view's
    cell shape (the base view is the whole image squashed)."""
    nr, nc = window
    w, h = _size(image)
    long_px = max(OBJ_LONG, object_head.cell_px * max(nr, nc))
    big_r, big_c = raw_grid(w, h, long_px, object_head.cell_px)
    target = (max(big_c * object_head.cell_px, nc * text_head.cell_px),
              max(big_r * object_head.cell_px, nr * text_head.cell_px))
    rgb = _open(image, target, "RGB")
    e = object_edges(object_head, rgb, big_r, big_c)
    b = pool_edges(e, nr, nc).numpy()
    b[0, :, -1] = 0.0
    b[1, -1, :] = 0.0
    bb = pool_edges(e, base, base).numpy()
    bb[0, :, -1] = 0.0
    bb[1, -1, :] = 0.0
    hi = ViewFields(text=text_probabilities(text_head, rgb, nr, nc).astype(np.float16), boundary=b.astype(np.float16))
    lo = ViewFields(text=text_probabilities(text_head, rgb, base, base).astype(np.float16),
                    boundary=bb.astype(np.float16))
    return lo, hi
