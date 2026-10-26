"""Image decoding and resizing on the GPU: the field heads' inputs and the CLIP input from ONE device-side decode.

The PIL path (:mod:`varest.fields` given a file path) is what the released checkpoints were trained and evaluated on.
This module keeps PIL's decode (exact) and reproduces everything after it on the device, to within one grey level:

  decode   PIL, then one upload (default): the decoded pixels are exactly the PIL path's. ``nvjpeg=True`` decodes JPEGs
           with nvJPEG instead (``torchvision.io.decode_jpeg(device=...)``), faster but NOT the same pixels: nvJPEG
           matches libjpeg bit for bit on most images, yet differs by > 2 levels on ~2.5% of pixels on average, and the
           object head's partition is sensitive to that (7B, 61 VarestBench JPEGs: 37 partitions identical vs 53 with
           the PIL decode); the text head and CLIP are not.
  grey     PIL's ``convert("L")`` integer luma, ``(19595 R + 38470 G + 7471 B + 2^15) >> 16``: exact.
  resize   PIL's BICUBIC as PIL computes it: antialiased (the kernel support scales with the reduction), a horizontal
           pass then a vertical pass, each rounded and clipped to 8 bits. The clip BETWEEN the passes matters: without
           it bicubic overshoot on sharp edges survives into the second pass (up to 23 grey levels off PIL); with it
           the result is within 1 of PIL on all but ~0.01% of pixels.
  draft    the object head's JPEG input was decoded by PIL with ``draft``: libjpeg's scaled IDCT at 1/2, 1/4 or 1/8 of
           full size. From a file this is PIL's own draft decode (a second, reduced-size decode, only for JPEGs at
           least twice the head's input). With ``nvjpeg=True`` it is emulated: the full decode, edge-replicated to
           whole blocks and area-averaged by the same factor, rounded to 8 bits (at 1/8 the scaled IDCT is exactly
           the block mean, the DC term).

Images are float32 ``[C, H, W]`` tensors holding integer values in [0, 255].
"""
from __future__ import annotations

from os import PathLike, fspath

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision.io import ImageReadMode, decode_jpeg
except ImportError:                                   # no torchvision: everything is decoded by PIL
    decode_jpeg = None


def luma(rgb: torch.Tensor) -> torch.Tensor:
    """``[3, H, W]`` -> ``[1, H, W]``, PIL's ``convert("L")`` exactly."""
    r, g, b = rgb.to(torch.int32).unbind(0)
    return ((r * 19595 + g * 38470 + b * 7471 + 32768) >> 16).to(torch.float32)[None]


def resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """PIL ``resize(size, BICUBIC)`` of a ``[C, H, W]`` image; ``size = (width, height)``."""
    w, h = size
    hh, ww = x.shape[-2:]
    y = x[None]
    if w != ww:
        y = F.interpolate(y, size=(hh, w), mode="bicubic", antialias=True, align_corners=False).round_().clamp_(0, 255)
    if h != hh:
        y = F.interpolate(y, size=(h, w), mode="bicubic", antialias=True, align_corners=False).round_().clamp_(0, 255)
    return y[0]


def draft_scale(size: tuple[int, int], target: tuple[int, int]) -> int:
    """The reduction PIL's JPEG ``draft(mode, target)`` picks: the largest of 8, 4, 2, 1 that keeps both sides
    at least ``target`` (1 when the image is already smaller)."""
    s = min(size[0] // target[0], size[1] // target[1])
    return next((d for d in (8, 4, 2) if s >= d), 1)


def _upload(a: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 ``[H, W, 3]`` -> float32 ``[3, H, W]`` on ``device``."""
    return torch.from_numpy(a).to(device).permute(2, 0, 1).float()


def as_uint8_chw(image: torch.Tensor) -> torch.Tensor:
    """A decoded image tensor as uint8 ``[3, H, W]`` with exactly the pixels :func:`varest.runtime.to_pil` reads from it:
    ``[C, H, W]`` unless the last axis is the channel axis, float in [0, 1] (scaled) or [0, 255], grey replicated,
    alpha dropped."""
    t = torch.as_tensor(image).detach()
    if t.ndim == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
        t = t.permute(1, 2, 0)
    if t.is_floating_point():
        t = (t * (255.0 if float(t.max()) <= 1.0 else 1.0)).round().clamp(0, 255)
    t = t.to(torch.uint8)
    if t.ndim == 2:
        t = t[..., None]
    if t.shape[-1] == 1:
        t = t.expand(*t.shape[:-1], 3)
    elif t.shape[-1] == 4:
        t = t[..., :3]
    return t.permute(2, 0, 1)


class DeviceImage:
    """A decoded image on the device. :meth:`draft` reduces it the way PIL's ``draft`` would have: from a file
    (``path``) PIL itself decides (JPEG and MPO files draft, others do not) and decodes the reduced image; for an
    nvJPEG decode (``jpeg``, no ``path``) the reduction is emulated."""

    def __init__(self, rgb: torch.Tensor, jpeg: bool = False, path: str | PathLike[str] | None = None) -> None:
        self.rgb = rgb                                   # float32 [3, H, W], integer values
        self.jpeg = jpeg
        self.path = path
        self._gray: torch.Tensor | None = None

    @classmethod
    def open(cls, path: str | PathLike[str], device: torch.device, nvjpeg: bool = False) -> DeviceImage:
        with Image.open(path) as im:
            if nvjpeg and decode_jpeg is not None and im.format == "JPEG" and im.mode in ("RGB", "L"):
                data = torch.from_numpy(np.fromfile(fspath(path), np.uint8))
                try:
                    return cls(decode_jpeg(data, mode=ImageReadMode.RGB, device=device).float(), jpeg=True)
                except RuntimeError:                     # a JPEG nvJPEG rejects: PIL below
                    pass
            im.load()
            a = np.array(im.convert("RGB"))
        return cls(_upload(a, device), path=path)

    @classmethod
    def from_pil(cls, image: Image.Image, device: torch.device) -> DeviceImage:
        """An already-decoded image (no draft: :mod:`varest.fields` given a PIL image uses the full decode too)."""
        return cls(_upload(np.array(image.convert("RGB")), device))

    @classmethod
    def from_tensor(cls, image: torch.Tensor, device: torch.device) -> DeviceImage:
        """A decoded image tensor (``[C, H, W]`` or ``[H, W, C]`` or ``[H, W]``; uint8, or float in [0, 1] / [0, 255]),
        read exactly as :func:`varest.runtime.to_pil` reads it (same layout rule, float scaling, grey replication,
        alpha dropped) but uploaded as is, without the PIL round trip. The Varest harness passes images this way."""
        return cls(as_uint8_chw(image).to(device).float())

    @property
    def size(self) -> tuple[int, int]:
        """(width, height), as PIL's ``Image.size``."""
        return int(self.rgb.shape[-1]), int(self.rgb.shape[-2])

    def gray(self) -> torch.Tensor:
        if self._gray is None:
            self._gray = luma(self.rgb)
        return self._gray

    def resized(self, size: tuple[int, int], gray: bool = False) -> torch.Tensor:
        """``[3 or 1, h, w]``: PIL's ``resize(size, BICUBIC)`` of the RGB (or, ``gray``, the L) image."""
        return resize(self.gray() if gray else self.rgb, size)

    def draft(self, target: tuple[int, int]) -> DeviceImage:
        """PIL's ``draft("RGB", target)`` decode: this image reduced 1/2, 1/4 or 1/8 for a JPEG, itself otherwise."""
        if self.path is not None:
            with Image.open(self.path) as im:
                im.draft("RGB", target)                  # header only; a no-op unless PIL can draft this file
                if im.size == self.size:
                    return self
                return DeviceImage(_upload(np.array(im.convert("RGB")), self.rgb.device))
        s = draft_scale(self.size, target) if self.jpeg else 1
        if s == 1:
            return self
        h, w = self.rgb.shape[-2:]
        x = F.pad(self.rgb[None], (0, -w % s, 0, -h % s), mode="replicate")
        return DeviceImage(F.avg_pool2d(x, s)[0].round_())
