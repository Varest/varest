"""varest.gpuimage reproduces PIL's conversions: luma exactly, BICUBIC resize within one grey level, the draft size."""
import io

import numpy as np
import pytest
import torch
from PIL import Image

from varest.gpuimage import DeviceImage, draft_scale, luma, resize


def _images():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (157, 211, 3), dtype=np.uint8)
    edges = np.zeros((240, 320, 3), np.uint8)               # black/white text-like bars: bicubic overshoots here
    edges[:, ::7] = 255
    edges[::5, :, 1] = 255
    edges[100:140, 50:300] = (255, 0, 128)
    return [Image.fromarray(noise), Image.fromarray(edges)]


def _t(im):
    return torch.from_numpy(np.asarray(im.convert("RGB"))).permute(2, 0, 1).float()


@pytest.mark.parametrize("k", [0, 1])
def test_luma_is_pil_convert_L(k):
    im = _images()[k]
    assert np.array_equal(luma(_t(im))[0].numpy().astype(np.uint8), np.asarray(im.convert("L")))


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("size", [(64, 48), (336, 336), (500, 160), (97, 400), (28, 28)])
def test_resize_matches_pil_bicubic(k, size):
    im = _images()[k]
    ref = np.asarray(im.resize(size, Image.BICUBIC)).astype(int)
    got = resize(_t(im), size).permute(1, 2, 0).numpy().astype(int)
    d = np.abs(ref - got)
    assert d.max() <= 1 and (d > 0).mean() < 0.01, (d.max(), (d > 0).mean())


@pytest.mark.parametrize("wh,target", [((1600, 1200), (768, 576)), ((1600, 1200), (190, 140)), ((1600, 1200), (1000, 900)),
                                       ((640, 480), (768, 576)), ((4000, 300), (768, 64)), ((3000, 2000), (300, 199))])
def test_draft_scale_matches_pil(wh, target):
    buf = io.BytesIO()
    Image.new("RGB", wh, (120, 30, 200)).save(buf, "JPEG")
    buf.seek(0)
    with Image.open(buf) as im:
        im.draft("RGB", target)
        pil_size = im.size
    s = draft_scale(wh, target)
    assert ((wh[0] + s - 1) // s, (wh[1] + s - 1) // s) == pil_size


def test_draft_is_a_block_mean_and_identity_off_jpeg():
    x = torch.arange(3 * 17 * 26, dtype=torch.float32).reshape(3, 17, 26) % 256
    assert DeviceImage(x).draft((5, 4)) is not None and DeviceImage(x).draft((5, 4)).rgb is x
    d = DeviceImage(x, jpeg=True).draft((6, 4)).rgb        # 26 // 6 = 4, 17 // 4 = 4 -> scale 4, size 7 x 5
    assert tuple(d.shape) == (3, 5, 7)
    assert torch.equal(d[:, 0, 0], x[:, :4, :4].mean((1, 2)).round())


@pytest.mark.parametrize("fmt", ["JPEG", "MPO", "PNG"])
def test_file_draft_is_pils_own(tmp_path, fmt):
    """JPEG and MPO (camera multi-picture JPEG) files draft, PNG does not -- exactly as PIL decides."""
    rng = np.random.default_rng(1)
    p = tmp_path / f"x.{fmt.lower()}"
    Image.fromarray(rng.integers(0, 256, (600, 400, 3), dtype=np.uint8)).save(p, fmt)
    img = DeviceImage.open(p, torch.device("cpu"))
    d = img.draft((180, 120))
    with Image.open(p) as im:
        im.draft("RGB", (180, 120))
        ref = np.asarray(im.convert("RGB"))
    assert np.array_equal(d.rgb.permute(1, 2, 0).numpy().astype(np.uint8), ref)
    assert (d is img) == (fmt == "PNG")


@pytest.mark.parametrize("make", ["chw_u8", "hwc_u8", "chw_f01", "chw_f255", "gray_hw", "gray_1hw", "rgba_chw"])
def test_from_tensor_reads_like_to_pil(make):
    """The harness hands images over as tensors; DeviceImage.from_tensor must see the pixels to_pil would."""
    from varest.runtime import to_pil
    g = torch.Generator().manual_seed(3)
    u8 = torch.randint(0, 256, (3, 37, 52), dtype=torch.uint8, generator=g)
    t = {"chw_u8": u8, "hwc_u8": u8.permute(1, 2, 0), "chw_f01": u8.float() / 255, "chw_f255": u8.float(),
         "gray_hw": u8[0], "gray_1hw": u8[:1], "rgba_chw": torch.cat([u8, u8[:1]], 0)}[make]
    want = np.asarray(to_pil(t)).astype(np.float32)
    got = DeviceImage.from_tensor(t, torch.device("cpu")).rgb.permute(1, 2, 0).numpy()
    assert np.array_equal(got, want)
