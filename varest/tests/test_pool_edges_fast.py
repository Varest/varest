"""The vectorised varest.heads.object.pool_edges equals the original per-cell loop (verbatim copy below) exactly."""
import numpy as np
import pytest
import torch

from varest.heads.object import pool_edges


def _pool_edges_ref(e, nr, nc):
    e = torch.as_tensor(np.asarray(e), dtype=torch.float32)
    big_r, big_c = e.shape[1], e.shape[2]
    out = torch.zeros(2, nr, nc)
    rb = np.linspace(0, big_r, nr + 1).round().astype(int)
    cb = np.linspace(0, big_c, nc + 1).round().astype(int)
    for i in range(nr):
        for j in range(nc):
            if j + 1 < nc:
                col = max(cb[j + 1] - 1, 0)
                seg = e[0, rb[i]:max(rb[i + 1], rb[i] + 1), col]
                out[0, i, j] = seg.max() if seg.numel() else 0.0
            if i + 1 < nr:
                row = max(rb[i + 1] - 1, 0)
                seg = e[1, row, cb[j]:max(cb[j + 1], cb[j] + 1)]
                out[1, i, j] = seg.max() if seg.numel() else 0.0
    return out


SHAPES = [((48, 48), (24, 24)), ((48, 36), (24, 18)), ((30, 48), (15, 24)), ((48, 48), (48, 32)), ((64, 40), (24, 24)),
          ((48, 48), (1, 24)), ((48, 48), (24, 1)), ((7, 5), (24, 24)), ((3, 3), (5, 2)), ((96, 64), (48, 32)), ((48, 41), (17, 29))]


@pytest.mark.parametrize("fine,coarse", SHAPES)
@pytest.mark.parametrize("kind", ["randn", "ties", "nonneg"])
def test_matches_loop(fine, coarse, kind):
    rng = np.random.default_rng(hash((fine, coarse, kind)) % 2**32)
    e = rng.standard_normal((2, *fine)).astype(np.float32)
    if kind == "ties":
        e = np.round(e * 2) / 2
    elif kind == "nonneg":
        e = np.abs(e).astype(np.float16).astype(np.float32)
    want, got = _pool_edges_ref(e, *coarse), pool_edges(e, *coarse)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)


def test_accepts_tensor_input():
    e = torch.rand(2, 48, 48)
    assert torch.equal(pool_edges(e, 24, 24), _pool_edges_ref(e, 24, 24))
