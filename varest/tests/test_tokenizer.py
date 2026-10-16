"""The W5 tokenizer's contracts: exact budget, raster emission, text pinning, the letterbox token, pooling."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from varest.tokenizer import GRID, build_partition, pin_threshold, pool_regions


def fields(seed: int = 0, band: tuple[int, int] = (4, 8)) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic image: one horizontal text band on a structured background."""
    rng = np.random.default_rng(seed)
    bnd = rng.random((2, GRID, GRID)).astype(np.float32) * 0.1
    bnd[:, GRID // 2:, :] += 0.4
    txt = np.zeros((GRID, GRID), np.float32)
    txt[band[0]:band[1], 2:GRID - 2] = 0.99
    txt += rng.random((GRID, GRID)).astype(np.float32) * 0.01
    return bnd, txt


def letterbox(rows: int = 18) -> np.ndarray:
    m = np.zeros((GRID, GRID), bool)
    r0 = (GRID - rows) // 2
    m[r0:r0 + rows] = True
    return m


class PinThresholdTest(unittest.TestCase):
    def test_linear_ramp(self) -> None:
        self.assertAlmostEqual(pin_threshold(0), 1.0)
        self.assertAlmostEqual(pin_threshold(576), 0.5)
        self.assertAlmostEqual(pin_threshold(10_000), 0.5)
        for k, want in ((32, 0.972), (64, 0.944), (128, 0.889), (224, 0.806)):
            self.assertAlmostEqual(pin_threshold(k), want, places=3)


class PartitionTest(unittest.TestCase):
    def test_exact_budget_and_full_cover(self) -> None:
        bnd, txt = fields()
        for inimg in (None, letterbox()):
            for k in (8, 32, 64, 128, 224, 400):
                lab, info = build_partition(bnd, txt, k, inimg=inimg)
                self.assertEqual(int(lab.max()) + 1, k)
                self.assertEqual(info["tokens"], k)
                self.assertEqual(set(np.unique(lab)), set(range(k)))

    def test_raster_emission_order(self) -> None:
        bnd, txt = fields(3)
        lab, _ = build_partition(bnd, txt, 96, inimg=letterbox())
        flat = lab.reshape(-1)
        firsts = [int(np.argmax(flat == j)) for j in range(int(flat.max()) + 1)]
        self.assertEqual(firsts, sorted(firsts))

    def test_text_pinned_at_native_resolution_when_affordable(self) -> None:
        bnd, txt = fields()
        lab, info = build_partition(bnd, txt, 224)
        self.assertEqual(info["regime"], "pin")
        cells = np.argwhere(txt > pin_threshold(224))
        sizes = np.bincount(lab.reshape(-1))
        self.assertTrue(all(sizes[lab[r, c]] == 1 for r, c in cells))

    def test_letterbox_is_one_token(self) -> None:
        bnd, txt = fields()
        m = letterbox()
        lab, _ = build_partition(bnd, txt, 64, inimg=m)
        self.assertEqual(len(np.unique(lab[~m])), 1)
        self.assertFalse(set(np.unique(lab[~m])) & set(np.unique(lab[m])))

    def test_rectangular_grid(self) -> None:
        rng = np.random.default_rng(1)
        bnd = rng.random((2, 30, 45)).astype(np.float32)
        txt = rng.random((30, 45)).astype(np.float32) * 0.3
        lab, _ = build_partition(bnd, txt, 150, grid=(30, 45), tau_k=128)
        self.assertEqual(lab.shape, (30, 45))
        self.assertEqual(int(lab.max()) + 1, 150)


class PoolTest(unittest.TestCase):
    def test_means_and_positions(self) -> None:
        bnd, txt = fields()
        lab, _ = build_partition(bnd, txt, 40)
        feats = torch.randn(GRID * GRID, 8)
        tok, pos, cnt = pool_regions(feats, lab)
        self.assertEqual(tok.shape, (40, 8))
        for j in (0, 17, 39):
            m = torch.from_numpy(lab.reshape(-1) == j)
            torch.testing.assert_close(tok[j], feats[m].mean(0))
            ys, xs = np.nonzero(lab == j)
            self.assertAlmostEqual(float(pos[j, 0]), (xs.mean() + .5) / GRID, places=5)
            self.assertAlmostEqual(float(pos[j, 1]), (ys.mean() + .5) / GRID, places=5)
            self.assertAlmostEqual(float(pos[j, 2]), np.sqrt(len(xs)) / GRID, places=5)
        self.assertEqual(int(cnt.sum()), GRID * GRID)


if __name__ == "__main__":
    unittest.main()
