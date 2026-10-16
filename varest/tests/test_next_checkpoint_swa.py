"""LLaVA-NeXT geometry and budgets, the release checkpoint format, and SWA."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

from varest import next as vnext
from varest.checkpoint import VarestCheckpoint
from varest.postag import PosTag
from varest.rel import RelHead
from varest.swa import average


class NextGeometryTest(unittest.TestCase):
    def test_layout_and_stitch(self) -> None:
        for w, h in [(640, 480), (480, 640), (1000, 1000), (1700, 2200), (3000, 900), (336, 336), (123, 987)]:
            lay = vnext.layout(w, h)
            self.assertTrue(0 < lay.nr <= lay.gh and 0 < lay.nc <= lay.gw)
            f = torch.arange(lay.nh * lay.nw * 576, dtype=torch.float32).reshape(-1, 576, 1)
            self.assertEqual(vnext.stitch(f, lay).shape[0], lay.nr * lay.nc)

    def test_budget_split(self) -> None:
        for nr, nc in [(24, 32), (48, 48), (18, 72)]:
            for ratio in vnext.STANDARD_RATIOS.values():
                k = vnext.kept(nr, nc, ratio)
                kk, kb, kh, tau = vnext.split(k, nr, nc)
                self.assertEqual(kb + kh, kk)
                self.assertAlmostEqual(tau, 576 * kk / vnext.n_native(nr, nc))
                self.assertLessEqual(kh, nr * nc)

    def test_plan_and_tokenize(self) -> None:
        lay = vnext.layout(900, 600)
        rng = np.random.default_rng(0)
        base = (rng.random((2, 24, 24)).astype(np.float32), rng.random((24, 24)).astype(np.float32) * .3, None)
        hi = (rng.random((2, lay.nr, lay.nc)).astype(np.float32),
              rng.random((lay.nr, lay.nc)).astype(np.float32) * .3, None)
        p = vnext.plan(base, hi, lay, vnext.kept(lay.nr, lay.nc, 0.2222))
        tok, pos = vnext.tokenize(torch.randn(576, 16), torch.randn(lay.nr * lay.nc, 16), p, lay)
        self.assertEqual(tok.shape[0], p.k)
        self.assertEqual(pos.shape, (p.k, 3))


def _training_ckpt(rel: bool) -> dict:
    lora = {"base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(4, 8),
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(8, 4)}
    proj = {"base_model.model.model.multi_modal_projector.linear_1.weight": torch.randn(8, 8)}
    ck = {"lora": lora, "proj": proj, "postag": PosTag(8).state_dict(), "cfg": {"REL_LAYER": 16}, "step": 1}
    if rel:
        ck["relhead"] = RelHead(d=8, nreg=2, rank=4).state_dict()
    return ck


class CheckpointTest(unittest.TestCase):
    def test_roundtrip_and_validation(self) -> None:
        heads = {"sd": {}}
        for rel, variant in ((True, "w5reljoint"), (False, "w5")):
            ck = VarestCheckpoint.from_training(_training_ckpt(rel), variant=variant, backbone="llava-1.5-7b",
                                                text_head=heads, object_head=heads)
            self.assertTrue(all(not k.startswith("base_model.") for k in ck.lora))
            self.assertEqual(ck.lora_rank, 4)
            self.assertEqual(ck.lora_targets, ["q_proj"])
            self.assertEqual(ck.rel_layer, 16 if rel else None)
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "c.pt")
                ck.save(p)
                back = VarestCheckpoint.load(p)
            self.assertEqual(back.variant, variant)
            self.assertEqual(sorted(back.lora), sorted(ck.lora))
        with self.assertRaises(ValueError):
            VarestCheckpoint.from_training(_training_ckpt(False), variant="w5reljoint", backbone="llava-1.5-7b",
                                           text_head=heads, object_head=heads)
        with self.assertRaises(ValueError):
            VarestCheckpoint.from_training(_training_ckpt(True), variant="w5", backbone="llava-1.5-7b",
                                           text_head=heads, object_head=heads)


class SwaTest(unittest.TestCase):
    def test_uniform_average_of_every_component(self) -> None:
        srcs = [_training_ckpt(True) for _ in range(3)]
        for s in srcs[1:]:
            s["cfg"] = srcs[0]["cfg"]
        out = average(srcs)
        for key in ("lora", "postag", "proj", "relhead"):
            for k in srcs[0][key]:
                torch.testing.assert_close(out[key][k], sum(s[key][k] for s in srcs) / 3)
        self.assertEqual(out["step"], -1)


if __name__ == "__main__":
    unittest.main()
