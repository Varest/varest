"""Adaptive halting (CPU, a tiny Llama): the cascade gate, and halting decoding with a KV cache equals a no-cache
reference that re-runs the whole prefix with the same rows deleted at every layer -- halting before [REL], after
[REL], and on W5 (no [REL]). A gate that never fires leaves decoding bit-identical to ``halt=None``."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from varest.halt import CascadeHalt, HaltConfig, lens_head
from varest.rel import RelHead, generate_pruned, register_keep_attention

T, A0, B0, R0, NR = 20, 2, 12, 13, 2


def tiny(layers: int = 6):
    from transformers import LlamaConfig, LlamaModel
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=4, vocab_size=50)
    tm = LlamaModel(cfg).eval()
    register_keep_attention(tm)
    return tm, RelHead(d=64, nreg=NR, rank=8), nn.Linear(64, 50, bias=False)


def prompt(head: RelHead) -> torch.Tensor:
    torch.manual_seed(1)
    e = torch.randn(1, T, 64)
    e[0, R0:R0 + NR] = head.tokens.detach()
    return e


@torch.no_grad()
def reference(tm, lm_head, embed, e, deleted: dict[int, set[int]], steps: int) -> list[int]:
    """No KV cache: every step re-runs layers on the whole prefix; ``deleted[i]`` = ORIGINAL prompt positions absent
    from layer ``i`` on (cumulative)."""
    out, x = [], e
    for _ in range(steps):
        tn = x.shape[1]
        h, keep = x, torch.arange(tn)
        gone: set[int] = set()
        for i, layer in enumerate(tm.layers):
            gone |= deleted.get(i, set())
            sel = torch.tensor([j for j in range(tn) if j not in gone])
            h = h[:, [int((keep == j).nonzero()) for j in sel]] if len(sel) != len(keep) else h
            keep = sel
            pe = tm.rotary_emb(h, keep[None])
            h = layer(h, attention_mask=None, position_ids=keep[None], position_embeddings=pe)
        nxt = int(lm_head(tm.norm(h[:, -1:]))[0, -1].argmax())
        out.append(nxt)
        x = torch.cat([x, embed(torch.tensor([[nxt]])).to(x.dtype)], 1)
    return out


class GateTest(unittest.TestCase):
    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            HaltConfig(entropy=None, maxp=None)
        with self.assertRaises(ValueError):
            HaltConfig(entropy=1.0, maxp=0.7)
        HaltConfig(entropy=None, maxp=0.7)

    def test_window_min_layer_and_gate(self) -> None:
        class Fixed(nn.Module):                  # lens head returning a scripted logit row per call
            def __init__(self, rows):
                super().__init__()
                self.rows, self.k = rows, 0

            def forward(self, x):
                r = self.rows[self.k]
                self.k += 1
                return r.view(1, 1, -1)
        peaked = torch.tensor([20.0, 0, 0, 0])      # entropy ~0, maxp ~1
        flat = torch.tensor([0.1, 0, 0, 0])         # argmax 0 but entropy ~ln 4
        other = torch.tensor([0, 20.0, 0, 0])
        h = torch.zeros(1, 3, 4)
        # stable from layer 0, peaked: halts exactly at min_layer
        c = CascadeHalt(HaltConfig(min_layer=4, window=3, entropy=1.0), nn.Identity(), Fixed([peaked] * 10))
        self.assertEqual([c.step(i, h) for i in range(6)], [False] * 4 + [True, False])
        self.assertEqual(c.layer, 4)
        # argmax switches at layer 4 -> needs 3 equal confident tops -> halts at 6
        rows = [flat] * 4 + [other] * 6            # stable but unconfident, then switches
        c = CascadeHalt(HaltConfig(min_layer=2, window=3, entropy=1.0), nn.Identity(), Fixed(rows))
        self.assertEqual(next(i for i in range(10) if c.step(i, h)), 6)
        # stable but never confident enough: never halts (entropy gate), halts under a low maxp gate
        c = CascadeHalt(HaltConfig(min_layer=0, window=2, entropy=1.0), nn.Identity(), Fixed([flat] * 10))
        self.assertFalse(any(c.step(i, h) for i in range(10)))
        c = CascadeHalt(HaltConfig(min_layer=0, window=2, entropy=None, maxp=0.2), nn.Identity(), Fixed([flat] * 10))
        self.assertEqual(next(i for i in range(10) if c.step(i, h)), 1)

    def test_lens_uses_base_layer(self) -> None:
        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = nn.Linear(4, 4)
        w = Wrapped()
        self.assertIs(lens_head(w, True), w.base_layer)
        self.assertIs(lens_head(w, False), w)


class DecodeTest(unittest.TestCase):
    def test_never_firing_gate_is_bit_identical(self) -> None:
        tm, head, lm_head = tiny()
        e = prompt(head)
        a, ia = generate_pruned(tm, lm_head, tm.embed_tokens, head, e, (A0, B0, R0, NR), 4, 3, 6, eos_id=-1)
        b, ib = generate_pruned(tm, lm_head, tm.embed_tokens, head, e, (A0, B0, R0, NR), 4, 3, 6, eos_id=-1,
                                halt=HaltConfig(min_layer=0, window=1, entropy=-1.0))
        self.assertEqual(a, b)
        self.assertNotIn("halt_layer", ib)
        self.assertTrue(torch.equal(ia["keep_idx"], ib["keep_idx"]))

    def test_halt_before_rel_equals_reference(self) -> None:
        tm, head, lm_head = tiny()
        e = prompt(head)
        fire = HaltConfig(min_layer=2, window=1, entropy=1e9)            # fires at layer 2, before [REL] at 4
        ids, info = generate_pruned(tm, lm_head, tm.embed_tokens, head, e, (A0, B0, R0, NR), 4, 4, 6, eos_id=-1,
                                    halt=fire)
        self.assertEqual((info["halt_layer"], info["halt_dropped"], info["kept"]), (2, B0 - A0, 0))
        gone = set(range(A0, B0)) | set(range(R0, R0 + NR))
        self.assertEqual(ids, reference(tm, lm_head, tm.embed_tokens, e, {2: gone}, 6))

    def test_halt_after_rel_equals_reference(self) -> None:
        tm, head, lm_head = tiny()
        e = prompt(head)
        fire = HaltConfig(min_layer=4, window=1, entropy=1e9)            # [REL] prunes at 2, halt at 4
        ids, info = generate_pruned(tm, lm_head, tm.embed_tokens, head, e, (A0, B0, R0, NR), 4, 2, 6, eos_id=-1,
                                    halt=fire)
        kept = {A0 + int(j) for j in info["keep_idx"]}
        self.assertEqual((info["halt_layer"], info["halt_dropped"], info["kept"]), (4, 4, 4))
        rel_gone = (set(range(A0, B0)) - kept) | set(range(R0, R0 + NR))
        self.assertEqual(ids, reference(tm, lm_head, tm.embed_tokens, e, {2: rel_gone, 4: kept}, 6))

    def test_w5_halting_equals_reference(self) -> None:
        tm, _, lm_head = tiny()
        torch.manual_seed(2)
        e = torch.randn(1, T, 64)
        fire = HaltConfig(min_layer=3, window=1, entropy=1e9)
        ids, info = generate_pruned(tm, lm_head, tm.embed_tokens, None, e, (A0, B0, B0, 0), 0, None, 6, eos_id=-1,
                                    halt=fire)
        self.assertEqual((info["halt_layer"], info["halt_dropped"]), (3, B0 - A0))
        self.assertEqual(ids, reference(tm, lm_head, tm.embed_tokens, e, {3: set(range(A0, B0))}, 6))
        none, _ = generate_pruned(tm, lm_head, tm.embed_tokens, None, e, (A0, B0, B0, 0), 0, None, 6, eos_id=-1)
        self.assertEqual(none, reference(tm, lm_head, tm.embed_tokens, e, {}, 6))


if __name__ == "__main__":
    unittest.main()
