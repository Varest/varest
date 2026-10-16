"""[REL]: the keep-masked training pass equals hard deletion, the loss reaches pruned tokens, and pruned greedy decoding
with a KV cache equals re-running the masked pass (CPU, a tiny Llama)."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from varest.rel import RelHead, budget_split, forward_loss, generate_pruned, keep_weights, register_keep_attention


def tiny():
    from transformers import LlamaConfig, LlamaModel
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, vocab_size=50)
    tm = LlamaModel(cfg).eval()
    register_keep_attention(tm)
    return tm, RelHead(d=64, nreg=2, rank=8), nn.Linear(64, 50, bias=False)


class BudgetSplitTest(unittest.TestCase):
    def test_released_pairs(self) -> None:
        self.assertEqual([budget_split(b) for b in (32, 64, 128, 224)], [(51, 13), (102, 26), (205, 51), (358, 90)])


class KeepMaskTest(unittest.TestCase):
    def test_masked_pass_equals_deletion_and_gradients_flow(self) -> None:
        tm, head, lm_head = tiny()
        t, a0, b0, r0, rl, kp = 20, 2, 12, 13, 2, 4
        x0 = torch.randn(1, t, 64)
        x = torch.cat([x0[:, :r0], head.tokens[None].detach(), x0[:, r0 + 2:]], 1)
        with torch.no_grad():
            pos = torch.arange(t).unsqueeze(0)
            pe = tm.rotary_emb(x, pos)
            h = x
            for i, layer in enumerate(tm.layers):
                if i == rl:
                    m = keep_weights(head.scores(h[0, r0:r0 + 2], h[0, a0:b0]), kp)
                    keep = torch.cat([torch.ones(1, a0), m[None], torch.ones(1, t - b0)], 1)
                    idx = torch.cat([torch.arange(a0), a0 + torch.nonzero(m > 0.5).squeeze(-1), torch.arange(b0, t)])
                    hd, pd = h[:, idx], pos[:, idx]
                    ped = tm.rotary_emb(hd, pd)
                if i < rl:
                    h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
                else:
                    h = layer(h, attention_mask=keep, position_ids=pos, position_embeddings=pe)
                    hd = layer(hd, attention_mask=None, position_ids=pd, position_embeddings=ped)
            self.assertLess((h[:, idx] - hd).abs().max().item(), 1e-4)
        lab = torch.full((1, t), -100)
        lab[0, 16:] = torch.randint(0, 50, (4,))
        tm.train()
        loss, info = forward_loss(tm, lm_head, head, torch.cat([x0[:, :r0], head.tokens[None], x0[:, r0 + 2:]], 1),
                                  lab, (a0, b0, r0, 2), kp, rl)
        loss.backward()
        self.assertEqual(info["kept"], kp)
        self.assertGreater(float(head.tokens.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.wq.weight.grad.abs().sum()), 0.0)

    def test_pruned_decoding_equals_masked_reference(self) -> None:
        tm, head, lm_head = tiny()
        t, a0, b0, r0, rl, kp = 20, 2, 12, 13, 2, 4
        e = torch.randn(1, t, 64)
        e[0, r0:r0 + 2] = head.tokens.detach()
        ids, info = generate_pruned(tm, lm_head, tm.embed_tokens, head, e, (a0, b0, r0, 2), kp, rl, 6, eos_id=-1)
        ref = []
        x = e
        with torch.no_grad():
            for _ in range(6):
                tn = x.shape[1]
                posn = torch.arange(tn).unsqueeze(0)
                pen = tm.rotary_emb(x, posn)
                h = x
                keep = None
                for i, layer in enumerate(tm.layers):
                    if i == rl:
                        m = keep_weights(head.scores(h[0, r0:r0 + 2], h[0, a0:b0]), kp)
                        keep = torch.cat([torch.ones(1, a0), m[None], torch.ones(1, r0 - b0), torch.zeros(1, 2),
                                          torch.ones(1, tn - r0 - 2)], 1)
                    h = layer(h, attention_mask=keep, position_ids=posn, position_embeddings=pen)
                nx = int(lm_head(tm.norm(h[:, -1:]))[0, -1].argmax())
                ref.append(nx)
                x = torch.cat([x, tm.embed_tokens(torch.tensor([[nx]]))], 1)
        self.assertEqual(ids, ref)
        self.assertEqual(info["kept"], kp)


if __name__ == "__main__":
    unittest.main()
