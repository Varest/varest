"""[REL]: learned relevance registers that prune visual tokens INSIDE the language model (W5RELJOINT).

R learned embeddings ([REL] x R) are placed after the prompt, before the assistant turn. At decoder layer L (the
middle layer: 16 of 32, 20 of 40) every visual token j is scored against them,

    s_j = < Wq LN(mean of the [REL] hidden states), Wk LN(h_j) > / sqrt(rank)

and only the top K' visual tokens continue past layer L. Because [REL] sits after the prompt, the ranking can only
come from the prompt and the image. It is trained jointly with the rest of the model, from scratch, with no relevance
labels: during training the tokens [REL] would drop are MASKED out of every later query at layers >= L (exactly
equivalent to deleting them; see :func:`keep_masked_attention`), with a straight-through soft keep so the loss reaches
the scores of pruned tokens too. At inference they are hard-deleted (:func:`generate_pruned`), keeping the survivors'
original position ids -- the same computation the masked training pass performs.

At inference the [REL] registers are deleted at layer L together with the pruned visual tokens: their only job is the
ranking, so they exist in layers < L only. Training (:func:`forward_loss`) leaves them visible after L, as the released
checkpoints were trained; deleting them at inference moves the answer log-probability by +0.002 nats (not significant;
activation-patching test on 800 RefCOCO and 398 GQA prompt pairs, 7B and 13B) -- the layers after L never read them.

BUDGET. A pre-LLM budget K and an in-LLM budget K' combine into the layer-averaged budget
``B = (L*K + (N-L)*K') / N``. The released checkpoints use ``K' = K / 4`` at ``L = N/2``, i.e. ``K = round(1.6 B)``,
``K' = 2B - K`` (:func:`budget_split`): B = 32/64/128/224 -> 51:13, 102:26, 205:51, 358:90.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from varest.halt import CascadeHalt, HaltConfig

KEEP_FRACTION = 0.25        # K' / K of the released checkpoints
N_REGISTERS = 4
RANK = 256


def budget_split(budget: float, keep_fraction: float = KEEP_FRACTION) -> tuple[int, int]:
    """Layer-averaged budget ``B`` at a middle-layer cut -> ``(K, K')`` with ``K + K' = 2B`` and ``K' ~ fraction*K``."""
    k = int(round(2 * budget / (1.0 + keep_fraction)))
    return k, int(round(2 * budget)) - k


class RelHead(nn.Module):
    """``nreg`` learned [REL] embeddings and a rank-``rank`` bilinear readout on the layer-L hidden states."""

    def __init__(self, d: int = 4096, nreg: int = N_REGISTERS, rank: int = RANK) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(nreg, d) * 0.02)
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.wq = nn.Linear(d, rank, bias=False)
        self.wk = nn.Linear(d, rank, bias=False)
        self.rank = rank

    def scores(self, h_rel: torch.Tensor, h_vis: torch.Tensor) -> torch.Tensor:
        """``h_rel [R, D]``, ``h_vis [N, D]`` (any dtype) -> float32 ``[N]`` scores."""
        q = self.wq(self.ln(h_rel.float()).mean(0))
        k = self.wk(self.ln(h_vis.float()))
        return (k @ q) / math.sqrt(self.rank)


def rel_head_from_state(sd: dict[str, torch.Tensor]) -> RelHead:
    head = RelHead(d=sd["tokens"].shape[1], nreg=sd["tokens"].shape[0], rank=sd["wq.weight"].shape[0])
    head.load_state_dict(sd)
    return head


def keep_weights(s: torch.Tensor, kp: int, temp: float = 0.1) -> torch.Tensor:
    """Scores ``[N]`` and a keep count -> keep weights ``[N]``: hard top-``kp`` forward, soft sigmoid backward.

    Scores are standardised first (with gradient) so the soft keep is scale-free; ``temp`` is in z-units and the
    threshold sits midway between the ``kp``-th and ``(kp+1)``-th scores."""
    n = s.numel()
    if n > 1:
        s = (s - s.mean()) / s.std().clamp(min=1e-3)
    hard = torch.zeros_like(s)
    order = torch.argsort(s.detach(), descending=True)
    if kp > 0:
        hard[order[:kp]] = 1.0
    sd = s.detach()
    t = torch.tensor(float(temp), device=s.device)
    if 0 < kp < n:
        tau = 0.5 * (sd[order[kp - 1]] + sd[order[kp]])
    elif kp <= 0:
        tau = sd.max() + t
    else:
        tau = sd.min() - t
    soft = torch.sigmoid((s - tau) / t)
    return hard + soft - soft.detach()


def keep_masked_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    """Attention for layers >= L during training. ``attention_mask`` semantics:
    ``None`` -> stock causal SDPA; a 2-D float ``[B, T]`` -> causal attention with per-KEY multiplicative keep weights
    (the weight multiplies the key's exp-logit, so a pruned key still receives gradient); anything else -> SDPA."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if attention_mask is None or not (torch.is_tensor(attention_mask) and attention_mask.dim() == 2
                                      and attention_mask.is_floating_point()):
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    g = getattr(module, "num_key_value_groups", 1)
    if g > 1:
        key = key.repeat_interleave(g, dim=1)
        value = value.repeat_interleave(g, dim=1)
    tq, tk = query.shape[2], key.shape[2]
    z = torch.matmul(query.float(), key.float().transpose(2, 3)) * scaling
    causal = torch.ones(tq, tk, dtype=torch.bool, device=z.device).tril(diagonal=tk - tq)
    z = z.masked_fill(~causal, float("-inf"))
    z = z - z.amax(-1, keepdim=True).detach()
    w = torch.exp(z) * attention_mask[:, None, None, :].float()
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-30)
    out = torch.matmul(w.to(value.dtype), value).transpose(1, 2).contiguous()
    return out, None


def register_keep_attention(text_model: nn.Module) -> None:
    """Route every decoder layer of ``text_model`` through :func:`keep_masked_attention` (SDPA when unmasked)."""
    from transformers import AttentionInterface
    AttentionInterface.register("rel_keep", keep_masked_attention)
    text_model.config._attn_implementation = "rel_keep"
    for layer in text_model.layers:
        layer.self_attn.config._attn_implementation = "rel_keep"


def forward_loss(text_model: nn.Module, lm_head: nn.Module, head: RelHead, embs: torch.Tensor, labels: torch.Tensor,
                 spans: tuple[int, int, int, int], kp: int, rel_layer: int, temp: float = 0.1
                 ) -> tuple[torch.Tensor, dict]:
    """Training pass for ONE example (B=1). ``embs [1, T, D]``, ``labels [1, T]`` (-100 ignored),
    ``spans = (a0, b0, r0, nreg)``: visual tokens ``[a0, b0)``, [REL] tokens ``[r0, r0 + nreg)``.
    Layers < ``rel_layer`` run the stock causal path; from ``rel_layer`` on, the keep weights computed from that
    layer's INPUT states mask the pruned visual keys. Requires :func:`register_keep_attention` on ``text_model``."""
    a0, b0, r0, nr = spans
    t = embs.shape[1]
    pos = torch.arange(t, device=embs.device).unsqueeze(0)
    h = embs
    pe = text_model.rotary_emb(h, pos)
    keep = None
    info: dict = {}
    for i, layer in enumerate(text_model.layers):
        if i == rel_layer:
            s = head.scores(h[0, r0:r0 + nr], h[0, a0:b0])
            m = keep_weights(s, kp, temp)
            keep = torch.ones(1, t, device=h.device, dtype=torch.float32)
            keep = torch.cat([keep[:, :a0], m.unsqueeze(0), keep[:, b0:]], 1)
            info = {"kept": int((m.detach() > 0.5).sum()), "n": b0 - a0, "s_std": float(s.detach().std())}
        h = layer(h, attention_mask=keep, position_ids=pos, position_embeddings=pe, use_cache=False)
    h = text_model.norm(h)
    tgt = labels[0, 1:]
    idx = torch.nonzero(tgt != -100).squeeze(-1)
    logits = lm_head(h[0, idx])
    return F.cross_entropy(logits.float(), tgt[idx]), info


@torch.no_grad()
def generate_pruned(text_model: nn.Module, lm_head: nn.Module, embed: nn.Module, head: RelHead | None, e: torch.Tensor,
                    spans: tuple[int, int, int, int], kp: int, rel_layer: int | None, max_new_tokens: int, eos_id: int,
                    halt: HaltConfig | None = None) -> tuple[list[int], dict]:
    """Greedy decoding with HARD pruning (B=1). ``e [1, T, D]`` are the prompt embeddings including the [REL]
    tokens; ``spans`` as in :func:`forward_loss`. Layers < ``rel_layer`` see all T positions; at ``rel_layer`` the
    visual tokens outside the top ``kp`` and the [REL] registers are deleted (their KV never enters later layers); the
    kept tokens keep their ORIGINAL position ids. Per-layer KV cache: early layers hold T entries, later layers only
    the kept ones.

    ``halt`` adds adaptive halting (:mod:`varest.halt`): at the layer the cascade fires, every visual token still
    present -- and the [REL] registers, if [REL] has not pruned yet, in which case it never runs -- is deleted. With
    ``head=None`` / ``rel_layer=None`` (W5, no [REL]; ``spans`` then carries ``nr = 0``) only halting prunes.
    ``halt=None`` is exactly the [REL]-only decoding above."""
    from transformers import DynamicCache
    a0, b0, r0, nr = spans
    t = e.shape[1]
    cache = DynamicCache()
    pos = torch.arange(t, device=e.device).unsqueeze(0)
    h = e
    pe = text_model.rotary_emb(h, pos)
    info: dict = {}
    halter = CascadeHalt(halt, text_model.norm, lm_head) if halt is not None else None
    idx = torch.arange(t, device=e.device)                  # ORIGINAL position of every row still in the sequence
    for i, layer in enumerate(text_model.layers):
        if halter is not None and halter.step(i, h):
            gone = ((idx >= a0) & (idx < b0)) | ((idx >= r0) & (idx < r0 + nr))
            info.update({"halt_layer": i, "halt_dropped": int(((idx >= a0) & (idx < b0)).sum())})
            h, pos, idx = h[:, ~gone], pos[:, ~gone], idx[~gone]
            pe = text_model.rotary_emb(h, pos)
        if i == rel_layer and head is not None and "halt_layer" in info:
            info.update({"n": b0 - a0, "kept": 0})            # halted first: nothing left for [REL] to rank
        elif i == rel_layer and head is not None:
            s = head.scores(h[0, r0:r0 + nr], h[0, a0:b0])
            n = b0 - a0
            k = max(0, min(int(kp), n))
            top = torch.argsort(s.float(), descending=True)[:k].sort().values + a0
            keep = torch.cat([torch.arange(a0, device=e.device), top, torch.arange(b0, r0, device=e.device),
                              torch.arange(r0 + nr, t, device=e.device)])          # [REL] removed with the pruned tokens
            h = h[:, keep]
            pos = pos[:, keep]
            idx = idx[keep]
            pe = text_model.rotary_emb(h, pos)
            info.update({"n": n, "kept": k, "scores": s.float().cpu(), "keep_idx": (top - a0).cpu()})
        h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe, past_key_values=cache,
                  use_cache=True)
    nxt = int(lm_head(text_model.norm(h[:, -1:]))[0, -1].argmax())
    out: list[int] = []
    p = t
    for _ in range(max_new_tokens):
        if nxt == eos_id:
            break
        out.append(nxt)
        x = embed(torch.tensor([[nxt]], device=e.device)).to(e.dtype)
        pp = torch.tensor([[p]], device=e.device)
        pep = text_model.rotary_emb(x, pp)
        for layer in text_model.layers:
            x = layer(x, attention_mask=None, position_ids=pp, position_embeddings=pep, past_key_values=cache,
                      use_cache=True)
        nxt = int(lm_head(text_model.norm(x))[0, -1].argmax())
        p += 1
    return out, info
