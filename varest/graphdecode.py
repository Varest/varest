"""Greedy decoding replayed from one CUDA graph per engine (Llama-family language models: LLaVA-1.5, LLaVA-NeXT-8B).

At batch 1 a decoded token is a few hundred small kernels (32-40 layers x norms, projections, the float32 LoRA updates,
RoPE, attention, MLP) and was launch-bound: ~13 ms/token for the bare model, ~18-21 ms with the adapter, while the
weights stream through an H100 in ~4-5 ms. :class:`GraphDecoder` captures one decode step and replays it:

* the KV cache is static: per layer a ``[1, capacity, kv_heads, head_dim]`` buffer that the prefill's cache is copied
  into, plus a per-layer length (LLaVA layers after the [REL] prune, or after an adaptive halt, hold fewer entries);
  ``capacity`` is sized to the request (its longest layer cache + ``max_new_tokens``, rounded up to ``block``
  positions). A request that needs more than is held, or at most half of it, reallocates the buffers and recaptures
  the graph; anything in between reuses them, so sizes that jitter around a block boundary do not recapture;
* one step calls each decoder layer's own modules in the order Hugging Face's ``LlamaDecoderLayer`` does (input norm,
  q/k/v projections, ``apply_rotary_pos_emb``, attention, ``o_proj``, residual, post-attention norm, MLP, residual),
  then the final norm, ``lm_head`` and the argmax on the GPU; the next token, its position and the cache lengths are
  updated inside the graph;
* the ONE operation that differs from the eager decoder is the attention kernel: FlashAttention-2's
  ``flash_attn_with_kvcache`` (it appends the new key/value in place and attends up to the per-layer length, which a
  static-shape graph needs) instead of PyTorch SDPA. Same float16 inputs and float32 softmax accumulation, a different
  kernel -- so the decoder is faithful, not bit-identical, and is validated by generated-answer agreement and a
  full-size accuracy A/B.

The decode step is otherwise the eager computation; the prefill is unchanged."""
from __future__ import annotations

import torch
from torch import nn


class GraphDecoder:
    def __init__(self, text_model: nn.Module, lm_head: nn.Module, embed: nn.Module, max_capacity: int,
                 block: int = 256) -> None:
        from flash_attn import flash_attn_with_kvcache
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        self._attn, self._rope = flash_attn_with_kvcache, apply_rotary_pos_emb
        self.tm, self.lm_head, self.embed = text_model, lm_head, embed
        cfg = text_model.config
        self.n_layers = len(text_model.layers)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        p = next(text_model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.max_capacity, self.block = int(max_capacity), int(block)
        self.capacity = 0                                # no KV buffers until the first request sizes them
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []
        with torch.inference_mode(False):                # normal tensors: updated in place under either mode
            self.lens = torch.zeros(self.n_layers, dtype=torch.int32, device=self.device)
            self.tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            self.pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.graph: torch.cuda.CUDAGraph | None = None

    # ------------------------------------------------------------------------------------------------ one step
    def _step(self) -> None:
        tm, hd = self.tm, self.head_dim
        x = self.embed(self.tok).to(self.dtype)
        cos, sin = tm.rotary_emb(x, self.pos)
        for i, layer in enumerate(tm.layers):
            a = layer.self_attn
            residual = x
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(1, 1, -1, hd).transpose(1, 2)
            k = a.k_proj(h).view(1, 1, -1, hd).transpose(1, 2)
            v = a.v_proj(h).view(1, 1, -1, hd).transpose(1, 2)
            q, k = self._rope(q, k, cos, sin)
            o = self._attn(q.transpose(1, 2), self.k[i], self.v[i], k=k.transpose(1, 2), v=v.transpose(1, 2),
                           cache_seqlens=self.lens[i:i + 1], softmax_scale=a.scaling, causal=True)
            x = residual + a.o_proj(o.reshape(1, 1, -1).contiguous())
            residual = x
            x = residual + layer.mlp(layer.post_attention_layernorm(x))
        nxt = self.lm_head(tm.norm(x))[0, -1].argmax()
        self.tok.copy_(nxt.view(1, 1))
        self.pos.add_(1)
        self.lens.add_(1)

    def _capture(self) -> None:
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        self.graph = g

    def _size(self, need: int) -> None:
        """Hold KV buffers for ``need`` positions (see the module docstring); drops the graph when they change."""
        target = min(-(-need // self.block) * self.block, self.max_capacity)
        if self.capacity >= need and 2 * target > self.capacity:
            return
        self.graph = None                                # release the old graph and buffers before allocating
        self.k, self.v = [], []
        shape = (1, target, self.kv_heads, self.head_dim)
        with torch.inference_mode(False):
            self.k = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.n_layers)]
            self.v = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.n_layers)]
        self.capacity = target

    # ------------------------------------------------------------------------------------------------ decoding
    def _need(self, cache, max_new_tokens: int) -> int:
        return max(int(cache.layers[i].keys.shape[-2]) for i in range(self.n_layers)) + max_new_tokens

    def fits(self, cache, max_new_tokens: int) -> bool:
        return self._need(cache, max_new_tokens) <= self.max_capacity

    @torch.no_grad()
    def decode(self, cache, first: int, position: int, max_new_tokens: int, eos_id: int) -> list[int]:
        """The eager loop of :func:`varest.rel.generate_pruned` from its first sampled token ``first`` (at ``position``),
        on the prefill's ``cache`` (a ``DynamicCache``; layer ``i`` holds ``[1, kv_heads, T_i, head_dim]``)."""
        self._size(self._need(cache, max_new_tokens))
        if self.graph is None:
            self._capture()
        for i in range(self.n_layers):
            lk, lv = cache.layers[i].keys, cache.layers[i].values
            t = lk.shape[-2]
            self.k[i][:, :t].copy_(lk.transpose(1, 2))
            self.v[i][:, :t].copy_(lv.transpose(1, 2))
            self.lens[i] = t
        self.tok.fill_(first)
        self.pos.fill_(position)
        out: list[int] = []
        nxt = first
        for _ in range(max_new_tokens):
            if nxt == eos_id:
                break
            out.append(nxt)
            self.graph.replay()
            nxt = int(self.tok)
        return out
