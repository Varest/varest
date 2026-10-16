"""GlimpsePrune: dynamic, glimpse-guided visual-token pruning inside the decoder.

Paper: Zeng et al., "A Glimpse to Compress: Dynamic Visual Token Pruning for
Large Vision-Language Models" (arXiv:2508.01548). Reference code:
https://github.com/HVision-NKU/GlimpsePrune.

One compressor, both released backbones. ``install()`` detects whether it is
sitting on LLaVA-1.5 or Qwen2.5-VL and selects the matching implementation, so
``--compressor GlimpsePrune`` works with either ``--backbone llava`` or
``--backbone qwen-vl``. The two differ in ways that cannot be papered over --
LLaVA uses standard RoPE and a square patch grid, Qwen uses M-RoPE and a
windowed ViT with per-image grids -- so the divergent halves live in two private
backends (``_LlavaBackend`` / ``_QwenBackend``) while everything else (config
parsing, weight loading, the glimpse token, the token-retention rules, cache
pruning, the decode handoff) is written once.

**Not training-free.** This is the paper's *non fine-tuned* variant -- the base
LVLM is frozen -- but it still needs the two released modules, so
``compressor_args.new_modules_dir`` is effectively required. The releases are
backbone-specific and not interchangeable:

===============  ==================================================
backbone         ``new_modules_dir``
===============  ==================================================
LLaVA-1.5-7B     ``ashun989/GlimpsePrune_LLaVA-1.5-7B``
Qwen2.5-VL-7B    ``ashun989/GlimpsePrune_Qwen2.5-VL-7B-Instruct``
===============  ==================================================

Loading is ``strict=True``, so pairing a checkpoint with the wrong backbone
fails loudly rather than silently producing nonsense. Without any
``new_modules_dir`` the modules are randomly initialised and warn.

What it does, per sample:

1. run the vision tower, keeping the intermediate features the VIP needs;
2. append the learned **glimpse token** after the prompt;
3. prefill decoder layers ``0..reduce_layer``, extracting only the
   glimpse-to-image attention row at each ``selected_layers`` entry;
4. score every visual token with the **VIP** (``attn_fuser``);
5. drop tokens whose ``sigmoid(score)`` fails ``reduce_threshold``, along with
   their KV-cache entries and the glimpse token itself;
6. finish the prefill on the reduced sequence and hand off to the stock
   generation loop. Retained tokens keep their original RoPE positions.

**No fixed budget.** The retained count is data-dependent, so ``max_tokens`` is 0
and unused, and the docs' placeholder-trimming route does not apply -- pruning
happens inside the prefill, which is why this wraps ``vlm.generate`` rather than
``get_image_features``. Use ``max_remain_ratio`` to cap retention (the paper's
11.1% headline is ``0.111``) or ``fixed_remain_ratio`` for the fixed-budget
ablation.

Harness adaptations
-------------------
Both paths drive the *stock* transformers modules (v5 API) rather than shipping
forked model classes, so nothing in the backbone is patched. Glimpse-row
attention is recomputed from the q/k projections, so any attention backend
(sdpa/eager/FA2) works and the full attention map is never materialised -- the
same trick ``fastv.py`` and ``vision_zip.py`` use.

With a keep-everything threshold either path reproduces stock
``model.generate`` **bit-exactly**, which is what pins down the positions,
masks, cache pruning and decode handoff.

Not supported on either backbone (the official image-path MVP has the same
limits): more than one image per sample, beam search, ``use_cache=False``, and
video. LLaVA additionally requires a square visual grid shared across the batch.
Text-only batches bypass GlimpsePrune and take the stock generation path.
"""

from __future__ import annotations

import json
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, override

import torch
import torch.nn.functional as F
from torch import nn

from llmoe.backbone.prototype import VLM
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor


# ---------------------------------------------------------------------------
# Released-checkpoint configuration
# ---------------------------------------------------------------------------


@dataclass
class GlimpsePruneConfig:
    """GP-specific fields of the official ``LlavaConfig_GP`` / ``Qwen2_5_VL_GPConfig``.

    The union of both releases' fields; each backbone's ``config.json`` fills in
    what it uses and the rest keep defaults that reproduce that backbone's
    reference behaviour. ``attn_fuse_global`` is Qwen-only (its windowed ViT
    chooses between per-image and per-window attention spans) and
    ``le_norm_type`` is Qwen-only (LLaVA's release is always ``rmsnorm``).

    Released 7B values:

    * LLaVA-1.5: ``selected_layers=[21]``, ``reduce_layer=21``,
      ``le_layers=[0..31]``, ``selected_visual_layers=[23, 17, 11, 5]``,
      ``visual_cond_size=512``.
    * Qwen2.5-VL: ``selected_layers=[18]``, ``reduce_layer=18``,
      ``le_layers=[0..27]``, ``use_attention_logits=True``,
      ``attn_fuse_global=True``, ``selected_visual_layers=[31, 23, 15, 7]``,
      ``visual_cond_size=512``.
    """

    selected_layers: tuple[int, ...] = ()
    use_attention_logits: bool = False
    attn_fuse_type: str = "AttnFuserV1"
    attn_fuse_size: int = 256
    attn_fuse_global: bool = False
    attn_fuse_num_heads: int = 4
    attn_fuse_hidden_act: str = "silu"
    selected_visual_layers: tuple[int, ...] = (8,)
    visual_cond_size: int = 256
    ori_attn_supervision: bool = True
    deep_supervision: bool = True
    le_layers: tuple[int, ...] = (0,)
    le_length: int = 1
    le_dropout_prob: float = 0.0
    le_norm_type: str = "rmsnorm"
    reduce_threshold: float = 0.5
    use_zero_masks: bool = False
    reduce_layer: int = 1000
    min_remain_num: int | None = 1
    max_remain_ratio: float | None = None
    fixed_remain_ratio: float | None = None

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "GlimpsePruneConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in blob.items() if key in known}
        config = cls(**kwargs)
        config.selected_layers = tuple(config.selected_layers)
        config.selected_visual_layers = tuple(config.selected_visual_layers)
        config.le_layers = tuple(config.le_layers)
        return config

    @classmethod
    def from_json(cls, path: str) -> "GlimpsePruneConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# VIP (attn fuser) modules — parameter names match the official state dicts
# ---------------------------------------------------------------------------


def _llama_rms_norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    return LlamaRMSNorm(hidden_size, eps=eps)


def _qwen_rms_norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRMSNorm

    return Qwen2_5_VLRMSNorm(hidden_size, eps=eps)


@dataclass
class _FuserDims:
    """Live-model dimensions the VIP depends on.

    ``spatial_merge_size`` is Qwen-only; LLaVA leaves it ``None``.
    """

    text_num_heads: int
    vision_hidden_size: int
    spatial_merge_size: int | None = None

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class _FuserMLP(nn.Module):
    """Official ``MLP`` used inside the attn fuser (bias=True gated MLP)."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, bias: bool) -> None:
        from transformers.activations import ACT2FN

        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class _LlavaVisionRotary(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        positions = torch.arange(length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(positions, self.inv_freq)

class _LlavaCondAttention(nn.Module):
    """Official LLaVA VIP attention with vision-conditioned queries/keys."""

    def __init__(self, hidden_size: int, cond_size: int, num_heads: int) -> None:
        super().__init__()
        qk_size = hidden_size + cond_size
        self.q_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.k_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_states: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, seq_length = hidden_states.shape[:2]
        qk = hidden_states if cond_states is None else torch.cat([hidden_states, cond_states], dim=-1)
        q = self.q_proj(qk).reshape(batch, seq_length, self.num_heads, -1)
        k = self.k_proj(qk).reshape(batch, seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(batch, seq_length, self.num_heads, -1)
        q_dtype, k_dtype = q.dtype, k.dtype
        q, k = q.float(), k.float()
        cos, sin = position_embeddings
        cos = cos.to(device=q.device, dtype=torch.float32)
        sin = sin.to(device=q.device, dtype=torch.float32)
        q = (q * cos.unsqueeze(-2)) + (_rotate_half(q) * sin.unsqueeze(-2))
        k = (k * cos.unsqueeze(-2)) + (_rotate_half(k) * sin.unsqueeze(-2))
        q, k = q.to(q_dtype), k.to(k_dtype)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0
        )
        output = output.transpose(1, 2).reshape(batch, seq_length, -1)
        return self.o_proj(output)

class _LlavaFuserLayer(nn.Module):
    def __init__(self, hidden_size: int, cond_size: int, num_heads: int, hidden_act: str) -> None:
        super().__init__()
        self.norm1 = _llama_rms_norm(hidden_size)
        self.norm2 = _llama_rms_norm(hidden_size)
        self.attn = _LlavaCondAttention(hidden_size, cond_size, num_heads)
        self.mlp = _FuserMLP(hidden_size, hidden_size * 2, hidden_act, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_states: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cond_states, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class _LlavaAttnFuserDummy(nn.Module):
    """Official ``_LlavaAttnFuserDummy``: normalized mean raw attention (no params)."""

    def __init__(self, gp: GlimpsePruneConfig, dims: _FuserDims) -> None:
        super().__init__()
        self._gp = gp

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds):
        del attn_grid_hw, selected_image_embeds
        mean = attn_map.mean(dim=-1)
        mean = torch.softmax(mean, dim=-1) if self._gp.use_attention_logits else torch.exp(mean)
        low = mean.min(dim=-1, keepdim=True).values
        high = mean.max(dim=-1, keepdim=True).values
        return ((mean - low) / (high - low + 1e-6)).unsqueeze(1)

class _LlavaAttnFuserMLP(nn.Module):
    """Official ``_LlavaAttnFuserMLP``: per-token MLP over fused attention heads."""

    def __init__(self, gp: GlimpsePruneConfig, dims: _FuserDims) -> None:
        from transformers.activations import ACT2FN

        super().__init__()
        input_size = len(gp.selected_layers) * dims.text_num_heads
        if input_size <= 0:
            raise ValueError("_LlavaAttnFuserMLP requires at least one selected attention layer.")
        self.attn_mlp = nn.Sequential(
            nn.Linear(input_size, gp.attn_fuse_size),
            ACT2FN[gp.attn_fuse_hidden_act],
            nn.Linear(gp.attn_fuse_size, 1),
        )

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds):
        del attn_grid_hw, selected_image_embeds
        return self.attn_mlp(attn_map).squeeze(-1).unsqueeze(1)

class _LlavaAttnFuserV1(nn.Module):
    """Official VIP used by the released LLaVA-1.5 checkpoints."""

    def __init__(self, gp: GlimpsePruneConfig, dims: _FuserDims) -> None:
        super().__init__()
        self._gp = gp
        attn_fuse_size = gp.attn_fuse_size
        num_layers = len(gp.selected_visual_layers)
        visual_cond_size = gp.visual_cond_size if num_layers > 0 else 0
        self.attn_in_proj = nn.Linear(len(gp.selected_layers) * dims.text_num_heads, attn_fuse_size)
        self.cond_in_projs = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.attn_out_projs = nn.ModuleList()
        for i in range(num_layers):
            self.cond_in_projs.append(nn.Linear(dims.vision_hidden_size, visual_cond_size))
            self.layers.append(
                _LlavaFuserLayer(attn_fuse_size, visual_cond_size, gp.attn_fuse_num_heads, gp.attn_fuse_hidden_act)
            )
            if not gp.deep_supervision and i < num_layers - 1:
                self.attn_out_projs.append(nn.Identity())
            else:
                self.attn_out_projs.append(nn.Linear(attn_fuse_size, 1))
        head_dim = (attn_fuse_size + visual_cond_size) // gp.attn_fuse_num_heads
        if (attn_fuse_size + visual_cond_size) % gp.attn_fuse_num_heads != 0:
            raise ValueError(
                f"attn_fuse_size {attn_fuse_size} + visual_cond_size {visual_cond_size} must be divisible by "
                f"num_heads {gp.attn_fuse_num_heads}"
            )
        self.rotary_pos_emb = _LlavaVisionRotary(head_dim // 2)

    def _position_embeddings(self, batch: int, height: int, width: int):
        device = self.rotary_pos_emb.inv_freq.device
        h_ids = torch.arange(height, device=device).unsqueeze(1).expand(-1, width).flatten()
        w_ids = torch.arange(width, device=device).unsqueeze(0).expand(height, -1).flatten()
        pos_ids = torch.stack([h_ids, w_ids], dim=-1)
        rotary = self.rotary_pos_emb(max(height, width))[pos_ids].flatten(1)
        rotary = rotary.unsqueeze(0).expand(batch, -1, -1)
        embedding = torch.cat((rotary, rotary), dim=-1)
        return embedding.cos(), embedding.sin()

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds):
        attn_outs = []
        batch = attn_map.shape[0]
        height, width = (int(value) for value in attn_grid_hw)
        if height * width != attn_map.shape[1]:
            raise ValueError(
                f"VIP grid {height}x{width} does not match {attn_map.shape[1]} visual tokens."
            )
        if self._gp.ori_attn_supervision and not self.training:
            attn_map_mean = attn_map.mean(dim=-1)
            attn_map_mean = (
                torch.softmax(attn_map_mean, dim=-1)
                if self._gp.use_attention_logits
                else torch.exp(attn_map_mean)
            )
            low = attn_map_mean.min(dim=-1, keepdim=True).values
            high = attn_map_mean.max(dim=-1, keepdim=True).values
            attn_outs.append((attn_map_mean - low) / (high - low + 1e-6))

        attn_hiddens = self.attn_in_proj(attn_map)
        position_embeddings = self._position_embeddings(batch, height, width)
        for layer_num, layer in enumerate(self.layers):
            curr_cond_hiddens = self.cond_in_projs[layer_num](
                selected_image_embeds[layer_num].to(attn_hiddens.dtype)
            )
            attn_hiddens = layer(attn_hiddens, curr_cond_hiddens, position_embeddings)
            if self.training or layer_num == len(self.layers) - 1:
                attn_out_proj = self.attn_out_projs[layer_num]
                if isinstance(attn_out_proj, nn.Identity):
                    continue  # not deep supervision
                aux_attn_out = attn_out_proj(attn_hiddens).squeeze(-1)
                attn_outs.append(aux_attn_out)

        if not attn_outs:
            raise RuntimeError("_LlavaAttnFuserV1 produced no output logits.")
        return torch.stack(attn_outs, dim=1)

class _QwenCondAttention(nn.Module):
    """Official ``CondSdpaAttention``: queries/keys conditioned on visual features.

    Weight shapes are identical to the FA2 variant, so released checkpoints
    load regardless of which attention backend they were trained with.
    """

    def __init__(self, hidden_size: int, cond_size: int, num_heads: int) -> None:
        super().__init__()
        qk_size = hidden_size + cond_size
        v_size = hidden_size
        self.q_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.k_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.v_proj = nn.Linear(v_size, v_size, bias=False)
        self.o_proj = nn.Linear(v_size, v_size, bias=False)
        self.num_heads = num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_states: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

        seq_length = hidden_states.shape[0]
        qk = hidden_states if cond_states is None else torch.cat([hidden_states, cond_states], dim=-1)
        q = self.q_proj(qk).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(qk).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Per-image SDPA over the packed sequence (official "better impl").
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (q, k, v)]
        attn_outputs = [
            F.scaled_dot_product_attention(q_split, k_split, v_split, None, dropout_p=0.0)[0]
            for q_split, k_split, v_split in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
        return self.o_proj(attn_output)

class _QwenFuserLayer(nn.Module):
    def __init__(self, hidden_size: int, cond_size: int, num_heads: int, hidden_act: str) -> None:
        super().__init__()
        self.norm1 = _qwen_rms_norm(hidden_size)
        self.norm2 = _qwen_rms_norm(hidden_size)
        self.attn = _QwenCondAttention(hidden_size, cond_size, num_heads)
        self.mlp = _FuserMLP(hidden_size, hidden_size * 2, hidden_act, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_states: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cond_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class _QwenAttnFuserDummy(nn.Module):
    """Official ``_QwenAttnFuserDummy``: normalized mean raw attention (no params)."""

    def __init__(self, gp: GlimpsePruneConfig, dims: _FuserDims) -> None:
        super().__init__()
        self._gp = gp

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        attn_map_mean = attn_map.mean(dim=-1)
        st = 0
        ori_attns = []
        for h, w in attn_grid_hw.tolist():
            ed = st + h * w
            one_attn_map = attn_map_mean[st:ed]
            st = ed
            if self._gp.use_attention_logits:
                one_attn_map = torch.softmax(one_attn_map, dim=-1)
            else:
                one_attn_map = torch.exp(one_attn_map)
            one_min, one_max = one_attn_map.min(), one_attn_map.max()
            ori_attns.append((one_attn_map - one_min) / (one_max - one_min + 1e-6))
        return torch.cat(ori_attns, dim=0).unsqueeze(0)

class _QwenAttnFuserMLP(nn.Module):
    """Official ``_QwenAttnFuserMLP``: per-token MLP over fused attention heads."""

    def __init__(self, gp: GlimpsePruneConfig, dims: _FuserDims) -> None:
        from transformers.activations import ACT2FN

        super().__init__()
        input_size = len(gp.selected_layers) * dims.text_num_heads
        if input_size <= 0:
            raise ValueError("_QwenAttnFuserMLP requires at least one selected attention layer.")
        self.attn_mlp = nn.Sequential(
            nn.Linear(input_size, gp.attn_fuse_size),
            ACT2FN[gp.attn_fuse_hidden_act],
            nn.Linear(gp.attn_fuse_size, 1),
        )

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        return self.attn_mlp(attn_map).squeeze(-1).unsqueeze(0)

class _QwenAttnFuserV1(nn.Module):
    """Official ``_QwenAttnFuserV1`` — the VIP used by the released Qwen checkpoints."""

    def __init__(self, gp: GlimpsePruneConfig, dims: _FuserDims) -> None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding

        super().__init__()
        self._gp = gp
        self._dims = dims
        attn_fuse_size = gp.attn_fuse_size
        num_layers = len(gp.selected_visual_layers)
        visual_cond_size = gp.visual_cond_size if num_layers > 0 else 0
        self._visual_cond_size = visual_cond_size
        self.attn_in_proj = nn.Linear(len(gp.selected_layers) * dims.text_num_heads, attn_fuse_size)
        self.cond_in_projs = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.attn_out_projs = nn.ModuleList()
        for i in range(num_layers):
            self.cond_in_projs.append(nn.Linear(dims.vision_hidden_size, visual_cond_size))
            self.layers.append(
                _QwenFuserLayer(attn_fuse_size, visual_cond_size, gp.attn_fuse_num_heads, gp.attn_fuse_hidden_act)
            )
            if not gp.deep_supervision and i < num_layers - 1:
                self.attn_out_projs.append(nn.Identity())
            else:
                self.attn_out_projs.append(nn.Linear(attn_fuse_size, 1))
        head_dim = (attn_fuse_size + visual_cond_size) // gp.attn_fuse_num_heads
        if (attn_fuse_size + visual_cond_size) % gp.attn_fuse_num_heads != 0:
            raise ValueError(
                f"attn_fuse_size {attn_fuse_size} + visual_cond_size {visual_cond_size} must be divisible by "
                f"num_heads {gp.attn_fuse_num_heads}"
            )
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    def rot_pos_emb(self, grid_hw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for h, w in grid_hw.tolist():
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))
        pos_ids = torch.cat(pos_ids, dim=0).to(self.rotary_pos_emb.inv_freq.device)
        max_grid_size = int(grid_hw.max().item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        attn_outs = []
        if self._gp.ori_attn_supervision and not self.training:
            attn_map_mean = attn_map.mean(dim=-1)
            st = 0
            ori_attns = []
            for h, w in attn_grid_hw.tolist():
                ed = st + h * w
                one_attn_map = attn_map_mean[st:ed]
                st = ed
                if self._gp.use_attention_logits:
                    one_attn_map = torch.softmax(one_attn_map, dim=-1)
                else:
                    one_attn_map = torch.exp(one_attn_map)
                one_min, one_max = one_attn_map.min(), one_attn_map.max()
                ori_attns.append((one_attn_map - one_min) / (one_max - one_min + 1e-6))
            attn_outs.append(torch.cat(ori_attns, dim=0))

        attn_hiddens = self.attn_in_proj(attn_map)
        attn_hiddens = attn_hiddens[window_index, :]
        cond_hiddens = [img_embeds[window_index, :] for img_embeds in selected_image_embeds]
        rotary_pos_emb = self.rot_pos_emb(attn_grid_hw).to(attn_hiddens.device)
        rotary_pos_emb = rotary_pos_emb[window_index, :]
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        reverse_indices = torch.argsort(window_index)

        merge_unit = self._dims.spatial_merge_size**2
        if self._gp.attn_fuse_global:
            cu_current_seqlens = cu_seqlens // merge_unit
        else:
            cu_current_seqlens = cu_window_seqlens // merge_unit
        for layer_num, layer in enumerate(self.layers):
            curr_cond_hiddens = self.cond_in_projs[layer_num](cond_hiddens[layer_num].to(attn_hiddens.dtype))
            attn_hiddens = layer(attn_hiddens, curr_cond_hiddens, cu_current_seqlens, position_embeddings)
            if self.training or layer_num == len(self.layers) - 1:
                attn_out_proj = self.attn_out_projs[layer_num]
                if isinstance(attn_out_proj, nn.Identity):
                    continue  # not deep supervision
                aux_attn_out = attn_out_proj(attn_hiddens).squeeze(-1)
                attn_outs.append(aux_attn_out[reverse_indices])

        return torch.stack(attn_outs, dim=0)

ATTN_FUSERS: dict[str, dict[str, type[nn.Module]]] = {
    "llava": {
        "AttnFuserDummy": _LlavaAttnFuserDummy,
        "AttnFuserMLP": _LlavaAttnFuserMLP,
        "AttnFuserV1": _LlavaAttnFuserV1,
    },
    "qwen": {
        "AttnFuserDummy": _QwenAttnFuserDummy,
        "AttnFuserMLP": _QwenAttnFuserMLP,
        "AttnFuserV1": _QwenAttnFuserV1,
    },
}


# ---------------------------------------------------------------------------
# Backends — the parts that genuinely differ between the two releases
# ---------------------------------------------------------------------------


class _Backend(ABC):
    """Per-backbone half of GlimpsePrune.

    Everything the two releases share lives on ``GlimpsePruneCompressor``; this
    holds only what standard-RoPE/square-grid LLaVA and M-RoPE/windowed-ViT Qwen
    genuinely cannot share -- the VIP variants, the vision forward, the position
    bookkeeping, the causal-mask shape, and the glimpse-attention recompute.
    """

    key: str

    @staticmethod
    def detect(model: Any) -> "_Backend":
        """Pick the backend for a loaded model, or explain why neither fits.

        Checked structurally rather than by class, so subclasses and local forks
        pass. The released weights are backbone-specific, so getting this wrong
        would fail at ``load_state_dict`` anyway -- but failing here names the
        problem instead.
        """
        inner = getattr(model, "model", None)
        config = getattr(model, "config", None)
        if getattr(inner, "visual", None) is not None:
            return _QwenBackend()
        if getattr(inner, "vision_tower", None) is not None:
            return _LlavaBackend()
        raise TypeError(
            f"GlimpsePrune supports the LLaVA-1.5 and Qwen2.5-VL backbones; "
            f"{type(model).__name__} exposes neither `model.model.vision_tower` nor "
            f"`model.model.visual`. Run with `--backbone llava` or `--backbone qwen-vl`."
        )

    @abstractmethod
    def rms_norm(self, hidden_size: int, eps: float) -> nn.Module:
        """RMSNorm flavour for the glimpse-token norm (mathematically identical;
        kept per-backbone so each matches its reference import)."""

    @abstractmethod
    def fuser_dims(self, model: Any) -> _FuserDims: ...

    @abstractmethod
    def validate(self, model: Any, gp: GlimpsePruneConfig) -> None:
        """Backbone-specific config checks, run before any weights are built."""

    @abstractmethod
    def prefill_inputs(self, model: Any, inputs: Any, input_ids: torch.Tensor) -> dict[str, Any] | None:
        """Extra tensors ``prefill`` needs, or ``None`` to skip pruning entirely."""

    @abstractmethod
    def prefill(self, c: "GlimpsePruneCompressor", model: Any, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, pixel_values: torch.Tensor,
                **extras: Any) -> tuple: ...

    @abstractmethod
    def continuation_kwargs(self, model: Any, prefill_out: tuple) -> dict[str, Any]:
        """How the stock generate loop is told where the pruned sequence resumes."""

    def fuser(self, gp: GlimpsePruneConfig, dims: _FuserDims) -> nn.Module:
        table = ATTN_FUSERS[self.key]
        if gp.attn_fuse_type not in table:
            raise ValueError(f"Unknown attn_fuse_type {gp.attn_fuse_type}. Options: {sorted(table)}")
        return table[gp.attn_fuse_type](gp, dims)


class _LlavaBackend(_Backend):
    """LLaVA-1.5: standard RoPE, a square CLIP patch grid, one causal mask."""

    key = "llava"

    def rms_norm(self, hidden_size: int, eps: float) -> nn.Module:
        return _llama_rms_norm(hidden_size, eps=eps)

    def fuser_dims(self, model: Any) -> _FuserDims:
        return _FuserDims(
            text_num_heads=int(model.config.text_config.num_attention_heads),
            vision_hidden_size=int(model.config.vision_config.hidden_size),
        )

    def validate(self, model: Any, gp: GlimpsePruneConfig) -> None:
        if getattr(model.config, "model_type", None) != "llava":
            raise TypeError(
                "GlimpsePrune's LLaVA path requires the LLMOE LLaVA backbone "
                "(transformers LlavaForConditionalGeneration)."
            )

    def prefill_inputs(self, model: Any, inputs: Any, input_ids: torch.Tensor) -> dict[str, Any] | None:
        return {}

    def continuation_kwargs(self, model: Any, prefill_out: tuple) -> dict[str, Any]:
        # LLaVA carries the retained positions forward explicitly; the next token
        # sits one past the last retained one.
        reduced_positions = prefill_out[2]
        return {"position_ids": torch.cat([reduced_positions, reduced_positions[:, -1:] + 1], dim=-1)}

    def prefill(
        self,
        c: "GlimpsePruneCompressor",
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor]:
            """Run the official LLaVA glimpse, reduction, and remaining layers."""
            from transformers.cache_utils import DynamicCache

            gp = c._gp
            inner = model.model
            lm = inner.language_model
            device = input_ids.device
            batch = input_ids.shape[0]
            image_token_id = int(model.config.image_token_id)
            num_layers = len(lm.layers)

            if attention_mask[:, -1].eq(0).any():
                raise NotImplementedError("GlimpsePrune requires left padding (the glimpse token is appended last).")

            # Stock HF LLaVA receives an expanded image-placeholder span. Capture
            # the same CLIP states used by the released VIP.
            vision_out = inner.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                output_hidden_states=True,
                return_dict=True,
            )
            image_embeds = torch.cat(list(vision_out.pooler_output), dim=0)
            image_positions = input_ids == image_token_id
            tokens_per_row = image_positions.sum(dim=1)
            expected_per_image = [int(features.shape[0]) for features in vision_out.pooler_output]
            if len(expected_per_image) != batch or tokens_per_row.tolist() != expected_per_image:
                raise NotImplementedError(
                    "GlimpsePrune expects exactly one image per sample; got image-token "
                    f"counts {tokens_per_row.tolist()} vs features {expected_per_image}."
                )
            visual_tokens = int(tokens_per_row[0].item())
            if tokens_per_row.ne(visual_tokens).any():
                raise NotImplementedError("LLaVA GlimpsePrune requires an equal visual grid across the batch.")
            grid_size = int(visual_tokens**0.5)
            if grid_size * grid_size != visual_tokens:
                raise ValueError(f"Expected a square LLaVA visual grid, got {visual_tokens} tokens.")
            grid_hw = (grid_size, grid_size)

            selected_image_embeds = []
            for layer_idx in gp.selected_visual_layers:
                try:
                    states = vision_out.hidden_states[layer_idx]
                except IndexError as exc:
                    raise ValueError(
                        f"selected_visual_layers contains {layer_idx}, but the vision tower returned "
                        f"{len(vision_out.hidden_states)} states."
                    ) from exc
                selected_image_embeds.append(states[:, 1:].to(image_embeds.dtype))

            # Text embeddings + image scatter, matching stock LlavaModel.forward.
            inputs_embeds = lm.embed_tokens(input_ids)
            scatter_mask = image_positions.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                scatter_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )

            # Match stock Llama prefill positions. Pruning retains these original
            # RoPE indices rather than compacting them.
            position_ids = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)

            # --- append the glimpse token(s) (official `_append_le`, inference path)
            le_len = 0
            if c.learnable_embeddings is not None and 0 in gp.le_layers:
                le_len = gp.le_length
                le = c._processed_le(gp.le_layers.index(0)).to(inputs_embeds.dtype)
                inputs_embeds = torch.cat([inputs_embeds, le.expand(batch, le_len, -1)], dim=1)
                eos_token_id = c._eos_token_id(model)
                input_ids_g = torch.cat(
                    [input_ids, torch.full((batch, le_len), eos_token_id, dtype=input_ids.dtype, device=device)], dim=1
                )
                attention_mask_g = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch, le_len))], dim=1
                )
                le_positions = position_ids[:, -1:] + torch.arange(
                    1, le_len + 1, device=device, dtype=position_ids.dtype
                )
                position_ids_g = torch.cat([position_ids, le_positions], dim=1)
            else:
                input_ids_g = input_ids
                attention_mask_g = attention_mask
                position_ids_g = position_ids
            total_len = input_ids_g.shape[1]
            kv_mask = input_ids_g == image_token_id
            q_index = total_len - 1  # official: `[seq_len - 1] * bsz` at inference

            # --- prefill decoder layers 0..reduce_layer with attention extraction
            try:
                cache = DynamicCache(config=lm.config)
            except TypeError:
                cache = DynamicCache()
            causal_mask = self._build_causal_mask(lm, inputs_embeds, attention_mask_g, position_ids_g)
            position_embeddings = lm.rotary_emb(inputs_embeds, position_ids_g)
            pad_additive = None
            if not gp.use_attention_logits:
                neg = torch.finfo(inputs_embeds.dtype).min
                pad_additive = torch.zeros(
                    (batch, 1, 1, total_len), dtype=inputs_embeds.dtype, device=device
                ).masked_fill(~attention_mask_g.bool().view(batch, 1, 1, total_len), neg)

            selected_layers = tuple(gp.selected_layers)
            if not selected_layers and not gp.use_zero_masks:
                raise ValueError("selected_layers cannot be empty unless use_zero_masks=True.")
            batched_attns: list[list[torch.Tensor | None]] = [[None] * len(selected_layers) for _ in range(batch)]
            max_forward_layer = max(list(selected_layers) + [gp.reduce_layer])
            max_forward_layer = min(max_forward_layer, num_layers - 1)

            hidden_states = inputs_embeds
            for layer_id in range(max_forward_layer + 1):
                layer = lm.layers[layer_id]
                if layer_id > 0 and c.learnable_embeddings is not None and layer_id in gp.le_layers:
                    # Official `_try_add_le`: add the per-layer glimpse embedding
                    # at the glimpse position(s) before the layer runs.
                    le = c._processed_le(gp.le_layers.index(layer_id)).to(hidden_states.dtype)
                    start = max(0, q_index - gp.le_length + 1)
                    width = q_index + 1 - start
                    hidden_states = torch.cat(
                        [
                            hidden_states[:, :start],
                            hidden_states[:, start : q_index + 1] + le[:, -width:],
                            hidden_states[:, q_index + 1 :],
                        ],
                        dim=1,
                    )
                if layer_id in selected_layers and not gp.use_zero_masks:
                    weights = self._extract_glimpse_attention(
                        c,
                        layer, hidden_states, position_embeddings, pad_additive, kv_mask
                    )
                    layer_pos = selected_layers.index(layer_id)
                    for b in range(batch):
                        batched_attns[b][layer_pos] = weights[b]
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids_g,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

            # --- trim the glimpse token(s) (official step 6)
            if le_len > 0:
                input_ids_g = input_ids_g[:, :-le_len]
                hidden_states = hidden_states[:, :-le_len]
                attention_mask_g = attention_mask_g[:, :-le_len]
                position_ids_g = position_ids_g[:, :-le_len]
                c._crop_cache(cache, input_ids_g.shape[1])

            # --- VIP: per-token keep logits (official `_decode_image_token_mask_logits`)
            if gp.use_zero_masks:
                mask_logits = torch.full(
                    (batch, 1, visual_tokens), float("-inf"),
                    device=device, dtype=hidden_states.dtype,
                )
            else:
                stacked = torch.stack(
                    [torch.stack(one, dim=1) for one in batched_attns], dim=0
                ).flatten(2)
                fuser_parameter = next(c.attn_fuser.parameters(), None)
                if fuser_parameter is not None:
                    stacked = stacked.to(fuser_parameter.device, fuser_parameter.dtype)
                    selected_image_embeds = [
                        states.to(fuser_parameter.device, fuser_parameter.dtype)
                        for states in selected_image_embeds
                    ]
                mask_logits = c.attn_fuser(stacked, grid_hw, selected_image_embeds)

            # --- keep masks + pruning (official `_get_remain_masks` / `_reduce_tokens`)
            importance_scores = [mask_logits[row, -1].sigmoid() for row in range(batch)]
            bool_masks = [c._apply_token_constraints(scores) for scores in importance_scores]
            remain = attention_mask_g.bool().clone()
            remain[image_positions] = torch.cat(bool_masks, dim=0)
            remain &= attention_mask_g.bool()

            if c._track_source:
                c._last_importance_maps = [
                    scores.detach().float().cpu().reshape(grid_hw)
                    for scores in importance_scores
                ]
                c._last_keep_masks = [
                    mask.detach().cpu().reshape(grid_hw)
                    for mask in bool_masks
                ]
                c._record_source(image_positions[0], remain[0])
            total_visual = int(image_positions.sum().item())
            kept_visual = int((remain & image_positions).sum().item())
            c._last_keep_ratio = kept_visual / total_visual if total_visual else None

            lengths = remain.sum(dim=1)
            reduced_len = int(lengths.max().item())
            repad = torch.zeros((batch, reduced_len), dtype=torch.bool, device=device)
            for b in range(batch):
                repad[b, reduced_len - int(lengths[b].item()) :] = True

            pad_token_id = getattr(model.config, "pad_token_id", None) or 0
            reduced_ids = torch.full((batch, reduced_len), pad_token_id, dtype=input_ids_g.dtype, device=device)
            reduced_ids[repad] = input_ids_g[remain]
            reduced_mask = torch.zeros((batch, reduced_len), dtype=attention_mask_g.dtype, device=device)
            reduced_mask[repad] = attention_mask_g[remain]
            reduced_positions = torch.full(
                (batch, reduced_len), 1, dtype=position_ids_g.dtype, device=device
            )
            reduced_positions[repad] = position_ids_g[remain]
            reduced_hidden = torch.zeros(
                (batch, reduced_len, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=device
            )
            reduced_hidden[repad] = hidden_states[remain]
            c._prune_cache(cache, remain, repad)

            # --- finish the prefill on the pruned sequence (official
            #     `_glimpse_forward_after_reduction`)
            final_hidden = reduced_hidden
            if max_forward_layer < num_layers - 1:
                causal_mask = self._build_causal_mask(
                    lm, reduced_hidden, reduced_mask, reduced_positions
                )
                position_embeddings = lm.rotary_emb(reduced_hidden, reduced_positions)
                for layer_id in range(max_forward_layer + 1, num_layers):
                    final_hidden = lm.layers[layer_id](
                        final_hidden,
                        attention_mask=causal_mask,
                        position_ids=reduced_positions,
                        past_key_values=cache,
                        use_cache=True,
                        position_embeddings=position_embeddings,
                    )
                    if isinstance(final_hidden, tuple):
                        final_hidden = final_hidden[0]

            # First-token logits under official semantics: layers <= reduce_layer
            # saw the full context, later layers the pruned one (`norm` is
            # positionwise, so norming the pruned rows matches norming pre-prune).
            first_logits = model.lm_head(lm.norm(final_hidden[:, -1:, :])).squeeze(1)

            return reduced_ids, reduced_mask, reduced_positions, cache, first_logits

    def _extract_glimpse_attention(
        self,
        c: "GlimpsePruneCompressor",
        layer: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        pad_additive: torch.Tensor | None,
        kv_mask: torch.Tensor,
        ) -> list[torch.Tensor]:
            """Glimpse-token attention over image tokens at one decoder layer.

            Equivalent to the official ``_cal_attn_weights``: computes the
            post-RoPE q·kᵀ/√d row for the glimpse query only (never materializing
            the full attention map), optionally log-softmax normalized, then
            selects the image-token columns per sample.
            """
            from transformers.models.llama.modeling_llama import repeat_kv

            attn = layer.self_attn
            normed = layer.input_layernorm(hidden_states)
            batch, total_len, _ = normed.shape
            head_dim = attn.head_dim
            query = attn.q_proj(normed[:, -1:, :]).view(batch, 1, -1, head_dim).transpose(1, 2)
            keys = attn.k_proj(normed).view(batch, total_len, -1, head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            query = (query * cos[:, -1:, :].unsqueeze(1)) + (
                _rotate_half(query) * sin[:, -1:, :].unsqueeze(1)
            )
            keys = (keys * cos.unsqueeze(1)) + (_rotate_half(keys) * sin.unsqueeze(1))
            groups = int(attn.config.num_attention_heads // attn.config.num_key_value_heads)
            keys = repeat_kv(keys, groups)
            weights = torch.matmul(query, keys.transpose(-1, -2)) * attn.scaling  # [B, heads, 1, L]
            if not c._gp.use_attention_logits:
                if pad_additive is not None:
                    weights = weights + pad_additive
                weights = torch.log_softmax(weights, dim=-1)
            weights = weights.squeeze(2).transpose(1, 2)  # [B, L, heads]
            return [weights[b][kv_mask[b]] for b in range(batch)]

    @staticmethod
    def _build_causal_mask(
        lm: Any,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        ) -> Any:
            from transformers.masking_utils import create_causal_mask

            return create_causal_mask(
                config=lm.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )

class _QwenBackend(_Backend):
    """Qwen2.5-VL: M-RoPE, a windowed ViT with per-image grids, per-layer masks."""

    key = "qwen"

    def rms_norm(self, hidden_size: int, eps: float) -> nn.Module:
        return _qwen_rms_norm(hidden_size, eps=eps)

    def fuser_dims(self, model: Any) -> _FuserDims:
        return _FuserDims(
            text_num_heads=int(model.config.text_config.num_attention_heads),
            vision_hidden_size=int(model.config.vision_config.hidden_size),
            spatial_merge_size=int(model.config.vision_config.spatial_merge_size),
        )

    def validate(self, model: Any, gp: GlimpsePruneConfig) -> None:
        # `prefill` indexes `block_states[layer + 1]`, so an out-of-range entry
        # would surface as a bare IndexError from deep inside the prefill.
        depth = int(model.config.vision_config.depth)
        if gp.selected_visual_layers and max(gp.selected_visual_layers) >= depth:
            raise ValueError(
                f"selected_visual_layers {gp.selected_visual_layers} exceed the vision-tower "
                f"depth ({depth}). The released config must match the backbone; check that "
                f"new_modules_dir belongs to this model."
            )

    def prefill_inputs(self, model: Any, inputs: Any, input_ids: torch.Tensor) -> dict[str, Any] | None:
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None:
            return None
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = (input_ids == model.config.image_token_id).to(torch.int)
        return {"image_grid_thw": image_grid_thw, "mm_token_type_ids": mm_token_type_ids}

    def continuation_kwargs(self, model: Any, prefill_out: tuple) -> dict[str, Any]:
        # Qwen resumes through `rope_deltas`, matching the official
        # `_update_model_kwargs_for_generation`.
        model.model.rope_deltas = prefill_out[3]
        return {}

    def prefill(
        self,
        c: "GlimpsePruneCompressor",
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor]:
            """Returns ``(reduced_ids, reduced_mask, pruned_cache, rope_deltas, first_logits)``."""
            from transformers.cache_utils import DynamicCache

            gp = c._gp
            inner = model.model  # Qwen2_5_VLModel
            lm = inner.language_model  # Qwen2_5_VLTextModel
            device = input_ids.device
            batch = input_ids.shape[0]
            image_token_id = int(model.config.image_token_id)
            num_layers = len(lm.layers)

            if attention_mask[:, -1].eq(0).any():
                raise NotImplementedError("GlimpsePrune requires left padding (the glimpse token is appended last).")

            # --- vision tower with intermediate feature capture (official `_visual_forward`)
            vision_out = inner.get_image_features(pixel_values, image_grid_thw, output_hidden_states=True)
            image_embeds = torch.cat(list(vision_out.pooler_output), dim=0)
            visual = inner.visual
            merge_unit = visual.spatial_merge_unit
            window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)
            window_index = window_index.to(device)
            cu_window_seqlens = torch.unique_consecutive(
                torch.tensor(cu_window_seqlens, device=device, dtype=torch.int32)
            )
            cu_seqlens = torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            reverse_indices = torch.argsort(window_index)
            block_states = vision_out.hidden_states  # (embed input, block 0, ..., block N-1), window order
            if len(block_states) != len(visual.blocks) + 1:
                raise RuntimeError(
                    f"Expected {len(visual.blocks) + 1} captured vision states, got {len(block_states)}"
                )
            selected_image_embeds = []
            for layer_idx in gp.selected_visual_layers:
                states = block_states[layer_idx + 1]
                states = states.reshape(states.shape[0] // merge_unit, merge_unit, -1).mean(dim=1)
                selected_image_embeds.append(states[reverse_indices])
            attn_grid = image_grid_thw[:, 1:] // visual.spatial_merge_size

            # --- text embeddings + image scatter
            inputs_embeds = lm.embed_tokens(input_ids)
            image_positions = input_ids == image_token_id
            tokens_per_row = image_positions.sum(dim=1)
            expected_per_image = (image_grid_thw.prod(dim=-1) // merge_unit).tolist()
            if len(expected_per_image) != batch or tokens_per_row.tolist() != expected_per_image:
                raise NotImplementedError(
                    "GlimpsePrune (this reimplementation) expects exactly one image per sample; got image token "
                    f"counts {tokens_per_row.tolist()} vs grids {expected_per_image}."
                )
            scatter_mask = image_positions.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(scatter_mask, image_embeds.to(inputs_embeds.dtype))

            # --- original M-RoPE positions
            position_ids, _ = inner.get_rope_index(
                input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
            )

            # --- append the glimpse token(s) (official `_append_le`, inference path)
            le_len = 0
            if c.learnable_embeddings is not None and 0 in gp.le_layers:
                le_len = gp.le_length
                le = c._processed_le(gp.le_layers.index(0)).to(inputs_embeds.dtype)
                inputs_embeds = torch.cat([inputs_embeds, le.expand(batch, le_len, -1)], dim=1)
                eos_token_id = c._eos_token_id(model)
                input_ids_g = torch.cat(
                    [input_ids, torch.full((batch, le_len), eos_token_id, dtype=input_ids.dtype, device=device)], dim=1
                )
                attention_mask_g = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch, le_len))], dim=1
                )
                last_pos = position_ids[-1, :, -1]  # [B]; all three grids agree on trailing text
                le_positions = last_pos.view(1, batch, 1) + torch.arange(1, le_len + 1, device=device).view(1, 1, -1)
                position_ids_g = torch.cat([position_ids, le_positions.expand(3, batch, le_len)], dim=2)
            else:
                input_ids_g = input_ids
                attention_mask_g = attention_mask
                position_ids_g = position_ids
            total_len = input_ids_g.shape[1]
            kv_mask = input_ids_g == image_token_id
            q_index = total_len - 1  # official: `[seq_len - 1] * bsz` at inference

            # --- prefill decoder layers 0..reduce_layer with attention extraction
            try:
                cache = DynamicCache(config=lm.config)
            except TypeError:
                cache = DynamicCache()
            mask_mapping = self._build_causal_masks(lm, inputs_embeds, attention_mask_g)
            position_embeddings = lm.rotary_emb(inputs_embeds, position_ids_g)
            merged_cos, merged_sin = self._merge_mrope(lm, *position_embeddings)
            pad_additive = None
            if not gp.use_attention_logits:
                neg = torch.finfo(inputs_embeds.dtype).min
                pad_additive = torch.zeros(
                    (batch, 1, 1, total_len), dtype=inputs_embeds.dtype, device=device
                ).masked_fill(~attention_mask_g.bool().view(batch, 1, 1, total_len), neg)

            selected_layers = tuple(gp.selected_layers)
            batched_attns: list[list[torch.Tensor | None]] = [[None] * len(selected_layers) for _ in range(batch)]
            max_forward_layer = max(list(selected_layers) + [gp.reduce_layer])
            max_forward_layer = min(max_forward_layer, num_layers - 1)

            layer_types = getattr(lm.config, "layer_types", None) or ["full_attention"] * num_layers
            hidden_states = inputs_embeds
            for layer_id in range(max_forward_layer + 1):
                layer = lm.layers[layer_id]
                if layer_id > 0 and c.learnable_embeddings is not None and layer_id in gp.le_layers:
                    # Official `_try_add_le`: add the per-layer glimpse embedding
                    # at the glimpse position(s) before the layer runs.
                    le = c._processed_le(gp.le_layers.index(layer_id)).to(hidden_states.dtype)
                    start = max(0, q_index - gp.le_length + 1)
                    width = q_index + 1 - start
                    hidden_states = torch.cat(
                        [
                            hidden_states[:, :start],
                            hidden_states[:, start : q_index + 1] + le[:, -width:],
                            hidden_states[:, q_index + 1 :],
                        ],
                        dim=1,
                    )
                if layer_id in selected_layers and not gp.use_zero_masks:
                    weights = self._extract_glimpse_attention(
                        c,
                        layer, hidden_states, merged_cos, merged_sin, pad_additive, kv_mask
                    )
                    layer_pos = selected_layers.index(layer_id)
                    for b in range(batch):
                        batched_attns[b][layer_pos] = weights[b]
                hidden_states = layer(
                    hidden_states,
                    attention_mask=mask_mapping[layer_types[layer_id]],
                    position_ids=None,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

            # --- trim the glimpse token(s) (official step 6)
            if le_len > 0:
                input_ids_g = input_ids_g[:, :-le_len]
                hidden_states = hidden_states[:, :-le_len]
                attention_mask_g = attention_mask_g[:, :-le_len]
                position_ids_g = position_ids_g[:, :, :-le_len]
                c._crop_cache(cache, input_ids_g.shape[1])

            # --- VIP: per-token keep logits (official `_decode_image_token_mask_logits`)
            if gp.use_zero_masks:
                mask_logits = [
                    torch.logit(torch.zeros((1, int(h * w)), device=device)) for h, w in attn_grid.tolist()
                ]
            else:
                stacked = [torch.stack(one, dim=1) for one in batched_attns]  # [n_tokens, n_layers, n_heads]
                num_tokens_by_batch = [one.shape[0] for one in stacked]
                catted = torch.cat(stacked, dim=0)
                catted = catted.reshape(catted.shape[0], -1).to(next(c.attn_fuser.parameters(), catted).dtype)
                fused = c.attn_fuser(
                    catted, attn_grid, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens
                )
                mask_logits = list(fused.split(num_tokens_by_batch, dim=-1))

            # --- keep masks + pruning (official `_get_remain_masks` / `_reduce_tokens`)
            is_visual = input_ids_g == image_token_id
            bool_masks = [
                c._apply_token_constraints(one_logits[-1].sigmoid().flatten()) for one_logits in mask_logits
            ]
            remain = attention_mask_g.bool().clone()
            remain[is_visual] = torch.cat(bool_masks, dim=0)
            remain &= attention_mask_g.bool()

            if c._track_source:
                c._record_source(is_visual[0], remain[0])
            total_visual = int(is_visual.sum().item())
            kept_visual = int((remain & is_visual).sum().item())
            c._last_keep_ratio = kept_visual / total_visual if total_visual else None

            lengths = remain.sum(dim=1)
            reduced_len = int(lengths.max().item())
            repad = torch.zeros((batch, reduced_len), dtype=torch.bool, device=device)
            for b in range(batch):
                repad[b, reduced_len - int(lengths[b].item()) :] = True

            pad_token_id = getattr(model.config, "pad_token_id", None) or 0
            reduced_ids = torch.full((batch, reduced_len), pad_token_id, dtype=input_ids_g.dtype, device=device)
            reduced_ids[repad] = input_ids_g[remain]
            reduced_mask = torch.zeros((batch, reduced_len), dtype=attention_mask_g.dtype, device=device)
            reduced_mask[repad] = attention_mask_g[remain]
            reduced_positions = torch.full((3, batch, reduced_len), 1, dtype=position_ids_g.dtype, device=device)
            reduced_positions[:, repad] = position_ids_g[:, remain]
            reduced_hidden = torch.zeros(
                (batch, reduced_len, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=device
            )
            reduced_hidden[repad] = hidden_states[remain]
            c._prune_cache(cache, remain, repad)

            # --- finish the prefill on the pruned sequence (official
            #     `_glimpse_forward_after_reduction`)
            final_hidden = reduced_hidden
            if max_forward_layer < num_layers - 1:
                mask_mapping = self._build_causal_masks(lm, reduced_hidden, reduced_mask)
                position_embeddings = lm.rotary_emb(reduced_hidden, reduced_positions)
                for layer_id in range(max_forward_layer + 1, num_layers):
                    final_hidden = lm.layers[layer_id](
                        final_hidden,
                        attention_mask=mask_mapping[layer_types[layer_id]],
                        position_ids=None,
                        past_key_values=cache,
                        use_cache=True,
                        position_embeddings=position_embeddings,
                    )
                    if isinstance(final_hidden, tuple):
                        final_hidden = final_hidden[0]

            # First-token logits under official semantics: layers <= reduce_layer
            # saw the full context, later layers the pruned one (`norm` is
            # positionwise, so norming the pruned rows matches norming pre-prune).
            first_logits = model.lm_head(lm.norm(final_hidden[:, -1:, :])).squeeze(1)

            # New tokens continue right after the last original position.
            last_original_pos = reduced_positions[0, :, -1]
            rope_deltas = (last_original_pos + 1 - lengths).view(batch, 1).to(torch.long)
            return reduced_ids, reduced_mask, cache, rope_deltas, first_logits

    def _extract_glimpse_attention(
        self,
        c: "GlimpsePruneCompressor",
        layer: Any,
        hidden_states: torch.Tensor,
        merged_cos: torch.Tensor,
        merged_sin: torch.Tensor,
        pad_additive: torch.Tensor | None,
        kv_mask: torch.Tensor,
        ) -> list[torch.Tensor]:
            """Glimpse-token attention over image tokens at one decoder layer.

            Equivalent to the official ``_cal_attn_weights``: computes the
            post-RoPE q·kᵀ/√d row for the glimpse query only (never materializing
            the full attention map), optionally log-softmax normalized, then
            selects the image-token columns per sample.
            """
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv, rotate_half

            attn = layer.self_attn
            normed = layer.input_layernorm(hidden_states)
            batch, total_len, _ = normed.shape
            head_dim = attn.head_dim
            query = attn.q_proj(normed[:, -1:, :]).view(batch, 1, -1, head_dim).transpose(1, 2)
            keys = attn.k_proj(normed).view(batch, total_len, -1, head_dim).transpose(1, 2)
            cos_q = merged_cos[:, -1:, :].unsqueeze(1)
            sin_q = merged_sin[:, -1:, :].unsqueeze(1)
            query = (query * cos_q) + (rotate_half(query) * sin_q)
            cos_k = merged_cos.unsqueeze(1)
            sin_k = merged_sin.unsqueeze(1)
            keys = (keys * cos_k) + (rotate_half(keys) * sin_k)
            keys = repeat_kv(keys, attn.num_key_value_groups)
            weights = torch.matmul(query, keys.transpose(-1, -2)) * attn.scaling  # [B, heads, 1, L]
            if not c._gp.use_attention_logits:
                if pad_additive is not None:
                    weights = weights + pad_additive
                weights = torch.log_softmax(weights, dim=-1)
            weights = weights.squeeze(2).transpose(1, 2)  # [B, L, heads]
            return [weights[b][kv_mask[b]] for b in range(batch)]

    @staticmethod
    def _build_causal_masks(lm: Any, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

            mask_kwargs = {
                "config": lm.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": None,
            }
            mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(lm, "has_sliding_layers", False):
                mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
            return mapping

    @staticmethod
    def _merge_mrope(lm: Any, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Collapse the 3-grid M-RoPE cos/sin into per-token tensors.

            Mirrors ``apply_multimodal_rotary_pos_emb``'s section interleaving so
            the glimpse-row attention is computed with exactly the rotary state
            the decoder layers use.
            """
            rope_parameters = getattr(lm.config, "rope_parameters", None) or getattr(lm.config, "rope_scaling", None)
            mrope_section = list(rope_parameters["mrope_section"]) * 2
            merged_cos = torch.cat(
                [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
            )
            merged_sin = torch.cat(
                [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
            )
            return merged_cos, merged_sin

# ---------------------------------------------------------------------------
# The compressor — one class, either backbone
# ---------------------------------------------------------------------------


class GlimpsePruneCompressor(TokenCompressor):
    """Dynamic glimpse-guided visual token pruning (paper's non fine-tuned GP).

    Works on both released backbones; ``install()`` detects which one it is on.
    See the module docstring for the checkpoint pairing, which is not
    interchangeable between them.

    Args:
        new_modules_dir: Local directory or HF repo id with the released
            ``config.json`` + ``new_modules_gp.pt`` (glimpse token + VIP) for the
            backbone in use. When ``None`` the modules are randomly initialized
            (smoke tests only) and a warning is emitted.
        reduce_layer: Optional override of the layer after which pruning
            happens. Like the official eval wrapper, overriding it also sets
            ``selected_layers=[reduce_layer]``.
        reduce_threshold: Keep threshold on ``sigmoid`` scores (paper: 0.5).
        min_remain_num: Keep at least this many visual tokens.
        max_remain_ratio: Optional cap on the retained ratio (top-k fallback).
        fixed_remain_ratio: Optional fixed retained ratio (pure top-k; used
            for the paper's fixed-budget ablations).
        use_zero_masks: Ablation that scores every token 0 (keeps
            ``min_remain_num`` tokens).
        config_overrides: Raw ``GlimpsePruneConfig`` field overrides applied on
            top of the loaded config (the official ``new_modules_config``
            escape hatch); mostly useful for tests.
        track_source: Record per-image keep matrices for ``llmoe.viz``.
    """

    def __init__(
        self,
        *,
        new_modules_dir: str | None = None,
        reduce_layer: int | None = None,
        reduce_threshold: float | None = None,
        min_remain_num: int | None = None,
        max_remain_ratio: float | None = None,
        fixed_remain_ratio: float | None = None,
        use_zero_masks: bool = False,
        config_overrides: dict[str, Any] | None = None,
        track_source: bool = False,
    ) -> None:
        super().__init__(max_tokens=0)  # dynamic: no fixed budget
        self._new_modules_dir = new_modules_dir
        self._config_overrides = dict(config_overrides or {})
        self._overrides: dict[str, Any] = {
            "reduce_layer": reduce_layer,
            "reduce_threshold": reduce_threshold,
            "min_remain_num": min_remain_num,
            "max_remain_ratio": max_remain_ratio,
            "fixed_remain_ratio": fixed_remain_ratio,
        }
        self._use_zero_masks = bool(use_zero_masks)
        self._track_source = bool(track_source)

        self._gp: GlimpsePruneConfig | None = None
        self._backend: _Backend | None = None
        self._vlm: VLM | None = None
        self._original_generate: Any | None = None
        self.attn_fuser: nn.Module | None = None
        self.le_proj: nn.Module | None = None
        self.le_norm: nn.Module | None = None
        self.le_dropout: nn.Module | None = None
        self._last_keep_ratio: float | None = None
        self._last_importance_maps: list[torch.Tensor] = []
        self._last_keep_masks: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # TokenCompressor interface
    # ------------------------------------------------------------------

    @property
    @override
    def budget(self) -> int | float:
        """Retention ratio in force. GlimpsePrune prunes dynamically, so it has no
        fixed token count to report; the ratio knobs are the only thing telling its
        runs apart. ``fixed_remain_ratio`` is the ablation, ``max_remain_ratio`` the
        cap, and zero-masks runs have neither.
        """
        if self._use_zero_masks:
            return 0
        fixed = self._overrides.get("fixed_remain_ratio")
        if fixed is not None:
            return float(fixed)
        capped = self._overrides.get("max_remain_ratio")
        return float(capped) if capped is not None else 0

    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs: Any) -> CompressedFeatures:
        # GlimpsePrune does not compress vision features; pruning happens
        # inside the LLM prefill (see install/_wrapped_generate).
        del attention, layer_idx, kwargs
        return CompressedFeatures(tokens=features)

    def get_source_history(self) -> list[torch.Tensor]:
        return list(self._source_history)

    def install(self, vlm: VLM) -> None:
        self._vlm = vlm
        model = vlm.model  # triggers lazy load
        self._backend = _Backend.detect(model)
        self._build_and_load_modules(model)
        self.to(device=model.device, dtype=model.dtype)
        self.eval()
        if getattr(vlm, "_glimpse_prune_wrapped", False):
            return
        self._original_generate = vlm.generate

        def wrapped_generate(inputs: Any, *, max_new_tokens: int = 256, **kwargs: Any) -> torch.Tensor:
            return self._wrapped_generate(inputs, max_new_tokens=max_new_tokens, **kwargs)

        vlm.generate = wrapped_generate
        vlm._glimpse_prune_wrapped = True

    def uninstall(self) -> None:
        if self._vlm is not None and self._original_generate is not None:
            self._vlm.generate = self._original_generate
            self._vlm._glimpse_prune_wrapped = False
        self._vlm = None
        self._original_generate = None
        self._backend = None

    @property
    def backend(self) -> str | None:
        """``"llava"`` or ``"qwen"`` once installed, else ``None``."""
        return None if self._backend is None else self._backend.key

    # ------------------------------------------------------------------
    # Module construction / weight loading
    # ------------------------------------------------------------------

    def _resolve_new_modules_dir(self) -> str | None:
        if self._new_modules_dir is None:
            return None
        if os.path.isdir(self._new_modules_dir):
            return self._new_modules_dir
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=self._new_modules_dir)

    def _build_and_load_modules(self, model: Any) -> None:
        backend = self._backend
        directory = self._resolve_new_modules_dir()
        if directory is not None:
            gp = GlimpsePruneConfig.from_json(os.path.join(directory, "config.json"))
        else:
            warnings.warn(
                "GlimpsePrune: no new_modules_dir given; the glimpse token and VIP are randomly "
                "initialized. Results are meaningless outside smoke tests."
            )
            gp = GlimpsePruneConfig()
        if self._config_overrides:
            merged = {**gp.__dict__, **self._config_overrides}
            gp = GlimpsePruneConfig.from_dict(merged)
        for key, value in self._overrides.items():
            if value is not None:
                setattr(gp, key, value)
        if self._overrides["reduce_layer"] is not None:
            # Mirrors the official eval wrapper: overriding the reduce layer
            # also moves the attention extraction to that layer.
            gp.selected_layers = (int(self._overrides["reduce_layer"]),)
        gp.use_zero_masks = gp.use_zero_masks or self._use_zero_masks
        self._gp = gp

        text_config = model.config.text_config
        backend.validate(model, gp)
        num_layers = int(text_config.num_hidden_layers)
        if not 0 <= gp.reduce_layer < num_layers:
            raise ValueError(f"reduce_layer {gp.reduce_layer} must be in [0, {num_layers - 1}]")
        if gp.le_layers and max(gp.le_layers) >= num_layers:
            raise ValueError(f"le_layers {gp.le_layers} exceed the decoder depth ({num_layers})")
        if gp.selected_layers and max(gp.selected_layers) > gp.reduce_layer:
            raise ValueError(
                f"selected_layers {gp.selected_layers} must be <= reduce_layer {gp.reduce_layer}"
            )

        self.attn_fuser = backend.fuser(gp, backend.fuser_dims(model))

        hidden_size = int(text_config.hidden_size)
        if gp.le_layers and gp.le_length > 0:
            self.register_parameter(
                "learnable_embeddings",
                nn.Parameter(torch.empty(len(gp.le_layers), gp.le_length, hidden_size)),
            )
            nn.init.normal_(self.learnable_embeddings, 0, 0.02)
            self.le_proj = nn.Linear(hidden_size, hidden_size)
            if gp.le_norm_type == "rmsnorm":
                self.le_norm = backend.rms_norm(hidden_size, eps=float(text_config.rms_norm_eps))
            elif gp.le_norm_type == "layernorm":
                self.le_norm = nn.LayerNorm(hidden_size)
            else:
                raise ValueError(f"Unsupported le_norm_type: {gp.le_norm_type}")
            self.le_dropout = nn.Dropout(gp.le_dropout_prob)
        else:
            self.learnable_embeddings = None

        if directory is not None:
            weights_path = os.path.join(directory, "new_modules_gp.pt")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"GlimpsePrune weights not found: {weights_path}")
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            # strict=True: a checkpoint built for the other backbone fails here
            # rather than silently mis-scoring every token.
            self.attn_fuser.load_state_dict(state["attn_fuser"], strict=True)
            if self.learnable_embeddings is not None:
                with torch.no_grad():
                    self.learnable_embeddings.copy_(state["learnable_embeddings"])
                self.le_proj.load_state_dict(state["le_proj"], strict=True)
                self.le_norm.load_state_dict(state["le_norm"], strict=True)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _wrapped_generate(self, inputs: Any, *, max_new_tokens: int = 256, **kwargs: Any) -> torch.Tensor:
        vlm = self._vlm
        backend = self._backend
        model = vlm.model
        vlm._configure_generation_tokenizer()
        tokenizer = getattr(vlm.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        if int(kwargs.get("num_beams", 1) or 1) != 1:
            raise ValueError("GlimpsePrune supports num_beams=1 only (greedy/sampled decoding).")
        if not bool(kwargs.pop("use_cache", True)):
            raise ValueError("GlimpsePrune requires use_cache=True: it prunes the prefill KV cache.")
        kwargs.setdefault("do_sample", False)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pixel_values = inputs.get("pixel_values")
        extras = None if pixel_values is None else backend.prefill_inputs(model, inputs, input_ids)
        if pixel_values is None or extras is None:
            # Text-only batch: nothing to prune, fall back to the stock path.
            with torch.inference_mode():
                return model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, **kwargs)

        with torch.inference_mode():
            prefill_out = backend.prefill(
                self, model, input_ids, attention_mask, pixel_values, **extras
            )
            reduced_ids, reduced_mask = prefill_out[0], prefill_out[1]
            first_logits = prefill_out[-1]
            cache = prefill_out[3] if backend.key == "llava" else prefill_out[2]
            # The first generated token comes from the pruned prefill's own
            # logits (official `_glimpse_forward_after_reduction` semantics:
            # layers <= reduce_layer saw the full context, later layers the
            # pruned one). The remaining tokens are decoded by the stock
            # `generate`, continuing from the pruned KV cache with retained
            # tokens keeping their original positions.
            reduced_len = reduced_ids.shape[1]
            first_token = self._sample_token(first_logits, kwargs)
            reduced_ids = torch.cat([reduced_ids, first_token.view(-1, 1)], dim=-1)
            reduced_mask = torch.cat([reduced_mask, reduced_mask.new_ones((reduced_mask.shape[0], 1))], dim=-1)
            continuation = backend.continuation_kwargs(model, prefill_out)
            if max_new_tokens > 1:
                output_ids = model.generate(
                    input_ids=reduced_ids,
                    attention_mask=reduced_mask,
                    past_key_values=cache,
                    max_new_tokens=max_new_tokens - 1,
                    use_cache=True,
                    **continuation,
                    **kwargs,
                )
            else:
                output_ids = reduced_ids
            new_tokens = output_ids[:, reduced_len:].to(input_ids.device)
            new_tokens = self._pad_after_eos(model, new_tokens, kwargs.get("pad_token_id"))
            # Re-attach the original (unpruned) prompt so the pipeline's
            # `decode_output_ids` can strip it by width.
            return torch.cat([input_ids, new_tokens], dim=-1)

    # ------------------------------------------------------------------
    # Helpers (shared by both backends)
    # ------------------------------------------------------------------

    def _processed_le(self, le_index: int) -> torch.Tensor:
            le = self.learnable_embeddings[le_index]
            le_dtype = le.dtype
            le = self.le_dropout(self.le_norm(self.le_proj(le)))
            return le.to(dtype=le_dtype).view(1, self._gp.le_length, -1)

    @staticmethod
    def _eos_token_id(model: Any) -> int:
            eos = getattr(model.config, "eos_token_id", None)
            if eos is None:
                eos = getattr(getattr(model.config, "text_config", None), "eos_token_id", None)
            if eos is None:
                eos = getattr(model.generation_config, "eos_token_id", None)
            if isinstance(eos, (list, tuple)):
                eos = eos[0]
            if eos is None:
                raise ValueError("Could not resolve an eos token id for the glimpse placeholder.")
            return int(eos)

    def _apply_token_constraints(self, one_prob: torch.Tensor) -> torch.Tensor:
            """Official ``_apply_token_constraints`` (threshold + remain bounds)."""
            gp = self._gp
            one_prob = one_prob.flatten()
            if one_prob.numel() == 0:
                return torch.zeros_like(one_prob, dtype=torch.bool)

            if gp.fixed_remain_ratio is not None:
                num_tokens = one_prob.numel()
                fixed_remain_num = int(float(gp.fixed_remain_ratio) * num_tokens)
                fixed_remain_num = max(1, min(num_tokens, fixed_remain_num))
                topk_idx = torch.topk(one_prob, fixed_remain_num, dim=-1).indices
                mask = torch.zeros_like(one_prob, dtype=torch.bool)
                mask[topk_idx] = True
                return mask

            mask = one_prob > gp.reduce_threshold

            if gp.max_remain_ratio is not None:
                remain_num = int(mask.sum().item())
                remain_ratio = remain_num / mask.numel() if mask.numel() > 0 else 0.0
                if remain_ratio > gp.max_remain_ratio:
                    max_remain_num = int(gp.max_remain_ratio * mask.numel())
                    max_remain_num = max(1, min(mask.numel(), max_remain_num))
                    topk_idx = torch.topk(one_prob, max_remain_num, dim=-1).indices
                    mask = torch.zeros_like(one_prob, dtype=torch.bool)
                    mask[topk_idx] = True

            if gp.min_remain_num is not None:
                remain_num = int(mask.sum().item())
                if remain_num < gp.min_remain_num:
                    k = max(1, min(mask.numel(), int(gp.min_remain_num)))
                    topk_idx = torch.topk(one_prob, k, dim=-1).indices
                    mask[topk_idx] = True

            return mask

    @staticmethod
    def _sample_token(logits: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
            """Sample the first generated token from the pruned-prefill logits.

            Greedy unless ``do_sample`` (with temperature / top-p), matching what
            the stock ``generate`` would do with the same logits.
            """
            if not bool(kwargs.get("do_sample", False)):
                return logits.argmax(dim=-1)
            scaled = logits.float()
            temperature = float(kwargs.get("temperature") or 1.0)
            scaled = scaled / max(temperature, 1e-5)
            top_p = kwargs.get("top_p")
            if top_p is not None and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
                probs = sorted_logits.softmax(dim=-1)
                remove = probs.cumsum(dim=-1) - probs > float(top_p)
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
            return torch.multinomial(scaled.softmax(dim=-1), num_samples=1).squeeze(-1)

    @staticmethod
    def _pad_after_eos(model: Any, new_tokens: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
            """Pad everything after the first EOS per row (the first token is
            sampled outside ``generate``, so ``generate`` cannot freeze rows that
            finished on it)."""
            eos = model.generation_config.eos_token_id
            if eos is None or new_tokens.shape[1] == 0:
                return new_tokens
            eos_ids = [int(eos)] if not isinstance(eos, (list, tuple)) else [int(t) for t in eos]
            if pad_token_id is None:
                pad_token_id = model.generation_config.pad_token_id
            if pad_token_id is None:
                pad_token_id = eos_ids[0]
            is_eos = torch.isin(new_tokens, torch.tensor(eos_ids, device=new_tokens.device))
            seen = is_eos.long().cumsum(dim=-1)
            after_first_eos = torch.zeros_like(seen)
            after_first_eos[:, 1:] = seen[:, :-1]
            return new_tokens.masked_fill(after_first_eos > 0, int(pad_token_id))

    @staticmethod
    def _crop_cache(cache: Any, max_length: int) -> None:
            for cache_layer in cache.layers:
                if getattr(cache_layer, "is_initialized", False):
                    cache_layer.crop(max_length)

    @staticmethod
    def _prune_cache(cache: Any, remain: torch.Tensor, repad: torch.Tensor) -> None:
            """Drop pruned positions from every cached layer, re-padding left."""
            batch, reduced_len = repad.shape
            for cache_layer in cache.layers:
                if not getattr(cache_layer, "is_initialized", False) or cache_layer.get_seq_length() == 0:
                    continue
                old_keys, old_values = cache_layer.keys, cache_layer.values
                num_heads, head_dim = old_keys.shape[1], old_keys.shape[3]
                # A [B, H, L] boolean over a [B, H, L, D] tensor selects [n, D]
                # rows, mirroring the official `_reduce_tokens` cache re-pack.
                remain_expanded = remain.to(old_keys.device).unsqueeze(1).expand(batch, num_heads, -1)
                repad_expanded = repad.to(old_keys.device).unsqueeze(1).expand(batch, num_heads, -1)
                new_keys = torch.zeros(
                    (batch, num_heads, reduced_len, head_dim), dtype=old_keys.dtype, device=old_keys.device
                )
                new_values = torch.zeros_like(new_keys)
                new_keys[repad_expanded] = old_keys[remain_expanded]
                new_values[repad_expanded] = old_values[remain_expanded]
                cache_layer.keys = new_keys
                cache_layer.values = new_values

    def _record_source(self, is_visual_row: torch.Tensor, remain_row: torch.Tensor) -> None:
            """Source matrices over the first sample's visual tokens for llmoe.viz."""
            visual_idx = is_visual_row.nonzero(as_tuple=True)[0]
            kept = remain_row[visual_idx]
            num_tokens = int(visual_idx.numel())
            source = torch.eye(num_tokens, dtype=torch.float32)[kept.cpu()]
            self._source_history = [torch.eye(num_tokens, dtype=torch.float32), source]

    @property
    def last_keep_ratio(self) -> float | None:
        """Retained visual-token ratio of the most recent batch."""
        return self._last_keep_ratio

    @property
    def last_importance_maps(self) -> list[torch.Tensor]:
        """VIP probabilities from the most recent tracked batch, one grid per image.

        Populated by the LLaVA path only; the Qwen path's per-image grids vary
        within a batch, and its reference never exposed them.
        """
        return [scores.clone() for scores in self._last_importance_maps]

    @property
    def last_keep_masks(self) -> list[torch.Tensor]:
        """Post-constraint VIP keep decisions from the most recent tracked batch."""
        return [mask.clone() for mask in self._last_keep_masks]
