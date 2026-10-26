"""Inference runtime: attach a Varest checkpoint to a Hugging Face LLaVA model and build its visual tokens.

The pieces a checkpoint adds to the frozen backbone:

* a LoRA adapter over q/k/v/o (vision tower and language model) and ``lm_head`` (:func:`attach_lora`);
* a fine-tuned multimodal projector (:func:`swap_projector`);
* the TEXT and OBJECT heads, the position tag and, for W5RELJOINT, the [REL] head (:class:`VarestRuntime`).

Everything here is backbone-agnostic torch; ``llmoe.compressor.varest`` wires it into the benchmark harness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from varest.checkpoint import VarestCheckpoint
from varest.fields import ImageLike, LlavaFields, ViewFields, llava_fields, next_fields
from varest.heads.object import object_head_from_state
from varest.heads.text import text_head_from_state
from varest.postag import postag_from_state
from varest.rel import budget_split, rel_head_from_state
from varest.tokenizer import GRID, build_partition, pool_regions
from varest import next as vnext


# ---------------------------------------------------------------------------------------------------- adapter
def _canonical(name: str) -> str:
    """Module path without the optional ``vision_model.`` level (transformers 5.5 nests CLIP's encoder under it,
    later versions do not), so one checkpoint loads on either."""
    return name.replace(".vision_tower.vision_model.", ".vision_tower.")


@dataclass
class LoraAttachment:
    model: nn.Module
    modules: list[str]


class _SharedCast:
    """The float32 copy of one input, shared by the LoRA modules that read the same tensor (q/k/v of one attention
    block). Converting a tensor once or three times gives the same values; the copy is dropped after its last reader."""

    def __init__(self, readers: int) -> None:
        self.readers = readers
        self.src: torch.Tensor | None = None
        self.x32: torch.Tensor | None = None
        self.left = 0

    def __call__(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.src is not x:
            self.src, self.x32, self.left = x, x.to(dtype), self.readers
        x32 = self.x32
        self.left -= 1
        if self.left <= 0:
            self.src = self.x32 = None
        return x32


def _power_of_two(v: float) -> bool:
    m, _ = math.frexp(v)
    return v > 0 and m == 0.5


class LoraLinear(nn.Module):
    """``base(x) + B(A(x)) * scaling`` with the adapter in float32, bit-identical to peft's ``lora.Linear.forward``
    (dropout 0, one adapter): base in the model dtype, ``x`` cast to float32, ``A`` then ``B`` as float32 GEMMs, the
    scale, the float32 sum, the cast back to the model dtype. What differs is only how many kernels that takes:

    * no per-call peft Python (adapter loops, merge/disable checks, kwarg filtering);
    * a power-of-two ``scaling`` (alpha / r = 2 for every released checkpoint) is folded into ``B``: scaling every
      product of a float32 GEMM by 2^k scales each rounded partial sum by exactly 2^k, so ``F.linear(t, 2^k B)`` equals
      ``F.linear(t, B) * 2^k`` bit for bit (outside float32's subnormal range, far below any LoRA update);
    * the float32 add and the cast back run as one kernel: ``result.add_(delta)`` on the model-dtype ``result``
      computes the sum in float32 (the promoted type) and rounds it to the model dtype on store -- the same two
      roundings, in the same order, as ``(result + delta).to(result.dtype)``;
    * q/k/v of one attention block share the float32 copy of their common input (``shared``).

    At batch-1 decoding a generated token goes through 129 of these modules."""

    def __init__(self, base_layer: nn.Linear, a: torch.Tensor, b: torch.Tensor, scaling: float,
                 shared: _SharedCast | None = None) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.scaling = float(scaling)
        self.folded = _power_of_two(self.scaling)
        self.lora_A = nn.Linear(a.shape[1], a.shape[0], bias=False, device=a.device, dtype=a.dtype)
        self.lora_B = nn.Linear(b.shape[1], b.shape[0], bias=False, device=b.device, dtype=b.dtype)
        self.lora_A.weight.data = a
        self.lora_B.weight.data = b * self.scaling if self.folded else b
        self.shared = shared
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        a = self.lora_A.weight
        x32 = self.shared(x, a.dtype) if self.shared is not None else x.to(a.dtype)
        delta = F.linear(F.linear(x32, a), self.lora_B.weight)
        if not self.folded:
            delta = delta * self.scaling
        if result.requires_grad:
            return (result + delta).to(result.dtype)
        return result.add_(delta)


def attach_lora(model: nn.Module, ckpt: VarestCheckpoint) -> LoraAttachment:
    """Wrap the checkpoint's LoRA targets in ``model`` (a ``Llava*ForConditionalGeneration``) with :class:`LoraLinear`
    IN PLACE.

    Every checkpoint tensor must land on exactly one module -- a partial match would silently score a different model.
    PRECISION, AS TRAINED AND EVALUATED: the adapter is float32 holding the checkpoint's values rounded to bfloat16 (the
    released numbers loaded the saved tensors through a bfloat16 cast and peft computed the update in float32).
    Computing the LoRA updates in bfloat16 instead moved CLIP features by up to ~4 and changed ~2.3% of TextVQA
    answers."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in ckpt.lora.items():
        mod, rest = k.split(".lora_", 1)                 # rest = "A.weight" or "A.default.weight"
        pairs.setdefault(_canonical(mod), {})[rest[0]] = v
    by_canon = {_canonical(name): name for name, _ in model.named_modules() if _canonical(name) in pairs}
    missing = sorted(set(pairs) - set(by_canon))
    if missing:
        raise ValueError(f"LoRA targets absent from the model: {missing[:3]} ({len(missing)} of {len(pairs)})")
    scaling = ckpt.lora_alpha / ckpt.lora_rank
    targets = sorted(by_canon.values())
    shared: dict[str, _SharedCast] = {}                  # attention block -> its q/k/v input's float32 copy
    for name in targets:
        parent, _, leaf = name.rpartition(".")
        if leaf in ("q_proj", "k_proj", "v_proj"):
            shared.setdefault(parent, _SharedCast(0)).readers += 1
    for canon, name in by_canon.items():
        ab = pairs[canon]
        if set(ab) != {"A", "B"}:
            raise ValueError(f"LoRA target {name}: expected an A and a B tensor, got {sorted(ab)}")
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, leaf)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target {name} is a {type(base).__name__}, not nn.Linear")
        a, b = (ab[x].to(device=base.weight.device, dtype=torch.bfloat16).to(torch.float32) for x in "AB")
        if a.shape != (ckpt.lora_rank, base.in_features) or b.shape != (base.out_features, ckpt.lora_rank):
            raise ValueError(f"LoRA tensor {name}: A {tuple(a.shape)} / B {tuple(b.shape)} do not fit "
                             f"{base.in_features} -> {base.out_features} at rank {ckpt.lora_rank}")
        setattr(parent, leaf, LoraLinear(base, a, b, scaling, shared.get(parent_name) if leaf in ("q_proj", "k_proj", "v_proj") else None))
    return LoraAttachment(model, targets)


def detach_lora(att: LoraAttachment) -> None:
    """Replace every :class:`LoraLinear` by its original base module."""
    for name in att.modules:
        parent_name, _, leaf = name.rpartition(".")
        parent = att.model.get_submodule(parent_name) if parent_name else att.model
        layer = getattr(parent, leaf)
        if isinstance(layer, LoraLinear):
            setattr(parent, leaf, layer.base_layer)


@dataclass
class ProjectorSwap:
    module: nn.Module
    original: dict[str, torch.Tensor]


def swap_projector(model: nn.Module, ckpt: VarestCheckpoint) -> ProjectorSwap:
    """Load the fine-tuned multimodal projector, keeping the original for :func:`restore_projector`."""
    proj = model.get_submodule("model.multi_modal_projector")
    prefix = "model.multi_modal_projector."
    sd = {k[len(prefix):]: v for k, v in ckpt.projector.items() if k.startswith(prefix)}
    if len(sd) != len(ckpt.projector):
        raise ValueError(f"projector tensors outside {prefix!r}: {sorted(set(ckpt.projector))[:3]}")
    original = {k: v.detach().clone() for k, v in proj.state_dict().items()}
    if set(sd) != set(original):
        raise ValueError(f"projector keys {sorted(sd)} != model {sorted(original)}")
    p0 = next(proj.parameters())
    proj.load_state_dict({k: v.to(device=p0.device, dtype=p0.dtype) for k, v in sd.items()})
    return ProjectorSwap(proj, original)


def restore_projector(swap: ProjectorSwap) -> None:
    swap.module.load_state_dict(swap.original)


# ---------------------------------------------------------------------------------------------------- tokens
def to_pil(image: Any) -> Image.Image:
    """A PIL image, a file path, or a ``[C, H, W]`` uint8/float tensor -> RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        with Image.open(image) as im:
            return im.convert("RGB")
    t = torch.as_tensor(image).detach().cpu()
    if t.ndim == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
        t = t.permute(1, 2, 0)
    if t.is_floating_point():
        t = (t * (255.0 if float(t.max()) <= 1.0 else 1.0)).round().clamp(0, 255)
    a = t.to(torch.uint8).numpy()
    if a.shape[-1] == 1:
        a = a[..., 0]
    return Image.fromarray(a).convert("RGB")


@dataclass
class VisualTokens:
    embeds: torch.Tensor          # [n, D] in the model dtype, position tag applied
    n: int                        # visual tokens emitted
    keep: int | None              # tokens [REL] keeps at its layer (None without [REL])
    info: dict


class VarestRuntime(nn.Module):
    """The checkpoint's small modules (heads, position tag, [REL]) and the per-image token builders."""

    def __init__(self, ckpt: VarestCheckpoint) -> None:
        super().__init__()
        self.ckpt = ckpt
        self.text_head = text_head_from_state(ckpt.text_head)
        self.object_head = object_head_from_state(ckpt.object_head)
        self.postag = postag_from_state(ckpt.postag)
        self.rel_head = rel_head_from_state(ckpt.relhead) if ckpt.relhead is not None else None
        self.rel_layer = ckpt.rel_layer

    def place(self, device: torch.device | str, dtype: torch.dtype) -> VarestRuntime:
        """Heads and [REL] stay float32 (as trained); the position tag runs in the language model's dtype."""
        self.text_head.to(device=device, dtype=torch.float32).eval()
        self.object_head.to(device=device, dtype=torch.float32).eval()
        self.postag.to(device=device, dtype=dtype).eval()
        if self.rel_head is not None:
            self.rel_head.to(device=device, dtype=torch.float32).eval()
        return self

    def split(self, budget: float) -> tuple[int, int | None]:
        """Layer-averaged budget -> ``(K before the LLM, K' after [REL])``; ``K' = None`` without [REL]."""
        if self.rel_head is None:
            return int(budget), None
        return budget_split(budget)

    @torch.no_grad()
    def llava_tokens(self, projected: torch.Tensor, image: ImageLike, budget: int, *,
                     fields: LlavaFields | None = None) -> VisualTokens:
        """LLaVA-1.5: projected CLIP features ``[576, D]`` of the padded image -> the W5 visual tokens.
        ``fields``: :func:`llava_fields` of ``image``, when the caller computed them (the engine overlaps them with
        CLIP); ``None`` computes them here."""
        k, kp = self.split(budget)
        f = fields if fields is not None else llava_fields(image, self.text_head, self.object_head)
        lab, info = build_partition(f.boundary.astype(np.float32), f.text.astype(np.float32), k, inimg=f.inimg)
        tokens, pos, _ = pool_regions(projected, lab, GRID)
        dt = projected.dtype
        p = torch.as_tensor(pos, device=projected.device).to(dt)
        embeds = self.postag.apply_to(tokens.to(dt), p)          # tokens cast to the model dtype BEFORE the tag
        n = embeds.shape[0]
        return VisualTokens(embeds, n, None if kp is None else min(kp, n), info)

    @torch.no_grad()
    def next_tokens(self, base: torch.Tensor, window: torch.Tensor, lay: vnext.Layout, image: ImageLike,
                    ratio: float, *, fields: tuple[ViewFields, ViewFields] | None = None) -> VisualTokens:
        """LLaVA-NeXT: projected base-view ``[576, D]`` and window ``[nr*nc, D]`` features -> the W5 visual tokens at
        retain ratio ``ratio`` (the layer-averaged per-image budget is ``floor(N_native * ratio)``)."""
        b = vnext.kept(lay.nr, lay.nc, ratio)
        k = int(round(2 * b / 1.25)) if self.rel_head is not None else b
        lo, hi = fields if fields is not None else next_fields(image, self.text_head, self.object_head, (lay.nr, lay.nc))
        p = vnext.plan((lo.boundary, lo.text, None), (hi.boundary, hi.text, None), lay, k)
        tokens, pos = vnext.tokenize(base, window, p, lay)
        dt = base.dtype
        tag = self.postag(torch.as_tensor(pos, device=base.device).to(dt))
        embeds = (tokens + tag).to(dt)                            # float32 tokens + tag, then cast (as trained)
        n = embeds.shape[0]
        keep = None if self.rel_head is None else max(0, min(2 * b - n, n))
        return VisualTokens(embeds, n, keep, {"budget": b, "k": p.k, "k_base": p.k_base, "k_hi": p.k_hi,
                                                 "tau_k": p.tau_k, "n_native": vnext.n_native(lay.nr, lay.nc)})
