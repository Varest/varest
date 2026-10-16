"""Inference runtime: attach a Varest checkpoint to a Hugging Face LLaVA model and build its visual tokens.

The pieces a checkpoint adds to the frozen backbone:

* a LoRA adapter over q/k/v/o (vision tower and language model) and ``lm_head`` (:func:`attach_lora`);
* a fine-tuned multimodal projector (:func:`swap_projector`);
* the TEXT and OBJECT heads, the position tag and, for W5RELJOINT, the [REL] head (:class:`VarestRuntime`).

Everything here is backbone-agnostic torch; ``llmoe.compressor.varest`` wires it into the benchmark harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from varest.checkpoint import VarestCheckpoint
from varest.fields import ImageLike, llava_fields, next_fields
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
    had_peft_config: bool = False


def attach_lora(model: nn.Module, ckpt: VarestCheckpoint) -> LoraAttachment:
    """Inject the checkpoint's LoRA into ``model`` (a ``Llava*ForConditionalGeneration``) IN PLACE and load it.

    Every checkpoint tensor must land on exactly one module -- a partial match would silently score a different model.
    Dropout is 0 (inference)."""
    from peft import LoraConfig, inject_adapter_in_model
    ours = {_canonical(k.split(".lora_")[0]) for k in ckpt.lora}
    by_canon = {}
    for name, _ in model.named_modules():
        if _canonical(name) in ours:
            by_canon[_canonical(name)] = name
    missing = sorted(ours - set(by_canon))
    if missing:
        raise ValueError(f"LoRA targets absent from the model: {missing[:3]} ({len(missing)} of {len(ours)})")
    targets = sorted(by_canon.values())
    had = hasattr(model, "peft_config")
    cfg = LoraConfig(r=ckpt.lora_rank, lora_alpha=ckpt.lora_alpha, lora_dropout=0.0, target_modules=targets)
    inject_adapter_in_model(cfg, model)
    own = dict(model.named_parameters())
    sd = {}
    for k, v in ckpt.lora.items():
        mod, rest = k.split(".lora_", 1)
        name = f"{by_canon[_canonical(mod)]}.lora_{rest}"
        if name not in own:
            raise ValueError(f"LoRA tensor {k!r} has no parameter {name!r} after injection")
        sd[name] = v
    # PRECISION, AS TRAINED AND EVALUATED: the adapter's parameters are float32 (peft's ``get_peft_model`` autocasts
    # them; ``inject_adapter_in_model`` does not, and left them in the backbone dtype), holding the checkpoint's values
    # rounded to bfloat16 (the released numbers loaded the saved tensors through a bfloat16 cast). Computing the LoRA
    # updates in bfloat16 instead moved CLIP features by up to ~4 and changed ~2.3% of TextVQA answers.
    for name, v in sd.items():
        p = own[name]
        if tuple(p.shape) != tuple(v.shape):
            raise ValueError(f"LoRA tensor {name}: checkpoint {tuple(v.shape)} != model {tuple(p.shape)}")
        with torch.no_grad():
            p.data = v.to(device=p.device, dtype=torch.bfloat16).to(torch.float32)
        p.requires_grad_(False)
    return LoraAttachment(model, targets, had)


def detach_lora(att: LoraAttachment) -> None:
    """Replace every injected LoRA layer by its original base module."""
    for name in att.modules:
        parent_name, _, leaf = name.rpartition(".")
        parent = att.model.get_submodule(parent_name) if parent_name else att.model
        layer = getattr(parent, leaf)
        base = getattr(layer, "base_layer", None)
        if base is not None:
            setattr(parent, leaf, base)
    if not att.had_peft_config and hasattr(att.model, "peft_config"):
        del att.model.peft_config


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
    def llava_tokens(self, projected: torch.Tensor, image: ImageLike, budget: int) -> VisualTokens:
        """LLaVA-1.5: projected CLIP features ``[576, D]`` of the padded image -> the W5 visual tokens."""
        k, kp = self.split(budget)
        f = llava_fields(image, self.text_head, self.object_head)
        lab, info = build_partition(f.boundary.astype(np.float32), f.text.astype(np.float32), k, inimg=f.inimg)
        tokens, pos, _ = pool_regions(projected, lab, GRID)
        dt = projected.dtype
        p = torch.as_tensor(pos, device=projected.device).to(dt)
        embeds = self.postag.apply_to(tokens.to(dt), p)          # tokens cast to the model dtype BEFORE the tag
        n = embeds.shape[0]
        return VisualTokens(embeds, n, None if kp is None else min(kp, n), info)

    @torch.no_grad()
    def next_tokens(self, base: torch.Tensor, window: torch.Tensor, lay: vnext.Layout, image: ImageLike,
                    ratio: float) -> VisualTokens:
        """LLaVA-NeXT: projected base-view ``[576, D]`` and window ``[nr*nc, D]`` features -> the W5 visual tokens at
        retain ratio ``ratio`` (the layer-averaged per-image budget is ``floor(N_native * ratio)``)."""
        b = vnext.kept(lay.nr, lay.nc, ratio)
        k = int(round(2 * b / 1.25)) if self.rel_head is not None else b
        lo, hi = next_fields(image, self.text_head, self.object_head, (lay.nr, lay.nc))
        p = vnext.plan((lo.boundary, lo.text, None), (hi.boundary, hi.text, None), lay, k)
        tokens, pos = vnext.tokenize(base, window, p, lay)
        dt = base.dtype
        tag = self.postag(torch.as_tensor(pos, device=base.device).to(dt))
        embeds = (tokens + tag).to(dt)                            # float32 tokens + tag, then cast (as trained)
        n = embeds.shape[0]
        keep = None if self.rel_head is None else max(0, min(2 * b - n, n))
        return VisualTokens(embeds, n, keep, {"budget": b, "k": p.k, "k_base": p.k_base, "k_hi": p.k_hi,
                                                 "tau_k": p.tau_k, "n_native": vnext.n_native(lay.nr, lay.nc)})
