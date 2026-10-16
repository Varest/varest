"""Adaptive halting: delete EVERY visual token once the language model has settled its answer, per example.

At each decoder layer the hidden state of the last prompt position is decoded through the final norm and the BASE
``lm_head`` (the logit lens; the checkpoint's ``lm_head`` LoRA is bypassed, as the rule was derived on the base head).
The cascade halts at the first layer ``i >= min_layer`` whose INPUT state satisfies both

  * stability: the lens argmax has been the same token for the last ``window`` layers (layers ``i-window+1 .. i``);
  * confidence: the lens distribution has entropy ``<= entropy`` nats (or, with ``maxp`` set instead, a top-1
    probability ``>= maxp``).

When it halts at layer ``i`` the visual tokens (and the [REL] registers, if [REL] has not pruned yet) are deleted
before layer ``i`` runs, so layers ``i .. N-1`` never see the image -- in the prompt pass and in every decoding step,
whose KV cache then holds no visual entries for those layers. The decision uses only layers already computed, needs no
labels and no training, and is inference-only.

``min_layer = 8`` lets the halt land before W5RELJOINT's [REL] layer (16 of 32), in which case [REL] never runs. The
research evaluator (``predres_src/mix_eval_rele2e.py``, ``VDROP_ADAPT=1 VDROP_CASC=1 CASC_W=3 CASC_E=1.0 LENS_BASE=1``)
implements the same cascade with an earliest halt at layer 7; ``min_layer=7`` reproduces it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HaltConfig:
    """The cascade's constants. Exactly one of ``entropy`` / ``maxp`` is the confidence gate."""

    min_layer: int = 8
    window: int = 3
    entropy: float | None = 1.0
    maxp: float | None = None
    base_lens: bool = True

    def __post_init__(self) -> None:
        if (self.entropy is None) == (self.maxp is None):
            raise ValueError("HaltConfig needs exactly one confidence gate: entropy= (nats) or maxp=")
        if self.window < 1 or self.min_layer < 0:
            raise ValueError(f"HaltConfig: window >= 1 and min_layer >= 0, got {self.window}, {self.min_layer}")


def lens_head(lm_head: nn.Module, base: bool) -> nn.Module:
    """The head the lens decodes through: the BASE linear under a peft LoRA wrapper when ``base``."""
    return getattr(lm_head, "base_layer", lm_head) if base else lm_head


class CascadeHalt:
    """Per-example state of the cascade. Call :meth:`step` with the INPUT state of every layer, in order, from 0."""

    def __init__(self, cfg: HaltConfig, norm: nn.Module, lm_head: nn.Module) -> None:
        self.cfg = cfg
        self.norm = norm
        self.head = lens_head(lm_head, cfg.base_lens)
        self.tops: list[int] = []
        self.layer: int | None = None          # the layer it halted at, once it has

    @torch.no_grad()
    def step(self, i: int, h: torch.Tensor) -> bool:
        """``h [1, T, D]`` = the input of decoder layer ``i``. -> True exactly once, at the layer it halts at."""
        if self.layer is not None:
            return False
        if i != len(self.tops):
            raise RuntimeError(f"CascadeHalt.step: layer {i} after {len(self.tops)} steps -- call it on every layer")
        lg = self.head(self.norm(h[:, -1:]))[0, -1].float()
        self.tops.append(int(lg.argmax()))
        c = self.cfg
        if i < c.min_layer or len(self.tops) < c.window or len(set(self.tops[-c.window:])) != 1:
            return False
        p = torch.softmax(lg, -1)
        if c.entropy is not None:
            ok = float(-(p * torch.log(p.clamp_min(1e-9))).sum()) <= c.entropy
        else:
            ok = float(p.max()) >= c.maxp
        if ok:
            self.layer = i
        return ok
