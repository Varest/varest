"""Position tag: a small MLP that maps a token's ``(cx, cy, size)`` to a bias added to its embedding.

A pooled region has no fixed place in LLaVA's patch raster, so each visual token carries its region's centroid and
extent explicitly. The tag is added AFTER the multimodal projector, in the language model's embedding space.
"""
from __future__ import annotations

import torch
from torch import nn

POS_DIM = 3
HIDDEN = 256


class PosTag(nn.Module):
    """``(cx, cy, size)`` -> ``d``-dim bias. ``apply_to(tokens, positions)`` returns ``tokens + tag(positions)``."""

    def __init__(self, d: int, in_dim: int = POS_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.d, self.in_dim = int(d), int(in_dim)
        self.net = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.GELU(), nn.Linear(hidden, self.d))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)

    def apply_to(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(positions)


def postag_from_state(sd: dict[str, torch.Tensor]) -> PosTag:
    """Build a :class:`PosTag` whose geometry is read off the weights."""
    tag = PosTag(d=sd["net.2.weight"].shape[0], in_dim=sd["net.0.weight"].shape[1], hidden=sd["net.0.weight"].shape[0])
    tag.load_state_dict(sd)
    return tag
