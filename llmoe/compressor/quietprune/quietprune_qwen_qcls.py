"""Query-guided ``[Q-CLS]`` scoring for QuietPrune on Qwen2.5-VL -- the trained path.

The merged Qwen adaptation of QuietPrune keeps the paper's ``2x2`` group selection
and its redundant-token aggregate, but ranks groups by the final vision block's
*received* attention. That is training-free, and it drops the one thing QuietPrune
actually trains: the text-to-vision adapter that turns the prompt into ``[Q-CLS]``
(Sec. 3.2). Without it QuietPrune on Qwen is no longer query-guided, which is the
paper's whole claim.

The stated obstacle is real -- "Qwen's vision encoder never receives prompt
embeddings, so its stock blocks have no faithful ``[Q-CLS]`` injection point". But
QuietPrune does not need ``[Q-CLS]`` *inside* the tower to be query-guided; it
needs a query vector living in the same space as whatever it scores. On Qwen the
reduction happens after ``visual.merger``, and two facts make that a natural fit:

* QuietPrune drops **whole ``2x2`` spatial groups**, and Qwen's merger consumes
  **exactly ``2x2`` groups** -- the granularity already matches.
* ``visual.merger.mlp`` is a ``Linear -> GELU -> Linear`` stack, which is precisely
  the case :func:`~llmoe.compressor.quietprune.quietprune.build_inverse_projector`
  is built to invert.

So the adapter here is the merger's MLP run backwards: prompt embeddings (decoder
width) map into the merger's *input* space, which is the concatenated ``2x2`` patch
group. Scoring a group against that ``[Q-CLS]`` is query-guided selection at
exactly the granularity QuietPrune prunes at, and it is trained by the same
self-distillation objective as the released LLaVA adapter.

``ln_q`` is deliberately not inverted. It is an RMSNorm, and the merger applies it
*before* the view into group space, so capturing the tensor that enters
``merger.mlp`` (a pre-hook) already gives the post-norm group vectors -- the
adapter and the captured groups meet in the same space with nothing left to undo.

Upstream publishes no Qwen QuietPrune recipe or checkpoint; this is an in-repo
adaptation, and a Qwen number from it is not a paper number.
"""

from typing import Any, override

import torch
from torch import nn

from llmoe.compressor.quietprune.quietprune import build_inverse_projector
from llmoe.utils.qwen import window_reverse_indices

__all__ = [
    "QwenQCLSAdapter",
    "MergerGroupCapture",
    "qcls_group_scores",
]


class QwenQCLSAdapter(nn.Module):
    """Prompt embeddings -> one ``[Q-CLS]`` in Qwen's merger-input space.

    The body is ``visual.merger.mlp`` inverted, so the map starts near-correct
    before a single gradient step -- the same property that makes QuietPrune's
    LLaVA adapter cheap to train. This is the only trainable module.
    """

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.body: nn.Module = body

    @classmethod
    def from_visual(cls, visual: Any) -> "QwenQCLSAdapter":
        """Build the adapter by inverting ``visual.merger.mlp``."""
        merger = getattr(visual, "merger", None)
        mlp = getattr(merger, "mlp", None)
        if mlp is None:
            raise TypeError(
                "QuietPrune's Qwen adapter inverts `visual.merger.mlp`; the active "
                "vision tower does not expose one."
            )
        return cls(build_inverse_projector(mlp))

    @property
    def group_dim(self) -> int:
        """Width of the merger's input, i.e. ``context_dim * spatial_merge_size ** 2``."""
        for module in reversed(list(self.body)):
            if isinstance(module, nn.Linear):
                return int(module.out_features)
        raise TypeError("QuietPrune's Qwen adapter has no Linear leaf to size.")

    @override
    def forward(self, query_embeds: torch.Tensor,
                query_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, T, d_decoder] -> [B, d_group]``.

        :param query_embeds: prompt token embeddings from the decoder's table.
        :param query_mask: ``[B, T]`` weights; zero out padding and the image
            placeholders, which carry no query information. ``None`` counts every
            position.
        """
        if query_embeds.ndim != 3:
            raise ValueError(
                f"query_embeds must be [B, T, d], got {tuple(query_embeds.shape)}")
        adapted = self.body(query_embeds)
        if query_mask is None:
            return adapted.mean(dim=1)
        if tuple(query_mask.shape) != tuple(query_embeds.shape[:2]):
            raise ValueError("query_mask must be [B, T] and match query_embeds")
        weights = query_mask.to(device=adapted.device, dtype=adapted.dtype).unsqueeze(-1)
        # clamp_min(1), as on LLaVA: an all-masked row is a degenerate prompt and a
        # zero [Q-CLS] is a defensible neutral query, not a division blow-up.
        return (adapted * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class MergerGroupCapture:
    """Capture the ``2x2`` group vectors entering ``visual.merger.mlp``.

    The tower runs its blocks over a window-permuted sequence and the merger
    reshapes it into groups, so what the MLP receives is ``[num_groups, group_dim]``
    in *permuted* group order. :meth:`raster` un-permutes it the same way the tower
    un-permutes the merger's output.
    """

    def __init__(self) -> None:
        self.groups: torch.Tensor | None = None
        self._handle: Any = None

    def install(self, visual: Any) -> None:
        merger = visual.merger

        def hook(_module: Any, args: tuple) -> None:
            if args and isinstance(args[0], torch.Tensor) and args[0].ndim == 2:
                self.groups = args[0].detach()

        self._handle = merger.mlp.register_forward_pre_hook(hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.groups = None

    def raster(self, visual: Any, image_grid_thw: torch.Tensor, num_patches: int,
               *, label: str = "QuietPrune/Qwen") -> torch.Tensor:
        """``[num_tokens, group_dim]`` in raster order, one row per merged token."""
        if self.groups is None:
            raise RuntimeError(
                f"{label} captured no merger input; the vision tower did not run.")
        _, reverse_indices = window_reverse_indices(
            visual, image_grid_thw, num_patches, label=label,
            device=self.groups.device)
        if self.groups.shape[0] != reverse_indices.numel():
            raise RuntimeError(
                f"{label} captured {self.groups.shape[0]} merger groups but the grid "
                f"implies {reverse_indices.numel()}."
            )
        return self.groups[reverse_indices]


def qcls_group_scores(q_cls: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between ``[Q-CLS]`` and each merged token's source group.

    Cosine rather than a raw dot product: the merger's input norms vary a lot
    across an image, and QuietPrune ranks *relevance to the query*, not magnitude.
    The result is the drop-in replacement for the received-attention score.

    :param q_cls: ``[d_group]`` or ``[1, d_group]``.
    :param groups: ``[num_tokens, d_group]`` in raster order.
    :return: ``[num_tokens]`` float32 scores.
    """
    query = q_cls.reshape(-1).to(device=groups.device, dtype=torch.float32)
    if query.numel() != groups.shape[-1]:
        raise ValueError(
            f"[Q-CLS] width {query.numel()} does not match the merger group width "
            f"{groups.shape[-1]}"
        )
    keys = groups.to(torch.float32)
    query = query / query.norm().clamp_min(1e-6)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return keys @ query
