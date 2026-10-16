"""QuietPrune inside Qwen2.5-VL's vision tower -- the paper's actual mechanism.

QuietPrune's claim is *early* pruning: tokens are dropped **inside the ViT**, so
every block after a cut skips them entirely. That is where its latency comes from,
and the paper says so directly -- the advantage "becomes more pronounced as model
size decreases, because ViT accounts for a larger fraction of the total
computation in smaller VLMs, making them more amenable to early pruning".

A post-merger reduction, however faithful its selection rule, saves none of that:
the tower has already run to completion. This module does the pruning where the
method puts it.

Why Qwen is the natural host, not the awkward one
-------------------------------------------------
QuietPrune drops whole ``2x2`` groups, and Sec. 4.2 says why: "adjacent 2x2 tokens
are merged into a single token for the LLM", and unstructured removal "breaks the
local structural prior assumed by the merging operation". The group *is* the
merger's unit. Qwen2.5-VL permutes patches in exactly those groups
(``spatial_merge_unit`` = 4) before its blocks run, so a surviving group is still a
complete merger input and ``visual.merger`` stays valid with no changes at all.

The bookkeeping
---------------
Qwen's tower is a flattened variable-length sequence, so dropping rows means
repairing four things at once:

* ``hidden_states`` and ``rotary_pos_emb`` -- gather the surviving rows.
* ``cu_seqlens`` -- the per-image boundaries used by the full-attention blocks.
* ``cu_window_seqlens`` -- the window boundaries used by the windowed blocks.
  Windows are contiguous runs of groups in permuted order (that is how
  ``get_window_index`` builds them), so each window's new length is just how many
  of its groups survived.
* the reverse permutation -- ``argsort(window_index)`` no longer applies once
  groups are gone, so the survivors' *original* group ids are what re-sorts the
  merger's output back into raster order.

Deviations, stated plainly
--------------------------
* ``[Q-CLS]`` is carried as a **virtual query**: it reads the sequence at each
  block and is updated by what it reads, but no patch attends *to* it. The paper
  prepends it to the sequence proper, so in the paper other tokens can see it.
  Inserting a real row would break the ``spatial_merge_unit`` divisibility, the
  window partition and the varlen ``cu_seqlens`` all at once, for a signal that is
  only ever used to rank groups.
* One image per prompt, matching the restriction the merged Qwen adapters already
  impose.
* The paper evaluates Qwen3-VL; this is Qwen2.5-VL.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import torch

__all__ = [
    "GroupLayout",
    "group_layout",
    "PrunedSequence",
    "prune_groups",
    "quietprune_layers",
]


def quietprune_layers(depth: int) -> tuple[int, ...]:
    """The paper's pruning depths: ``1/4``, ``1/2`` and ``3/4`` of the blocks.

    Returned as block indices to prune *after*, de-duplicated and clamped inside
    the stack so a shallow tower still yields a strictly increasing schedule.
    """
    if depth < 2:
        raise ValueError(f"a vision tower needs at least 2 blocks, got {depth}")
    picked: list[int] = []
    for fraction in (0.25, 0.5, 0.75):
        index = max(0, min(depth - 2, int(round(fraction * depth)) - 1))
        if not picked or index > picked[-1]:
            picked.append(index)
    return tuple(picked)


@dataclass(frozen=True)
class GroupLayout:
    """How a permuted Qwen ViT sequence divides into merger groups and windows.

    :param merge_unit: patches per merger group (``spatial_merge_unit``, 4).
    :param original: ``[G]`` original group id of each permuted group, i.e.
        ``window_index``. Re-sorting the merger's output by this restores raster
        order.
    :param window_of: ``[G]`` which window each permuted group belongs to.
    :param num_windows: number of windows.
    """

    merge_unit: int
    original: torch.Tensor
    window_of: torch.Tensor
    num_windows: int

    @property
    def num_groups(self) -> int:
        return int(self.original.numel())


def group_layout(visual: Any, image_grid_thw: torch.Tensor, num_patches: int,
                 cu_window_seqlens: Sequence[int] | torch.Tensor) -> GroupLayout:
    """Derive the group/window layout the tower built for this image.

    ``cu_window_seqlens`` counts *patches*, and windows are contiguous in permuted
    group order, so dividing its deltas by ``merge_unit`` gives each window's group
    count -- which is all that is needed to label every group with its window.
    """
    merge_unit = int(visual.spatial_merge_unit)
    if num_patches % merge_unit:
        raise RuntimeError(
            f"QuietPrune/Qwen expected the patch count ({num_patches}) to divide by the "
            f"merger unit ({merge_unit})."
        )
    window_index, _ = visual.get_window_index(image_grid_thw)
    device = window_index.device
    bounds = torch.as_tensor(list(cu_window_seqlens), device=device, dtype=torch.long)
    bounds = torch.unique_consecutive(bounds)
    lengths = (bounds[1:] - bounds[:-1]) // merge_unit
    lengths = lengths[lengths > 0]
    window_of = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device), lengths)
    if window_of.numel() != window_index.numel():
        raise RuntimeError(
            f"QuietPrune/Qwen window lengths cover {window_of.numel()} groups but the "
            f"permutation has {window_index.numel()}."
        )
    return GroupLayout(merge_unit=merge_unit, original=window_index,
                       window_of=window_of, num_windows=int(lengths.numel()))


@dataclass(frozen=True)
class PrunedSequence:
    """A ViT sequence after whole merger groups have been removed."""

    hidden_states: torch.Tensor
    rotary_pos_emb: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_window_seqlens: torch.Tensor
    layout: GroupLayout
    kept_groups: torch.Tensor
    dropped_groups: torch.Tensor


def prune_groups(hidden_states: torch.Tensor, rotary_pos_emb: torch.Tensor,
                 layout: GroupLayout, keep_groups: torch.Tensor) -> PrunedSequence:
    """Drop whole merger groups and repair every index the tower depends on.

    :param hidden_states: ``[seq, dim]`` in permuted patch order.
    :param rotary_pos_emb: ``[seq, rot]`` in the same order.
    :param layout: the layout :func:`group_layout` derived.
    :param keep_groups: ``[K]`` permuted group ids to keep, ascending.
    :return: the pruned sequence and the repaired boundaries.
    """
    merge_unit = layout.merge_unit
    device = hidden_states.device
    keep_groups = keep_groups.to(device=device, dtype=torch.long)
    if keep_groups.numel() == 0:
        raise ValueError("QuietPrune/Qwen must keep at least one merger group")

    # Groups are contiguous runs of `merge_unit` rows, so a kept group expands to
    # its own rows and nothing else moves.
    rows = (keep_groups[:, None] * merge_unit
            + torch.arange(merge_unit, device=device)[None, :]).reshape(-1)
    new_hidden = hidden_states[rows]
    new_rotary = rotary_pos_emb[rows]

    # One image per prompt, so the full-attention boundary is the whole survivor set.
    cu_seqlens = torch.tensor([0, int(rows.numel())], device=device, dtype=torch.int32)

    # Each window keeps however many of its groups survived; empty windows vanish.
    window_of = layout.window_of.to(device)
    kept_windows = window_of[keep_groups]
    counts = torch.bincount(kept_windows, minlength=layout.num_windows)
    counts = counts[counts > 0] * merge_unit
    cu_window_seqlens = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        counts.to(torch.int32).cumsum(0),
    ])

    mask = torch.ones(layout.num_groups, dtype=torch.bool, device=device)
    mask[keep_groups] = False
    dropped = mask.nonzero(as_tuple=True)[0]

    new_layout = GroupLayout(
        merge_unit=merge_unit,
        original=layout.original.to(device)[keep_groups],
        window_of=torch.repeat_interleave(
            torch.arange(counts.numel(), device=device), counts // merge_unit),
        num_windows=int(counts.numel()),
    )
    return PrunedSequence(
        hidden_states=new_hidden,
        rotary_pos_emb=new_rotary,
        cu_seqlens=cu_seqlens,
        cu_window_seqlens=cu_window_seqlens,
        layout=new_layout,
        kept_groups=keep_groups,
        dropped_groups=dropped,
    )
