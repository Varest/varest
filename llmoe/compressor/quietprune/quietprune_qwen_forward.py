"""The QuietPrune schedule as a replacement for Qwen2.5-VL's vision forward.

This reproduces ``Qwen2_5_VisionTransformerPretrainedModel.forward`` step for step
and inserts QuietPrune between the blocks: at ``1/4``, ``1/2`` and ``3/4`` of the
stack, a fraction ``R`` of the merger groups is dropped by ``[Q-CLS]`` relevance,
and the dropped tokens are folded into one aggregate (Sec. 4.2-4.3).

``[Q-CLS]`` as a virtual query
------------------------------
The paper prepends ``[Q-CLS]`` to the sequence so it is updated by attention at
every block -- "the first row of ``O`` corresponds to the updated ``[Q-CLS]``
token, which fuses textual semantics with global visual context". Here it is
carried *alongside* the sequence instead: it reads the patches through each
block's own attention and is updated by what it reads, but no patch attends to it.

That is a real deviation, and it is deliberate. Qwen's tower is a flattened
variable-length sequence whose length must stay divisible by
``spatial_merge_unit``, partitioned into windows by ``cu_window_seqlens`` and into
images by ``cu_seqlens``. Inserting one real row breaks all three at once, for a
signal that is only ever read back out to rank groups. Everything ``[Q-CLS]``
contributes -- the query-guided score, updated with visual context as depth grows
-- is preserved.

``[Q-CLS]`` carries no rotary position, matching the paper's prepend (it is not a
patch and has no grid coordinate).
"""

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from llmoe.compressor.quietprune.quietprune_qwen_vit import (
    GroupLayout,
    group_layout,
    prune_groups,
)

__all__ = ["QuietPruneVisionRunner", "groups_kept_after_schedule"]


def groups_kept_after_schedule(num_groups: int, rate: float, cuts: int) -> list[int]:
    """Surviving group count after each cut, so the token count is known up front.

    QuietPrune drops ``floor(G * R)`` groups at every pruning layer, which makes the
    final visual-token count a pure function of the grid and ``R`` -- known before
    the image is seen, which is what lets the placeholders be trimmed ahead of the
    forward.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"pruning rate must be in [0, 1), got {rate}")
    counts: list[int] = []
    alive = int(num_groups)
    for _ in range(int(cuts)):
        alive = max(1, alive - int(alive * float(rate)))
        counts.append(alive)
    return counts


class QuietPruneVisionRunner:
    """Run Qwen's vision tower with QuietPrune's in-ViT schedule.

    :param visual: the ``Qwen2_5_VisionTransformerPretrainedModel``.
    :param pruning_layers: block indices to prune *after*.
    :param rate: ``R``, the fraction of surviving groups dropped at each layer.
    :param aggregate: fold the dropped tokens into one aggregate token (Sec. 4.3).
    :param normalize_aggregation: divide by the relevance mass. ``False`` is the
        paper's ``x_m = sum_i a_i x_i``.
    """

    def __init__(self, visual: Any, *, pruning_layers: Sequence[int], rate: float,
                 aggregate: bool = True, normalize_aggregation: bool = False,
                 eps: float = 1e-6) -> None:
        self.visual = visual
        self.pruning_layers = {int(index) for index in pruning_layers}
        self.rate = float(rate)
        self.aggregate = bool(aggregate)
        self.normalize_aggregation = bool(normalize_aggregation)
        self.eps = float(eps)
        #: Original group ids of the survivors, in output order -- what the LLM-side
        #: M-RoPE gather needs to give each kept token its true grid coordinate.
        self.survivor_groups: torch.Tensor | None = None

    # ------------------------------------------------------------------ [Q-CLS]

    def _block_attention(self, block: Any, q_cls: torch.Tensor,
                         hidden_states: torch.Tensor,
                         position_embeddings: tuple[torch.Tensor, torch.Tensor],
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """One block applied to ``[Q-CLS]`` as a query over the patches.

        :return: ``(updated_q_cls, relevance)`` where ``relevance`` is the
            head-averaged ``[Q-CLS] -> patch`` attention row (Sec. 4.2).
        """
        attn = block.attn
        heads, head_dim = attn.num_heads, attn.head_dim
        normed = block.norm1(hidden_states)
        query_normed = block.norm1(q_cls)

        keys = attn.qkv(normed).reshape(-1, 3, heads, head_dim)[:, 1]
        values = attn.qkv(normed).reshape(-1, 3, heads, head_dim)[:, 2]
        query = attn.qkv(query_normed).reshape(-1, 3, heads, head_dim)[:, 0]

        # Patch keys carry vision RoPE; [Q-CLS] has no grid position, so it is left
        # unrotated -- the same status the paper's prepended token has.
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_rotary_pos_emb_vision,
        )

        cos, sin = position_embeddings
        _, keys = apply_rotary_pos_emb_vision(keys.unsqueeze(0), keys.unsqueeze(0), cos, sin)
        keys = keys.squeeze(0)

        logits = torch.einsum("qhd,khd->hqk", query.float(), keys.float()) * attn.scaling
        weights = logits.softmax(dim=-1)
        relevance = weights.mean(dim=0).squeeze(0)          # [N], head-averaged
        context = torch.einsum("hqk,khd->qhd", weights.to(values.dtype), values)
        updated = q_cls + attn.proj(context.reshape(1, -1))
        return updated + block.mlp(block.norm2(updated)), relevance

    # ------------------------------------------------------------------ pruning

    def _select(self, relevance: torch.Tensor, layout: GroupLayout
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Group relevance by averaging (Sec. 4.2), then keep the top groups."""
        grouped = relevance.reshape(layout.num_groups, layout.merge_unit).mean(dim=-1)
        drop = int(layout.num_groups * self.rate)
        keep_count = max(1, layout.num_groups - drop)
        if keep_count >= layout.num_groups:
            return torch.arange(layout.num_groups, device=relevance.device), grouped
        keep = grouped.topk(keep_count).indices.sort().values
        return keep, grouped

    def _aggregate_dropped(self, hidden_states: torch.Tensor, relevance: torch.Tensor,
                           layout: GroupLayout, dropped: torch.Tensor) -> torch.Tensor | None:
        """``x_m = sum_i a_i x_i`` over the dropped patches (Sec. 4.3)."""
        if not self.aggregate or dropped.numel() == 0:
            return None
        rows = (dropped[:, None] * layout.merge_unit
                + torch.arange(layout.merge_unit, device=dropped.device)[None, :]).reshape(-1)
        weights = relevance[rows].float().clamp_min(0.0)
        if self.normalize_aggregation:
            weights = weights / weights.sum().clamp_min(self.eps)
        return (hidden_states[rows] * weights.to(hidden_states.dtype)[:, None]).sum(dim=0)

    # ------------------------------------------------------------------ forward

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor,
                q_cls: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Qwen's vision forward with QuietPrune between the blocks.

        :param hidden_states: raw ``pixel_values`` as the tower expects them.
        :param grid_thw: ``[1, 3]`` -- one image per prompt.
        :param q_cls: ``[1, context_dim]`` query from the text-to-vision adapter.
        :return: merged visual tokens in raster order, plus the aggregate when on.
        """
        visual = self.visual
        if grid_thw.shape[0] != 1:
            raise NotImplementedError(
                "QuietPrune/Qwen supports one image per prompt.")

        hidden_states = visual.patch_embed(hidden_states)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
        seq_len = hidden_states.shape[0]
        unit = visual.spatial_merge_unit

        # The tower's own permutation into window order, groups intact.
        hidden_states = hidden_states.reshape(seq_len // unit, unit, -1)[window_index]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // unit, unit, -1)[window_index]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        layout = group_layout(visual, grid_thw, seq_len, cu_window_seqlens)
        cu_seqlens = torch.tensor([0, seq_len], device=hidden_states.device,
                                  dtype=torch.int32)
        cu_window = torch.unique_consecutive(
            torch.as_tensor(list(cu_window_seqlens), device=hidden_states.device,
                            dtype=torch.int32))
        aggregates: torch.Tensor | None = None

        for index, block in enumerate(visual.blocks):
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
            windowed = index not in visual.fullatt_block_indexes
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_window if windowed else cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            q_cls, relevance = self._block_attention(
                block, q_cls, hidden_states, position_embeddings)
            if index not in self.pruning_layers:
                continue

            keep, _ = self._select(relevance, layout)
            pruned = prune_groups(hidden_states, rotary_pos_emb, layout, keep)
            folded = self._aggregate_dropped(
                hidden_states, relevance, layout, pruned.dropped_groups)
            if folded is not None:
                aggregates = folded if aggregates is None else aggregates + folded
            hidden_states = pruned.hidden_states
            rotary_pos_emb = pruned.rotary_pos_emb
            cu_seqlens, cu_window = pruned.cu_seqlens, pruned.cu_window_seqlens
            layout = pruned.layout

        merged = visual.merger(hidden_states)
        # argsort over the survivors' *original* ids: the tower's plain
        # argsort(window_index) no longer applies once groups are gone.
        order = torch.argsort(layout.original)
        merged = merged[order]
        self.survivor_groups = layout.original[order]

        if aggregates is not None:
            # One aggregate token, appended after the retained tokens and passed to
            # the language model (Sec. 4.3).
            padded = aggregates.reshape(1, -1).repeat(1, unit).reshape(unit, -1)
            merged = torch.cat([merged, visual.merger(padded)], dim=0)
        return merged
