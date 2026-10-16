"""DUET Stage 1 -- VisionZip(C) dominant/contextual reduction inside the ViT.

The selection runs on the vision tower's penultimate-layer output (the tensor
LLaVA itself projects) and splits the patches into two groups:

* **Dominant** -- the ``dominant`` patches receiving the most ``[CLS]`` attention,
  kept verbatim in spatial order. ``[CLS]`` itself is preserved separately by the
  caller, so ``dominant`` here counts patches only.
* **Contextual** -- everything else is *first narrowed to a pool* of the
  ``cluster_width * contextual`` best-scoring leftovers, and only that pool is
  merged: ``contextual`` anchors are taken at even strides through the
  score-ordered pool, every other pool member joins its most cosine-similar
  anchor (over the self-attention **keys**), and each anchor absorbs the mean of
  its group. Leftovers outside the pool are dropped outright.

The pool is the one place DUET departs from stock VisionZip, whose contextual
targets are spread over *all* the leftovers. It is reproduced here because DUET's
published budgets are tuned against it -- see ``AMD-AGI/DUET-VLM``.

Both released backbones run exactly this selection; only the salience signal and
where the reduction is applied differ (``[CLS]`` attention inside the CLIP tower
on LLaVA-1.5, received attention from the last ViT block on Qwen2.5-VL). So the
index arithmetic lives in ``duet_stage1_plan`` and each backbone realises the
plan the way its release does: ``duet_stage1`` gathers the tokens into
``[dominant | contextual]`` order for LLaVA, while ``duet_qwen.py``'s ``QwenDuetAdapter``
writes each merged token back into its anchor's own slot so raster order -- and
with it the M-RoPE grid -- survives.

Everything in this module is a pure tensor op so it is testable without a model;
the hook wiring lives in ``duet.py``.
"""

from typing import NamedTuple

import torch


def duet_contextual_anchors(pool_size: int, contextual: int, device: torch.device) -> torch.Tensor:
    """Positions of the merge anchors inside the score-ordered pool.

    Even strides through the pool, so the anchors span the whole salience range
    rather than clustering at the top. When the pool is smaller than the budget
    the reference pads by repeating the last anchor, which yields duplicate
    contextual tokens; that is reproduced rather than fixed, so the token count
    stays fixed at ``contextual`` and the ``<image>`` placeholder trim stays valid.

    :param pool_size: number of leftovers entering the merge, ``>= 1``.
    :param contextual: number of anchors (= contextual tokens) to produce.
    :return: ``[contextual]`` positions into the score-ordered pool.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    step = max(1, pool_size // contextual)
    anchors = torch.arange(0, pool_size, step, device=device, dtype=torch.long)[:contextual]
    if anchors.numel() < contextual:
        anchors = torch.cat([anchors, anchors[-1:].expand(contextual - anchors.numel())], dim=0)
    return anchors


class Stage1Plan(NamedTuple):
    """Which tokens Stage 1 keeps, merges away and drops, as indices into ``[B, N]``.

    Every index addresses the *input* rows, so a caller can realise the plan by
    gathering (LLaVA) or by writing merged tokens back in place (Qwen). Tokens
    named by neither ``dominant_index`` nor ``anchor_index`` are gone: the merged
    ones fold into their anchor, the rest -- the leftovers outside the pool -- are
    dropped outright.
    """

    dominant_index: torch.Tensor  # [B, D] kept verbatim, ascending
    anchor_index: torch.Tensor  # [B, C] leftovers promoted to merge anchors
    merge_index: torch.Tensor  # [B, M] leftovers folded into an anchor
    assign_one_hot: torch.Tensor | None  # [B, M, C] which anchor each merge joined
    counts: torch.Tensor | None  # [B, C, 1] tokens merged into each anchor, floored at 1


def duet_stage1_plan(scores: torch.Tensor, metric: torch.Tensor, *, dominant: int, contextual: int,
                     cluster_width: int, dtype: torch.dtype = torch.float32) -> Stage1Plan:
    """Run the dominant/contextual selection over ``[B, N]`` salience scores.

    :param scores: ``[B, N]`` salience, one value per token (``[CLS]`` attention on
        LLaVA, received attention on Qwen).
    :param metric: ``[B, N, D]`` per-token self-attention keys, the merge similarity.
    :param dominant: tokens kept verbatim, ranked by ``scores``.
    :param contextual: merged tokens produced from the leftovers.
    :param cluster_width: the merge pool is the ``cluster_width * contextual``
        best-scoring leftovers; the rest are discarded.
    :param dtype: dtype of ``assign_one_hot``, so the caller's aggregation matmul
        can stay in the model's precision as upstream's does.
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected [B, N] scores, got {tuple(scores.shape)}")
    if metric.ndim != 3 or metric.shape[:2] != scores.shape:
        raise ValueError(
            f"metric {tuple(metric.shape)} does not match the {tuple(scores.shape)} scores"
        )
    batch, n_tokens = scores.shape
    device = scores.device
    scores = scores.float()
    dominant = max(0, min(int(dominant), n_tokens))
    contextual = max(0, min(int(contextual), n_tokens - dominant))
    if dominant + contextual < 1:
        raise ValueError("DUET Stage 1 must keep at least one token")
    empty = torch.zeros(batch, 0, dtype=torch.long, device=device)

    # --- Dominant: the top-scoring tokens, verbatim, in spatial order. ---------
    keep_mask = torch.zeros(batch, n_tokens, dtype=torch.bool, device=device)
    if dominant > 0:
        top_index = scores.topk(dominant, dim=1).indices.sort(dim=1).values
        keep_mask.scatter_(1, top_index, True)
    else:
        top_index = empty

    residual_count = n_tokens - dominant
    pool_size = min(int(cluster_width) * contextual, residual_count)
    if contextual < 1 or pool_size < 1:
        return Stage1Plan(top_index, empty, empty, None, None)

    # --- Contextual: narrow the leftovers to a pool, then merge into anchors. --
    # `nonzero` is ascending per row, so leftover k of every row is the k-th
    # surviving token in spatial order and the rows stay aligned.
    residual_index = (~keep_mask).nonzero(as_tuple=True)[1].view(batch, residual_count)
    residual_scores = scores.gather(1, residual_index)
    # Descending by salience: anchor strides are over score rank, not position.
    pool_local = residual_scores.topk(pool_size, dim=1).indices
    pool_index = residual_index.gather(1, pool_local)  # [B, P] token ids

    anchor_pool = duet_contextual_anchors(pool_size, contextual, device)  # [C]
    anchor_index = pool_index.gather(1, anchor_pool.unsqueeze(0).expand(batch, contextual))
    merge_mask = torch.ones(pool_size, dtype=torch.bool, device=device)
    merge_mask[anchor_pool.unique()] = False  # a repeated anchor is excluded once
    merge_pool = merge_mask.nonzero(as_tuple=True)[0]  # [M]
    merge_index = pool_index.gather(1, merge_pool.unsqueeze(0).expand(batch, merge_pool.shape[0]))
    if merge_pool.numel() == 0:
        return Stage1Plan(top_index, anchor_index, merge_index, None, None)

    metric_dim = metric.shape[-1]
    anchor_metric = metric.gather(
        1, anchor_index.unsqueeze(-1).expand(batch, contextual, metric_dim)).float()
    merge_metric = metric.gather(
        1, merge_index.unsqueeze(-1).expand(batch, merge_index.shape[1], metric_dim)).float()
    anchor_metric = anchor_metric / anchor_metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    merge_metric = merge_metric / merge_metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    assignment = torch.bmm(merge_metric, anchor_metric.transpose(1, 2)).argmax(dim=2)  # [B, M]
    assign_one_hot = torch.zeros(
        batch, merge_index.shape[1], contextual, dtype=dtype, device=device)
    assign_one_hot.scatter_(2, assignment.unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B, C, 1]
    return Stage1Plan(top_index, anchor_index, merge_index, assign_one_hot, counts)


def duet_stage1(patch_features: torch.Tensor, cls_attention: torch.Tensor, metric: torch.Tensor, *,
                dominant: int, contextual: int, cluster_width: int,
                track_source: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reduce ``[B, N, C]`` patch features to ``dominant + contextual`` tokens.

    :param patch_features: ``[B, N, C]`` patch hidden states, ``[CLS]`` stripped.
    :param cls_attention: ``[B, N]`` ``[CLS]``->patch attention, summed over heads.
    :param metric: ``[B, N, D]`` per-patch self-attention keys, averaged over heads,
        used as the merge similarity.
    :param dominant: patches kept verbatim, ranked by ``cls_attention``.
    :param contextual: merged tokens produced from the leftovers.
    :param cluster_width: the merge pool is the ``cluster_width * contextual``
        best-scoring leftovers; the rest are discarded.
    :param track_source: also return the ``[B, out, N]`` attribution matrix.
    :return: ``(tokens, source)``; ``source`` is ``None`` unless ``track_source``.
    """
    if patch_features.ndim != 3:
        raise ValueError(f"Expected [B, N, C] patch features, got {tuple(patch_features.shape)}")
    batch, n_tokens, channels = patch_features.shape
    if cls_attention.shape != (batch, n_tokens):
        raise ValueError(
            f"cls_attention {tuple(cls_attention.shape)} does not match the {batch} x {n_tokens} "
            f"patch features being reduced"
        )
    if metric.ndim != 3 or metric.shape[:2] != (batch, n_tokens):
        raise ValueError(
            f"metric {tuple(metric.shape)} does not match the {batch} x {n_tokens} patch features"
        )
    device = patch_features.device
    plan = duet_stage1_plan(cls_attention, metric, dominant=dominant, contextual=contextual,
                            cluster_width=cluster_width, dtype=patch_features.dtype)
    top_index, anchor_index, merge_index = plan.dominant_index, plan.anchor_index, plan.merge_index
    dominant = int(top_index.shape[1])
    contextual = int(anchor_index.shape[1])

    # --- Dominant: the top-scoring patches, verbatim, in spatial order. --------
    dominant_tokens = patch_features.gather(
        1, top_index.unsqueeze(-1).expand(batch, dominant, channels))

    source: torch.Tensor | None = None
    if track_source:
        # Selection rows are one-hot, which the visualiser reads as pruning.
        source = torch.zeros(batch, dominant, n_tokens, dtype=torch.float32, device=device)
        if dominant > 0:
            source.scatter_(2, top_index.unsqueeze(-1), 1.0)

    if contextual < 1:
        return dominant_tokens, source

    # --- Contextual: each anchor absorbs the mean of the tokens assigned to it. -
    assign_one_hot, counts = plan.assign_one_hot, plan.counts
    anchor_tokens = patch_features.gather(
        1, anchor_index.unsqueeze(-1).expand(batch, contextual, channels))
    if assign_one_hot is None or counts is None:
        contextual_tokens = anchor_tokens
    else:
        merge_tokens = patch_features.gather(
            1, merge_index.unsqueeze(-1).expand(batch, merge_index.shape[1], channels))
        # An anchor with no assignment adds nothing, matching the reference's
        # `contextual.append(target_hidden[idx])` branch.
        contextual_tokens = anchor_tokens + torch.bmm(
            assign_one_hot.transpose(1, 2), merge_tokens) / counts

    if track_source:
        # Contextual row i weights its anchor by 1 and each assigned patch by
        # 1/count[i] -- fractional rows read as merging to the visualiser.
        contextual_source = torch.zeros(
            batch, contextual, n_tokens, dtype=torch.float32, device=device)
        contextual_source.scatter_(2, anchor_index.unsqueeze(-1), 1.0)
        if assign_one_hot is not None and counts is not None:
            contextual_source.scatter_add_(
                2,
                merge_index.unsqueeze(1).expand(batch, contextual, merge_index.shape[1]),
                assign_one_hot.transpose(1, 2).float() / counts.float(),
            )
        source = torch.cat([source, contextual_source], dim=1)

    tokens = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(patch_features.dtype)
    return tokens, source
