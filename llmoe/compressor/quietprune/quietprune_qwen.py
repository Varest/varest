"""Post-merger Qwen2.5-VL adaptation of QuietPrune.

Qwen's vision encoder never receives prompt embeddings, so its stock blocks have
no faithful ``[Q-CLS]`` injection point. This adapter keeps QuietPrune's spatial
group selection and redundant-token aggregate, but ranks groups by the final
vision block's received attention (the same backend-independent signal used by
VisionZip). Reduction is applied to the projected visual span after it has been
concatenated with text, preserving Qwen's native vision forward and M-RoPE.
"""

from dataclasses import dataclass
from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    place_visual_tokens,
    ratio_keep_count,
)


@dataclass(frozen=True)
class QuietQwenResult:
    tokens: torch.Tensor
    positions: torch.Tensor
    source: torch.Tensor


def quiet_qwen_reduce(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    *,
    grid: tuple[int, int],
    keep: int,
    group_size: int,
    keep_border_tokens: bool,
    aggregate: bool,
    normalize_aggregation: bool,
    eps: float,
) -> QuietQwenResult:
    """Keep spatial groups, using individual fill only for ratio remainders."""
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [N, D], got {tuple(tokens.shape)}")
    num_tokens = int(tokens.shape[0])
    height, width = (int(value) for value in grid)
    if height * width != num_tokens:
        raise ValueError(f"grid {grid} does not match {num_tokens} tokens")
    if scores.reshape(-1).numel() != num_tokens:
        raise ValueError(
            f"scores {tuple(scores.shape)} do not match {num_tokens} tokens"
        )
    keep = max(1, min(num_tokens, int(keep)))
    scores = scores.reshape(-1).to(device=tokens.device, dtype=torch.float32)
    if keep == num_tokens:
        return QuietQwenResult(
            tokens=tokens,
            positions=torch.arange(num_tokens, device=tokens.device),
            source=torch.eye(num_tokens, dtype=torch.float32, device=tokens.device),
        )

    aggregate_count = int(bool(aggregate))
    native_target = keep - aggregate_count
    keep_mask = torch.zeros(num_tokens, dtype=torch.bool, device=tokens.device)
    full_height = height - height % group_size
    full_width = width - width % group_size
    members = [
        [
            (top + dy) * width + left + dx
            for dy in range(group_size)
            for dx in range(group_size)
        ]
        for top in range(0, full_height, group_size)
        for left in range(0, full_width, group_size)
    ]
    grouped = torch.zeros(num_tokens, dtype=torch.bool, device=tokens.device)
    if members:
        group_members = torch.tensor(members, dtype=torch.long, device=tokens.device)
        grouped[group_members.reshape(-1)] = True
    else:
        group_members = torch.empty(
            (0, group_size * group_size), dtype=torch.long, device=tokens.device
        )

    if keep_border_tokens and native_target:
        border = (~grouped).nonzero(as_tuple=True)[0]
        if border.numel() > native_target:
            border = border[scores[border].topk(native_target).indices]
        keep_mask[border] = True

    remaining = native_target - int(keep_mask.sum())
    group_area = group_size * group_size
    if remaining >= group_area and group_members.numel():
        group_scores = scores[group_members].mean(dim=1)
        group_count = min(int(group_members.shape[0]), remaining // group_area)
        chosen = group_scores.topk(group_count).indices
        keep_mask[group_members[chosen].reshape(-1)] = True

    # A ratio is not generally divisible by a 2x2 group. Fill only that remainder
    # with individual high-relevance tokens so the dynamic budget is exact.
    remaining = native_target - int(keep_mask.sum())
    if remaining:
        candidates = (~keep_mask).nonzero(as_tuple=True)[0]
        fill = candidates[scores[candidates].topk(remaining).indices]
        keep_mask[fill] = True

    native = keep_mask.nonzero(as_tuple=True)[0]
    output_tokens = tokens[native]
    output_positions = native
    source = torch.zeros(
        native.numel(), num_tokens, dtype=torch.float32, device=tokens.device
    )
    if native.numel():
        source[torch.arange(native.numel(), device=tokens.device), native] = 1.0

    if aggregate_count:
        dropped = (~keep_mask).nonzero(as_tuple=True)[0]
        weights = scores[dropped].clamp_min(0.0)
        mass = weights.sum()
        if float(mass) <= eps:
            normalized = torch.full_like(weights, 1.0 / max(1, weights.numel()))
        else:
            normalized = weights / mass
        token_weights = normalized if normalize_aggregation else weights
        aggregate_token = (
            tokens[dropped] * token_weights.to(tokens.dtype).unsqueeze(-1)
        ).sum(dim=0, keepdim=True)
        representative = dropped[scores[dropped].argmax()].reshape(1)
        aggregate_source = torch.zeros(
            1, num_tokens, dtype=torch.float32, device=tokens.device
        )
        aggregate_source[0, dropped] = normalized
        output_tokens = torch.cat([output_tokens, aggregate_token], dim=0)
        output_positions = torch.cat([output_positions, representative], dim=0)
        source = torch.cat([source, aggregate_source], dim=0)

    order = output_positions.argsort()
    return QuietQwenResult(
        tokens=output_tokens[order],
        positions=output_positions[order],
        source=source[order],
    )


class QwenQuietPruneAdapter(QwenPruningAdapter):
    """Run QuietPrune's group reducer over Qwen's merged visual grid."""

    method = "QuietPrune"
    capture_vision_attention = True

    def __init__(
        self,
        *,
        compressor: Any,
        retain_ratio: float,
        attention_chunk: int,
        track_source: bool,
    ) -> None:
        super().__init__(
            attention_chunk=attention_chunk,
            track_source=track_source,
        )
        self._compressor = compressor
        self._retain_ratio = float(retain_ratio)
        self._logged = False

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        super().install(vlm, image_token_id=image_token_id)
        print(
            f"[QuietPrune/Qwen] install: retain_ratio={self._retain_ratio:g}, "
            f"group_size={self._compressor._group_size}, "
            f"aggregate={self._compressor._aggregate}, score @vision block "
            f"{len(self._visual.blocks) - 1}. This is the documented "
            "received-attention adaptation, not the released Q-CLS path.",
            flush=True,
        )

    def _prefill(
        self,
        args: tuple,
        kwargs: dict,
        inputs_embeds: torch.Tensor,
    ) -> tuple[tuple, dict] | None:
        span = self._locate_image_span(inputs_embeds)
        if span is None:
            return None
        if self._received is None or self._keys is None:
            raise RuntimeError(
                "QuietPrune/Qwen found image tokens but captured no final-block "
                "vision attention."
            )
        first, num_visual = span
        keep_count = ratio_keep_count(
            num_visual,
            self._retain_ratio,
            label="QuietPrune/Qwen",
        )
        if keep_count >= num_visual:
            if self._track_source:
                identity = torch.eye(num_visual, dtype=torch.float32)
                self._source_history = [identity, identity.clone()]
            return None

        self._require_kv_cache(kwargs)
        received, _ = self._merged_token_signals()
        if received.numel() != num_visual:
            raise RuntimeError(
                f"QuietPrune/Qwen scored {received.numel()} tokens but the "
                f"prompt carries {num_visual}."
            )
        visual = inputs_embeds[0, first:first + num_visual]
        result = quiet_qwen_reduce(
            visual,
            received.to(visual.device),
            grid=self._merged_image_grid(num_visual),
            keep=keep_count,
            group_size=self._compressor._group_size,
            keep_border_tokens=self._compressor._keep_border_tokens,
            aggregate=self._compressor._aggregate,
            normalize_aggregation=self._compressor._normalize_aggregation,
            eps=self._compressor._eps,
        )
        modified, keep = place_visual_tokens(
            inputs_embeds,
            start=first,
            visual_count=num_visual,
            tokens=result.tokens.unsqueeze(0),
            positions=result.positions,
            label="QuietPrune/Qwen",
        )
        state = self._apply_keep(kwargs, inputs_embeds, modified, keep)
        if self._track_source:
            self._source_history = [
                torch.eye(num_visual, dtype=torch.float32),
                result.source.detach().cpu(),
            ]
        if not self._logged:
            self._logged = True
            print(
                f"[QuietPrune/Qwen] first prefill: visual {num_visual} -> "
                f"{result.tokens.shape[0]}, sequence {inputs_embeds.shape[1]} "
                f"-> {state['prefill_len']}",
                flush=True,
            )
        return args, kwargs
