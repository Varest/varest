"""Qwen2.5-VL integration for VisionZip.

The shared Qwen adapter owns vision-attention capture, prompt geometry, M-RoPE
narrowing, decode re-addressing, and hook lifecycle. This module owns only
VisionZip's dominant-token selection and contextual merge policy.
"""

from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import QwenPruningAdapter
from llmoe.compressor.vision_zip.vision_zip import ContextualMerge, contextual_merge
from llmoe.utils.qwen import DEFAULT_ATTENTION_CHUNK


class QwenVisionZipAdapter(QwenPruningAdapter):
    """Apply VisionZip after Qwen's patch merger and before its decoder."""

    method = "VisionZip"
    capture_vision_attention = True

    def __init__(
        self,
        *,
        dominant_ratio: float,
        contextual_ratio: float,
        track_source: bool = False,
        attention_chunk: int = DEFAULT_ATTENTION_CHUNK,
    ) -> None:
        super().__init__(
            attention_chunk=attention_chunk,
            track_source=track_source,
        )
        self._dominant_ratio = float(dominant_ratio)
        self._contextual_ratio = float(contextual_ratio)
        self._logged = False

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        super().install(vlm, image_token_id=image_token_id)
        blocks = self._visual.blocks
        print(
            f"[VisionZip/Qwen] install: dominant={self._dominant_ratio:g}, "
            f"contextual={self._contextual_ratio:g} "
            f"(retain={self._dominant_ratio + self._contextual_ratio:g}), "
            f"score @vision block {len(blocks) - 1}, "
            f"image_token_id={self._image_token_id}",
            flush=True,
        )

    def _prefill(
        self,
        args: tuple,
        kwargs: dict,
        inputs_embeds: torch.Tensor,
    ) -> tuple[tuple, dict] | None:
        if (
            self._received is None
            or self._keys is None
            or self._image_grid_thw is None
        ):
            return None
        self._require_kv_cache(kwargs)

        span = self._locate_image_span(inputs_embeds)
        if span is None:
            return None
        first, num_visual = span
        last = first + num_visual
        scores, merged_keys = self._merged_token_signals()
        if scores.numel() != num_visual:
            raise RuntimeError(
                f"VisionZip scored {scores.numel()} visual tokens but the prompt "
                f"carries {num_visual} image placeholders."
            )

        dominant_num = min(
            int(self._dominant_ratio * num_visual), num_visual
        )
        contextual_num = max(
            int(self._contextual_ratio * num_visual), 1
        )
        contextual_num = min(
            contextual_num, num_visual - dominant_num
        )
        if contextual_num < 1 or dominant_num + contextual_num >= num_visual:
            self._prefill_state = None
            return None

        device = inputs_embeds.device
        dominant_mask = torch.zeros(
            num_visual, dtype=torch.bool, device=device
        )
        if dominant_num:
            dominant_mask[scores.topk(dominant_num).indices] = True
        leftover = (~dominant_mask).nonzero(as_tuple=True)[0]

        metric = merged_keys[leftover].float()
        metric = metric / metric.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        merged = contextual_merge(
            inputs_embeds[:, first:last][:, leftover],
            metric.unsqueeze(0),
            contextual_num,
        )

        target_positions = leftover[merged.target_indices]
        keep_visual = dominant_mask.clone()
        keep_visual[target_positions] = True
        keep = torch.ones(
            inputs_embeds.shape[1], dtype=torch.bool, device=device
        )
        keep[first:last] = keep_visual

        modified = inputs_embeds.clone()
        modified[:, first + target_positions] = merged.tokens.to(
            modified.dtype
        )
        state = self._apply_keep(
            kwargs, inputs_embeds, modified, keep
        )

        if self._track_source:
            self._record_source(
                num_visual,
                keep_visual,
                leftover,
                target_positions,
                merged,
            )
        if not self._logged:
            self._logged = True
            print(
                f"[VisionZip/Qwen] first prefill: visual {num_visual} -> "
                f"{dominant_num} dominant + {contextual_num} contextual, "
                f"sequence {inputs_embeds.shape[1]} -> "
                f"{state['prefill_len']}",
                flush=True,
            )
        return args, kwargs

    def _record_source(
        self,
        num_visual: int,
        keep_visual: torch.Tensor,
        leftover: torch.Tensor,
        target_positions: torch.Tensor,
        merged: ContextualMerge,
    ) -> None:
        """Record attribution in surviving raster order."""
        kept = keep_visual.nonzero(as_tuple=True)[0]
        num_kept = int(kept.numel())
        source = torch.zeros(
            num_kept, num_visual, dtype=torch.float32
        )
        kept_cpu = kept.cpu()
        source[torch.arange(num_kept), kept_cpu] = 1.0

        row_of = torch.full((num_visual,), -1, dtype=torch.long)
        row_of[kept_cpu] = torch.arange(num_kept)
        target_rows = row_of[target_positions.cpu()]
        merged_from = leftover[merged.nontarget_mask].cpu()
        if merged_from.numel():
            weights = (
                merged.assign_one_hot[0].transpose(0, 1)
                / merged.counts[0]
            ).float().cpu()
            source[
                target_rows.unsqueeze(1), merged_from.unsqueeze(0)
            ] += weights

        self._source_history = [
            torch.eye(num_visual, dtype=torch.float32),
            source,
        ]
