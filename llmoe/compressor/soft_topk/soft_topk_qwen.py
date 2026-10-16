"""Qwen2.5-VL adapter for learned SoftTopk selection.

The scorer runs on the projected visual block after Qwen has concatenated it
with text. The full block therefore passes placeholder validation and M-RoPE
construction first; the shared adapter then gathers the ratio-based survivors.
"""

from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    place_visual_tokens,
    ratio_keep_count,
)


class QwenSoftTopkAdapter(QwenPruningAdapter):
    """Apply SoftTopk to Qwen's concatenated prompt embeddings."""

    method = "SoftTopk"

    def __init__(
        self,
        *,
        compressor: Any,
        retain_ratio: float,
        for_training: bool = False,
    ) -> None:
        super().__init__(track_source=True)
        self._compressor = compressor
        self._retain_ratio = float(retain_ratio)
        self._for_training = bool(for_training)
        self._logged = False

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        super().install(vlm, image_token_id=image_token_id)
        print(
            f"[SoftTopk/Qwen] install: retain_ratio={self._retain_ratio:g}, "
            f"image_token_id={self._image_token_id}",
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
        first, num_visual = span
        keep_count = ratio_keep_count(
            num_visual,
            self._retain_ratio,
            label="SoftTopk/Qwen",
        )

        visual = inputs_embeds[:, first:first + num_visual]
        reduced, indices = self._compressor._compress_to(
            visual, keep_count
        )
        self._compressor._record_selection(
            indices,
            num_visual,
            visual.device,
        )
        self._source_history = list(
            self._compressor._source_history
        )
        if keep_count >= num_visual:
            self._prefill_state = None
            return None
        if not self._for_training:
            self._require_kv_cache(kwargs)

        modified, keep = place_visual_tokens(
            inputs_embeds,
            start=first,
            visual_count=num_visual,
            tokens=reduced,
            positions=indices[0],
            label="SoftTopk/Qwen",
        )
        state = self._apply_keep(
            kwargs, inputs_embeds, modified, keep
        )
        if not self._logged:
            self._logged = True
            print(
                f"[SoftTopk/Qwen] first prefill: visual "
                f"{num_visual} -> {keep_count}, sequence "
                f"{inputs_embeds.shape[1]} -> {state['prefill_len']}",
                flush=True,
            )
        return args, kwargs

