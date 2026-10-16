"""Qwen2.5-VL adapter for OTPrune's released DPP policy."""

from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    place_visual_tokens,
    ratio_keep_count,
)


class QwenOTPruneAdapter(QwenPruningAdapter):
    """Run OTPrune over Qwen's projected, concatenated visual block."""

    method = "OTPrune"

    def __init__(
        self,
        *,
        compressor: Any,
        retain_ratio: float,
        track_source: bool,
    ) -> None:
        super().__init__(track_source=track_source)
        self._compressor = compressor
        self._retain_ratio = float(retain_ratio)
        self._logged = False

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        super().install(vlm, image_token_id=image_token_id)
        print(
            f"[OTPrune/Qwen] install: retain_ratio="
            f"{self._retain_ratio:g}, gamma={self._compressor._gamma:g}, "
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
            label="OTPrune/Qwen",
        )
        visual = inputs_embeds[0, first:first + num_visual]
        if keep_count >= num_visual:
            if self._track_source:
                identity = torch.eye(
                    num_visual, dtype=torch.float32
                )
                self._source_history = [
                    identity, identity.clone()
                ]
            self._prefill_state = None
            return None

        self._require_kv_cache(kwargs)
        indices = self._compressor._select_indices(
            visual, keep=keep_count
        )
        reduced = visual.index_select(0, indices).unsqueeze(0)
        modified, keep = place_visual_tokens(
            inputs_embeds,
            start=first,
            visual_count=num_visual,
            tokens=reduced,
            positions=indices,
            label="OTPrune/Qwen",
        )
        state = self._apply_keep(
            kwargs, inputs_embeds, modified, keep
        )

        if self._track_source:
            source = torch.zeros(
                keep_count, num_visual, dtype=torch.float32
            )
            source[
                torch.arange(keep_count), indices.detach().cpu()
            ] = 1.0
            self._source_history = [
                torch.eye(num_visual, dtype=torch.float32),
                source,
            ]
        if not self._logged:
            self._logged = True
            print(
                f"[OTPrune/Qwen] first prefill: visual "
                f"{num_visual} -> {keep_count}, sequence "
                f"{inputs_embeds.shape[1]} -> {state['prefill_len']}",
                flush=True,
            )
        return args, kwargs

