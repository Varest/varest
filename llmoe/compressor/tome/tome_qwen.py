"""Qwen2.5-VL integration for Token Merging (ToMe).

Qwen's windowed NaViT does not expose a CLIP-style layer stack that can be
patched safely. Following the VisionZip integration, this adapter captures the
last vision block's RoPE-adjusted keys, folds them through Qwen's patch merger,
and uses those keys for ToMe matching over the projected visual span after it
has been concatenated with text.
"""

from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    place_visual_tokens,
    ratio_keep_count,
)
from llmoe.compressor.tome.tome import _bipartite_soft_matching
from llmoe.utils.qwen import DEFAULT_ATTENTION_CHUNK


class QwenToMeAdapter(QwenPruningAdapter):
    """Apply exact-ratio ToMe merging to Qwen's visual prompt block."""

    method = "ToMe"
    capture_vision_attention = True

    def __init__(
        self,
        *,
        compressor: Any,
        retain_ratio: float,
        track_source: bool,
        attention_chunk: int = DEFAULT_ATTENTION_CHUNK,
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
        rounds = (
            self._compressor._num_merge_layers
            or len(self._visual.blocks)
        )
        print(
            f"[ToMe/Qwen] install: retain_ratio="
            f"{self._retain_ratio:g}, rounds={rounds}, "
            f"metric={self._compressor._metric_source}, score "
            f"@vision block {len(self._visual.blocks) - 1}, "
            f"image_token_id={self._image_token_id}",
            flush=True,
        )

    @staticmethod
    def _representative_positions(
        centroids: torch.Tensor, num_visual: int
    ) -> torch.Tensor:
        """Map ordered group centroids to distinct nearby M-RoPE columns."""
        raw = centroids.round().to(torch.long).tolist()
        count = len(raw)
        positions: list[int] = []
        previous = -1
        for index, value in enumerate(raw):
            lower = previous + 1
            upper = int(num_visual) - (count - index)
            position = max(lower, min(int(value), upper))
            positions.append(position)
            previous = position
        return torch.tensor(
            positions,
            dtype=torch.long,
            device=centroids.device,
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
            label="ToMe/Qwen",
        )
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
        if (
            self._received is None
            or self._keys is None
            or self._image_grid_thw is None
        ):
            raise RuntimeError(
                "ToMe/Qwen found image tokens but captured no last-block "
                "vision keys."
            )

        self._require_kv_cache(kwargs)
        _, merged_keys = self._merged_token_signals()
        if merged_keys.shape[0] != num_visual:
            raise RuntimeError(
                f"ToMe/Qwen folded {merged_keys.shape[0]} key vectors for "
                f"{num_visual} visual prompt tokens."
            )

        visual = inputs_embeds[:, first:first + num_visual]
        metric = (
            visual
            if self._compressor._metric_source == "hidden_state"
            else merged_keys.unsqueeze(0).to(
                device=visual.device, dtype=visual.dtype
            )
        )
        token_width = visual.shape[-1]
        metric_width = metric.shape[-1]
        coordinate = torch.arange(
            num_visual,
            device=visual.device,
            dtype=visual.dtype,
        ).view(1, num_visual, 1)
        combined = torch.cat(
            [visual, metric, coordinate], dim=-1
        )
        size = visual.new_ones((1, num_visual, 1))
        source = (
            torch.eye(
                num_visual,
                device=visual.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            if self._track_source
            else None
        )
        history = (
            [torch.eye(num_visual, dtype=torch.float32)]
            if self._track_source
            else []
        )

        rounds = (
            self._compressor._num_merge_layers
            or len(self._visual.blocks)
        )
        schedule = self._compressor._resolve_schedule(
            num_visual,
            num_visual - keep_count,
            rounds,
            protect_cls=False,
        )
        for reduction in schedule:
            if reduction < 1:
                continue
            combined, size, source = _bipartite_soft_matching(
                combined,
                metric,
                reduction,
                size,
                source,
                protect_cls=False,
            )
            order = combined[0, :, -1].argsort()
            combined = combined[:, order]
            size = size[:, order]
            if source is not None:
                source = source[:, order]
                history.append(source[0].detach().cpu())
            metric = combined[
                :, :, token_width:token_width + metric_width
            ]

        reduced = combined[:, :, :token_width]
        centroids = combined[0, :, -1]
        positions = self._representative_positions(
            centroids, num_visual
        )
        modified, keep = place_visual_tokens(
            inputs_embeds,
            start=first,
            visual_count=num_visual,
            tokens=reduced,
            positions=positions,
            label="ToMe/Qwen",
        )
        state = self._apply_keep(
            kwargs, inputs_embeds, modified, keep
        )
        if self._track_source:
            self._source_history = history
        if not self._logged:
            self._logged = True
            print(
                f"[ToMe/Qwen] first prefill: visual "
                f"{num_visual} -> {keep_count}, sequence "
                f"{inputs_embeds.shape[1]} -> {state['prefill_len']}",
                flush=True,
            )
        return args, kwargs

