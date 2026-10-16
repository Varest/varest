"""InternVL adapter for VFlowOpt's tiled vision and decoder stages."""

from typing import Any

import torch

from llmoe.compressor.internvl import (
    concatenate_tile_signal,
    image_sequences,
    is_internvl,
    pool_tile_signal,
    recorded_tile_counts,
    tokens_per_tile,
    trim_image_placeholders,
)
from llmoe.compressor.vflowopt.vflowopt import (
    StageLayers,
    grid_cell_ids,
    importance_score,
    layer_average_tokens,
    patch_entropy,
    vision_stage,
)


class InternVLVFlowOptAdapter:
    """Run VFlowOpt once over a source image's globally concatenated tiles."""

    def __init__(self, compressor: Any, vlm: Any, *, decoder_layers: int) -> None:
        self._compressor = compressor
        self._vlm = vlm
        self.tokens_per_tile = tokens_per_tile(vlm)
        self.stages = StageLayers.even(int(decoder_layers))
        self._logged_geometry = False

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_internvl(vlm)

    def grid(self, source_tokens: int) -> tuple[int, int]:
        if source_tokens % self.tokens_per_tile:
            raise ValueError(
                f"VFlowOpt expected whole InternVL tiles, got {source_tokens} tokens"
            )
        side = int(round(self.tokens_per_tile**0.5))
        if side * side != self.tokens_per_tile:
            raise ValueError(
                f"InternVL tokens per tile must form a square, got {self.tokens_per_tile}"
            )
        return side * (source_tokens // self.tokens_per_tile), side

    def configure(self, source_tokens: int) -> None:
        self.grid(source_tokens)
        owner = self._compressor
        n1 = min(source_tokens, owner.vision_stage_tokens)
        n2 = n1 - int(round(n1 * (1.0 - owner.strategy.r2)))
        n2 = max(1, min(n1, n2))
        slack = (
            owner._budget_tokens * self.stages.total
            - n1 * self.stages.l1
            - n2 * self.stages.l2
        )
        n3 = int(slack // self.stages.l3) if slack > 0 else 1
        n3 = max(1, min(n2, n3))
        owner._active_source_tokens = source_tokens
        owner._active_keeps = (n1, n2, n3)

        if owner._verbose and not self._logged_geometry:
            self._logged_geometry = True
            measured = layer_average_tokens(owner._active_keeps, self.stages)
            grid_h, grid_w = self.grid(source_tokens)
            print(
                f"[VFlowOpt/InternVL] global sequence {source_tokens} tokens "
                f"({source_tokens // self.tokens_per_tile} tiles) -> "
                f"keeps={n1}/{n2}/{n3}, grid={grid_h}x{grid_w}, "
                f"layer-average={measured:.2f} (budget {owner._budget_tokens})",
                flush=True,
            )

    def log_install(self) -> None:
        k1, k2 = self.stages.prune_layers
        print(
            f"[VFlowOpt/InternVL] install budget={self._compressor._budget_tokens}; "
            "concatenate all tiles before the global vision stage; "
            f"decoder pruning at layers {k1}/{k2}. Active keeps are "
            "resolved from each source image.",
            flush=True,
        )

    def compress_outputs(self, image_outputs: Any) -> Any:
        """Concatenate every tile, then run one global VFlowOpt vision stage."""
        owner = self._compressor
        counts = recorded_tile_counts(self._vlm)
        if len(counts) != 1:
            raise NotImplementedError(
                "VFlowOpt on InternVL requires batch_size=1 so progressive decoder "
                "pruning has one dynamic tile geometry"
            )
        sequences = image_sequences(
            image_outputs,
            tile_counts=counts,
            output_tokens_per_tile=self.tokens_per_tile,
        )
        if len(sequences) != 1:
            raise RuntimeError(f"Expected one InternVL source image, got {len(sequences)}")

        context = owner._context_for_forward(1)
        pixels = owner._captured_pixels
        if not isinstance(pixels, torch.Tensor):
            raise RuntimeError("VFlowOpt did not capture InternVL pixel_values")
        entropy = patch_entropy(
            pixels,
            patch_size=owner._vision_patch_size,
            mode=owner._entropy_mode,
            image_mean=owner._image_mean,
            image_std=owner._image_std,
        )
        if context.shape[-1] == entropy.shape[-1] + 1:
            context = context[:, 1:]
        if context.shape != entropy.shape:
            raise RuntimeError(
                f"InternVL context {tuple(context.shape)} and entropy "
                f"{tuple(entropy.shape)} do not align"
            )

        pooled_context = pool_tile_signal(
            context, self.tokens_per_tile, has_cls=False, reduction="sum"
        )
        pooled_entropy = pool_tile_signal(
            entropy, self.tokens_per_tile, has_cls=False, reduction="mean"
        )
        context_groups = concatenate_tile_signal(
            pooled_context, sequences, self.tokens_per_tile
        )
        entropy_groups = concatenate_tile_signal(
            pooled_entropy, sequences, self.tokens_per_tile
        )
        features = sequences[0]
        source_tokens = int(features.shape[0])
        self.configure(source_tokens)
        importance = importance_score(
            context_groups[0].unsqueeze(0),
            entropy_groups[0].unsqueeze(0),
            owner.strategy.alpha,
        )[0].to(features.device)

        grid_h, grid_w = self.grid(source_tokens)
        cell_ids = grid_cell_ids(
            source_tokens,
            owner.strategy.grid_side,
            owner.strategy.a,
            features.device,
            grid_h=grid_h,
            grid_w=grid_w,
            reference_side=owner.strategy.grid_side,
        )
        n_cells = int(torch.unique(cell_ids).numel())
        result = vision_stage(
            features,
            importance,
            keep_target=owner._active_keeps[0],
            a=owner.strategy.a,
            side=owner.strategy.grid_side,
            n_cells=n_cells,
            grid_h=grid_h,
            grid_w=grid_w,
            reference_side=owner.strategy.grid_side,
        )

        owner._vit_importance = [result.importance]
        owner.stats["merged_cells_sum"] += float(result.merged_cells)
        owner.stats["topped_up_sum"] += float(result.topped_up)
        if owner._track_source:
            owner._source_history = owner._build_source_history(result, source_tokens)
        else:
            owner._source_history = []
        owner._captured_hidden = None
        owner._captured_pixels = None
        image_outputs.pooler_output = result.features.unsqueeze(0)
        return image_outputs

    def trim_inputs(self, inputs: Any, *, pad_token_id: int) -> Any:
        owner = self._compressor
        counts = recorded_tile_counts(self._vlm)
        if len(counts) != 1:
            raise NotImplementedError(
                "VFlowOpt on InternVL requires batch_size=1; set batch_size: 1"
            )
        source_tokens = counts[0] * self.tokens_per_tile
        input_ids = inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            placeholders = int((input_ids == owner._image_token_id).sum())
            if placeholders != source_tokens:
                raise RuntimeError(
                    f"InternVL recorded {source_tokens} source tokens but the prompt "
                    f"contains {placeholders} image placeholders"
                )
        self.configure(source_tokens)
        keep = owner._active_keeps[0]
        return trim_image_placeholders(
            inputs,
            token_id=owner._image_token_id,
            keep_for_run=lambda _run_length: keep,
            pad_token_id=pad_token_id,
        )
