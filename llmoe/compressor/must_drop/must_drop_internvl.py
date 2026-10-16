"""InternVL adapter for MustDrop's tiled vision stage."""

from typing import Any

import torch

from llmoe.compressor.internvl import (
    concatenate_tile_signal,
    image_sequences,
    is_internvl,
    pool_tile_signal,
    recorded_tile_counts,
    tokens_per_tile,
)


class InternVLMustDropAdapter:
    """Align tile-local ViT signals, then run one global MustDrop reduction."""

    def __init__(self, compressor: Any, vlm: Any, *, window_size: int) -> None:
        self._compressor = compressor
        self._vlm = vlm
        self.tokens_per_tile = tokens_per_tile(vlm)
        tile_side = int(round(self.tokens_per_tile**0.5))
        if tile_side * tile_side != self.tokens_per_tile:
            raise ValueError(
                f"MustDrop needs square InternVL tiles, got "
                f"{self.tokens_per_tile} tokens per tile"
            )
        requested = int(window_size)
        if tile_side % requested:
            if tile_side % 2:
                raise ValueError(
                    f"MustDrop window_size={requested} does not tile InternVL's "
                    f"{tile_side}x{tile_side} token grid"
                )
            print(
                f"[MustDrop/InternVL] window_size={requested} does not tile a "
                f"{tile_side}x{tile_side} tile; using 2x2 local windows.",
                flush=True,
            )
            requested = 2
        self.window_size = requested

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_internvl(vlm)

    def grid_divisible(self, n_patch: int, window_size: int) -> bool:
        side = int(round(self.tokens_per_tile**0.5))
        return (
            n_patch % self.tokens_per_tile == 0
            and side * side == self.tokens_per_tile
            and side % int(window_size) == 0
        )

    def window_partition(self, n_patch: int, window_size: int) -> torch.Tensor | None:
        if not self.grid_divisible(n_patch, window_size):
            return None
        k = int(window_size)
        side = int(round(self.tokens_per_tile**0.5))
        tile_grid = torch.arange(self.tokens_per_tile).reshape(side, side)
        local = tile_grid.unfold(0, k, k).unfold(1, k, k).reshape(-1, k * k)
        tile_count = n_patch // self.tokens_per_tile
        offsets = torch.arange(tile_count)[:, None, None] * self.tokens_per_tile
        # Select which windows merge globally after concatenation; the budget is
        # never spent once per tile.
        return (local[None, :, :] + offsets).reshape(-1, k * k)

    def compress_outputs(
        self,
        image_outputs: Any,
        cls_attention: torch.Tensor | None,
    ) -> Any:
        sequences = image_sequences(
            image_outputs,
            tile_counts=recorded_tile_counts(self._vlm),
            output_tokens_per_tile=self.tokens_per_tile,
        )

        captured = self._compressor._captured_vit_hidden
        if isinstance(captured, torch.Tensor):
            pooled_hidden = pool_tile_signal(
                captured, self.tokens_per_tile, has_cls=True
            )
            hidden_groups: list[torch.Tensor | None] = list(
                concatenate_tile_signal(
                    pooled_hidden, sequences, self.tokens_per_tile
                )
            )
        else:
            hidden_groups = [None] * len(sequences)

        if isinstance(cls_attention, torch.Tensor):
            pooled_attention = pool_tile_signal(
                cls_attention,
                self.tokens_per_tile,
                has_cls=False,
                reduction="sum",
            )
            attention_groups: list[torch.Tensor | None] = list(
                concatenate_tile_signal(
                    pooled_attention, sequences, self.tokens_per_tile
                )
            )
        else:
            attention_groups = [None] * len(sequences)

        compressed: list[torch.Tensor] = []
        key_indices: list[torch.Tensor] = []
        for index, sequence in enumerate(sequences):
            score_group = hidden_groups[index]
            self._compressor._captured_vit_hidden = (
                None if score_group is None else score_group.unsqueeze(0)
            )
            attention_group = attention_groups[index]
            attention = None if attention_group is None else attention_group.unsqueeze(0)
            result = self._compressor.compress(sequence, attention=attention)
            compressed.append(result.tokens)
            key_indices.extend(self._compressor._key_indices)

        self._compressor._captured_vit_hidden = None
        self._compressor._key_indices = key_indices
        image_outputs.pooler_output = torch.cat(compressed, dim=0).unsqueeze(0)
        return image_outputs
