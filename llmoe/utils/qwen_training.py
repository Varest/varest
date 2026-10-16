"""Shared row-wise Qwen2.5-VL training geometry for token compressors.

Qwen's NaViT encoder gives each image a different visual-token count. A
compression-aware training forward therefore has to reduce every row
independently, preserve that row's M-RoPE coordinates and labels, then right-pad
the reduced rows back into a rectangular microbatch. DUET and VisionZip share
that plumbing; their token-selection policies remain method-local.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from llmoe.utils.qwen import (
    chunked_received_attention,
    window_reverse_indices,
)


IGNORE_INDEX = -100


@dataclass(frozen=True)
class QwenVisualSpan:
    start: int
    end: int

    @property
    def count(self) -> int:
        return int(self.end - self.start)


class QwenVisionSignalCapture:
    """Capture per-image received-attention scores and post-RoPE key metrics."""

    def __init__(
        self,
        visual: nn.Module,
        *,
        attention_chunk: int,
        label: str = "Qwen training",
    ) -> None:
        self.visual = visual
        self.attention_chunk = int(attention_chunk)
        self.label = str(label)
        self.received: torch.Tensor | None = None
        self.keys: torch.Tensor | None = None
        blocks = getattr(visual, "blocks", None)
        if not blocks:
            raise TypeError(f"{self.label} needs model.visual.blocks.")
        self.handle: RemovableHandle = blocks[-1].attn.register_forward_pre_hook(
            self._capture, with_kwargs=True
        )

    def _capture(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        cu_seqlens = kwargs.get("cu_seqlens")
        position_embeddings = kwargs.get("position_embeddings")
        if (
            not isinstance(hidden_states, torch.Tensor)
            or cu_seqlens is None
            or position_embeddings is None
        ):
            return None
        self.received, self.keys = chunked_received_attention(
            module,
            hidden_states,
            cu_seqlens,
            position_embeddings,
            attention_chunk=self.attention_chunk,
        )
        return None

    def consume(
        self, image_grid_thw: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        received, keys = self.received, self.keys
        self.received = None
        self.keys = None
        if received is None or keys is None:
            raise RuntimeError(
                f"{self.label} did not capture the last Qwen vision block before "
                "get_image_features returned."
            )
        merge_unit, reverse_indices = window_reverse_indices(
            self.visual,
            image_grid_thw,
            int(received.numel()),
            label=self.label,
            device=received.device,
        )
        scores = received.view(-1, merge_unit).mean(dim=-1)[reverse_indices]
        num_heads, _, head_dim = keys.shape
        metric = keys.view(
            num_heads, -1, merge_unit, head_dim
        ).mean(dim=2)
        metric = metric[:, reverse_indices, :].mean(dim=0)
        split_sizes = [
            int(value)
            for value in (
                image_grid_thw.prod(dim=-1) // int(merge_unit)
            ).tolist()
        ]
        return (
            list(torch.split(scores, split_sizes)),
            list(torch.split(metric, split_sizes)),
        )

    def remove(self) -> None:
        self.handle.remove()


def qwen_valid_length(
    row_length: int, mask_row: torch.Tensor | None
) -> int:
    if mask_row is None:
        return int(row_length)
    return int(mask_row.to(torch.long).sum().item())


def pad_qwen_training_rows(
    hidden_rows: list[torch.Tensor],
    mrope_rows: list[torch.Tensor],
    id_rows: list[torch.Tensor],
    mask_rows: list[torch.Tensor] | None,
    label_rows: list[torch.Tensor] | None,
    *,
    pad_token_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Right-pad independently reduced Qwen rows back to one microbatch."""

    max_length = max(int(row.shape[0]) for row in hidden_rows)

    def pad_hidden(row: torch.Tensor) -> torch.Tensor:
        pad = max_length - int(row.shape[0])
        if pad == 0:
            return row
        return torch.cat(
            (row, row.new_zeros((pad, row.shape[-1]))), dim=0
        )

    def pad_mrope(row: torch.Tensor) -> torch.Tensor:
        pad = max_length - int(row.shape[-1])
        if pad == 0:
            return row
        return torch.cat(
            (row, row.new_zeros((row.shape[0], pad))), dim=-1
        )

    def pad_vector(row: torch.Tensor, value: int | bool) -> torch.Tensor:
        pad = max_length - int(row.shape[0])
        if pad == 0:
            return row
        return torch.cat((row, row.new_full((pad,), value)), dim=0)

    hidden_states = torch.stack(
        [pad_hidden(row) for row in hidden_rows], dim=0
    )
    mrope_ids = torch.stack(
        [pad_mrope(row) for row in mrope_rows], dim=1
    )
    input_ids = torch.stack(
        [pad_vector(row, pad_token_id) for row in id_rows], dim=0
    )
    ragged = any(
        int(row.shape[0]) != max_length for row in hidden_rows
    )
    if mask_rows is None and ragged:
        mask_rows = [
            torch.ones(
                row.shape[0], dtype=torch.bool, device=row.device
            )
            for row in hidden_rows
        ]
    attention_mask = (
        torch.stack(
            [pad_vector(row.to(torch.bool), False) for row in mask_rows],
            dim=0,
        )
        if mask_rows is not None
        else None
    )
    labels = (
        torch.stack(
            [pad_vector(row, ignore_index) for row in label_rows], dim=0
        )
        if label_rows is not None
        else None
    )
    return hidden_states, mrope_ids, input_ids, attention_mask, labels


def qwen_training_row_views(
    hidden_states: torch.Tensor,
    mrope_ids: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    row: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    mask_row = attention_mask[row] if attention_mask is not None else None
    valid = qwen_valid_length(int(hidden_states.shape[1]), mask_row)
    return (
        hidden_states[row, :valid],
        mrope_ids[:, row, :valid],
        input_ids[row, :valid],
        mask_row[:valid] if mask_row is not None else None,
        labels[row, :valid] if labels is not None else None,
    )


def qwen_visual_spans(
    input_ids: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
    *,
    label: str = "Qwen training",
) -> tuple[QwenVisualSpan | None, ...]:
    # Locate one contiguous visual-token span in each valid batch row.
    spans: list[QwenVisualSpan | None] = []
    for row in range(int(input_ids.shape[0])):
        valid = qwen_valid_length(
            int(input_ids.shape[1]),
            attention_mask[row] if attention_mask is not None else None,
        )
        positions = torch.where(
            input_ids[row, :valid] == int(image_token_id)
        )[0]
        if positions.numel() == 0:
            spans.append(None)
            continue
        start = int(positions[0].item())
        end = int(positions[-1].item()) + 1
        if int(positions.numel()) != end - start:
            raise NotImplementedError(
                f"{label} expects one contiguous visual-token span per row; "
                f"row {row} has {int(positions.numel())} placeholders spread "
                f"over [{start}, {end})."
            )
        spans.append(QwenVisualSpan(start, end))
    return tuple(spans)
