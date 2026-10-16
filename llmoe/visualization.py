"""Shared helpers for compressor-provided visualization views."""

from typing import Any

import torch


def pixel_values_to_canvas(pixel_values: torch.Tensor, image_processor: Any) -> torch.Tensor:
    """Convert one normalized processor image to a displayable CPU tensor."""

    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    image = pixel_values.detach().cpu().float() * std + mean
    return image.clamp(0, 1)
