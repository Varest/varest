from pathlib import Path
from typing import Any

import torch
from torchvision.io import ImageReadMode, decode_image, read_image


def coerce_image_tensor(image: Any, *, source: str) -> torch.Tensor:
    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            image = image["bytes"]
        elif "path" in image and image["path"] is not None:
            image = image["path"]
        else:
            raise TypeError(f"Unrecognized {source} image dict keys: {sorted(image.keys())}")

    if isinstance(image, (str, Path)):
        tensor = read_image(str(image), mode=ImageReadMode.RGB)
    elif isinstance(image, (bytes, bytearray)):
        tensor = decode_image(torch.frombuffer(bytearray(image), dtype=torch.uint8), mode=ImageReadMode.RGB)
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        raise TypeError(f"Unrecognized {source} image type: {type(image)}")

    if tensor.ndim != 3:
        raise ValueError(f"Expected {source} image tensor to have 3 dimensions, got {tensor.ndim}")
    if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    elif tensor.shape[0] == 4:
        tensor = tensor[:3]
    elif tensor.shape[0] != 3:
        raise ValueError(f"Expected {source} image tensor to have 1, 3, or 4 channels, got {tensor.shape[0]}")

    if tensor.dtype.is_floating_point:
        if tensor.numel() and float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0:
            tensor = tensor * 255
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    elif tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    return tensor.contiguous()
