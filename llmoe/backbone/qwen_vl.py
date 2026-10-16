"""Qwen2.5-VL backbone loading and input preprocessing."""

from collections.abc import Sequence
from typing import Any, override

import torch
from mipcandy import Device, auto_device
from transformers import AutoProcessor, PreTrainedModel, ProcessorMixin

from llmoe.backbone.prototype import ImagePreprocessor, PromptPreprocessor, VLM


class QwenVLImagePreprocessor(ImagePreprocessor):
    @override
    def forward(self, vlm: VLM,
                images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError(f"Expected 4 dimensions (2D), got {images.ndim}")
            return list(images.unbind(0))
        return list(images)


class QwenVLPromptPreprocessor(PromptPreprocessor):
    @override
    def forward(self, vlm: VLM, prompts: Sequence[str]) -> list[str]:
        processor = vlm.processor
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for prompt in prompts
        ]
        return [
            processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]


class QwenVL(VLM):
    weights_key = "qwen-vl"

    def __init__(self, *, model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 image_preprocessor: ImagePreprocessor | None = None,
                 prompt_preprocessor: PromptPreprocessor | None = None,
                 local_files_only: bool = True, device: Device = auto_device(),
                 dtype: torch.dtype = torch.float16,
                 attn_implementation: str | None = "sdpa",
                 min_pixels: int | None = None, max_pixels: int | None = None) -> None:
        # Qwen's NaViT encoder emits a token count that follows the image's own
        # resolution, so a token budget is set by capping pixels: each visual
        # token covers patch_size * spatial_merge_size = 28 px square, i.e. 784
        # pixels, so `max_pixels = tokens * 784`. The cap is an upper bound --
        # a smaller image still yields fewer tokens.
        self._min_pixels: int | None = None if min_pixels is None else int(min_pixels)
        self._max_pixels: int | None = None if max_pixels is None else int(max_pixels)
        super().__init__(
            model_name_or_path,
            image_preprocessor or QwenVLImagePreprocessor(),
            prompt_preprocessor or QwenVLPromptPreprocessor(),
            local_files_only=local_files_only,
            device=device,
            dtype=dtype,
        )
        self._attn_implementation = attn_implementation

    @override
    def load_model(self) -> tuple[ProcessorMixin, PreTrainedModel]:
        from transformers import Qwen2_5_VLForConditionalGeneration

        processor_kwargs: dict[str, Any] = {}
        if self._min_pixels is not None:
            processor_kwargs["min_pixels"] = self._min_pixels
        if self._max_pixels is not None:
            processor_kwargs["max_pixels"] = self._max_pixels
        processor = AutoProcessor.from_pretrained(
            self._model_id, local_files_only=self._local_files_only, **processor_kwargs
        )
        if not isinstance(processor, ProcessorMixin):
            raise TypeError(f"Expected an instance of `ProcessorMixin`, got {type(processor)}")

        model_kwargs: dict[str, Any] = {
            "dtype": self._dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self._local_files_only,
        }
        if self._attn_implementation is not None:
            model_kwargs["attn_implementation"] = self._attn_implementation
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_id, **model_kwargs
        ).to(self._device).eval()
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"Expected an instance of `PreTrainedModel`, got {type(model)}")
        return processor, model
