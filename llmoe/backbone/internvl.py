from collections.abc import Sequence
from typing import Any, override

import torch
from mipcandy import Device, auto_device
from transformers import AutoProcessor, BatchFeature, PreTrainedModel, ProcessorMixin

from llmoe.backbone.prototype import VLM, ImagePreprocessor, PromptPreprocessor


class InternVLImagePreprocessor(ImagePreprocessor):
    """Hand the processor one image per prompt; it does its own dynamic tiling.

    InternVL cuts each image into up to ``max_patches`` tiles of ``448x448`` plus a
    thumbnail, so a stacked ``[B, C, H, W]`` batch has to be split before the
    processor sees it -- each image gets its own tile count.
    """

    @override
    def forward(self, vlm: VLM, images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError(f"Expected 4 dimensions (2D), got {images.ndim}")
            return list(images.unbind(0))
        return list(images)


class InternVLPromptPreprocessor(PromptPreprocessor):
    """Wrap each prompt in InternVL's chat template, one image before the text."""

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
            processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in conversations
        ]


class InternVL(VLM):
    """InternVL (3.x and 2.5, ``-hf`` checkpoints) through transformers.

    The tiled encoder gives each image its own visual-token count -- ``256`` per
    tile, ``max_patches`` tiles plus a thumbnail -- so compressors written against
    LLaVA-1.5's fixed 576-patch grid do not transfer, and batching images of
    different sizes changes the token count per row. Prefer ``batch_size: 1``.

    :param model_name_or_path: any ``InternVLForConditionalGeneration`` checkpoint.
    :param attn_implementation: passed to ``from_pretrained``; ``None`` leaves the
        checkpoint's own default. Compressors that need attention weights switch
        the relevant submodel to eager themselves.
    """

    weights_key = "internvl"

    def __init__(self, *, model_name_or_path: str = "OpenGVLab/InternVL3-8B-hf",
                 image_preprocessor: ImagePreprocessor | None = None,
                 prompt_preprocessor: PromptPreprocessor | None = None, local_files_only: bool = True,
                 device: Device = auto_device(), dtype: torch.dtype = torch.float16,
                 attn_implementation: str | None = "sdpa") -> None:
        super().__init__(model_name_or_path, image_preprocessor or InternVLImagePreprocessor(),
                         prompt_preprocessor or InternVLPromptPreprocessor(),
                         local_files_only=local_files_only, device=device, dtype=dtype)
        # The processor flattens every source image's dynamic tiles into one
        # pixel batch and removes its `num_patches` metadata. Preserve the same
        # boundaries from the untrimmed placeholder runs so compressors can
        # concatenate tiles per source image before making a selection.
        self._tile_counts: list[int] = []
        self._attn_implementation: str | None = attn_implementation

    @property
    def tile_counts(self) -> tuple[int, ...]:
        """Tile count for each source image in the most recently prepared batch."""

        return tuple(self._tile_counts)

    @override
    def prepare_inputs(self, images: torch.Tensor | Sequence[torch.Tensor],
                       prompts: Sequence[str], *,
                       pixel_dtype: torch.dtype | None = None) -> BatchFeature:
        inputs = super().prepare_inputs(images, prompts, pixel_dtype=pixel_dtype)
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            self._tile_counts = []
            return inputs

        token_id = int(self.model.config.image_token_id)
        per_tile = int(
            getattr(
                self.model.config,
                "image_seq_length",
                getattr(self.processor, "image_seq_length", 256),
            )
        )
        counts: list[int] = []
        for ids in input_ids:
            positions = (ids == token_id).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            splits = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=True)[0] + 1
            for run in torch.tensor_split(positions, splits.tolist()):
                length = int(run.numel())
                if length % per_tile:
                    raise RuntimeError(
                        f"InternVL produced an image-placeholder run of {length} tokens, "
                        f"which is not divisible by its {per_tile} tokens per tile."
                    )
                counts.append(length // per_tile)
        self._tile_counts = counts
        return inputs

    @override
    def load_model(self) -> tuple[ProcessorMixin, PreTrainedModel]:
        from transformers import InternVLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self._model_id, local_files_only=self._local_files_only)
        if not isinstance(processor, ProcessorMixin):
            raise TypeError(f"Expected an instance of `ProcessorMixin`, got {type(processor)}")

        model_kwargs: dict[str, Any] = {
            "dtype": self._dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self._local_files_only,
        }
        if self._attn_implementation is not None:
            model_kwargs["attn_implementation"] = self._attn_implementation
        model = InternVLForConditionalGeneration.from_pretrained(self._model_id, **model_kwargs).to(
            self._device
        ).eval()
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"Expected an instance of `PreTrainedModel`, got {type(model)}")
        return processor, model
