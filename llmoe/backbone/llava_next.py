"""LLaVA-NeXT backbone loading and native AnyRes input preprocessing."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

import torch
from mipcandy import Device, auto_device
from transformers import AutoProcessor, PreTrainedModel, ProcessorMixin

from llmoe.backbone.prototype import ImagePreprocessor, PromptPreprocessor, VLM


LLAVA_NEXT_8B_MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
LLAVA_NEXT_8B_REVISION = "b041c0d0ea0dd0196d147206c210c8d1752fc2da"


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    value = getattr(torch, str(dtype).removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported torch dtype {dtype!r}")
    return value


class LLaVANextImagePreprocessor(ImagePreprocessor):
    """Hand the processor one source image at a time for AnyRes crop selection.

    LLaVA-NeXT chooses a different number and arrangement of ``336 x 336``
    crops from each image's native aspect ratio.  Splitting a tensor batch into
    individual images preserves those source-image boundaries for the processor.
    """

    @override
    def forward(
        self,
        vlm: VLM,
        images: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        del vlm
        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError(f"Expected 4 dimensions (2D), got {images.ndim}")
            return list(images.unbind(0))
        return list(images)


class LLaVANextPromptPreprocessor(PromptPreprocessor):
    """Render one image-plus-text turn with the checkpoint's Llama 3 template."""

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
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]


class LLaVANext8B(VLM):
    """Llama-3-LLaVA-NeXT-8B through Transformers' native NeXT model.

    This is deliberately separate from :class:`~llmoe.backbone.llava.LLaVA`.
    NeXT uses AnyRes crops, spatial unpadding and newline tokens, so its visual
    sequence is variable-length and cannot use the fixed 576-token assumptions
    of LLaVA-1.5 compressors without a method-specific adapter.
    """

    # Compressor adapters use an explicit marker because LLaVA-1.5, NeXT, and
    # InternVL all expose similarly named vision_tower/projector modules.
    compressor_family: str = "llava_next"
    weights_key: str = "llava-next-8b"

    def __init__(
        self,
        *,
        model_name_or_path: str = LLAVA_NEXT_8B_MODEL_ID,
        model_revision: str | None = LLAVA_NEXT_8B_REVISION,
        image_preprocessor: ImagePreprocessor | None = None,
        prompt_preprocessor: PromptPreprocessor | None = None,
        local_files_only: bool = True,
        device: Device = auto_device(),
        dtype: torch.dtype | str = torch.float16,
        attn_implementation: str | None = "sdpa",
        device_map: str | None = None,
        max_memory_per_gpu: str | None = None,
    ) -> None:
        super().__init__(
            model_name_or_path,
            image_preprocessor or LLaVANextImagePreprocessor(),
            prompt_preprocessor or LLaVANextPromptPreprocessor(),
            local_files_only=local_files_only,
            device=device,
            dtype=_resolve_dtype(dtype),
        )
        self._model_revision: str | None = model_revision
        self._attn_implementation: str | None = attn_implementation
        # When set (e.g. "auto"), accelerate shards the decoder across the
        # visible GPUs. LearnPruner's training attention retains fp32
        # [1, H, L, L] tensors in EVERY layer, so a 2880-token AnyRes span
        # cannot fit one 80GB card; sharding splits that retention per device.
        self._device_map: str | None = device_map
        # device_map alone does NOT shard: accelerate fills each GPU to capacity,
        # so a ~16GB model lands entirely on cuda:0 (verified with
        # infer_auto_device_map). Capping per-GPU weight memory forces the decoder
        # layers to spread, which is what splits the fp32 attention retention.
        self._max_memory_per_gpu: str | None = max_memory_per_gpu

    @override
    def validate_compressor(self, compressor: Any) -> None:
        validator = getattr(compressor, "validate_llava_next", None)
        if callable(validator):
            validator()
            return
        if isinstance(compressor, str):
            # Only the training and tuning entry points validate by registry key:
            # they build the compressor afterwards, so no instance exists yet, and
            # a `str` never carries `validate_llava_next`. That made the gate
            # reject every method by name regardless of support, which is why
            # "VisionZip++" was special-cased here.
            #
            # A name is accepted when it is both adapted for this backbone and
            # something this repository can train or tune. Both halves matter:
            # `validate_llava_next` alone means the method runs on NeXT at
            # evaluation, not that a trainer exists -- Nuwa and eight others are
            # inference-only, and naming one here is still an error.
            #
            # The knob-level checks are not skipped: the trainers call
            # `validate_compressor` again with the built compressor.
            from llmoe.compressor import (  # local: avoids an import cycle
                COMPRESSORS,
                COMPRESSOR_TRAINERS,
                COMPRESSOR_TUNERS,
            )

            registered = COMPRESSORS.get(compressor)
            if (
                registered is not None
                and callable(getattr(registered, "validate_llava_next", None))
                and (compressor in COMPRESSOR_TRAINERS or compressor in COMPRESSOR_TUNERS)
            ):
                return
        compressor_name = (
            compressor if isinstance(compressor, str) else compressor.__class__.__name__
        )
        raise NotImplementedError(
            f"{compressor_name} has not been adapted for llava-next-8b. "
            "LLaVA-NeXT uses a variable-length AnyRes visual sequence with "
            "spatial unpadding and image-newline tokens; run the full-token "
            "backbone without a compressor until a method-specific adapter is available."
        )

    def _revision_kwargs(self) -> dict[str, str]:
        if self._model_revision and not Path(self._model_id).expanduser().is_dir():
            return {"revision": str(self._model_revision)}
        return {}

    @override
    def load_model(self) -> tuple[ProcessorMixin, PreTrainedModel]:
        from transformers import LlavaNextForConditionalGeneration

        revision_kwargs = self._revision_kwargs()
        processor = AutoProcessor.from_pretrained(
            self._model_id,
            local_files_only=self._local_files_only,
            **revision_kwargs,
        )
        if not isinstance(processor, ProcessorMixin):
            raise TypeError(f"Expected an instance of `ProcessorMixin`, got {type(processor)}")

        model_kwargs: dict[str, Any] = {
            "dtype": self._dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self._local_files_only,
            **revision_kwargs,
        }
        if self._attn_implementation is not None:
            model_kwargs["attn_implementation"] = self._attn_implementation
        if self._device_map is not None:
            model_kwargs["device_map"] = self._device_map
            if self._max_memory_per_gpu is not None:
                import torch as _torch
                count = _torch.cuda.device_count()
                if count < 2:
                    raise RuntimeError(
                        "max_memory_per_gpu needs >=2 visible GPUs; saw "
                        f"{count}. Request more GPUs or drop device_map."
                    )
                model_kwargs["max_memory"] = {
                    index: self._max_memory_per_gpu for index in range(count)
                }
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self._model_id,
            **model_kwargs,
        )
        # `.to(device)` would collapse an accelerate device_map back onto one GPU.
        model = (model if self._device_map is not None else model.to(self._device)).eval()
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"Expected an instance of `PreTrainedModel`, got {type(model)}")

        # Modern converted checkpoints carry these fields in processor_config,
        # but populate missing values from the model so local conversions still
        # expand each <image> placeholder to the correct AnyRes sequence length.
        config = model.config
        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = int(config.vision_config.patch_size)
        if getattr(processor, "vision_feature_select_strategy", None) is None:
            processor.vision_feature_select_strategy = config.vision_feature_select_strategy
        if getattr(processor, "num_additional_image_tokens", None) is None:
            processor.num_additional_image_tokens = 1
        return processor, model
