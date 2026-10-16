from abc import ABCMeta, abstractmethod
from contextlib import nullcontext
from typing import Any, Sequence

import torch
from mipcandy import HasDevice, Device, auto_device
from transformers import ProcessorMixin, PreTrainedModel, BatchFeature


class ImagePreprocessor(object, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, vlm: "VLM", images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def __call__(self, vlm: "VLM", images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        return self.forward(vlm, images)


class IdentityImagePreprocessor(ImagePreprocessor):
    def forward(self, vlm: "VLM", images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        return images if isinstance(images, torch.Tensor) else list(images)


class PromptPreprocessor(object, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, vlm: "VLM", prompts: Sequence[str]) -> list[str]:
        raise NotImplementedError

    def __call__(self, vlm: "VLM", prompts: Sequence[str]) -> list[str]:
        return self.forward(vlm, prompts)


class VLM(HasDevice, metaclass=ABCMeta):
    # The `--backbone` key this class is registered under. Weight artifacts are keyed by
    # it, so a compressor asks the backbone what it is rather than inferring it from the
    # loaded model -- two backbones can share an architecture and still need different
    # weights (LLaVA-1.5 7B and 13B differ only in decoder depth and width).
    weights_key: str = ""
    # Benchmarks read prompt suffixes from the selected backbone, not from model-specific globals.
    short_answer_suffix: str = "\nAnswer the question using a single word or phrase."
    option_letter_suffix: str = "\nAnswer with the option's letter from the given choices directly."
    direct_answer_suffix: str = "\nAnswer the question directly."

    def __init__(self, model_name_or_path: str, image_preprocessor: ImagePreprocessor,
                 prompt_preprocessor: PromptPreprocessor, *, local_files_only: bool = True,
                 device: Device = auto_device(), dtype: torch.dtype = torch.float16) -> None:
        super().__init__(device)
        self._model_id: str = model_name_or_path
        self._image_preprocessor: ImagePreprocessor = image_preprocessor
        self._prompt_preprocessor: PromptPreprocessor = prompt_preprocessor
        self._local_files_only: bool = local_files_only
        self._dtype: torch.dtype = dtype
        self._processor: ProcessorMixin | None = None
        self._model: PreTrainedModel | None = None

    def _lazy_load_model(self) -> None:
        if self._processor and self._model:
            return
        if (self._processor is None) ^ (self._model is None):
            raise ValueError("Unexpected error: processor and model must be loaded together")
        self._processor, self._model = self.load_model()
        self._configure_generation_tokenizer()

    def _configure_generation_tokenizer(self) -> None:
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is None:
                    raise ValueError("Tokenizer needs a pad token or eos token for batched generation.")
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            self._model.generation_config.pad_token_id = tokenizer.pad_token_id

    @property
    def processor(self) -> ProcessorMixin:
        self._lazy_load_model()
        return self._processor

    @property
    def model(self) -> PreTrainedModel:
        self._lazy_load_model()
        return self._model

    @abstractmethod
    def load_model(self) -> tuple[ProcessorMixin, PreTrainedModel]:
        raise NotImplementedError

    def validate_compressor(self, compressor: Any) -> None:
        """Reject compressor/backbone combinations that need an explicit adapter.

        Most backbones keep the permissive default. Architecturally distinct
        backbones can override this hook so structurally similar module names do
        not make an incompatible compressor appear supported.
        """
        del compressor

    def vision_tokens_per_layer(self) -> list[int] | None:
        """Vision tokens entering each decoder layer during the last generate call's prefill, if the backbone knows.

        ``None`` (the default) leaves the count to tracing, which finds the vision tokens through the image tokens of
        the prompt. Compressors that feed the decoder their own embeddings, so no image token ever reaches it, install
        an override (``varest_bench`` uses it for the theoretical FLOPs).
        """
        return None

    def format_prompts(self, prompts: Sequence[str]) -> list[str]:
        return self._prompt_preprocessor(self, prompts)

    def format_images(self, images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        return self._image_preprocessor(self, images)

    def prepare_inputs(self, images: torch.Tensor | Sequence[torch.Tensor], prompts: Sequence[str], *,
                       pixel_dtype: torch.dtype | None = None) -> BatchFeature:
        formatted_prompts = self.format_prompts(prompts)
        formatted_images = self.format_images(images)
        processor = self.processor
        self._configure_generation_tokenizer()
        inputs = processor(images=formatted_images, text=formatted_prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(self._device)
        if pixel_dtype and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(pixel_dtype)
        return inputs

    def generate(self, inputs: BatchFeature, *, max_new_tokens: int = 256,
                 generation_profiler: Any | None = None, **kwargs) -> torch.Tensor:
        self._configure_generation_tokenizer()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        model = self.model
        measure_generate = getattr(generation_profiler, "measure_generate", None)
        generate_context = (
            measure_generate(model)
            if callable(measure_generate)
            else nullcontext()
        )
        with torch.inference_mode():
            with generate_context:
                return model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **{"do_sample": False, **kwargs},
                )

    def decode_output_ids(self, output_ids: torch.Tensor, input_ids: torch.Tensor) -> list[str]:
        prompt_width = input_ids.shape[-1]
        generated_ids = output_ids[:, prompt_width:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def forward(self, images: torch.Tensor | Sequence[torch.Tensor], prompts: Sequence[str], *,
                max_new_tokens: int = 256, generation_profiler: Any | None = None,
                **kwargs) -> list[str]:
        """
        :param images: [B, C, H, W] or a sequence of [C, H, W]
        :param prompts: a sequence of string prompts
        :param max_new_tokens: max number of tokens to generate
        :param kwargs: additional arguments for generation
        :return: a list of decoded output strings
        """
        if isinstance(images, torch.Tensor) and images.ndim != 4:
            raise ValueError(f"Expected 4 dimensions (2D), got {images.ndim}")
        if len(images) != len(prompts):
            raise ValueError(f"Unmatched number of images and prompts: {len(images)} != {len(prompts)}")
        inputs = self.prepare_inputs(images, prompts, pixel_dtype=self._dtype)
        output_ids = self.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            generation_profiler=generation_profiler,
            **kwargs,
        )
        return self.decode_output_ids(output_ids, inputs["input_ids"])

    def __call__(self, images: torch.Tensor | Sequence[torch.Tensor], prompts: str | Sequence[str], *,
                max_new_tokens: int = 256, generation_profiler: Any | None = None,
                **kwargs) -> list[str]:
        return self.forward(
            images,
            prompts,
            max_new_tokens=max_new_tokens,
            generation_profiler=generation_profiler,
            **kwargs,
        )
