from typing import override, Sequence

import torch
from mipcandy import Device, auto_device
from transformers import AutoProcessor, PreTrainedModel, LlavaForConditionalGeneration, ProcessorMixin, \
    LlavaImageProcessor

from llmoe.backbone.prototype import IdentityImagePreprocessor, ImagePreprocessor, PromptPreprocessor, VLM


class LLaVAPromptPreprocessor(PromptPreprocessor):
    def __init__(self) -> None:
        self.prompt_template: str = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: <image>\n{prompt} ASSISTANT:"
        )

    @override
    def forward(self, vlm: VLM, prompts: Sequence[str]) -> list[str]:
        return [self.prompt_template.format(prompt=prompt) for prompt in prompts]


class LLaVA(VLM):
    weights_key = "llava"

    def __init__(self, *, model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
                 image_preprocessor: ImagePreprocessor | None = None,
                 prompt_preprocessor: PromptPreprocessor | None = None, local_files_only: bool = True,
                 device: Device = auto_device(), dtype: torch.dtype = torch.float16) -> None:
        super().__init__(model_name_or_path, image_preprocessor or IdentityImagePreprocessor(),
                         prompt_preprocessor or LLaVAPromptPreprocessor(), local_files_only=local_files_only,
                         device=device, dtype=dtype)

    @override
    def load_model(self) -> tuple[ProcessorMixin, PreTrainedModel]:
        processor = AutoProcessor.from_pretrained(self._model_id, local_files_only=self._local_files_only)
        if not isinstance(processor, ProcessorMixin):
            raise TypeError(f"Expected an instance of `ProcessorMixin`, got {type(processor)}")
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise TypeError("Expected LLaVA processor to have an `image_processor`.")
        image_processor_config = image_processor.to_dict()
        image_processor_config.pop("do_pad", None)
        image_processor_config["image_processor_type"] = "LlavaImageProcessor"
        processor.image_processor = LlavaImageProcessor(**image_processor_config, do_pad=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            self._model_id, dtype=self._dtype, low_cpu_mem_usage=True, local_files_only=self._local_files_only
        ).to(self._device).eval()
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"Expected an instance of `PreTrainedModel`, got {type(model)}")
        return processor, model


class LLaVA13B(LLaVA):
    """LLaVA-1.5 with the 13B Vicuna language-model backbone.

    Identical to :class:`LLaVA` in everything a compressor hooks -- the same CLIP
    tower, the same 576-token square grid, the same
    ``LlavaForConditionalGeneration`` layout -- so the structural backbone checks
    every compressor makes still route it down the LLaVA path. The two
    differences are in the decoder: 40 layers instead of 32, and a 5120-wide
    hidden state instead of 4096.

    Both are load-bearing, and neither can be read off the model *class*:

    * Layer-scheduled compressors state their budgets as a layer average over the
      decoder, so their published boundaries have to be re-solved for 40 layers.
      They do that from the loaded model's own depth via
      :mod:`llmoe.utils.depth`, which is why this class needs no per-method
      special-casing.
    * Learned projectors and adapters are 4096-wide and do not fit. A distinct
      :attr:`weights_key` keeps the 7B artifacts from being resolved onto this
      backbone. Where a genuine 13B release exists (HiDrop, QuietPrune) it is
      registered under this key; asking for one that does not fails naming what
      does exist instead of silently loading the 7B weights.
    """

    weights_key = "llava-13b"
    def __init__(self, *, model_name_or_path: str = "llava-hf/llava-1.5-13b-hf",
                 image_preprocessor: ImagePreprocessor | None = None,
                 prompt_preprocessor: PromptPreprocessor | None = None, local_files_only: bool = True,
                 device: Device = auto_device(), dtype: torch.dtype = torch.float16) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            image_preprocessor=image_preprocessor,
            prompt_preprocessor=prompt_preprocessor,
            local_files_only=local_files_only,
            device=device,
            dtype=dtype,
        )
