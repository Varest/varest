from typing import Literal as _Literal

from llmoe.backbone.internvl import InternVLImagePreprocessor, InternVLPromptPreprocessor, InternVL
from llmoe.backbone.llava import LLaVAPromptPreprocessor, LLaVA, LLaVA13B
from llmoe.backbone.llava_next import (
    LLaVANext8B,
    LLaVANextImagePreprocessor,
    LLaVANextPromptPreprocessor,
)
from llmoe.backbone.prototype import IdentityImagePreprocessor, ImagePreprocessor, PromptPreprocessor, VLM
from llmoe.backbone.qwen_vl import QwenVLImagePreprocessor, QwenVLPromptPreprocessor, QwenVL
# CORE's model ships with its compressor; imported after VLM is bound above so
# the backbone/compressor import cycle resolves.
from llmoe.compressor.core.core import CoreLlama3VLM, CoreVLM, CoreVLM13B

type BackboneLiteral = _Literal[
    "llava", "llava-13b", "llava-next-8b", "qwen-vl", "internvl", "core",
    "core-13b", "core-llama3-8b"
]
BACKBONES: dict[str, type[VLM]] = {
    "llava": LLaVA,
    "llava-13b": LLaVA13B,
    "llava-next-8b": LLaVANext8B,
    "qwen-vl": QwenVL,
    "internvl": InternVL,
    # Its own vision stack (ConvNeXt-L + OMG-Seg) and Vicuna decoder; pairs with
    # the CORE compressor, which carries the merge strategy.
    "core": CoreVLM,
    # Same vision stack and merge strategy, Vicuna-13B decoder; its stage-2
    # release is keyed separately because the projector/LoRA are decoder-shaped.
    "core-13b": CoreVLM13B,
    # The same CORE ConvNeXt-L + OMG-Seg/Mask2Former architecture, with only
    # Vicuna replaced by the pinned Llama-3-8B-Instruct decoder/tokenizer.
    "core-llama3-8b": CoreLlama3VLM,
}
