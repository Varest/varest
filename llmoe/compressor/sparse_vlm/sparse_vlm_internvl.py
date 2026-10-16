"""InternVL runtime policy for SparseVLM."""

from typing import Any

from llmoe.compressor.internvl import is_internvl


class InternVLSparseVLMAdapter:
    """Describe SparseVLM's concatenate-first InternVL decoder path.

    SparseVLM already prunes the visual columns inside the language model, so
    InternVL needs no alternate tensor kernel. The adapter owns backbone
    detection and the policy metadata that differs from the fixed-grid LLaVA
    path, keeping those branches out of ``sparse_vlm.py``.
    """

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_internvl(vlm)

    @property
    def label(self) -> str:
        return "SparseVLM/InternVL"

    @property
    def install_note(self) -> str:
        return ", targets apply once to the globally concatenated tile span"
