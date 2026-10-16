"""InternVL adapter for DivPrune's projected-feature selector."""

from typing import Any

from llmoe.compressor.internvl import InternVLProjectedFeatureAdapter


class InternVLDivPruneAdapter(InternVLProjectedFeatureAdapter):
    """Apply DivPrune once to each source image's concatenated tile sequence."""

    def __init__(self, compressor: Any) -> None:
        super().__init__(
            label="DivPrune/InternVL",
            compress_one=lambda tokens, _index: compressor.compress(tokens).tokens,
            keep_for_run=compressor._resolve_keep,
        )
