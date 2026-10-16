"""InternVL adapter for ToMe's projected-feature merger."""

from typing import Any

from llmoe.compressor.internvl import InternVLProjectedFeatureAdapter


class InternVLToMeAdapter(InternVLProjectedFeatureAdapter):
    """Merge one globally concatenated tile sequence per source image."""

    def __init__(self, compressor: Any) -> None:
        super().__init__(
            label="ToMe/InternVL",
            compress_one=lambda tokens, _index: compressor.compress(tokens).tokens,
            keep_for_run=lambda run_length: min(compressor._max_tokens, run_length),
        )
