"""InternVL adapter for OTPrune's projected-feature selector."""

from typing import Any

from llmoe.compressor.internvl import InternVLProjectedFeatureAdapter


class InternVLOTPruneAdapter(InternVLProjectedFeatureAdapter):
    """Apply OTPrune once to each source image's concatenated tile sequence."""

    def __init__(self, compressor: Any) -> None:
        super().__init__(
            label="OTPrune/InternVL",
            compress_one=lambda tokens, _index: compressor.compress(tokens).tokens,
            keep_for_run=lambda run_length: min(compressor._max_tokens, run_length),
        )
