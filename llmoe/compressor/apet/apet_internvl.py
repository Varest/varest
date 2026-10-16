"""InternVL adapter for ApET's projected-feature stage."""

from typing import Any

from llmoe.compressor.internvl import InternVLProjectedFeatureAdapter


class InternVLApETAdapter(InternVLProjectedFeatureAdapter):
    """Run ApET once over every source image's concatenated dynamic tiles.

    The decoder cut is not handled here: ``install`` moves it to the same fraction
    of whatever decoder is loaded (``llmoe.utils.depth.scale_layer``), which covers
    InternVL's Qwen decoder and LLaVA-1.5-13B alike.
    """

    def __init__(self, compressor: Any) -> None:
        super().__init__(
            label="ApET/InternVL",
            compress_one=lambda tokens, _index: compressor.compress(tokens).tokens,
            keep_for_run=lambda run_length: min(compressor._vision_budget, run_length),
        )
