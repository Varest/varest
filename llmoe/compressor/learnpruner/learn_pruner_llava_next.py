"""LLaVA-NeXT lifecycle for LearnPruner's post-packed AnyRes adaptation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING


from llmoe.compressor.llava_next import (
    image_sequences,
    is_llava_next,
    placeholder_run_lengths,
    tokenizer_pad_token_id,
    trim_image_placeholders_to_lengths,
)

if TYPE_CHECKING:
    from llmoe.compressor.learnpruner.learn_pruner import LearnPrunerCompressor


class LlavaNextLearnPrunerAdapter:
    """Run the LPM once over each complete native packed image sequence.

    The wrapped native ``get_image_features`` performs CLIP encoding, projection,
    tile-grid reconstruction, spatial unpadding, and newline insertion first.
    LearnPruner then sees exactly the sequence consumed by the decoder. This is
    intentionally image-global: no tile receives an independent quota or LPM
    decision.
    """

    def __init__(self, compressor: LearnPrunerCompressor) -> None:
        self.compressor = compressor
        self._vlm: Any | None = None
        self._inner: Any | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._pending: tuple[int, int, int] | None = None

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_llava_next(vlm)

    def install(self, vlm: Any) -> None:
        if self._vlm is vlm:
            return
        if self._vlm is not None:
            raise RuntimeError(
                "LearnPruner's LLaVA-NeXT adapter is already installed on another backbone"
            )
        if not self.supports(vlm):
            raise TypeError("LearnPruner's packed AnyRes adapter needs llava-next-8b")
        inner = getattr(getattr(vlm, "model", None), "model", None)
        original_get = getattr(inner, "get_image_features", None)
        if not callable(original_get):
            raise TypeError("LearnPruner/LLaVA-NeXT needs model.model.get_image_features")

        self._vlm = vlm
        self._inner = inner
        self._original_get_image_features = original_get
        self._original_prepare_inputs = vlm.prepare_inputs

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            inputs = self._original_prepare_inputs(*args, **kwargs)
            if self.compressor.for_training:
                return inputs
            lengths = placeholder_run_lengths(inputs, token_id=int(self.compressor._image_token_id))
            self._pending = None
            if not lengths:
                return inputs
            if len(lengths) != 1:
                raise NotImplementedError(
                    "LearnPruner/LLaVA-NeXT inference requires batch_size=1 with "
                    "exactly one contiguous image span."
                )
            native = lengths[0]
            stage1, stage2, _ = self.compressor.use_next_stage_counts(native)
            self._pending = native, stage1, stage2
            return trim_image_placeholders_to_lengths(
                inputs,
                token_id=int(self.compressor._image_token_id),
                lengths=(stage1,),
                pad_token_id=tokenizer_pad_token_id(vlm),
            )

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            # Native NeXT packs every tile before this returns. Calling it first
            # is the architectural boundary this adapter must preserve.
            outputs = self._original_get_image_features(*args, **kwargs)
            sequences = tuple(image_sequences(outputs))
            if self.compressor.for_training:
                self.compressor.set_training_features(sequences)
                return outputs
            if len(sequences) != 1:
                raise NotImplementedError(
                    "LearnPruner/LLaVA-NeXT inference requires one packed image sequence"
                )

            sequence = sequences[0]
            native = int(sequence.shape[0])
            pending, self._pending = self._pending, None
            stage1, stage2, _ = self.compressor.use_next_stage_counts(native)
            if pending is not None and pending != (native, stage1, stage2):
                raise RuntimeError(
                    "LearnPruner's packed AnyRes feature length does not match its "
                    f"placeholder run: prepared={pending}, packed={(native, stage1, stage2)}."
                )
            selected = self.compressor.compress(sequence).tokens
            if selected.ndim != 2 or int(selected.shape[0]) != stage1:
                raise RuntimeError(
                    f"LearnPruner selected {tuple(selected.shape)} from {native} packed "
                    f"LLaVA-NeXT rows; stage 1 requires {stage1}."
                )
            outputs.pooler_output = [selected]
            return outputs

        inner.get_image_features = wrapped_get_image_features
        vlm.prepare_inputs = wrapped_prepare_inputs

    def uninstall(self) -> None:
        if self._inner is not None and self._original_get_image_features is not None:
            self._inner.get_image_features = self._original_get_image_features
        if self._vlm is not None and self._original_prepare_inputs is not None:
            self._vlm.prepare_inputs = self._original_prepare_inputs
        self._vlm = None
        self._inner = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._pending = None
