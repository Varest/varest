"""InternVL policy adapter for FastV's decoder-stage pruning."""

from typing import Any

from llmoe.compressor.internvl import is_internvl


class InternVLFastVAdapter:
    """Resolve FastV's layer-average keep against one concatenated tile span."""

    def __init__(self, *, max_tokens: int, k: int, decoder_layers: int) -> None:
        self._max_tokens = int(max_tokens)
        self._k = int(k)
        self._decoder_layers = int(decoder_layers)
        self._warned_unreachable = False

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_internvl(vlm)

    @property
    def label(self) -> str:
        return "FastV/InternVL"

    @property
    def install_note(self) -> str:
        return ", dynamic global keep resolved from the concatenated InternVL span"

    def resolve_keep(self, n_tokens: int) -> int:
        # Back-solve the fixed layer-average budget from this image's one
        # concatenated dynamic tile sequence. If the unpruned shallow layers
        # alone exceed the label, one token is the closest reachable result.
        numerator = self._decoder_layers * self._max_tokens - self._k * n_tokens
        keep = numerator // (self._decoder_layers - self._k)
        if keep < 1 and not self._warned_unreachable:
            self._warned_unreachable = True
            floor = self._k * n_tokens / self._decoder_layers
            print(
                f"[FastV/InternVL] WARNING: b{self._max_tokens} is unreachable for "
                f"{n_tokens} globally concatenated image tokens at k={self._k}; "
                f"the unpruned-layer floor is {floor:.2f}. Keeping one token.",
                flush=True,
            )
        return max(1, min(int(n_tokens), int(keep)))
