"""InternVL schedule adapter for PyramidDrop."""

from collections.abc import Callable
from typing import Any

from llmoe.compressor.internvl import is_internvl


class InternVLPyramidDropAdapter:
    """Derive a global stage schedule from each concatenated tile sequence."""

    def __init__(
        self,
        *,
        max_tokens: int,
        decoder_layers: int,
        dynamic: bool,
        schedule_solver: Callable[[float, int], tuple[tuple[int, ...], tuple[float, ...]]],
    ) -> None:
        self._max_tokens = int(max_tokens)
        self._decoder_layers = int(decoder_layers)
        self._dynamic = bool(dynamic)
        self._schedule_solver = schedule_solver
        if self._dynamic and self._decoder_layers < 4:
            raise ValueError(
                f"PyramidDrop needs at least four decoder layers, got {self._decoder_layers}"
            )

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_internvl(vlm)

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @property
    def label(self) -> str:
        return "PyramidDrop/InternVL"

    def configure(
        self, visual_count: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, int]]:
        if not self._dynamic:
            raise RuntimeError("A caller-supplied PyramidDrop schedule is already resolved")
        if visual_count < 1:
            raise ValueError("PyramidDrop cannot schedule an empty InternVL span")
        ratio = min(1.0, self._max_tokens / float(visual_count))
        try:
            boundaries, stage_ratios = self._schedule_solver(ratio, self._decoder_layers)
        except ValueError as error:
            floor = (visual_count + self._decoder_layers - 1) / self._decoder_layers
            raise ValueError(
                f"PyramidDrop/InternVL b{self._max_tokens} is unreachable for "
                f"{visual_count} globally concatenated tokens on a "
                f"{self._decoder_layers}-layer decoder; the one-full-layer "
                f"floor is {floor:.2f}."
            ) from error

        targets = tuple(
            max(1, min(visual_count, int(visual_count * stage_ratio)))
            for stage_ratio in stage_ratios
        )
        counts = (visual_count, *targets)
        stages = {layer: stage for stage, layer in enumerate(boundaries)}
        return boundaries, counts, stages
