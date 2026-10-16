from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from llmoe.backbone.prototype import VLM


@dataclass
class CompressedFeatures:
    tokens: torch.Tensor


@dataclass(frozen=True)
class VisualizationView:
    """A display canvas and the source columns that correspond to its patches."""

    name: str
    canvas: torch.Tensor
    source_columns: torch.Tensor | None = None
    canvas_columns: torch.Tensor | None = None


class TokenCompressor(nn.Module, metaclass=ABCMeta):
    def __init__(self, max_tokens: int) -> None:
        super().__init__()
        self._max_tokens: int = max_tokens
        # One [T_cur, T_orig] source tensor per compressor layer (see get_source_history).
        self._source_history: list[torch.Tensor] = []

    @abstractmethod
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None, layer_idx: int | None = None,
                 **kwargs) -> CompressedFeatures:
        raise NotImplementedError

    def get_source_history(self) -> list[torch.Tensor]:
        # One [T_cur, T_orig] tensor per layer (batch size 1, no batch dim): row i
        # (a current token) weights the original tokens it now subsumes.
        raise NotImplementedError

    def get_benchmark_record_metadata(self, batch_size: int) -> list[dict[str, Any]]:
        """Return one optional result-record extension per evaluated sample."""

        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        return [{} for _ in range(int(batch_size))]

    def get_visualization_views(
        self,
        vlm: VLM,
        inputs: Mapping[str, Any],
        image_processor: Any,
    ) -> list[VisualizationView] | None:
        """Describe a nonstandard source coordinate system for visualization."""

        del vlm, inputs, image_processor
        return None

    @property
    def budget(self) -> int | float | str:
        """The token budget this run is labelled with — the only part of ``name``
        that varies, so it is what keeps a sweep's result directories apart.

        Defaults to ``max_tokens``. Override *this*, never ``name``, when the
        reported budget is not what ``__init__`` was handed: a two-stage method
        whose ``max_tokens`` is an internal stage count, or a dynamic pruner driven
        by a retention ratio rather than a count.

        Normally a number. A method whose output size is genuinely not a quantity —
        CORE's adaptive rate emits one token per detected object, so the count
        changes with every image — may return a short word instead, which is
        formatted verbatim. Prefer a number whenever one honestly exists: only
        numbers line a sweep up against the other methods.
        """
        return self._max_tokens

    @property
    def name(self) -> str:
        """Run label, one format for every compressor: ``<Method>-b<budget>``.

        ``run.py`` writes results to
        ``<output_dir>/<benchmark>/<name>/<benchmark>_<name>_<date>.json`` and
        ``viz.py`` writes under ``<output_dir>/<name>/``, so this names every
        artifact directory. Do not override it — one format is what makes methods
        comparable by path across a budget sweep. Override ``budget`` instead.
        """
        budget = self.budget
        label = budget if isinstance(budget, str) else f"{budget:g}"
        return f"{self.__class__.__name__.replace('Compressor', '')}-b{label}"

    def wrap_get_image_features(self, vlm: VLM) -> None:
        inner = vlm.model.model
        original_get_image_features = inner.get_image_features

        def wrapped_get_image_features(*args, **kwargs):
            image_outputs = original_get_image_features(*args, **kwargs)
            feats_list = image_outputs.pooler_output
            stacked = torch.stack(list(feats_list), dim=0)
            result: CompressedFeatures = self.compress(stacked)
            image_outputs.pooler_output = list(result.tokens.unbind(0))
            return image_outputs

        inner.get_image_features = wrapped_get_image_features

    @staticmethod
    def register_llm_layer_pre_hook(vlm: VLM, layer_idx: int, fn: Any) -> None:
        layer = vlm.model.model.language_model.layers[layer_idx]
        layer.register_forward_pre_hook(fn, with_kwargs=True)

    def install(self, vlm: VLM) -> None:
        self.wrap_get_image_features(vlm)
