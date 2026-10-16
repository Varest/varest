from json import dumps
from abc import ABCMeta, abstractmethod
from typing import Any, Iterator, Sequence

import torch

from llmoe.config import Config
from llmoe.backbone import VLM

type DataType = str | int | Sequence[str] | torch.Tensor | Any


class Benchmark(object, metaclass=ABCMeta):
    # Benchmarks that time their own `generate` run at batch size 1 without the generic profiler.
    measures_performance: bool = False

    def __init__(self, config: Config, backbone: type[VLM], *, paired: bool = False, dataset_length: int | None = None,
                 **kwargs) -> None:
        self._config: Config = config
        self._backbone: type[VLM] = backbone
        self._paired: bool = paired
        self._dataset_length: int | None = dataset_length

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Benchmark", "")

    @property
    def paired(self) -> bool:
        return self._paired

    @property
    def dataset_length(self) -> int | None:
        return self._dataset_length

    @abstractmethod
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        raise NotImplementedError

    @abstractmethod
    def format_prompt(self, raw_question: str) -> str:
        raise NotImplementedError

    def format_short_answer_prompt(self, raw_question: str) -> str:
        prompt = str(raw_question).strip()
        suffix = self._backbone.short_answer_suffix
        if not suffix or prompt.endswith(suffix.strip()):
            return prompt
        return prompt + suffix

    def format_option_letter_prompt(self, raw_question: str) -> str:
        prompt = str(raw_question).strip()
        suffix = self._backbone.option_letter_suffix
        if not suffix or prompt.endswith(suffix.strip()):
            return prompt
        return prompt + suffix

    def format_direct_answer_prompt(self, raw_question: str) -> str:
        prompt = str(raw_question).strip()
        suffix = self._backbone.direct_answer_suffix
        if not suffix or prompt.endswith(suffix.strip()):
            return prompt
        return prompt + suffix

    def generate(self, backbone: VLM, batch: list[dict[str, DataType]], *, max_new_tokens: int,
                 generation_profiler: Any | None = None, **generation_kwargs: Any) -> list[str]:
        """Run the model on one batch and return raw responses; overrides may add to ``record_extra``."""
        return backbone(
            images=[item["image"] for item in batch],
            prompts=[item["question"] for item in batch],
            max_new_tokens=max_new_tokens,
            generation_profiler=generation_profiler,
            **generation_kwargs,
        )

    @abstractmethod
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        raise NotImplementedError

    def validate_config(self) -> None:
        pass

    def build_artifacts(self, records: list[dict[str, Any]], compressor: Any, path: Any) -> dict[str, Any]:
        """
        Write optional benchmark-specific artifacts after inference.
        """
        return {}

    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool | None:
        """
        Per-row correctness flag stored on the record. Default: exact match.
        """
        return prediction == row.get("answer")

    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict.get("metrics", metrics_dict)
        if isinstance(metrics, dict):
            pairs: list[str] = []
            for key, value in metrics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, float):
                    pairs.append(f"{key}={value:.4f}")
                elif isinstance(value, int):
                    pairs.append(f"{key}={value}")
                if len(pairs) >= 6:
                    break
            if pairs:
                return "  ".join(pairs)
        return dumps(metrics, default=str)[:200]
