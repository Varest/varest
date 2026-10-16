from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, override

from datasets import Image as DatasetImage, load_dataset

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.utils import coerce_image_tensor


SAMPLES_PER_L1 = 250
TOTAL_SAMPLES = 1500


def parse_leading_letter(text: str) -> str | None:
    predict = str(text).lower().strip().replace("\n", " ")
    if not predict:
        return None
    for answer in "abcd":
        try:
            if predict[0] == answer:
                return answer.upper()
            if predict[0] == "(" and predict[1] == answer:
                return answer.upper()
            if predict[:7] == "option " and predict[7] == answer:
                return answer.upper()
            if predict[:14] == "the answer is " and predict[14] == answer:
                return answer.upper()
        except IndexError:
            continue
    return None


class MMStarBenchmark(Benchmark):
    """MMStar val evaluation with the official prompt and strict scorer."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[str | None, str, int | None] | None = None
        self._items: list[dict[str, Any]] | None = None
        self._dataset_length = len(self._load_items())

    def _load_items(self) -> list[dict[str, Any]]:
        split = str(self._config.custom_args.get("dataset_split", "val"))
        key = (self._config.dataset_dir, split, self._config.dataset_limit)
        if self._cache_key == key and self._items is not None:
            return self._items

        root = Path(self._config.dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"MMStar directory not found: {root}")
        try:
            dataset = load_dataset(str(root), split=split)
        except Exception:
            parquet_files = sorted(root.rglob("*.parquet"))
            selected = [path for path in parquet_files if split.lower() in path.name.lower()] or parquet_files
            if not selected:
                raise
            dataset = load_dataset("parquet", data_files=[str(path) for path in selected], split="train")
        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", DatasetImage(decode=False))
        if self._config.dataset_limit is not None:
            dataset = dataset.select(range(min(self._config.dataset_limit, len(dataset))))
        self._cache_key = key
        self._items = list(dataset)
        return self._items

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for row_idx, item in enumerate(self._load_items()):
            index = item.get("index", row_idx)
            l1 = str(item.get("category") or "unknown")
            l2 = str(item.get("l2_category") or "unknown")
            raw_question = str(item["question"])
            yield {
                "question_id": index,
                "item_id": f"mmstar/{index}",
                "category": l1,
                "image_path": None,
                "question": self.format_prompt(raw_question),
                "question_raw": raw_question,
                "answer": str(item.get("answer", "")).strip().upper(),
                "image": coerce_image_tensor(item["image"], source="MMStar"),
                "record_extra": {
                    "l2_category": l2,
                    "meta_info": None if item.get("meta_info") is None else str(item.get("meta_info")),
                },
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        # MMStar is multiple-choice, but the historical prompt here is the
        # free-form "answer directly" one. Verbose backbones then answer in prose
        # and the strict leading-letter scorer rejects them. "letter" switches to
        # the option-letter instruction used by VLMEvalKit.
        style = str(self._config.custom_args.get("mmstar_prompt_style", "direct")).lower()
        if style == "letter":
            return self.format_option_letter_prompt(raw_question)
        if style != "direct":
            raise ValueError(f"mmstar_prompt_style must be 'direct' or 'letter', got {style!r}")
        return self.format_direct_answer_prompt(raw_question)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return raw_text.strip()

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return parse_leading_letter(prediction) == str(row.get("answer", "")).strip().upper()

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        correct_by_l1: dict[str, int] = defaultdict(int)
        total_by_l1: dict[str, int] = defaultdict(int)
        correct_by_l1l2: dict[tuple[str, str], int] = defaultdict(int)
        total_by_l1l2: dict[tuple[str, str], int] = defaultdict(int)

        correct = 0
        unparseable = 0
        for record in records:
            l1 = str(record.get("category") or "unknown")
            l2 = str(record.get("l2_category") or "unknown")
            total_by_l1[l1] += 1
            total_by_l1l2[(l1, l2)] += 1
            if record.get("correct"):
                correct += 1
                correct_by_l1[l1] += 1
                correct_by_l1l2[(l1, l2)] += 1
            if parse_leading_letter(str(record.get("prediction", ""))) is None:
                unparseable += 1

        n_total = len(records)
        if n_total == TOTAL_SAMPLES:
            per_category_l1 = {
                l1: round(100 * correct_by_l1[l1] / SAMPLES_PER_L1, 4)
                for l1 in sorted(total_by_l1)
            }
            accuracy = round(100 * correct / TOTAL_SAMPLES, 4)
            scoring = "official_per250"
        else:
            per_category_l1 = {
                l1: round(100 * correct_by_l1[l1] / max(1, total_by_l1[l1]), 4)
                for l1 in sorted(total_by_l1)
            }
            accuracy = round(100 * correct / max(1, n_total), 4)
            scoring = "observed_counts"

        return {
            "num_examples": n_total,
            "valid_examples": n_total,
            "skipped_examples": 0,
            "num_cases": n_total,
            "scoring": scoring,
            "accuracy": accuracy,
            "metrics": {
                "accuracy": accuracy,
                "unparseable": unparseable,
                "unparseable_pct": round(100 * unparseable / max(1, n_total), 4),
            },
            "per_category_l1": per_category_l1,
            "per_category_l2": {
                f"{l1}({l2})": round(
                    100 * correct_by_l1l2[(l1, l2)] / max(1, total_by_l1l2[(l1, l2)]),
                    4,
                )
                for (l1, l2) in sorted(total_by_l1l2)
            },
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        return (
            f"accuracy={metrics['accuracy']:.4f}  "
            f"unparseable={metrics['unparseable_pct']:.4f}  "
            f"n={metrics_dict['num_examples']}  "
            f"scoring={metrics_dict['scoring']}"
        )
