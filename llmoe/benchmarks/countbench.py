from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, override

from datasets import Image as DatasetImage, load_dataset

from llmoe.backbone import VLM
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.config import Config
from llmoe.utils import coerce_image_tensor


EXPECTED_TEST_SIZE = 491
OFFICIAL_PRE_PROMPT = (
    "Look at the image carefully and count the objects.\n"
    "Answer with just a number, without any additional text. "
)
NUMBER_WORD_TO_NUMERAL = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def normalize_count_answer(answer: Any) -> str:
    """Match lmms-eval's CountBench answer normalization exactly."""
    normalized = str(answer).strip().lower()
    return NUMBER_WORD_TO_NUMERAL.get(normalized, normalized)


class CountBenchBenchmark(Benchmark):
    """CountBenchQA test evaluation using lmms-eval's official prompt and scorer."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[str | None, str, int | None] | None = None
        self._items: list[dict[str, Any]] | None = None
        self._dataset_length = len(self._load_items())

    def _load_items(self) -> list[dict[str, Any]]:
        split = str(self._config.custom_args.get("dataset_split", "test"))
        key = (self._config.dataset_dir, split, self._config.dataset_limit)
        if self._cache_key == key and self._items is not None:
            return self._items

        root = Path(self._config.dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"CountBenchQA directory not found: {root}")
        try:
            dataset = load_dataset(str(root), split=split)
        except Exception as dataset_error:
            parquet_files = sorted(root.rglob("*.parquet"))
            selected = [
                path for path in parquet_files
                if split.lower() in path.name.lower()
            ] or parquet_files
            if not selected:
                raise FileNotFoundError(
                    f"No CountBenchQA {split} dataset or parquet files found under {root}"
                ) from dataset_error
            dataset = load_dataset(
                "parquet",
                data_files=[str(path) for path in selected],
                split="train",
            )

        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", DatasetImage(decode=False))
        full_items = list(dataset)
        if self._config.dataset_limit is None and split == "test":
            if len(full_items) != EXPECTED_TEST_SIZE:
                raise ValueError(
                    f"CountBenchQA full test run expected {EXPECTED_TEST_SIZE} examples, "
                    f"got {len(full_items)}."
                )

        self._cache_key = key
        self._items = (
            full_items[: self._config.dataset_limit]
            if self._config.dataset_limit is not None
            else full_items
        )
        return self._items

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for row_idx, item in enumerate(self._load_items()):
            raw_question = str(item["question"]).strip()
            answer = normalize_count_answer(item["number"])
            image = item["image"]
            image_path = (
                str(image["path"])
                if isinstance(image, dict) and image.get("path") is not None
                else None
            )
            yield {
                "question_id": row_idx,
                "item_id": f"countbench/{row_idx}",
                "category": f"count_{answer}",
                "image_path": image_path,
                "question": self.format_prompt(raw_question),
                "question_raw": raw_question,
                "answer": answer,
                "image": coerce_image_tensor(image, source="CountBenchQA"),
                "record_extra": {
                    "count": answer,
                    "source_text": None if item.get("text") is None else str(item["text"]),
                },
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return f"{OFFICIAL_PRE_PROMPT}{str(raw_question).strip()}"

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return normalize_count_answer(raw_text)

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return normalize_count_answer(prediction) == normalize_count_answer(row.get("answer", ""))

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        correct_by_count: dict[str, int] = defaultdict(int)
        total_by_count: dict[str, int] = defaultdict(int)
        correct = 0
        for record in records:
            count = normalize_count_answer(record.get("answer", ""))
            total_by_count[count] += 1
            if bool(record.get("correct")):
                correct += 1
                correct_by_count[count] += 1

        n = len(records)
        accuracy = round(100 * correct / max(1, n), 4)
        return {
            "metrics": {
                "accuracy": accuracy,
                "correct": correct,
                "n": n,
            },
            "accuracy": accuracy,
            "num_examples": n,
            "valid_examples": n,
            "skipped_examples": 0,
            "num_cases": n,
            "scoring": "lmms_eval_exact_match",
            "per_count": {
                count: {
                    "accuracy": round(
                        100 * correct_by_count[count] / max(1, total_by_count[count]),
                        4,
                    ),
                    "correct": correct_by_count[count],
                    "total": total_by_count[count],
                }
                for count in sorted(total_by_count, key=lambda value: int(value))
            },
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        return (
            f"accuracy={metrics['accuracy']:.4f}  "
            f"correct={metrics['correct']}  "
            f"n={metrics['n']}  "
            f"scoring={metrics_dict['scoring']}"
        )
