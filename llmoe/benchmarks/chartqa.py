import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, override

from datasets import Image as DatasetImage, load_dataset

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.utils import coerce_image_tensor


NUMERIC_TOLERANCE = 0.05
CHARTQA_SPLITS = ("human", "augmented")
EXPECTED_TEST_SIZE = 2500
EXPECTED_TEST_SPLIT_SIZE = EXPECTED_TEST_SIZE // 2
NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*(?:\.\d+)?")


def normalize_chartqa_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _try_parse_number(text: str) -> float | None:
    match = NUM_RE.search(str(text).strip())
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def relaxed_correctness(prediction: str, answer: str, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    if normalize_chartqa_text(prediction) == normalize_chartqa_text(answer):
        return True
    pred_num = _try_parse_number(prediction)
    answer_num = _try_parse_number(answer)
    if pred_num is None or answer_num is None:
        return False
    if answer_num == 0.0:
        return pred_num == 0.0
    return abs(pred_num - answer_num) / abs(answer_num) <= tolerance


class ChartQABenchmark(Benchmark):
    """ChartQA test evaluation using official-release relaxed scoring."""

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
            raise FileNotFoundError(f"ChartQA directory not found: {root}")
        try:
            dataset = load_dataset(str(root), split=split)
        except Exception:
            parquet_files = sorted(root.rglob("*.parquet"))
            selected = [path for path in parquet_files if split.lower() in path.name.lower()]
            if not selected:
                raise FileNotFoundError(f"No ChartQA {split} parquet files found under {root}")
            dataset = load_dataset("parquet", data_files=[str(path) for path in selected], split="train")
        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", DatasetImage(decode=False))
        full_items = list(dataset)
        split_counts: dict[str, int] = defaultdict(int)
        for item in full_items:
            split_counts[str(item.get("type") or "unknown")] += 1
        if self._config.dataset_limit is None and split == "test":
            if len(full_items) != EXPECTED_TEST_SIZE:
                raise ValueError(f"ChartQA full test run expected {EXPECTED_TEST_SIZE} examples, got {len(full_items)}.")
            expected = {name: EXPECTED_TEST_SPLIT_SIZE for name in CHARTQA_SPLITS}
            observed = {name: split_counts.get(name, 0) for name in CHARTQA_SPLITS}
            if observed != expected:
                raise ValueError(f"ChartQA full test run expected split balance {expected}, got {observed}.")
            unexpected = sorted(set(split_counts) - set(CHARTQA_SPLITS))
            if unexpected:
                raise ValueError(f"ChartQA full test run found unexpected type values: {unexpected}")

        self._cache_key = key
        self._items = full_items[: self._config.dataset_limit] if self._config.dataset_limit is not None else full_items
        return self._items

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for row_idx, item in enumerate(self._load_items()):
            split_name = str(item.get("type") or "unknown")
            imgname = None if item.get("imgname") is None else str(item.get("imgname"))
            label = item.get("label", "")
            answer = str(label[0]) if isinstance(label, list) and label else str(label)
            question = str(item["query"])
            yield {
                "question_id": imgname or row_idx,
                "item_id": f"chartqa/{imgname or row_idx}",
                "category": split_name,
                "image_path": None,
                "question": self.format_prompt(question),
                "question_raw": question,
                "answer": answer,
                "image": coerce_image_tensor(item["image"], source="ChartQA"),
                "record_extra": {"imgname": imgname, "type": split_name},
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return self.format_short_answer_prompt(raw_question)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return raw_text.strip()

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return relaxed_correctness(prediction, str(row.get("answer", "")))

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        by_split: dict[str, list[dict[str, DataType]]] = defaultdict(list)
        for record in records:
            by_split[str(record.get("category") or "unknown")].append(record)

        per_split: dict[str, dict[str, Any]] = {}
        relaxed_correct = 0
        exact_correct = 0
        for split_name, split_records in sorted(by_split.items()):
            split_relaxed = sum(bool(record.get("correct")) for record in split_records)
            split_exact = sum(
                normalize_chartqa_text(str(record.get("prediction", ""))) == normalize_chartqa_text(str(record.get("answer", "")))
                for record in split_records
            )
            relaxed_correct += split_relaxed
            exact_correct += split_exact
            per_split[split_name] = {
                "accuracy_relaxed": round(100 * split_relaxed / max(1, len(split_records)), 4),
                "accuracy_exact_normalized": round(100 * split_exact / max(1, len(split_records)), 4),
                "correct_relaxed": int(split_relaxed),
                "correct_exact_normalized": int(split_exact),
                "total": len(split_records),
            }

        n_total = len(records)
        is_full_run = (
            n_total == EXPECTED_TEST_SIZE
            and set(by_split) == set(CHARTQA_SPLITS)
            and all(len(by_split[split]) == EXPECTED_TEST_SPLIT_SIZE for split in CHARTQA_SPLITS)
        )
        if is_full_run:
            accuracy_relaxed = round(sum(per_split[split]["accuracy_relaxed"] for split in CHARTQA_SPLITS) / len(CHARTQA_SPLITS), 4)
            accuracy_exact = round(sum(per_split[split]["accuracy_exact_normalized"] for split in CHARTQA_SPLITS) / len(CHARTQA_SPLITS), 4)
            scoring = "official_split_average"
        else:
            accuracy_relaxed = round(100 * relaxed_correct / max(1, n_total), 4)
            accuracy_exact = round(100 * exact_correct / max(1, n_total), 4)
            scoring = "example_weighted"

        return {
            "num_examples": n_total,
            "valid_examples": n_total,
            "skipped_examples": 0,
            "num_cases": n_total,
            "scoring": scoring,
            "accuracy": accuracy_relaxed,
            "metrics": {
                "accuracy_relaxed": accuracy_relaxed,
                "accuracy_exact_normalized": accuracy_exact,
                "human_relaxed": per_split.get("human", {}).get("accuracy_relaxed"),
                "augmented_relaxed": per_split.get("augmented", {}).get("accuracy_relaxed"),
                "human_exact_normalized": per_split.get("human", {}).get("accuracy_exact_normalized"),
                "augmented_exact_normalized": per_split.get("augmented", {}).get("accuracy_exact_normalized"),
                "n": n_total,
            },
            "per_split": per_split,
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        human = metrics.get("human_relaxed")
        augmented = metrics.get("augmented_relaxed")
        return (
            f"relaxed={metrics['accuracy_relaxed']:.4f}  "
            f"exact={metrics['accuracy_exact_normalized']:.4f}  "
            f"human={'nan' if human is None else f'{float(human):.4f}'}  "
            f"augmented={'nan' if augmented is None else f'{float(augmented):.4f}'}  "
            f"n={metrics['n']}"
        )
