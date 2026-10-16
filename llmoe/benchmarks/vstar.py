import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, override

from datasets import Image as DatasetImage, load_dataset

from llmoe.backbone import VLM
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.config import Config
from llmoe.utils import coerce_image_tensor


EXPECTED_CATEGORY_COUNTS = {
    "direct_attributes": 115,
    "relative_position": 76,
}
EXPECTED_TEST_SIZE = sum(EXPECTED_CATEGORY_COUNTS.values())
OFFICIAL_POST_PROMPT = (
    "\nAnswer with the option's letter from the given choices directly."
)
OPTION_RE = re.compile(r"\([A-D]\)\s*([^()]+?)(?=\s*\([A-D]\)|$)")
ANSWER_PATTERNS = (
    re.compile(r"^([A-D])\s*[\.)\]]*", re.IGNORECASE),
    re.compile(
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",
        re.IGNORECASE,
    ),
    re.compile(r"\(([A-D])\)", re.IGNORECASE),
    re.compile(r"([A-D])\s*(?:\.|\)|])", re.IGNORECASE),
    re.compile(r"(?:^|\s)([A-D])(?:\s|$)", re.IGNORECASE),
)


def format_vstar_prompt(
        raw_question: str,
        *,
        pre_prompt: str = "",
        post_prompt: str = OFFICIAL_POST_PROMPT,
) -> str:
    """Reproduce lmms-eval's V*-Bench prompt transformation."""
    question = str(raw_question)
    options = OPTION_RE.findall(question)
    if options:
        question_text = question.split("(A)")[0].strip()
        option_letters = ("A", "B", "C", "D")
        options_formatted = [
            f"{option_letters[index]}. {option.strip()}"
            for index, option in enumerate(options)
            if index < len(option_letters)
        ]
        question = f"{question_text}\n" + "\n".join(options_formatted)
    if pre_prompt:
        question = f"{pre_prompt}{question}"
    if post_prompt:
        question = f"{question}{post_prompt}"
    return question


def extract_answer_letter(response: str) -> str | None:
    """Reproduce lmms-eval's ordered V*-Bench answer extraction."""
    normalized = str(response).strip().upper()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return match.group(1).upper()

    letters = re.findall(r"[A-D]", normalized)
    if len(letters) == 1:
        return letters[0]
    if normalized and normalized[0] in "ABCD":
        return normalized[0]
    return None


class VStarBenchmark(Benchmark):
    """V*-Bench test evaluation using lmms-eval's official prompt and scorer."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[str | None, str, str | None, int | None] | None = None
        self._items: list[dict[str, Any]] | None = None
        self._dataset_length = len(self._load_items())

    def _selected_category(self) -> str | None:
        raw_category = self._config.custom_args.get("vstar_category")
        if raw_category is None:
            return None
        category = str(raw_category).strip()
        if category not in EXPECTED_CATEGORY_COUNTS:
            raise ValueError(
                f"Unknown V*-Bench category {category!r}. "
                f"Valid categories: {tuple(EXPECTED_CATEGORY_COUNTS)}"
            )
        return category

    def _load_items(self) -> list[dict[str, Any]]:
        split = str(self._config.custom_args.get("dataset_split", "test"))
        selected_category = self._selected_category()
        key = (
            self._config.dataset_dir,
            split,
            selected_category,
            self._config.dataset_limit,
        )
        if self._cache_key == key and self._items is not None:
            return self._items

        root = Path(self._config.dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"V*-Bench directory not found: {root}")
        try:
            dataset = load_dataset(str(root), split=split)
        except Exception as dataset_error:
            parquet_files = sorted(root.rglob("*.parquet"))
            selected_files = [
                path for path in parquet_files
                if split.lower() in path.name.lower()
            ] or parquet_files
            if not selected_files:
                raise FileNotFoundError(
                    f"No V*-Bench {split} dataset or parquet files found under {root}"
                ) from dataset_error
            dataset = load_dataset(
                "parquet",
                data_files=[str(path) for path in selected_files],
                split="train",
            )

        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", DatasetImage(decode=False))
        full_items = list(dataset)
        items = [
            item for item in full_items
            if selected_category is None or str(item.get("category")) == selected_category
        ]

        if self._config.dataset_limit is None and split == "test":
            expected = (
                EXPECTED_TEST_SIZE
                if selected_category is None
                else EXPECTED_CATEGORY_COUNTS[selected_category]
            )
            if len(items) != expected:
                scope = "full test set" if selected_category is None else selected_category
                raise ValueError(
                    f"V*-Bench {scope} expected {expected} examples, got {len(items)}."
                )
            if selected_category is None:
                observed_counts: dict[str, int] = defaultdict(int)
                for item in items:
                    observed_counts[str(item.get("category"))] += 1
                if dict(observed_counts) != EXPECTED_CATEGORY_COUNTS:
                    raise ValueError(
                        "V*-Bench full test set expected category counts "
                        f"{EXPECTED_CATEGORY_COUNTS}, got {dict(observed_counts)}."
                    )

        self._cache_key = key
        self._items = (
            items[: self._config.dataset_limit]
            if self._config.dataset_limit is not None
            else items
        )
        return self._items

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for row_idx, item in enumerate(self._load_items()):
            question_id = str(item.get("question_id", row_idx))
            category = str(item.get("category", "unknown"))
            raw_question = str(item["text"])
            answer = str(item["label"]).strip().upper()
            if answer not in "ABCD" or len(answer) != 1:
                raise ValueError(
                    f"V*-Bench question {question_id} has invalid label {answer!r}."
                )
            image = item["image"]
            image_path = (
                str(image["path"])
                if isinstance(image, dict) and image.get("path") is not None
                else None
            )
            options = [option.strip() for option in OPTION_RE.findall(raw_question)]
            yield {
                "question_id": question_id,
                "item_id": f"vstar/{question_id}",
                "category": category,
                "image_path": image_path,
                "question": self.format_prompt(raw_question),
                "question_raw": raw_question,
                "answer": answer,
                "image": coerce_image_tensor(image, source="V*-Bench"),
                "record_extra": {
                    "options": options,
                    "n_choices": len(options),
                },
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return format_vstar_prompt(raw_question)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return extract_answer_letter(raw_text) or ""

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return prediction == str(row.get("answer", "")).strip().upper()

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        correct_by_category: dict[str, int] = defaultdict(int)
        total_by_category: dict[str, int] = defaultdict(int)
        correct = 0
        unparseable = 0
        for record in records:
            category = str(record.get("category", "unknown"))
            total_by_category[category] += 1
            if bool(record.get("correct")):
                correct += 1
                correct_by_category[category] += 1
            if not record.get("prediction"):
                unparseable += 1

        n = len(records)
        accuracy = round(100 * correct / max(1, n), 4)
        per_category = {
            category: {
                "accuracy": round(
                    100 * correct_by_category[category]
                    / max(1, total_by_category[category]),
                    4,
                ),
                "correct": correct_by_category[category],
                "total": total_by_category[category],
            }
            for category in sorted(total_by_category)
        }
        return {
            "metrics": {
                "accuracy": accuracy,
                "direct_attributes_accuracy": per_category.get(
                    "direct_attributes", {}
                ).get("accuracy"),
                "relative_position_accuracy": per_category.get(
                    "relative_position", {}
                ).get("accuracy"),
                "unparseable": unparseable,
                "unparseable_pct": round(100 * unparseable / max(1, n), 4),
                "n": n,
            },
            "accuracy": accuracy,
            "num_examples": n,
            "valid_examples": n,
            "skipped_examples": 0,
            "num_cases": n,
            "scoring": "lmms_eval_example_weighted",
            "per_category": per_category,
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        direct = metrics.get("direct_attributes_accuracy")
        relative = metrics.get("relative_position_accuracy")
        return (
            f"accuracy={metrics['accuracy']:.4f}  "
            f"direct={'nan' if direct is None else f'{float(direct):.4f}'}  "
            f"relative={'nan' if relative is None else f'{float(relative):.4f}'}  "
            f"unparseable={metrics['unparseable_pct']:.4f}  "
            f"n={metrics['n']}"
        )
