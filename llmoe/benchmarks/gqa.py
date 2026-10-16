import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, override

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.utils import coerce_image_tensor, strip_punct_and_articles


def _mean_pct(values: list[bool]) -> float:
    return round(100 * sum(values) / max(1, len(values)), 4)


def _group_accuracy(records: list[dict[str, DataType]], key: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        if value is not None:
            grouped[str(value)].append(bool(record.get("correct")))
    return {name: _mean_pct(values) for name, values in sorted(grouped.items())}


class GQABenchmark(Benchmark):
    """GQA testdev-balanced evaluation using Roby's short-answer recipe."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[str | None, int | None] | None = None
        self._items: list[dict[str, DataType]] | None = None
        self._source: str | None = None
        self._eval_dir: str | None = None
        self._dataset_length = len(self._load_items())

    def _load_items(self) -> list[dict[str, DataType]]:
        key = (self._config.dataset_dir, self._config.dataset_limit)
        if self._cache_key == key and self._items is not None:
            return self._items

        root = Path(self._config.dataset_dir)
        question_file = root / "testdev_balanced_questions.json"
        image_dir = root / "images"
        if not question_file.exists():
            raise FileNotFoundError(f"GQA question file not found: {question_file}")
        if not image_dir.exists():
            raise FileNotFoundError(f"GQA image directory not found: {image_dir}")

        questions_dict = json.loads(question_file.read_text(encoding="utf-8"))
        questions = [
            {"question_id": str(question_id), **question}
            for question_id, question in questions_dict.items()
        ]
        if self._config.dataset_limit is not None:
            questions = questions[: self._config.dataset_limit]

        items: list[dict[str, DataType]] = []
        for question in questions:
            image_id = str(question["imageId"])
            image_path = image_dir / f"{image_id}.jpg"
            if not image_path.exists():
                print(f"[warn] GQA: missing image for {question['question_id']}: {image_path}")
                continue
            types = question.get("types") or {}
            semantic = question.get("semantic") or []
            raw_question = str(question["question"])
            items.append(
                {
                    "question_id": str(question["question_id"]),
                    "image_id": image_id,
                    "image_path": image_path,
                    "question_raw": raw_question,
                    "answer": str(question.get("answer", "")),
                    "full_answer": None if question.get("fullAnswer") is None else str(question.get("fullAnswer")),
                    "structural_type": None if types.get("structural") is None else str(types.get("structural")),
                    "semantic_type": None if types.get("semantic") is None else str(types.get("semantic")),
                    "detailed_type": None if types.get("detailed") is None else str(types.get("detailed")),
                    "steps": len(semantic),
                    "word_count": len(raw_question.split()),
                }
            )

        input_dir = Path(getattr(self._config, "input_dir", root.parent))
        eval_dir = input_dir / "eval" / "gqa"
        self._cache_key = key
        self._items = items
        self._source = str(question_file)
        self._eval_dir = str(eval_dir) if eval_dir.exists() else None
        print(f"[info] GQA: using {question_file.name} ({len(items)} questions)")
        return items

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for item in self._load_items():
            image_path = Path(item["image_path"])
            yield {
                "question_id": item["question_id"],
                "item_id": f"gqa/{item['question_id']}",
                "category": item.get("structural_type") or "unknown",
                "image_path": str(image_path),
                "question": self.format_prompt(str(item["question_raw"])),
                "question_raw": item["question_raw"],
                "answer": item["answer"],
                "image": coerce_image_tensor(image_path, source="GQA"),
                "record_extra": {
                    "image_id": item["image_id"],
                    "full_answer": item.get("full_answer"),
                    "structural_type": item.get("structural_type"),
                    "semantic_type": item.get("semantic_type"),
                    "detailed_type": item.get("detailed_type"),
                    "steps": item.get("steps"),
                    "word_count": item.get("word_count"),
                },
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return self.format_short_answer_prompt(raw_question)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return raw_text.strip()

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return strip_punct_and_articles(prediction) == strip_punct_and_articles(str(row.get("answer", "")))

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        exact = [
            str(record.get("prediction", "")).strip().lower()
            == str(record.get("answer", "")).strip().lower()
            for record in records
        ]
        binary_records = [record for record in records if record.get("structural_type") != "query"]
        open_records = [record for record in records if record.get("structural_type") == "query"]
        eval_dir = Path(self._eval_dir) if self._eval_dir else None
        return {
            "metrics": {
                "n": len(records),
                "accuracy": _mean_pct([bool(record.get("correct")) for record in records]),
                "exact_match_accuracy": _mean_pct(exact),
                "binary_accuracy": _mean_pct([bool(record.get("correct")) for record in binary_records]),
                "open_accuracy": _mean_pct([bool(record.get("correct")) for record in open_records]),
                "per_structural_type": _group_accuracy(records, "structural_type"),
                "per_semantic_type": _group_accuracy(records, "semantic_type"),
                "per_steps": _group_accuracy(records, "steps"),
                "per_word_count": _group_accuracy(records, "word_count"),
            },
            "num_examples": len(records),
            "valid_examples": len(records),
            "skipped_examples": 0,
            "num_cases": len(records),
            "source": self._source,
            "official_predictions": [
                {
                    "questionId": str(record.get("question_id")),
                    "prediction": strip_punct_and_articles(str(record.get("prediction", ""))),
                }
                for record in records
            ],
            "official_eval": {
                "eval_dir": str(eval_dir) if eval_dir else None,
                "eval_py": str(eval_dir / "eval.py") if eval_dir and (eval_dir / "eval.py").exists() else None,
                "choices": (
                    str(eval_dir / "val_choices.json")
                    if eval_dir and (eval_dir / "val_choices.json").exists()
                    else None
                ),
                "prediction_format": '[{"questionId": str, "prediction": str}]',
                "note": (
                    "This benchmark mirrors Roby's normalized testdev-balanced accuracy. "
                    "The official GQA script also computes validity, plausibility, "
                    "consistency, and grounding when support files are present."
                ),
            },
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        return (
            f"accuracy={metrics['accuracy']:.4f}  "
            f"exact={metrics['exact_match_accuracy']:.4f}  "
            f"binary={metrics['binary_accuracy']:.4f}  "
            f"open={metrics['open_accuracy']:.4f}  "
            f"n={metrics['n']}"
        )
