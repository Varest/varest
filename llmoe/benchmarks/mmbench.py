import base64
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, override

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.utils import coerce_image_tensor, strip_punct_and_articles


def _is_present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in ("", "nan", "none")


class MMBenchBenchmark(Benchmark):
    """MMBench dev evaluation with Roby's LLaVA-style CircularEval recipe."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[str | None, int | None] | None = None
        self._rotation_rows: list[dict[str, DataType]] | None = None
        self._dataset_length = len(self._load_rotation_rows())

    def _load_rotation_rows(self) -> list[dict[str, DataType]]:
        key = (self._config.dataset_dir, self._config.dataset_limit)
        if self._cache_key == key and self._rotation_rows is not None:
            return self._rotation_rows

        root = Path(self._config.dataset_dir)
        candidates = sorted(root.glob("*.tsv"))
        if not candidates:
            raise FileNotFoundError(f"No MMBench .tsv file found under {root}")
        tsv_file = candidates[0]
        for candidate in candidates:
            if "dev" in candidate.name.lower():
                tsv_file = candidate
                break

        csv.field_size_limit(100_000_000)
        source_rows: list[dict[str, str]] = []
        with tsv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                source_rows.append(row)
                if self._config.dataset_limit is not None and len(source_rows) >= self._config.dataset_limit:
                    break

        rotation_rows: list[dict[str, DataType]] = []
        for row_idx, row in enumerate(source_rows):
            original_choices: list[tuple[str, str]] = []
            for letter in ("A", "B", "C", "D"):
                if _is_present(row.get(letter)):
                    original_choices.append((letter, str(row[letter])))
            n_choices = len(original_choices)
            gt_letter = str(row.get("answer", "")).strip().upper()
            if len(gt_letter) != 1:
                continue
            gt_idx = ord(gt_letter) - 65
            if not (0 <= gt_idx < n_choices):
                continue

            question_id = str(row.get("index") or row_idx)
            for shift in range(n_choices):
                choices: dict[str, str] = {}
                for new_pos in range(n_choices):
                    original_pos = (new_pos + shift) % n_choices
                    choices[chr(65 + new_pos)] = original_choices[original_pos][1]
                question = str(row["question"])
                if _is_present(row.get("hint")):
                    question = f"{row['hint']}\n{question}"
                for letter, value in choices.items():
                    question = f"{question}\n{letter}. {value}"
                rotation_rows.append(
                    {
                        "question_id": question_id,
                        "item_id": f"mmbench/{question_id}/{shift}",
                        "category": row.get("category") or "all",
                        "image_path": None,
                        "question": self.format_prompt(question),
                        "question_raw": row.get("question"),
                        "answer": chr(65 + ((gt_idx - shift) % n_choices)),
                        "image_b64": row["image"],
                        "choices": choices,
                        "n_choices": n_choices,
                        "record_extra": {"shift": shift},
                    }
                )

        self._cache_key = key
        self._rotation_rows = rotation_rows
        return rotation_rows

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for row in self._load_rotation_rows():
            yield {
                **row,
                "image": coerce_image_tensor(base64.b64decode(str(row["image_b64"])), source="MMBench"),
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return self.format_option_letter_prompt(raw_question)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        choices = row.get("choices")
        choices = choices if isinstance(choices, dict) else {}
        valid = {chr(ord("A") + i) for i in range(int(row.get("n_choices", 4)))}
        text = raw_text.strip()
        match = re.match(r"^\s*\(?([A-Z])\)?[\.\)\s,:]", text)
        if match and match.group(1) in valid:
            return match.group(1)
        if len(text) == 1 and text.upper() in valid:
            return text.upper()

        norm_reply = strip_punct_and_articles(text)
        matches = []
        for letter, value in choices.items():
            norm_option = strip_punct_and_articles(str(value))
            if norm_option and (norm_option == norm_reply or norm_option in norm_reply):
                matches.append(str(letter))
        return matches[0] if len(matches) == 1 else ""

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return prediction == row.get("answer")

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        by_question: dict[str, dict[str, Any]] = {}
        for record in records:
            question_id = str(record["question_id"])
            state = by_question.setdefault(
                question_id,
                {"n": 0, "correct_count": 0, "vanilla_ok": False, "category": record.get("category", "all")},
            )
            state["n"] += 1
            if record.get("correct"):
                state["correct_count"] += 1
            if int(record.get("shift") or 0) == 0:
                state["vanilla_ok"] = bool(record.get("correct"))

        by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"circ": 0, "vanilla": 0, "total": 0})
        circular_correct = 0
        vanilla_correct = 0
        for state in by_question.values():
            category = str(state["category"])
            by_category[category]["total"] += 1
            if state["n"] and state["correct_count"] == state["n"]:
                by_category[category]["circ"] += 1
                circular_correct += 1
            if state["vanilla_ok"]:
                by_category[category]["vanilla"] += 1
                vanilla_correct += 1

        n_questions = len(by_question)
        return {
            "metrics": {
                "n_questions": n_questions,
                "n_inferences": len(records),
                "accuracy_circular": round(100 * circular_correct / max(1, n_questions), 4),
                "accuracy_vanilla": round(100 * vanilla_correct / max(1, n_questions), 4),
                "per_category_circular": {
                    category: round(100 * values["circ"] / max(1, values["total"]), 4)
                    for category, values in by_category.items()
                },
                "note": (
                    "CircularEval on validation split; paper test split requires "
                    "opencompass.org.cn submission."
                ),
            }
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        return (
            f"circular={metrics['accuracy_circular']:.4f}  "
            f"vanilla={metrics['accuracy_vanilla']:.4f}  "
            f"questions={metrics['n_questions']}  "
            f"inferences={metrics['n_inferences']}"
        )
