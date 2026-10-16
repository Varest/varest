import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, override

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.utils import coerce_image_tensor


EXPECTED_SPLIT_ROWS = {"train": 39463, "val": 5349, "test": 5188}
ANLS_THRESHOLD = 0.5
SPLIT_FILE_NAMES = {
    "test": ("test_v1.0.json", "test_v1_0.json", "test.json"),
    "val": ("val_v1.0_withQT.json", "val_v1_0_withQT.json", "val.json"),
    "train": ("train_v1.0_withQT.json", "train_v1_0_withQT.json", "train.json"),
}
SPLIT_ALIASES = {"validation": "val", "valid": "val", "dev": "val"}


try:
    import Levenshtein as _Levenshtein

    def _lev_distance(a: str, b: str) -> int:
        return _Levenshtein.distance(a, b)

except ImportError:

    def _lev_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous = list(range(len(b) + 1))
        for i, char_a in enumerate(a, 1):
            current = [i]
            for j, char_b in enumerate(b, 1):
                cost = 0 if char_a == char_b else 1
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
            previous = current
        return previous[-1]


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    records = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a list or top-level data list.")
    return records


def _find_split_file(root: Path, split: str) -> Path:
    names = SPLIT_FILE_NAMES.get(split)
    if names is None:
        valid = ", ".join(["auto", *SPLIT_FILE_NAMES])
        raise ValueError(f"Unknown DocVQA split {split!r}. Valid: {valid}")
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"DocVQA {split} JSON not found under {root}. Expected: {', '.join(names)}")


def _coerce_answers(raw_answers: Any) -> list[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, list):
        return [str(answer) for answer in raw_answers]
    return [str(raw_answers)]


def anls_score(prediction: str, gold_answers: list[str], threshold: float = ANLS_THRESHOLD) -> float:
    pred = str(prediction).lower().strip()
    if not gold_answers:
        return 0.0
    best = 0.0
    for answer in gold_answers:
        gold = str(answer).lower().strip()
        if not pred and not gold:
            score = 1.0
        elif not pred or not gold:
            score = 0.0
        else:
            score = 1.0 - _lev_distance(pred, gold) / max(len(pred), len(gold))
        best = max(best, score)
    return best if best >= threshold else 0.0


class DocVQABenchmark(Benchmark):
    """SP-DocVQA v1.0 offline validation using official ANLS scoring."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[str | None, str, int | None] | None = None
        self._items: list[dict[str, DataType]] | None = None
        self._split: str | None = None
        self._source: str | None = None
        self._dataset_length = len(self._load_items())

    def _load_items(self) -> list[dict[str, DataType]]:
        requested_split = SPLIT_ALIASES.get(
            str(self._config.custom_args.get("dataset_split", "auto")).strip().lower(),
            str(self._config.custom_args.get("dataset_split", "auto")).strip().lower(),
        )
        key = (self._config.dataset_dir, requested_split, self._config.dataset_limit)
        if self._cache_key == key and self._items is not None:
            return self._items

        root = Path(self._config.dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"DocVQA root not found: {root}")

        if requested_split == "auto":
            try:
                test_path = _find_split_file(root, "test")
                test_records = _read_json_records(test_path)
            except FileNotFoundError:
                test_path = None
                test_records = []
            if test_path is not None and any(_coerce_answers(record.get("answers")) for record in test_records[:32]):
                split = "test"
                source_path = test_path
                records = test_records
            else:
                split = "val"
                source_path = _find_split_file(root, "val")
                records = _read_json_records(source_path)
        else:
            split = requested_split
            source_path = _find_split_file(root, split)
            records = _read_json_records(source_path)
            if not any(_coerce_answers(record.get("answers")) for record in records[:32]):
                raise RuntimeError(
                    f"DocVQA split {split!r} at {source_path} has no answers. "
                    "Use the default auto split or validation for offline ANLS scoring."
                )

        expected_rows = EXPECTED_SPLIT_ROWS.get(split)
        if self._config.dataset_limit is None and expected_rows is not None and len(records) != expected_rows:
            raise ValueError(f"DocVQA {split} split expected {expected_rows} rows, got {len(records)} from {source_path}.")

        items: list[dict[str, DataType]] = []
        for record in records:
            question_id = record.get("questionId", record.get("question_id"))
            answers = _coerce_answers(record.get("answers"))
            if not answers:
                raise RuntimeError(
                    f"DocVQA {split} record questionId={question_id!r} has no answers. "
                    "Offline ANLS scoring requires answered annotations."
                )
            relative_image = record.get("image")
            if not relative_image:
                raise ValueError(f"DocVQA record questionId={question_id!r} has no image field.")
            image_path = root / str(relative_image)
            if not image_path.exists():
                found = None
                for suffix in (".png", ".jpg", ".jpeg"):
                    candidate = image_path.with_suffix(suffix)
                    if candidate.exists():
                        found = candidate
                        break
                if found is None:
                    raise FileNotFoundError(f"DocVQA image not found for questionId={question_id}: {image_path}")
                image_path = found

            raw_question = str(record["question"])
            raw_types = record.get("question_types")
            if raw_types is None:
                question_types: list[str] = []
            elif isinstance(raw_types, list):
                question_types = [str(question_type) for question_type in raw_types]
            else:
                question_types = [str(raw_types)]
            items.append(
                {
                    "question_id": question_id,
                    "item_id": f"docvqa/{question_id}",
                    "category": question_types[0] if question_types else "docvqa",
                    "image_path": image_path,
                    "question_raw": raw_question,
                    "answer": answers[0],
                    "answers": answers,
                    "question_types": question_types,
                    "doc_id": record.get("ucsf_document_id", record.get("docId")),
                    "page_no": None if record.get("ucsf_document_page_no") is None else str(record.get("ucsf_document_page_no")),
                    "split": split,
                }
            )

        if self._config.dataset_limit is not None:
            items = items[: self._config.dataset_limit]

        self._cache_key = key
        self._items = items
        self._split = split
        self._source = str(source_path)
        print(f"[info] DocVQA: using {source_path.name} ({len(items)} questions, split={split})")
        return items

    @override
    def validate_config(self) -> None:
        self._load_items()

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for item in self._load_items():
            image_path = Path(item["image_path"])
            yield {
                "question_id": item["question_id"],
                "item_id": item["item_id"],
                "category": item["category"],
                "image_path": str(image_path),
                "question": self.format_prompt(str(item["question_raw"])),
                "question_raw": item["question_raw"],
                "answer": item["answer"],
                "answers": item["answers"],
                "image": coerce_image_tensor(image_path, source="DocVQA"),
                "record_extra": {
                    "question_types": item["question_types"],
                    "doc_id": item["doc_id"],
                    "page_no": item["page_no"],
                    "split": item["split"],
                },
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return self.format_short_answer_prompt(raw_question)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return str(raw_text).strip()

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return anls_score(prediction, [str(answer) for answer in row.get("answers") or []]) >= ANLS_THRESHOLD

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        scores: list[float] = []
        exact_matches: list[bool] = []
        per_type_scores: dict[str, list[float]] = defaultdict(list)
        case_wise_metrics: list[dict[str, Any]] = []
        for record in records:
            answers = [str(answer) for answer in record.get("answers") or []]
            prediction = str(record.get("prediction", ""))
            score = anls_score(prediction, answers)
            scores.append(score)
            exact = any(prediction.lower().strip() == str(answer).lower().strip() for answer in answers)
            exact_matches.append(exact)
            question_types = [str(question_type) for question_type in record.get("question_types") or ["<untagged>"]]
            for question_type in question_types:
                per_type_scores[question_type].append(score)
            case_wise_metrics.append(
                {
                    "question_id": record.get("question_id"),
                    "doc_id": record.get("doc_id"),
                    "question_types": question_types,
                    "anls": score,
                    "exact_normalized": exact,
                }
            )

        n = len(records)
        mean_anls = sum(scores) / max(1, n)
        exact_accuracy = sum(exact_matches) / max(1, n)
        return {
            "split": self._split,
            "num_examples": n,
            "valid_examples": n,
            "skipped_examples": 0,
            "num_cases": n,
            "scoring": "official_anls",
            "accuracy": round(100.0 * mean_anls, 4),
            "metrics": {
                "anls": round(100.0 * mean_anls, 4),
                "anls_normalized": mean_anls,
                "exact_match_accuracy": round(100.0 * exact_accuracy, 4),
                "threshold": ANLS_THRESHOLD,
                "n": n,
            },
            "per_question_type": {
                question_type: {
                    "anls": round(100.0 * sum(type_scores) / max(1, len(type_scores)), 4),
                    "n": len(type_scores),
                }
                for question_type, type_scores in sorted(per_type_scores.items(), key=lambda item: (-len(item[1]), item[0]))
            },
            "case_wise_metrics": case_wise_metrics,
            "source": self._source,
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        return (
            f"anls={metrics['anls']:.4f}  "
            f"exact={metrics['exact_match_accuracy']:.4f}  "
            f"split={metrics_dict.get('split')}  "
            f"n={metrics['n']}"
        )
