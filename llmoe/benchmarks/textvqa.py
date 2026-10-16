import json
import re
from pathlib import Path
from typing import Any, Iterator, override

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.utils import coerce_image_tensor


TEXTVQA_ANNOTATION_NAMES = ("TextVQA_0.5.1_val.json", "TextVQA_0_5_1_val.json")
REFERENCE_OCR_MARKER = "\nReference OCR token:"
TEXTVQA_OCR_RELPATH = Path("eval") / "textvqa" / "llava_textvqa_val_v051_ocr.jsonl"
_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
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
}
_PUNCT_TO_REPLACE = set(';/[]"{}()=+\\_-><@`?!')
_PERIOD_RE = re.compile(r"(?<!\d)\.(?!\d)|(?<=\d)\.(?!\d)|(?<!\d)\.(?=\d)")
_COMMA_RE = re.compile(r"(?<!\d),(?!\d)|(?<=\d),(?!\d)|(?<!\d),(?=\d)")


def _evalai_normalize(text: str) -> str:
    chars = [" " if ch in _PUNCT_TO_REPLACE else ch for ch in str(text).replace("\n", " ").replace("\t", " ")]
    text = _COMMA_RE.sub(" ", _PERIOD_RE.sub(" ", "".join(chars).strip().lower()))
    tokens = []
    for token in text.split():
        token = _NUMBER_WORDS.get(token, token)
        if token not in _ARTICLES:
            tokens.append(token)
    return " ".join(tokens)


def _textvqa_soft_accuracy(prediction: str, answers: list[str]) -> float:
    norm_pred = _evalai_normalize(prediction)
    matches = sum(1 for answer in answers if _evalai_normalize(answer) == norm_pred)
    return min(matches / 3.0, 1.0)


class TextVQABenchmark(Benchmark):
    """TextVQA validation with LLaVA OCR prompts and EvalAI-style soft accuracy."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[str | None, int | None] | None = None
        self._items: list[dict[str, DataType]] | None = None
        self._source: str | None = None
        self._n_with_ocr_tokens = 0
        self._dataset_length = len(self._load_items())

    def _load_items(self) -> list[dict[str, DataType]]:
        key = (self._config.dataset_dir, self._config.dataset_limit)
        if self._cache_key == key and self._items is not None:
            return self._items

        root = Path(self._config.dataset_dir)
        data_dir = root
        ann_file = None
        candidate_dirs = [root]
        if root.name.lower() != "textvqa":
            candidate_dirs.append(root / "textvqa")
        for candidate_dir in candidate_dirs:
            for name in TEXTVQA_ANNOTATION_NAMES:
                candidate = candidate_dir / name
                if candidate.exists():
                    data_dir = candidate_dir
                    ann_file = candidate
                    break
            if ann_file is not None:
                break
        if ann_file is None:
            names = ", ".join(TEXTVQA_ANNOTATION_NAMES)
            raise FileNotFoundError(f"TextVQA annotation file not found under {root}. Expected one of: {names}")

        image_dir = data_dir / "train_images"
        if not image_dir.exists():
            raise FileNotFoundError(f"TextVQA image directory not found: {image_dir}")

        annotation_blob = json.loads(ann_file.read_text(encoding="utf-8"))
        annotations = annotation_blob.get("data")
        if not isinstance(annotations, list):
            raise ValueError(f"Expected {ann_file} to contain a top-level 'data' list.")

        shared_root = Path(getattr(self._config, "input_dir", data_dir.parent))
        ocr_jsonl = None
        for candidate in (
            root / "llava_textvqa_val_v051_ocr.jsonl",
            root / TEXTVQA_OCR_RELPATH,
            data_dir / "llava_textvqa_val_v051_ocr.jsonl",
            data_dir.parent / TEXTVQA_OCR_RELPATH,
            shared_root / TEXTVQA_OCR_RELPATH,
        ):
            if candidate.exists():
                ocr_jsonl = candidate
                break

        items: list[dict[str, DataType]] = []
        gt_by_key = {
            (str(row["image_id"]), str(row["question"]).strip().lower()): row
            for row in annotations
        }
        if ocr_jsonl is not None:
            unmatched = 0
            for line in ocr_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                text = self.format_short_answer_prompt(str(row["text"]))
                raw_question = text.split(REFERENCE_OCR_MARKER, 1)[0].strip()
                image_name = Path(str(row["image"])).name
                image_id = Path(image_name).stem
                annotation = gt_by_key.get((image_id, raw_question.lower()))
                if annotation is None:
                    unmatched += 1
                    continue
                answers = [str(answer) for answer in (annotation.get("answers") or [])]
                question_id = annotation.get("question_id", row.get("question_id"))
                items.append(
                    {
                        "question_id": question_id,
                        "item_id": f"textvqa/{question_id}",
                        "category": str(row.get("category", "textvqa")),
                        "image_path": image_dir / image_name,
                        "question": text,
                        "question_raw": raw_question,
                        "answers": answers,
                        "answer": answers[0] if answers else "",
                        "has_ocr_tokens": REFERENCE_OCR_MARKER in text,
                    }
                )
            if unmatched:
                print(f"[warn] TextVQA: skipped {unmatched} OCR rows with no matching annotation in {ann_file.name}")
            self._source = str(ocr_jsonl)
        else:
            for row in annotations:
                raw_question = str(row["question"]).strip()
                tokens: list[str] = []
                for token in row.get("ocr_tokens") or row.get("ocr_info") or []:
                    if isinstance(token, dict):
                        if token.get("word"):
                            tokens.append(str(token["word"]))
                    else:
                        tokens.append(str(token))
                prompt = f"{raw_question}{REFERENCE_OCR_MARKER} {', '.join(tokens)}" if tokens else raw_question
                answers = [str(answer) for answer in (row.get("answers") or [])]
                question_id = row.get("question_id")
                items.append(
                    {
                        "question_id": question_id,
                        "item_id": f"textvqa/{question_id}",
                        "category": "textvqa",
                        "image_path": image_dir / f"{row['image_id']}.jpg",
                        "question": self.format_short_answer_prompt(prompt),
                        "question_raw": raw_question,
                        "answers": answers,
                        "answer": answers[0] if answers else "",
                        "has_ocr_tokens": bool(tokens),
                    }
                )
            self._source = str(ann_file)

        if self._config.dataset_limit is not None:
            items = items[: self._config.dataset_limit]

        self._n_with_ocr_tokens = (
            sum(1 for item in items if bool(item["has_ocr_tokens"])) if self._use_ocr_tokens() else 0
        )
        if ocr_jsonl is None and self._n_with_ocr_tokens == 0:
            print(f"[warn] TextVQA: no OCR JSONL or annotation OCR tokens found. Expected OCR JSONL at {shared_root / TEXTVQA_OCR_RELPATH}.")
        else:
            print(f"[info] TextVQA: using {Path(self._source).name} ({len(items)} questions, {self._n_with_ocr_tokens} with OCR tokens)")
        self._cache_key = key
        self._items = items
        return items

    def _use_ocr_tokens(self) -> bool:
        return bool(self._config.custom_args.get("textvqa_use_ocr_tokens", True))

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        # LLaVA-1.5 appends Rosetta OCR tokens to every question. Backbones that
        # read text natively from high-res images are evaluated without them, so
        # the tokens are strippable rather than baked in.
        keep_ocr = self._use_ocr_tokens()
        for item in self._load_items():
            image_path = Path(item["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"TextVQA image not found: {image_path}")
            question = item["question"]
            has_ocr = item["has_ocr_tokens"]
            if not keep_ocr and REFERENCE_OCR_MARKER in question:
                question = self.format_short_answer_prompt(item["question_raw"])
                has_ocr = False
            yield {
                "question_id": item["question_id"],
                "item_id": item["item_id"],
                "category": item["category"],
                "image_path": str(image_path),
                "question": question,
                "question_raw": item["question_raw"],
                "answers": item["answers"],
                "answer": item["answer"],
                "image": coerce_image_tensor(image_path, source="TextVQA"),
                "record_extra": {"has_ocr_tokens": has_ocr},
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return self.format_short_answer_prompt(raw_question)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return raw_text.strip()

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return _textvqa_soft_accuracy(prediction, [str(answer) for answer in row.get("answers") or []]) > 0.0

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        soft_scores = [
            _textvqa_soft_accuracy(str(record.get("prediction", "")), [str(answer) for answer in record.get("answers") or []])
            for record in records
        ]
        primary_scores = [
            _evalai_normalize(str(record.get("prediction", ""))) == _evalai_normalize(str(record.get("answer", "")))
            for record in records
        ]
        return {
            "metrics": {
                "textvqa_soft_accuracy": round(100 * sum(soft_scores) / max(1, len(soft_scores)), 4),
                "primary_answer_accuracy": round(100 * sum(primary_scores) / max(1, len(primary_scores)), 4),
                "any_answer_accuracy": round(100 * sum(bool(record.get("correct")) for record in records) / max(1, len(records)), 4),
                "n_examples": len(records),
                "n_with_ocr_tokens": self._n_with_ocr_tokens,
            },
            "source": self._source,
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        return (
            f"textvqa_soft_acc={metrics['textvqa_soft_accuracy']:.4f}  "
            f"any_acc={metrics['any_answer_accuracy']:.4f}  "
            f"ocr={metrics['n_with_ocr_tokens']}  "
            f"n={metrics['n_examples']}"
        )
