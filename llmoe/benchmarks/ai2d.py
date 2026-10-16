import csv
import json
import re
from pathlib import Path
from typing import Any, Iterator, override

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.utils import coerce_image_tensor


def _find_ai2d_root(path: Path) -> Path:
    if (path / "ai2d" / "images").is_dir() and (path / "ai2d" / "questions").is_dir():
        return path / "ai2d"
    if (path / "images").is_dir() and (path / "questions").is_dir():
        return path
    raise FileNotFoundError(
        f"Could not find AI2D images/ and questions/ under {path}. "
        f"Expected either {path / 'ai2d' / 'images'} or {path / 'images'}."
    )


def _find_abc_images(dataset_dir: Path, ai2d_root: Path) -> Path | None:
    """The ABC-labelled image variant, if the distribution ships it.

    AI2D-ABC re-renders each diagram with its answer options replaced by A/B/C/D
    labels. The ``abcLabel`` questions ask which letter marks something, so they
    are only answerable from that variant -- against the plain diagram the
    letters are not on the page at all, and the model is reduced to guessing.
    The variant ships beside the ai2d tree, so it is found rather than
    configured; ``ai2d_abc_images_dir`` overrides, and a falsy value opts out.
    """
    for candidate in (
        dataset_dir / "ai2d_abc" / "images",
        ai2d_root.parent / "ai2d_abc" / "images",
        ai2d_root / "ai2d_abc" / "images",
    ):
        if candidate.is_dir():
            return candidate
    return None


def _load_test_ids(path: Path, ai2d_root: Path) -> set[str]:
    filenames = ("ai2d-test-ids.txt", "ai2d_test_ids.csv")
    search_dirs = (path, ai2d_root, ai2d_root.parent)
    for candidate in (directory / name for directory in search_dirs for name in filenames):
        if not candidate.exists():
            continue
        ids: set[str] = set()
        with candidate.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                cell = row[0].strip()
                if cell and cell.lower() != "image_id":
                    ids.add(cell)
        if not ids:
            raise ValueError(f"AI2D test-id file is empty: {candidate}")
        return ids
    searched = ", ".join(str(directory) for directory in search_dirs)
    raise FileNotFoundError(
        "AI2D test-id file not found. Looked for "
        + " or ".join(filenames)
        + f" in: {searched}"
    )


def _sort_key(image_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(image_id))
    except ValueError:
        return (1, image_id)


def _extract_letter(text: str, n_choices: int) -> str | None:
    valid = {chr(ord("A") + i) for i in range(n_choices)}
    match = re.search(r"\b([A-Z])\b", text.strip())
    if match and match.group(1) in valid:
        return match.group(1)
    return None


def _enumerate_test_questions(ai2d_root: Path, test_ids: set[str], *,
                              abc_images_dir: Path | None = None) -> list[dict[str, DataType]]:
    """Enumerate the AI2D test questions.

    ``abc_images_dir`` supplies the ABC-labelled ("masked") image variant, where
    in-diagram text labels are replaced by the letters the ``abcLabel`` questions
    refer to. The plain 2016 release ships no such images, so those questions are
    unanswerable without it. Only the images that actually carry abc questions
    differ from the originals, so this directory is used for every question whose
    image is present in it -- matching the official lmms-eval/VLMEvalKit setup.
    """
    questions_dir = ai2d_root / "questions"
    images_dir = ai2d_root / "images"
    records: list[dict[str, DataType]] = []
    missing_question_examples: list[str] = []
    missing_image_examples: list[str] = []

    for image_id in sorted(test_ids, key=_sort_key):
        question_file = questions_dir / f"{image_id}.png.json"
        image_path = images_dir / f"{image_id}.png"
        # One diagram serves both question kinds, so the variant is chosen per
        # question below rather than here: the ABC rendering replaces the
        # diagram's own labels with A/B/C/D, which is what the abcLabel
        # questions ask about and exactly what the text-option questions need
        # to still be able to read.
        abc_image_path = None
        if abc_images_dir is not None:
            candidate = abc_images_dir / f"{image_id}.png"
            if candidate.exists():
                abc_image_path = candidate
        if not question_file.exists():
            if len(missing_question_examples) < 5:
                missing_question_examples.append(str(question_file))
            continue
        if not image_path.exists():
            if len(missing_image_examples) < 5:
                missing_image_examples.append(str(image_path))
            continue

        data = json.loads(question_file.read_text(encoding="utf-8"))
        for ordinal, (question_text, question_meta) in enumerate(data.get("questions", {}).items()):
            options = list(question_meta.get("answerTexts") or [])
            answer = question_meta.get("correctAnswer")
            if not options or answer is None:
                continue
            try:
                answer_idx = int(answer)
            except (TypeError, ValueError):
                print(f"[warn] AI2D: bad correctAnswer={answer!r} for {image_id}, skipping.")
                continue
            if not (0 <= answer_idx < len(options)):
                print(
                    f"[warn] AI2D: bad correctAnswer={answer_idx} for {image_id} "
                    f"(n_opts={len(options)}), skipping."
                )
                continue
            question_id = str(question_meta.get("questionId") or f"{image_id}-{ordinal}")
            abc_label = bool(question_meta.get("abcLabel", False))
            records.append(
                {
                    "question_id": question_id,
                    "image_id": image_id,
                    "image_path": abc_image_path if (abc_label and abc_image_path) else image_path,
                    "question_raw": str(question_text),
                    "options": options,
                    "answer_idx": answer_idx,
                    "abc_label": abc_label,
                }
            )
    missing_questions = sum(
        not (questions_dir / f"{image_id}.png.json").exists()
        for image_id in test_ids
    )
    missing_images = sum(
        (questions_dir / f"{image_id}.png.json").exists()
        and not (images_dir / f"{image_id}.png").exists()
        for image_id in test_ids
    )
    if missing_questions:
        print(
            f"[warn] AI2D: skipped {missing_questions} test IDs with no questions file "
            f"(examples: {', '.join(missing_question_examples)})"
        )
    if missing_images:
        print(
            f"[warn] AI2D: skipped {missing_images} test IDs with no image "
            f"(examples: {', '.join(missing_image_examples)})"
        )
    return records


class AI2DBenchmark(Benchmark):
    """AI2D test-set evaluation using Roby's raw-dataset prompt and scoring."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[str | None, int | None, str | None] | None = None
        self._items: list[dict[str, DataType]] | None = None
        self._dataset_length: int = len(self._load_items(config))

    def _load_items(self, cfg: Config) -> list[dict[str, DataType]]:
        root = Path(cfg.dataset_dir)
        ai2d_root = _find_ai2d_root(root)
        # Unset means "use the ABC variant if it is there": the abcLabel questions
        # need it, and silently answering them from the plain diagram scores near
        # chance without anything looking wrong. A falsy value opts out; a path
        # overrides the search.
        setting = cfg.custom_args.get("ai2d_abc_images_dir", True)
        if setting is True:
            abc_images_dir = _find_abc_images(root, ai2d_root)
        elif not setting:
            abc_images_dir = None
        else:
            abc_images_dir = Path(setting)
            if not abc_images_dir.is_dir():
                raise FileNotFoundError(f"ai2d_abc_images_dir does not exist: {abc_images_dir}")

        key = (cfg.dataset_dir, cfg.dataset_limit, str(abc_images_dir) if abc_images_dir else None)
        if self._cache_key == key and self._items is not None:
            return self._items

        test_ids = _load_test_ids(root, ai2d_root)
        items = _enumerate_test_questions(ai2d_root, test_ids, abc_images_dir=abc_images_dir)
        if cfg.dataset_limit is not None:
            items = items[: cfg.dataset_limit]
        self._cache_key = key
        self._items = items
        n_abc = sum(1 for item in items if item.get("abc_label"))
        if abc_images_dir is None and n_abc:
            print(
                f"[warn] AI2D: {n_abc} abcLabel questions will be answered from the plain "
                f"diagrams -- the ABC-labelled variant was not found beside {ai2d_root} and "
                f"was not given as ai2d_abc_images_dir. Those questions ask about letters "
                f"that are only drawn in that variant, so expect near-chance on them.",
                flush=True,
            )
        print(
            f"[info] AI2D: {len(items)} test questions across "
            f"{len({item['image_id'] for item in items})} images"
            + (f", {n_abc} abcLabel from {abc_images_dir}" if abc_images_dir else "")
        )
        return items

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for item in self._load_items(self._config):
            image_path = Path(item["image_path"])
            options = list(item["options"])
            answer = chr(65 + int(item["answer_idx"]))
            abc_label = bool(item["abc_label"])
            yield {
                "question_id": item["question_id"],
                "item_id": f"ai2d/{item['question_id']}",
                "category": "abc_options" if abc_label else "text_options",
                "image_path": str(image_path),
                "question": self.format_prompt(item["question_raw"], options=options),
                "question_raw": item["question_raw"],
                "answer": answer,
                "image": coerce_image_tensor(image_path, source="AI2D"),
                "n_choices": len(options),
                "record_extra": {
                    "image_id": item["image_id"],
                    "options": options,
                    "answer_idx": item["answer_idx"],
                    "abc_label": abc_label,
                },
            }

    @override
    def format_prompt(self, raw_question: str, *, options: list[str] | None = None) -> str:
        rendered = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(options or []))
        return self.format_option_letter_prompt(f"{raw_question}\nOptions:\n{rendered}")

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return _extract_letter(raw_text, int(row.get("n_choices", 4))) or ""

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return prediction == row.get("answer")

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        correct = sum(bool(record.get("correct")) for record in records)
        abc_records = [record for record in records if record.get("abc_label")]
        text_records = [record for record in records if not record.get("abc_label")]
        abc_correct = sum(bool(record.get("correct")) for record in abc_records)
        text_correct = sum(bool(record.get("correct")) for record in text_records)
        n = len(records)
        metrics = {
            "accuracy": round(100 * correct / max(1, n), 4),
            "accuracy_text_options": round(100 * text_correct / max(1, len(text_records)), 4),
            "accuracy_abc_options": round(100 * abc_correct / max(1, len(abc_records)), 4),
            "n_text_options": len(text_records),
            "n_abc_options": len(abc_records),
        }
        return {
            "metrics": metrics,
            "num_examples": n,
            "valid_examples": n,
            "skipped_examples": 0,
            "num_cases": n,
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        m = metrics_dict["metrics"]
        return (
            f"accuracy={m['accuracy']:.4f}  "
            f"text={m['accuracy_text_options']:.4f}  "
            f"abc={m['accuracy_abc_options']:.4f}  "
            f"n={metrics_dict['num_examples']}"
        )
