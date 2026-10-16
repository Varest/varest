from pathlib import Path
from typing import Any, Iterator, override

import numpy as np

from llmoe.backbone import VLM
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.config import Config
from llmoe.utils import binary_label_metrics, coerce_image_tensor, normalize_yes_no, pair_accuracy, safe_div

PERCEPTION: tuple[str, ...] = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
)
COGNITION: tuple[str, ...] = (
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)
MME_CATEGORIES: tuple[str, ...] = PERCEPTION + COGNITION
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

MME_YES_NO_SUFFIX = " Please answer yes or no."
MME_YES_NO_SUFFIX_DOUBLE = "  Please answer yes or no."


def is_official_mme_dir(path: str | Path | None) -> bool:
    if path is None:
        return False
    root = Path(path)
    return all((root / category).exists() for category in ("existence", "OCR", "scene"))


def _require_official_mme_dir(dataset_dir: str | Path | None) -> Path:
    if dataset_dir is None:
        raise FileNotFoundError("MME requires cfg.dataset_dir to point at the official MME release folder.")
    root = Path(dataset_dir)
    if is_official_mme_dir(root):
        return root
    nested_root = root / "MME_Benchmark"
    if is_official_mme_dir(nested_root):
        return nested_root
    raise FileNotFoundError(
        "MME requires the official release layout with category folders "
        f"such as existence, OCR, and scene. Got: {root}"
    )


def _question_files(category_dir: Path) -> list[Path]:
    qa_dir = category_dir / "questions_answers_YN"
    if qa_dir.exists():
        return sorted(qa_dir.glob("*.txt"))
    return sorted(category_dir.glob("*.txt"))


def _resolve_image_path(category_dir: Path, stem: str) -> Path:
    search_dirs = [category_dir / "images", category_dir]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in IMAGE_EXTENSIONS:
            candidate = search_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not find image for {category_dir.name}/{stem}")


def _parse_qa_lines(txt_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            raise ValueError(f"Expected tab-separated question/answer line in {txt_path}: {line}")
        question, answer = line.rsplit("\t", 1)
        pairs.append((question.strip(), answer.strip().lower().replace(".", "")))
    if len(pairs) != 2:
        raise ValueError(f"Official MME expects 2 yes/no rows per file, got {len(pairs)} in {txt_path}")
    return pairs


def _iter_official_mme_rows(dataset_dir: str | Path) -> Iterator[dict[str, DataType]]:
    root = Path(dataset_dir)
    for category in MME_CATEGORIES:
        category_dir = root / category
        if not category_dir.exists():
            raise FileNotFoundError(f"Missing official MME category folder: {category_dir}")
        for txt_path in _question_files(category_dir):
            image_path = _resolve_image_path(category_dir, txt_path.stem)
            for question_index, (question, answer) in enumerate(_parse_qa_lines(txt_path)):
                yield {
                    "question_id": txt_path.stem,
                    "item_id": f"{category}/{txt_path.stem}/{question_index}",
                    "category": category,
                    "question": question,
                    "question_mme": question,
                    "answer": answer,
                    "image_path": str(image_path),
                    "image": coerce_image_tensor(image_path, source="MME"),
                }


def get_mme_dataset_length(*, limit: int | None = None, dataset_dir: str | Path | None = None) -> int:
    root = _require_official_mme_dir(dataset_dir)
    total = sum(2 * len(_question_files(root / category)) for category in MME_CATEGORIES)
    return total if limit is None else min(limit, total)


def load_mme_dataset(
        split: str = "test",
        limit: int | None = None,
        dataset_dir: str | Path | None = None,
) -> Iterator[dict[str, DataType]]:
    """Stream MME rows as `{question_id, category, question, answer, image}`.

    MME uses the official release folder configured by `cfg.dataset_dir`.
    Rows stay in paired order so each image has its two yes/no questions.
    """
    root = _require_official_mme_dir(dataset_dir)
    source = _iter_official_mme_rows(root)
    for i, row in enumerate(source):
        if limit is not None and i >= limit:
            break
        yield row


def parse_pred_ans(pred: str) -> str:
    """Normalize a model's free-text answer into 'yes' | 'no' | 'other'."""
    return normalize_yes_no(pred, fallback="other")


def _classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records or not all("answer" in record and "prediction" in record for record in records):
        return {}

    golds = [str(record["answer"]).lower().strip().replace(".", "") for record in records]
    preds = [str(record["prediction"]).lower().strip().replace(".", "") for record in records]
    yes = binary_label_metrics(golds, preds, "yes")
    no = binary_label_metrics(golds, preds, "no")
    supports = {
        "yes": sum(gold == "yes" for gold in golds),
        "no": sum(gold == "no" for gold in golds),
    }
    n = len(records)
    accuracy = safe_div(sum(g == p for g, p in zip(golds, preds)), n)
    macro_precision = (float(yes["precision"]) + float(no["precision"])) / 2
    macro_recall = (float(yes["recall"]) + float(no["recall"])) / 2
    macro_f1 = (float(yes["f1_score"]) + float(no["f1_score"])) / 2
    weighted_f1 = safe_div(
        float(yes["f1_score"]) * supports["yes"] + float(no["f1_score"]) * supports["no"],
        supports["yes"] + supports["no"],
    )
    pred_labels = ("yes", "no", "other")
    confusion = {
        gold: {pred: sum(g == gold and p == pred for g, p in zip(golds, preds)) for pred in pred_labels}
        for gold in ("yes", "no")
    }

    return {
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "f1_score": macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_yes": yes["f1_score"],
        "f1_no": no["f1_score"],
        "precision_yes": yes["precision"],
        "precision_no": no["precision"],
        "recall_yes": yes["recall"],
        "recall_no": no["recall"],
        "valid_samples": n,
        "support": supports,
        "prediction_distribution": {label: preds.count(label) for label in pred_labels},
        "confusion_matrix": confusion,
    }


def compute_case_wise_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["category"], record["question_id"]), []).append(record)

    case_rows: list[dict[str, Any]] = []
    for (category, question_id), rows in groups.items():
        rows = sorted(rows, key=lambda row: row.get("item_id", ""))
        num_correct = sum(bool(row["correct"]) for row in rows)
        case_rows.append(
            {
                "case_id": f"{category}/{question_id}",
                "category": category,
                "question_id": question_id,
                "image_path": rows[0].get("image_path"),
                "num_questions": len(rows),
                "num_correct": int(num_correct),
                "accuracy": safe_div(num_correct, len(rows)),
                "accuracy_plus": 1.0 if num_correct == len(rows) else 0.0,
                "both_correct": num_correct == len(rows),
                "questions": [
                    {
                        "item_id": row.get("item_id"),
                        "question": row["question"],
                        "answer": row["answer"],
                        "prediction": row["prediction"],
                        "prediction_raw": row["prediction_raw"],
                        "correct": bool(row["correct"]),
                    }
                    for row in rows
                ],
            }
        )
    return sorted(case_rows, key=lambda row: row["case_id"])


def compute_mme_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute official MME scores plus binary yes/no classification metrics."""
    if not records:
        raise ValueError("compute_mme_metrics requires at least one record")

    cats = np.asarray([r["category"] for r in records])
    qids = np.asarray([r["question_id"] for r in records])
    correct = np.fromiter((bool(r["correct"]) for r in records), dtype=bool, count=len(records))

    cat_uniq, cat_idx = np.unique(cats, return_inverse=True)
    cat_counts = np.bincount(cat_idx)
    acc_per_cat = np.bincount(cat_idx, weights=correct.astype(np.int64)) / cat_counts

    sep = "\x1f"
    pair_keys = np.char.add(np.char.add(cats.astype(str), sep), qids.astype(str))
    pair_uniq, pair_idx, pair_counts = np.unique(pair_keys, return_inverse=True, return_counts=True)
    if not (pair_counts == 2).all():
        bad = pair_uniq[pair_counts != 2].tolist()
        raise AssertionError(
            f"MME expects exactly 2 questions per (category, question_id); "
            f"found {len(bad)} group(s) with count != 2: {bad[:5]}"
        )

    pair_correct = pair_accuracy(correct, pair_idx)
    pair_cat_idx = cat_idx[np.unique(pair_idx, return_index=True)[1]]
    pair_cat_counts = np.bincount(pair_cat_idx, minlength=len(cat_uniq))
    accplus_per_cat = (
            np.bincount(pair_cat_idx, weights=pair_correct.astype(np.int64), minlength=len(cat_uniq))
            / pair_cat_counts
    )

    categories: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(cat_uniq):
        acc = float(acc_per_cat[i])
        accp = float(accplus_per_cat[i])
        categories[str(name)] = {
            "accuracy": acc,
            "accuracy_plus": accp,
            "score": 100.0 * acc + 100.0 * accp,
            "n_questions": int(cat_counts[i]),
            "n_images": int(pair_cat_counts[i]),
        }

    def _rollup(group: tuple[str, ...]) -> dict[str, float | int]:
        q_mask = np.isin(cats, np.asarray(group))
        p_cat_names = cat_uniq[pair_cat_idx]
        p_mask = np.isin(p_cat_names, np.asarray(group))
        n_q = int(q_mask.sum())
        n_p = int(p_mask.sum())
        acc = float(correct[q_mask].mean()) if n_q else 0.0
        accp = float(pair_correct[p_mask].mean()) if n_p else 0.0
        return {
            "accuracy": acc,
            "accuracy_plus": accp,
            "score": 100.0 * acc + 100.0 * accp,
            "n_questions": n_q,
            "n_images": n_p,
        }

    perception = _rollup(PERCEPTION)
    cognition = _rollup(COGNITION)
    perception["score_sum"] = float(sum(categories[c]["score"] for c in PERCEPTION if c in categories))
    cognition["score_sum"] = float(sum(categories[c]["score"] for c in COGNITION if c in categories))

    overall_acc = float(correct.mean())
    overall_accp = float(pair_correct.mean())
    total_score = float(sum(c["score"] for c in categories.values()))
    classification = _classification_metrics(records)

    return {
        "categories": categories,
        "perception": perception,
        "cognition": cognition,
        "classification": classification,
        "overall": {
            "accuracy": overall_acc,
            "accuracy_plus": overall_accp,
            "total_score": total_score,
            "mme_perception_score": float(perception["score_sum"]),
            "mme_cognition_score": float(cognition["score_sum"]),
        },
        "n_questions": int(len(records)),
        "n_images": int(len(pair_uniq)),
    }


class MMEBenchmark(Benchmark):
    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=True, dataset_length=get_mme_dataset_length(
            limit=config.dataset_limit, dataset_dir=config.dataset_dir))

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for row in load_mme_dataset(
            split=self._config.custom_args.get("dataset_split", "test"),
            limit=self._config.dataset_limit,
            dataset_dir=self._config.dataset_dir,
        ):
            yield {**row, "question": self.format_prompt(str(row["question_mme"]))}

    @override
    def format_prompt(self, raw_question: str) -> str:
        stripped = str(raw_question).strip()
        for suffix in (MME_YES_NO_SUFFIX_DOUBLE, MME_YES_NO_SUFFIX):
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].rstrip()
                break
        return self.format_short_answer_prompt(stripped)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return parse_pred_ans(raw_text)

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        gold = str(row["answer"]).lower().strip().replace(".", "")
        return prediction == gold

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        return {
            "metrics": compute_mme_metrics(records),
            "case_wise": compute_case_wise_metrics(records),
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        m = metrics_dict["metrics"]["overall"]
        cls = metrics_dict["metrics"]["classification"]
        return (
            f"acc={m['accuracy']:.4f}  acc+={m['accuracy_plus']:.4f}  "
            f"f1={cls.get('f1_score', float('nan')):.4f}  "
            f"MME-P={m['mme_perception_score']:.4f}  "
            f"MME-C={m['mme_cognition_score']:.4f}  "
            f"total={m['total_score']:.4f}"
        )


def _self_test() -> None:
    """Hand-checked correctness test, no model required."""
    records = [
        {"category": "existence", "question_id": "ex_a", "answer": "yes", "prediction": "yes", "correct": True},
        {"category": "existence", "question_id": "ex_a", "answer": "no", "prediction": "no", "correct": True},
        {"category": "existence", "question_id": "ex_b", "answer": "yes", "prediction": "yes", "correct": True},
        {"category": "existence", "question_id": "ex_b", "answer": "no", "prediction": "yes", "correct": False},
        {"category": "code_reasoning", "question_id": "cr_a", "answer": "yes", "prediction": "yes", "correct": True},
        {"category": "code_reasoning", "question_id": "cr_a", "answer": "no", "prediction": "no", "correct": True},
        {"category": "code_reasoning", "question_id": "cr_b", "answer": "yes", "prediction": "yes", "correct": True},
        {"category": "code_reasoning", "question_id": "cr_b", "answer": "no", "prediction": "no", "correct": True},
    ]

    out = compute_mme_metrics(records)
    ex = out["categories"]["existence"]
    cr = out["categories"]["code_reasoning"]
    assert ex["accuracy"] == 0.75, ex
    assert ex["accuracy_plus"] == 0.5, ex
    assert ex["score"] == 125.0, ex
    assert ex["n_questions"] == 4 and ex["n_images"] == 2, ex
    assert cr["accuracy"] == 1.0 and cr["accuracy_plus"] == 1.0 and cr["score"] == 200.0, cr

    p = out["perception"]
    assert p["accuracy"] == 0.75 and p["accuracy_plus"] == 0.5 and p["score"] == 125.0, p
    assert p["score_sum"] == 125.0, p
    c = out["cognition"]
    assert c["accuracy"] == 1.0 and c["accuracy_plus"] == 1.0 and c["score"] == 200.0, c
    assert c["score_sum"] == 200.0, c

    o = out["overall"]
    assert o["accuracy"] == 7 / 8, o
    assert o["accuracy_plus"] == 0.75, o
    assert out["n_questions"] == 8 and out["n_images"] == 4, out
    assert out["classification"]["valid_samples"] == 8, out["classification"]

    bad = list(records)
    bad[0] = {**bad[0], "question_id": "ex_orphan"}
    try:
        compute_mme_metrics(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected unpaired records to raise")

    for raw, expected in [
        ("Yes.", "yes"),
        ("no", "no"),
        ("Y", "yes"),
        ("N", "no"),
        ("Yes, the cat is on the mat.", "yes"),
        ("Maybe", "other"),
        ("", "other"),
    ]:
        got = parse_pred_ans(raw)
        assert got == expected, (raw, got, expected)

    print("benchmarks.mme self-test passed")


if __name__ == "__main__":
    _self_test()
