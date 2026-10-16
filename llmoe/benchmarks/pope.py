import json
from pathlib import Path
from typing import Any, Iterator, override

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.utils import binary_label_metrics, coerce_image_tensor, normalize_yes_no
from llmoe.benchmarks.prototype import Benchmark, DataType


POPE_SPLITS = ("random", "popular", "adversarial")


def _label_file(root: Path, split: str) -> Path:
    candidates = [
        root / f"coco_pope_{split}.json",
        root / "coco" / f"coco_pope_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_entries(path: Path) -> list[dict[str, DataType]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class POPEBenchmark(Benchmark):
    """POPE evaluation following LLaVA's eval_pope.py recipe."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        total = 0
        found = False
        root = Path(config.dataset_dir)
        for split in POPE_SPLITS:
            label_file = _label_file(root, split)
            if not label_file.exists():
                continue
            found = True
            count = len(_load_entries(label_file))
            total += min(config.dataset_limit, count) if config.dataset_limit is not None else count
        super().__init__(config, backbone, paired=False, dataset_length=total if found else config.dataset_limit)

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        root = Path(self._config.dataset_dir)
        image_dir_candidates = [root / "val2014", root / "coco" / "val2014"]
        image_dir = image_dir_candidates[0]
        for path in image_dir_candidates:
            if path.exists():
                image_dir = path
                break
        if not image_dir.exists():
            raise FileNotFoundError(f"POPE image directory not found: {image_dir}")

        for split in POPE_SPLITS:
            label_file = _label_file(root, split)
            if not label_file.exists():
                print(f"[warn] POPE: {label_file} not found, skipping {split}")
                continue
            entries = _load_entries(label_file)
            if self._config.dataset_limit is not None:
                entries = entries[: self._config.dataset_limit]
            for i, row in enumerate(entries):
                image_path = image_dir / str(row["image"])
                if not image_path.exists():
                    raise FileNotFoundError(f"POPE image not found: {image_path}")
                question_raw = str(row["text"]).strip()
                label = str(row["label"]).lower().strip()
                question_id = row.get("question_id", i)
                yield {
                    "question_id": f"{split}/{question_id}",
                    "item_id": f"pope/{split}/{question_id}",
                    "category": split,
                    "image_path": str(image_path),
                    "question": self.format_prompt(question_raw),
                    "question_raw": question_raw,
                    "answer": label,
                    "image": coerce_image_tensor(image_path, source="POPE"),
                }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return self.format_short_answer_prompt(raw_question)

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return normalize_yes_no(raw_text, fallback=None) or "no"

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        return prediction == str(row.get("answer", "")).lower().strip()

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        by_split: dict[str, list[dict[str, DataType]]] = {split: [] for split in POPE_SPLITS}
        for record in records:
            by_split.setdefault(str(record.get("category", "")), []).append(record)

        metrics: dict[str, Any] = {}
        for split in POPE_SPLITS:
            split_records = by_split.get(split) or []
            if not split_records:
                continue
            golds = [record["answer"] for record in split_records]
            preds = [record["prediction"] for record in split_records]
            split_metrics = binary_label_metrics(golds, preds, "yes")
            yes_count = split_metrics["tp"] + split_metrics["fp"]
            yes_ratio = yes_count / split_metrics["n"] if split_metrics["n"] else 0.0
            metrics[split] = {
                "n": split_metrics["n"],
                "accuracy": round(100 * float(split_metrics["accuracy"]), 4),
                "precision": round(100 * float(split_metrics["precision"]), 4),
                "recall": round(100 * float(split_metrics["recall"]), 4),
                "f1": round(100 * float(split_metrics["f1"]), 4),
                "yes_ratio": round(100 * yes_ratio, 4),
            }
        f1_values = [split_metrics["f1"] for split_metrics in metrics.values()]
        metrics["average_f1"] = round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0
        return {"metrics": metrics}

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        return (
            f"avg_f1={metrics['average_f1']:.4f}  "
            f"random={metrics.get('random', {}).get('f1', float('nan')):.4f}  "
            f"popular={metrics.get('popular', {}).get('f1', float('nan')):.4f}  "
            f"adversarial={metrics.get('adversarial', {}).get('f1', float('nan')):.4f}"
        )
