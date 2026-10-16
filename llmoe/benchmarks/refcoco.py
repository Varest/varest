import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, override

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.utils import coerce_image_tensor


DEFAULT_EVAL_MATRIX: dict[str, tuple[str, tuple[str, ...]]] = {
    "refcoco": ("unc", ("val", "testA", "testB")),
    "refcoco+": ("unc", ("val", "testA", "testB")),
    "refcocog": ("umd", ("val", "test")),
}
EXPECTED_SPLIT_COUNTS: dict[tuple[str, str], int] = {
    ("refcoco", "val"): 10834,
    ("refcoco", "testA"): 5657,
    ("refcoco", "testB"): 5095,
    ("refcoco+", "val"): 10758,
    ("refcoco+", "testA"): 5726,
    ("refcoco+", "testB"): 4889,
    ("refcocog", "val"): 4896,
    ("refcocog", "test"): 9602,
}
VALID_DATASETS = tuple(DEFAULT_EVAL_MATRIX)
VALID_SPLITS = tuple(sorted({split for _, splits in DEFAULT_EVAL_MATRIX.values() for split in splits}))
IOU_THRESHOLD = 0.5
PROMPT_TEMPLATE = (
    "Please provide the bounding box coordinate of the region this sentence "
    "describes: {expression}. Return only normalized [x1, y1, x2, y2] "
    "coordinates between 0 and 1."
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
BRACKET_RE = re.compile(r"[\[\(]([^\]\)]{3,})[\]\)]")


@dataclass(frozen=True)
class BoxParseResult:
    parsed: bool
    valid: bool
    box: list[float] | None
    coord_format: str
    error: str | None = None
    ambiguous_scale: bool = False


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _extract_four_numbers(text: str) -> list[float] | None:
    if not text:
        return None
    for match in BRACKET_RE.finditer(text):
        nums = NUMBER_RE.findall(match.group(1))
        if len(nums) >= 4:
            return [float(value) for value in nums[:4]]
    nums = NUMBER_RE.findall(text)
    return [float(value) for value in nums[:4]] if len(nums) >= 4 else None


def _normalize_prediction_box(text: str, row: dict[str, DataType]) -> BoxParseResult:
    coords = _extract_four_numbers(text)
    if coords is None:
        return BoxParseResult(False, False, None, "invalid", "no_four_numbers")

    max_abs = max(abs(value) for value in coords)
    padded_side = float(row.get("padded_side") or 0)
    raw_width = float(row.get("raw_width") or 0)
    raw_height = float(row.get("raw_height") or 0)
    use_padding = bool(row.get("refcoco_use_padding", True))
    ambiguous = False

    if max_abs <= 1.5:
        coord_format = "normalized"
        scaled = coords
    else:
        x_divisor = padded_side if use_padding else raw_width
        y_divisor = padded_side if use_padding else raw_height
        pixel_limit = max(x_divisor, y_divisor) * 1.05
        ambiguous = max_abs <= min(1000.0, pixel_limit)
        if x_divisor > 0 and y_divisor > 0 and max_abs <= pixel_limit:
            coord_format = "pixel_scale"
            scaled = [
                coords[0] / x_divisor,
                coords[1] / y_divisor,
                coords[2] / x_divisor,
                coords[3] / y_divisor,
            ]
        elif max_abs <= 1050.0:
            coord_format = "thousand_scale"
            scaled = [value / 1000.0 for value in coords]
        elif x_divisor > 0 and y_divisor > 0 and max_abs <= max(x_divisor, y_divisor) * 1.5:
            coord_format = "pixel_scale"
            scaled = [
                coords[0] / x_divisor,
                coords[1] / y_divisor,
                coords[2] / x_divisor,
                coords[3] / y_divisor,
            ]
        else:
            return BoxParseResult(True, False, None, "invalid", "unsupported_scale")

    box = [min(1.0, max(0.0, value)) for value in scaled]
    if box[2] <= box[0] or box[3] <= box[1]:
        return BoxParseResult(True, False, box, "invalid", "degenerate_box", ambiguous)
    return BoxParseResult(True, True, box, coord_format, None, ambiguous)


class RefCOCOBenchmark(Benchmark):
    """RefCOCO/RefCOCO+/RefCOCOg grounding with standard splits and IoU@0.5."""

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cache_key: tuple[Any, ...] | None = None
        self._items: list[dict[str, DataType]] | None = None
        self._source_counts: dict[str, int] = {}
        self._dataset_length = len(self._load_items())

    def _load_items(self) -> list[dict[str, DataType]]:
        raw_datasets = self._config.custom_args.get("refcoco_datasets")
        if raw_datasets is None:
            requested_datasets = VALID_DATASETS
        elif isinstance(raw_datasets, str):
            requested_datasets = (raw_datasets,)
        else:
            requested_datasets = tuple(str(value) for value in raw_datasets)
        unknown_datasets = sorted(set(requested_datasets) - set(VALID_DATASETS))
        if unknown_datasets:
            raise ValueError(f"Unknown RefCOCO dataset(s): {unknown_datasets}. Valid: {VALID_DATASETS}")
        requested_datasets = tuple(dataset for dataset in VALID_DATASETS if dataset in requested_datasets)

        raw_splits = self._config.custom_args.get("refcoco_splits")
        if raw_splits is None:
            generic_split = str(self._config.custom_args.get("dataset_split", "val"))
            requested_splits = None if generic_split in {"standard", "all"} else (generic_split,)
        elif isinstance(raw_splits, str):
            requested_splits = (raw_splits,)
        else:
            requested_splits = tuple(str(value) for value in raw_splits)
        if requested_splits is not None:
            unknown_splits = sorted(set(requested_splits) - set(VALID_SPLITS))
            if unknown_splits:
                raise ValueError(f"Unknown RefCOCO split(s): {unknown_splits}. Valid: {VALID_SPLITS}")

        use_padding = not bool(self._config.custom_args.get("refcoco_padding_off", False))
        key = (
            self._config.dataset_dir,
            requested_datasets,
            requested_splits,
            use_padding,
            self._config.dataset_limit,
        )
        if self._cache_key == key and self._items is not None:
            return self._items

        root = Path(self._config.dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"RefCOCO directory not found: {root}")

        selected_pairs: list[tuple[str, str, str]] = []
        for dataset in requested_datasets:
            split_by, splits = DEFAULT_EVAL_MATRIX[dataset]
            for split in splits:
                if requested_splits is not None and split not in requested_splits:
                    continue
                selected_pairs.append((dataset, split_by, split))
        if not selected_pairs:
            raise ValueError("RefCOCO selection produced no splits. Check refcoco_datasets/refcoco_splits.")

        items: list[dict[str, DataType]] = []
        source_counts: dict[str, int] = {}
        missing_images: list[str] = []
        for dataset, split_by, split in selected_pairs:
            dataset_dir = root / "annotations" / dataset
            if not (dataset_dir / "instances.json").exists():
                dataset_dir = root / dataset
            if not (dataset_dir / "instances.json").exists():
                raise FileNotFoundError(
                    f"Could not find {dataset} annotations under {root}. Expected instances.json "
                    f"in {root / 'annotations' / dataset} or {root / dataset}."
                )
            refs_path = dataset_dir / f"refs({split_by}).p"
            instances_path = dataset_dir / "instances.json"
            if not refs_path.exists():
                raise FileNotFoundError(f"Missing RefCOCO refs file: {refs_path}")

            with refs_path.open("rb") as handle:
                refs = pickle.load(handle, encoding="latin1")
            instances = json.loads(instances_path.read_text(encoding="utf-8"))
            annotations_by_id = {item["id"]: item for item in instances["annotations"]}
            images_by_id = {item["id"]: item for item in instances["images"]}

            split_rows: list[dict[str, Any]] = []
            for ref in refs:
                source_split = str(ref["split"])
                if split in {"testA", "testB", "testC"}:
                    split_matches = split[-1] in source_split
                elif split in {"testAB", "testBC", "testAC"}:
                    split_matches = source_split == split
                elif split == "test":
                    split_matches = "test" in source_split
                else:
                    split_matches = source_split == split
                if not split_matches:
                    continue
                annotation = annotations_by_id.get(ref["ann_id"])
                image = images_by_id.get(ref["image_id"])
                if annotation is None or image is None:
                    continue
                for sentence in ref["sentences"]:
                    split_rows.append(
                        {
                            "ref_id": ref["ref_id"],
                            "sent_id": sentence["sent_id"],
                            "expression": sentence["sent"],
                            "image_id": ref["image_id"],
                            "file_name": image["file_name"],
                            "raw_width": int(image["width"]),
                            "raw_height": int(image["height"]),
                            "bbox_xywh": annotation["bbox"],
                            "source_split": source_split,
                            "category_id": ref["category_id"],
                        }
                    )
            if not split_rows:
                raise ValueError(f"RefCOCO selected split is empty: {dataset}/{split}")
            expected = EXPECTED_SPLIT_COUNTS.get((dataset, split))
            if expected is not None and len(split_rows) != expected:
                raise ValueError(f"RefCOCO expected {expected} sentence examples for {dataset}/{split}, got {len(split_rows)}.")
            source_counts[f"{dataset}_{split}"] = len(split_rows)

            image_path_cache: dict[str, str] = {}
            for row in split_rows:
                file_name = str(row["file_name"])
                if file_name not in image_path_cache:
                    image_path = None
                    for candidate in (
                        root / "images" / "train2014" / file_name,
                        root / "train2014" / file_name,
                        root / "images" / "mscoco" / "images" / "train2014" / file_name,
                        root / "mscoco" / "images" / "train2014" / file_name,
                        root / file_name,
                    ):
                        if candidate.exists():
                            image_path = candidate
                            break
                    if image_path is None:
                        missing_images.append(file_name)
                        continue
                    image_path_cache[file_name] = str(image_path)

                x, y, width, height = [float(value) for value in row["bbox_xywh"]]
                raw_width = int(row["raw_width"])
                raw_height = int(row["raw_height"])
                if use_padding:
                    side = max(raw_width, raw_height)
                    x_offset = (side - raw_width) // 2 if raw_width < side else 0
                    y_offset = (side - raw_height) // 2 if raw_height < side else 0
                    gt_box = [
                        (x + x_offset) / side,
                        (y + y_offset) / side,
                        (x + width + x_offset) / side,
                        (y + height + y_offset) / side,
                    ]
                else:
                    side = max(raw_width, raw_height)
                    x_offset = 0
                    y_offset = 0
                    gt_box = [x / raw_width, y / raw_height, (x + width) / raw_width, (y + height) / raw_height]
                question_id = f"{dataset}/{split}/{row['ref_id']}/{row['sent_id']}"
                items.append(
                    {
                        "question_id": question_id,
                        "item_id": f"refcoco/{question_id}",
                        "category": dataset,
                        "image_path": image_path_cache[file_name],
                        "question_raw": str(row["expression"]).strip(),
                        "answer": gt_box,
                        "gt_box_normalized": gt_box,
                        "padded_side": side,
                        "raw_width": raw_width,
                        "raw_height": raw_height,
                        "refcoco_use_padding": use_padding,
                        "record_extra": {
                            "dataset": dataset,
                            "split": split,
                            "split_by": split_by,
                            "ref_id": row["ref_id"],
                            "sent_id": row["sent_id"],
                            "image_id": row["image_id"],
                            "file_name": file_name,
                            "expression": str(row["expression"]).strip(),
                            "bbox_xywh": [float(value) for value in row["bbox_xywh"]],
                            "gt_box_normalized": gt_box,
                            "gt_coord_space": "padded_normalized" if use_padding else "raw_normalized",
                            "padding_x_offset": x_offset,
                            "padding_y_offset": y_offset,
                            "padded_side": side,
                            "raw_width": raw_width,
                            "raw_height": raw_height,
                            "refcoco_use_padding": use_padding,
                        },
                    }
                )

        if missing_images:
            sample = ", ".join(sorted(set(missing_images))[:5])
            raise FileNotFoundError(f"RefCOCO missing {len(set(missing_images))} referenced images. Examples: {sample}")

        if self._config.dataset_limit is not None:
            items = items[: self._config.dataset_limit]

        self._cache_key = key
        self._items = items
        self._source_counts = source_counts
        return items

    @override
    def validate_config(self) -> None:
        self._load_items()

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        for row in self._load_items():
            image_path = Path(row["image_path"])
            expression = str(row["question_raw"]).strip()
            yield {
                "question_id": row["question_id"],
                "item_id": row["item_id"],
                "category": row["category"],
                "image_path": str(image_path),
                "question": self.format_prompt(expression),
                "question_raw": expression,
                "answer": row["answer"],
                "image": coerce_image_tensor(image_path, source="RefCOCO"),
                "gt_box_normalized": row["gt_box_normalized"],
                "padded_side": row["padded_side"],
                "raw_width": row["raw_width"],
                "raw_height": row["raw_height"],
                "refcoco_use_padding": row["refcoco_use_padding"],
                "record_extra": row["record_extra"],
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return PROMPT_TEMPLATE.format(expression=str(raw_question).strip())

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return str(raw_text).strip()

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> bool:
        parsed = _normalize_prediction_box(prediction, row)
        if not parsed.valid or parsed.box is None:
            return False
        return _iou(parsed.box, [float(value) for value in row.get("gt_box_normalized") or []]) >= IOU_THRESHOLD

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        by_split: dict[tuple[str, str], list[dict[str, DataType]]] = defaultdict(list)
        parsed_total = 0
        valid_total = 0
        coord_counts: Counter[str] = Counter()
        ambiguous_total = 0
        for record in records:
            parsed = _normalize_prediction_box(str(record.get("prediction", "")), record)
            iou_value = 0.0
            correct = False
            if parsed.valid and parsed.box is not None:
                iou_value = _iou(parsed.box, [float(value) for value in record.get("gt_box_normalized") or []])
                correct = iou_value >= IOU_THRESHOLD
            record["parsed_box"] = parsed.box
            record["box_parse_valid"] = parsed.valid
            record["box_parse_error"] = parsed.error
            record["coord_format"] = parsed.coord_format
            record["coord_scale_ambiguous"] = parsed.ambiguous_scale
            record["iou"] = round(iou_value, 4)
            record["correct"] = correct
            parsed_total += int(parsed.parsed)
            valid_total += int(parsed.valid)
            coord_counts[parsed.coord_format] += 1
            ambiguous_total += int(parsed.ambiguous_scale)
            by_split[(str(record.get("dataset")), str(record.get("split")))].append(record)

        per_split: dict[str, dict[str, Any]] = {}
        split_accuracies: list[float] = []
        correct_total = 0
        for (dataset, split), split_records in sorted(by_split.items()):
            n = len(split_records)
            correct = sum(bool(record.get("correct")) for record in split_records)
            parsed = sum(record.get("box_parse_error") != "no_four_numbers" for record in split_records)
            valid = sum(bool(record.get("box_parse_valid")) for record in split_records)
            split_coord_counts = Counter(str(record.get("coord_format")) for record in split_records)
            split_acc = round(100 * correct / max(1, n), 4)
            split_accuracies.append(split_acc)
            correct_total += correct
            per_split[f"{dataset}_{split}"] = {
                "dataset": dataset,
                "split": split,
                "split_by": DEFAULT_EVAL_MATRIX[dataset][0],
                "num_examples": n,
                "accuracy@0.5": split_acc,
                "parse_rate": round(100 * parsed / max(1, n), 4),
                "valid_box_rate": round(100 * valid / max(1, n), 4),
                "invalid_box_count": n - valid,
                "coord_format_counts": {
                    "normalized": split_coord_counts.get("normalized", 0),
                    "thousand_scale": split_coord_counts.get("thousand_scale", 0),
                    "pixel_scale": split_coord_counts.get("pixel_scale", 0),
                    "invalid": split_coord_counts.get("invalid", 0),
                },
            }

        n_total = len(records)
        return {
            "num_examples": n_total,
            "valid_examples": n_total,
            "skipped_examples": 0,
            "num_cases": n_total,
            "scoring": "iou@0.5_padded" if not records or records[0].get("refcoco_use_padding") else "iou@0.5_raw",
            "accuracy": round(sum(split_accuracies) / len(split_accuracies), 4) if split_accuracies else 0.0,
            "metrics": {
                "accuracy@0.5_macro": round(sum(split_accuracies) / len(split_accuracies), 4) if split_accuracies else 0.0,
                "accuracy@0.5_example_weighted": round(100 * correct_total / max(1, n_total), 4),
                "parse_rate": round(100 * parsed_total / max(1, n_total), 4),
                "valid_box_rate": round(100 * valid_total / max(1, n_total), 4),
                "invalid_box_count": n_total - valid_total,
                "ambiguous_coord_scale_count": ambiguous_total,
                "n": n_total,
                "coord_format_counts": {
                    "normalized": coord_counts.get("normalized", 0),
                    "thousand_scale": coord_counts.get("thousand_scale", 0),
                    "pixel_scale": coord_counts.get("pixel_scale", 0),
                    "invalid": coord_counts.get("invalid", 0),
                },
            },
            "per_split": per_split,
            "source_counts": self._source_counts,
            "case_wise_metrics": [
                {
                    "question_id": record.get("question_id"),
                    "dataset": record.get("dataset"),
                    "split": record.get("split"),
                    "ref_id": record.get("ref_id"),
                    "sent_id": record.get("sent_id"),
                    "iou": record.get("iou"),
                    "correct": int(bool(record.get("correct"))),
                    "coord_format": record.get("coord_format"),
                    "box_parse_valid": bool(record.get("box_parse_valid")),
                }
                for record in records
            ],
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        return (
            f"acc_macro={metrics['accuracy@0.5_macro']:.4f}  "
            f"acc_weighted={metrics['accuracy@0.5_example_weighted']:.4f}  "
            f"parse={metrics['parse_rate']:.4f}  "
            f"valid_box={metrics['valid_box_rate']:.4f}  "
            f"n={metrics['n']}"
        )
