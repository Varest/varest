"""Run manifests: what a HiDrop checkpoint was trained from, and with what.

HiDrop is trained per schedule and per stage, and its two stages chain: the 665K
full finetune must start from *its own* budget's 558K projector. Nothing in a bare
`state_dict` records that, so every artifact this trainer writes carries a
`run_manifest.json` fingerprinting the base model, dataset, schedule, batching,
software versions, and training semantics that produced it.

Reuse then fails closed rather than silently:

* `ft` refuses a projector trained for another schedule, dataset, or software set;
* a resume refuses a checkpoint whose run settings differ;
* inference refuses a checkpoint that never went through full finetuning.

Ported from the sidecar's ``train/provenance.py``.
"""

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import platform
import subprocess
import warnings

import torch
from packaging.version import Version


MANIFEST_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
LLAVA_NEXT_8B_REVISION = "b041c0d0ea0dd0196d147206c210c8d1752fc2da"
LLAVA_NEXT_8B_CACHE_REPOSITORY = "models--llava-hf--llama3-llava-next-8b-hf"


def is_pinned_llava_next_snapshot(path: str | Path) -> bool:
    """Whether a local directory is the exact cached HiDrop base snapshot."""

    candidate = Path(path).expanduser()
    return (
        candidate.is_dir()
        and candidate.name == LLAVA_NEXT_8B_REVISION
        and candidate.parent.name == "snapshots"
        and candidate.parent.parent.name == LLAVA_NEXT_8B_CACHE_REPOSITORY
    )
# The fixed LLMOE runtime. Release-level equality is required: these versions
# decide loss normalization, tokenizer behavior, and attention dispatch.
FIXED_RUNTIME_RELEASES = {
    "torch": "2.11.0",
    "transformers": "5.5.4",
    "tokenizers": "0.22.2",
    "accelerate": "1.13.0",
}
REQUIRED_DISTRIBUTED_PACKAGES = ("deepspeed", "flash-attn")
PINNED_DISTRIBUTED_RELEASES = {"flash-attn": "2.8.3.post1"}
MODEL_ARTIFACT_SUFFIXES = {".bin", ".json", ".model", ".safetensors", ".txt"}
# Training corpora ship as one file (LLaVA-1.5) or sharded tables (LLaVA-NeXT).
DATASET_SHARD_SUFFIXES = {".parquet", ".arrow", ".json", ".jsonl"}
RELEASED_TRAINING_OPTIMIZER = "adamw_torch"
RELEASED_LOSS_SEMANTICS = "transformers_4_37_causal_ce_mean_per_microbatch"
RELEASED_TRAINING_SEMANTICS = {
    "optimizer": RELEASED_TRAINING_OPTIMIZER,
    "loss": RELEASED_LOSS_SEMANTICS,
    "gradient_accumulation_loss_normalization": "mean_each_microbatch_without_cross_device_token_rescaling",
    "selector_gradient": "detached_hard_indices",
    "sampler": "released_llava_modality_length_grouping_without_repetition",
    "text_only_multimodal_graph": "zero_image_with_empty_projector_feature_slice",
}
LLAVA_NEXT_TRAINING_SEMANTICS = {
    "optimizer": RELEASED_TRAINING_OPTIMIZER,
    "loss": RELEASED_LOSS_SEMANTICS,
    "attention_backend": "flash_attention_2_only",
    "flash_attn_release": "2.8.3.post1",
    "gradient_accumulation_loss_normalization": "mean_each_microbatch_without_cross_device_token_rescaling",
    "selector_gradient": "detached_hard_indices",
    "sampler": "local_parquet_random_without_repetition",
    "data": "LLaVA-NeXT-Data_738601_image_rows_plus_40688_text_only_rows",
    "text_only_multimodal_graph": "dummy_anyres_image_with_empty_projector_feature_slice",
    "stage1": "stock_projector_compression_adaptation_without_reinitialization",
    "stage2": "language_model_and_projector_compression_aware_finetuning",
}


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def validate_fixed_training_environment(*, require_distributed_dependencies: bool) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, expected_release in FIXED_RUNTIME_RELEASES.items():
        installed = _distribution_version(package)
        if installed is None:
            raise RuntimeError(f"Required fixed training package {package}=={expected_release} is not installed.")
        versions[package] = installed
        if Version(installed).release != Version(expected_release).release:
            raise RuntimeError(
                f"HiDrop training is pinned to {package}=={expected_release} semantics; found {installed}. "
                "Use the unchanged LLMOE training environment."
            )
    for package in REQUIRED_DISTRIBUTED_PACKAGES:
        installed = _distribution_version(package)
        if require_distributed_dependencies and installed is None:
            raise RuntimeError(
                f"HiDrop cluster training requires {package}; it is missing from the active LLMOE environment."
            )
        if installed is not None:
            versions[package] = installed
            expected_release = PINNED_DISTRIBUTED_RELEASES.get(package)
            if (
                require_distributed_dependencies
                and expected_release is not None
                and Version(installed) != Version(expected_release)
            ):
                raise RuntimeError(
                    f"HiDrop cluster training requires {package}=={expected_release}; "
                    f"found {installed}. Install the pinned Conrad wheel in the active Varest environment."
                )
    if require_distributed_dependencies and not torch.cuda.is_available():
        raise RuntimeError("HiDrop FA2 training requires a CUDA GPU on the training cluster.")
    return versions


def resolve_model_directory(model_name_or_path: str) -> Path | None:
    """Local directory holding the base model, or ``None`` for an unresolvable hub id.

    ``backbone_args.model_name_or_path`` is usually the hub id
    ``llava-hf/llava-1.5-7b-hf`` rather than a path, so the cached snapshot is
    resolved here to keep the fingerprint meaningful. The sidecar only ever
    pointed at a directory.
    """
    candidate = Path(model_name_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=model_name_or_path, local_files_only=True)).resolve()
    except Exception:  # noqa: BLE001 - fingerprint degrades, the run does not fail
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected a provenance input file, got {resolved}.")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _fingerprint_dataset(path: str | Path) -> dict[str, Any]:
    """Fingerprint a training corpus that is either one file or a shard directory.

    LLaVA-1.5 trains from a single JSON, so :func:`_fingerprint_file` was enough.
    LLaVA-NeXT trains from a directory of Parquet shards -- the NeXT trainer sets
    ``data_path`` to ``<root>/data`` -- and fingerprinting that as a file raised
    ``FileNotFoundError`` before the first step, so no NeXT run could start.

    A directory is aggregated exactly like :func:`_fingerprint_model_directory`:
    every shard is content-hashed and the digest is taken over
    ``(relative path, size, sha256)`` in sorted order. That keeps the guarantee
    the single-file fingerprint gave -- a resume is refused if the corpus changed
    at all -- rather than trusting names and sizes. Hashing the 92 GB LLaVA-NeXT
    corpus measures ~1.8 min against multi-hour runs.
    """
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_file():
        return _fingerprint_file(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Expected a provenance input file or directory, got {resolved}.")
    shards = sorted(
        candidate
        for candidate in resolved.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in DATASET_SHARD_SUFFIXES
    )
    if not shards:
        raise FileNotFoundError(
            f"No dataset shards ({', '.join(sorted(DATASET_SHARD_SUFFIXES))}) found under {resolved}."
        )
    aggregate = hashlib.sha256()
    records = []
    total = 0
    for shard in shards:
        relative = shard.relative_to(resolved).as_posix()
        shard_sha = _sha256_file(shard)
        size = int(shard.stat().st_size)
        total += size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(shard_sha.encode("ascii"))
        aggregate.update(b"\0")
        records.append({"path": relative, "size_bytes": size, "sha256": shard_sha})
    return {
        "path": str(resolved),
        "size_bytes": total,
        "sha256": aggregate.hexdigest(),
        "shards": records,
    }


def _fingerprint_model_directory(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Expected a model directory, got {resolved}.")
    files = sorted(
        candidate
        for candidate in resolved.rglob("*")
        if candidate.is_file()
        and ".git" not in candidate.relative_to(resolved).parts
        and candidate.suffix.lower() in MODEL_ARTIFACT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No model/config/tokenizer artifacts found under {resolved}.")
    aggregate = hashlib.sha256()
    records = []
    for candidate in files:
        relative = candidate.relative_to(resolved).as_posix()
        file_sha = _sha256_file(candidate)
        size = int(candidate.stat().st_size)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(file_sha.encode("ascii"))
        aggregate.update(b"\0")
        records.append({"path": relative, "size_bytes": size, "sha256": file_sha})
    return {
        "path": str(resolved),
        "sha256": aggregate.hexdigest(),
        "files": records,
    }


def _fingerprint_base_model(model_name_or_path: str) -> dict[str, Any]:
    directory = resolve_model_directory(model_name_or_path)
    if directory is None:
        warnings.warn(
            f"HiDrop could not resolve {model_name_or_path!r} to a local snapshot; the run manifest "
            "records the identifier without a weight fingerprint.",
            stacklevel=2,
        )
        return {"path": str(model_name_or_path), "sha256": None, "files": []}
    fingerprint = _fingerprint_model_directory(directory)
    fingerprint["model_name_or_path"] = str(model_name_or_path)
    return fingerprint


def _git_state() -> dict[str, Any]:
    # Git resolves the work tree from any directory inside it, so the package
    # directory is enough regardless of where the repo is checked out.
    root = Path(__file__).resolve().parent

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _software_state() -> dict[str, Any]:
    packages = sorted(set(FIXED_RUNTIME_RELEASES) | set(REQUIRED_DISTRIBUTED_PACKAGES))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _distribution_version(name) for name in packages},
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def build_run_manifest(
    *,
    args: dict[str, Any],
    hidrop: dict[str, Any],
    dataset_length: int,
    optimizer: str,
    loss_semantics: str,
    training_semantics: dict[str, str] | None = None,
) -> dict[str, Any]:
    if optimizer != RELEASED_TRAINING_OPTIMIZER or loss_semantics != RELEASED_LOSS_SEMANTICS:
        raise ValueError("HiDrop run manifests only support the released optimizer and loss semantics.")
    model = _fingerprint_base_model(args["model_name_or_path"])
    dataset = _fingerprint_dataset(args["data_path"])
    dataset["rows_used"] = int(dataset_length)
    projector_path = args.get("pretrain_mm_mlp_adapter")
    projector = _fingerprint_file(projector_path) if projector_path else None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": hidrop,
        "training_semantics": dict(training_semantics or RELEASED_TRAINING_SEMANTICS),
        "batching": {
            "world_size": world_size,
            "per_device_train_batch_size": int(args["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(args["gradient_accumulation_steps"]),
            "effective_global_batch_size": (
                world_size
                * int(args["per_device_train_batch_size"])
                * int(args["gradient_accumulation_steps"])
            ),
        },
        "inputs": {
            "model": model,
            "dataset": dataset,
            "image_folder": str(Path(args["image_folder"]).expanduser().resolve(strict=True)),
            "pretrained_projector": projector,
        },
        "run": args,
        "software": _software_state(),
        "code": _git_state(),
        "parity_test": {
            "status": "deferred",
            "target": "H100 training cluster before full training",
        },
    }


def _manifest_candidates(artifact_path: str | Path) -> list[Path]:
    artifact = Path(artifact_path).expanduser().resolve()
    directory = artifact if artifact.is_dir() else artifact.parent
    candidates = [directory / MANIFEST_FILENAME]
    if directory.name.startswith("checkpoint-"):
        candidates.append(directory.parent / MANIFEST_FILENAME)
    return candidates


def _read_artifact_manifest(artifact_path: str | Path, artifact_name: str) -> dict[str, Any]:
    manifest_path = next((path for path in _manifest_candidates(artifact_path) if path.is_file()), None)
    if manifest_path is None:
        raise ValueError(f"Cannot use {artifact_name}: required {MANIFEST_FILENAME} is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("manifest_version", -1)) != MANIFEST_VERSION:
        raise ValueError(
            f"Cannot use {artifact_name}: unsupported manifest version "
            f"{manifest.get('manifest_version')!r}; expected {MANIFEST_VERSION}."
        )
    return manifest


def validate_artifact_provenance(artifact_path: str | Path, current: dict[str, Any], *, artifact_name: str,
                                 resume: bool) -> None:
    recorded = _read_artifact_manifest(artifact_path, artifact_name)
    for component in ("model",):
        expected_sha = current["inputs"][component]["sha256"]
        recorded_sha = recorded.get("inputs", {}).get(component, {}).get("sha256")
        if recorded_sha != expected_sha:
            raise ValueError(
                f"Cannot use {artifact_name}: {component} fingerprint differs from this run "
                f"({recorded_sha!r} != {expected_sha!r})."
            )
    if recorded.get("implementation") != current.get("implementation"):
        raise ValueError(f"Cannot use {artifact_name}: HiDrop implementation/profile metadata differs.")
    if recorded.get("training_semantics") != current.get("training_semantics"):
        raise ValueError(f"Cannot use {artifact_name}: optimizer/loss/selector training semantics differ.")
    recorded_software = recorded.get("software", {})
    current_software = current.get("software", {})
    for key in ("python", "packages", "torch_cuda", "cudnn"):
        if recorded_software.get(key) != current_software.get(key):
            raise ValueError(f"Cannot use {artifact_name}: software setting {key!r} differs.")
    if not resume and current.get("inputs", {}).get("pretrained_projector") is not None:
        if recorded.get("run", {}).get("stage") != "pretrain":
            raise ValueError(f"Cannot use {artifact_name}: its manifest is not from projector pretraining.")
        expected_projector = current["inputs"]["pretrained_projector"]["sha256"]
        recorded_projector = recorded.get("outputs", {}).get("mm_projector", {}).get("sha256")
        if recorded_projector != expected_projector:
            raise ValueError(
                f"Cannot use {artifact_name}: projector fingerprint is missing or differs from its "
                "pretraining manifest."
            )
    if resume:
        recorded_dataset = recorded.get("inputs", {}).get("dataset", {}).get("sha256")
        current_dataset = current["inputs"]["dataset"]["sha256"]
        if recorded_dataset != current_dataset:
            raise ValueError(f"Cannot use {artifact_name}: training dataset fingerprint differs.")
        recorded_rows = recorded.get("inputs", {}).get("dataset", {}).get("rows_used")
        current_rows = current["inputs"]["dataset"]["rows_used"]
        if recorded_rows != current_rows:
            raise ValueError(f"Cannot use {artifact_name}: number of training rows differs.")
        if recorded.get("inputs", {}).get("image_folder") != current.get("inputs", {}).get("image_folder"):
            raise ValueError(f"Cannot use {artifact_name}: training image folder differs.")
        fidelity_settings = (
            "stage",
            "seed",
            "global_batch_size",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "lr_scheduler_type",
            "num_train_epochs",
            "model_max_length",
            "deepspeed_stage",
            "bf16",
            "tf32",
            "gradient_checkpointing",
            "group_by_modality_length",
            "attn_implementation",
            "max_steps",
        )
        for key in fidelity_settings:
            if recorded.get("run", {}).get(key) != current.get("run", {}).get(key):
                raise ValueError(f"Cannot use {artifact_name}: run setting {key!r} differs.")


def validate_inference_checkpoint_provenance(checkpoint_path: str | Path, implementation: dict[str, Any], *,
                                             expected_stage: str = "finetune") -> dict[str, Any]:
    """Refuse to benchmark a checkpoint that is not a finished HiDrop model."""
    manifest = _read_artifact_manifest(checkpoint_path, "inference checkpoint")
    recorded_stage = manifest.get("run", {}).get("stage")
    if recorded_stage != expected_stage:
        raise ValueError(
            f"Cannot use inference checkpoint: its manifest records stage {recorded_stage!r}, "
            f"expected {expected_stage!r}."
        )
    if manifest.get("implementation") != implementation:
        raise ValueError("Cannot use inference checkpoint: HiDrop implementation/profile metadata differs.")
    if expected_stage == "finetune" and manifest.get("inputs", {}).get("pretrained_projector") is None:
        raise ValueError("Cannot use inference checkpoint: no pretrained projector provenance was recorded.")
    expected_training_semantics = (
        LLAVA_NEXT_TRAINING_SEMANTICS
        if implementation.get("backbone_family") == "llava_next"
        else RELEASED_TRAINING_SEMANTICS
    )
    if manifest.get("training_semantics", {}) != expected_training_semantics:
        raise ValueError("Cannot use inference checkpoint: recorded training semantics are incompatible.")
    return manifest


def write_run_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / MANIFEST_FILENAME
    temporary = output / f".{MANIFEST_FILENAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)
    return path


def record_output_file(manifest: dict[str, Any], name: str, path: str | Path) -> None:
    manifest.setdefault("outputs", {})[str(name)] = _fingerprint_file(path)
