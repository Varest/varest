"""LLMOE ``tr`` / ``ft`` entry points for QuietPrune's adapter self-distillation.

QuietPrune trains one thing: the text-to-vision adapter that turns the prompt into
``[Q-CLS]`` (Sec. 4.1). The vision tower, the projector, the token embeddings, and
the language model are all frozen, and nothing about the pruning itself is learned
-- the 2x2 geometry and the drop schedule are fixed. So this is a small, cheap job
next to the two-stage projector/LoRA recipes the other trained baselines here run:
the paper reports roughly 40 minutes on one A100 for 10K samples.

The objective is self-distillation against the *same* model without pruning
(Sec. 4.1)::

    L = kl_weight * KL(teacher_answer || student_answer) + ce_weight * CE(student, answer)

The teacher is the frozen full-token LLaVA forward; the student is the same
weights with QuietPrune installed. Both see the same prompt and the same answer,
so the KL term asks the adapter to pick a ``[Q-CLS]`` whose pruning leaves the
answer distribution where the unpruned model put it, while the CE term keeps it
anchored to the ground truth rather than to a teacher that may itself be wrong.

``tr`` starts from the inverse initialisation, which is already close enough to
useful that the run is short; ``ft`` continues from a checkpoint and is otherwise
the same loop, exactly as ``SoftTopk`` does. Both write an adapter checkpoint that
``compressor_args.checkpoint_path`` reads back for benchmarking.

Training data
-------------
The paper's mix is LLaVA-Instruct, ShareGPT4V, ChartQA, ScreenQA, SynthDoG, and
MultiVQA, sampled to 10K. The NeXT configuration streams the staged
``LLaVA-NeXT-Data/data/train-*.parquet`` shards and their embedded image bytes;
the legacy LLaVA-1.5 configuration can still read a normal LLaVA JSON manifest.
``max_samples`` defaults to the paper's 10K.

Configuration::

    compressor_args:
      pruning_rate: 0.25          # Sec. 4.1; the default when no rate knob is set
    training_args:
      dataset_dir: LLaVA-Tuning
      num_epochs: 1
      batch_size: 1
      learning_rate: 0.0001
      gradient_accumulation_steps: 8
      max_samples: 10000
      kl_weight: 1.0
      ce_weight: 1.0
      checkpoint_name: quietprune.pt

    llmoe --config local --compressor QuietPrune --custom_args quietprune_train.yaml tr

The adapter is budget-agnostic on purpose: the paper trains at ``R = 25%`` and
evaluates at whatever ``R`` the budget asks for, so one checkpoint serves every
row of a token-budget sweep.
"""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from json import load
from os import environ
from pathlib import Path
import random
from typing import Any

import pyarrow.parquet as pq
import torch
from PIL import Image
from mle.vars import ExpConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from llmoe.backbone import VLM
from llmoe.compressor.quietprune import TRAINING_PRUNING_RATE, QuietPruneCompressor
from llmoe.compressor.quietprune.quietprune_next import (
    QUIETPRUNE_NEXT_FLASH_ATTN_VERSION,
    QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION,
)
from llmoe.config import Config
from llmoe.weights import BACKBONE_SLUGS, backbone_key


# The harness dataset layout shared with the other trained baselines. QuietPrune's
# own mixture is not staged here; `LLaVA-Tuning` carries the LLaVA-Instruct rows
# the paper names, and `training_args.data_json` overrides both entries.
DEFAULT_DATASET_DIR = "LLaVA-Tuning"
PREFERRED_ANNOTATIONS = (
    "llava_v1_5_mix665k.json",
    "llava_instruct_150k.json",
    "llava_next_raw_format_processed.json",
    "blip_laion_cc_sbu_558k.json",
)
IGNORE_INDEX = -100
# Sec. 4.1: the paper's 10K mixed sample, ~40 minutes on one A100.
DEFAULT_MAX_SAMPLES = 10_000


# ------------------------------------------------------------ environment check


def _dataset_dir(config: ExpConfig) -> Path:
    """Where the mixture lives, honouring ``training_args.dataset_dir``.

    The environment check runs against a bare :class:`ExpConfig`, which carries no
    ``custom_args``, so the override is read defensively rather than assumed.
    """
    custom_args = getattr(config, "custom_args", None)
    training_args = custom_args.get("training_args") if isinstance(custom_args, Mapping) else None
    name = (
        training_args.get("dataset_dir", DEFAULT_DATASET_DIR)
        if isinstance(training_args, Mapping)
        else DEFAULT_DATASET_DIR
    )
    return Path(config.input_dir) / str(name)


def _find_annotations(root: Path) -> Path:
    for name in PREFERRED_ANNOTATIONS:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    matches = sorted(root.rglob("*.json"))
    if not matches:
        raise FileNotFoundError(f"No JSON annotations found under {root}")
    return matches[0]


def _find_parquet_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("data/train-*.parquet")))


def _dataset_check(config: ExpConfig) -> str:
    dataset_dir = _dataset_dir(config)
    if not dataset_dir.is_dir():
        return f"Missing {dataset_dir.name} dataset under {config.input_dir}"
    parquet_files = _find_parquet_files(dataset_dir)
    if parquet_files:
        return f"OK with {dataset_dir.name} ({len(parquet_files)} parquet shards)"
    try:
        annotations = _find_annotations(dataset_dir)
    except FileNotFoundError:
        return f"Missing {dataset_dir.name} JSON annotations under {dataset_dir}"
    return f"OK with {dataset_dir.name} ({annotations.name})"


def quietprune_tr_fn_cd(config: ExpConfig) -> str:
    return _dataset_check(config)


def quietprune_ft_fn_cd(config: ExpConfig) -> str:
    return _dataset_check(config)


# --------------------------------------------------------------------- dataset


class _ConversationDataset(Dataset):
    """Single-turn ``(image, question, answer)`` rows from a LLaVA-style JSON mixture.

    QuietPrune distils one answer distribution per sample, so multi-turn rows are
    reduced to their last assistant turn and the question that prompted it -- the
    reference's own normalisation. Text-only rows are dropped: without an image
    there is nothing to prune and the sample would train the adapter on a forward
    that never reaches the vision tower.
    """

    def __init__(self, root: Path, annotations: Path, *, limit: int | None = None) -> None:
        self.root: Path = root
        with annotations.open(encoding="utf-8") as file:
            records = load(file)
        if isinstance(records, Mapping):
            for key in ("data", "annotations", "samples"):
                if isinstance(records.get(key), list):
                    records = records[key]
                    break
        if not isinstance(records, list):
            raise TypeError(f"Expected a JSON list in {annotations}")
        usable = [
            record
            for record in records
            if isinstance(record, Mapping) and (record.get("image") or record.get("image_path"))
        ]
        self.records: list[Mapping[str, Any]] = usable[:limit] if limit is not None else usable
        if not self.records:
            raise ValueError(f"No image-bearing training samples found in {annotations}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.records[index]

    def image_path(self, image: str) -> Path:
        candidate = Path(image)
        for path in (
            candidate,
            self.root / candidate,
            self.root / "images" / candidate,
            self.root / "images" / candidate.name,
        ):
            if path.is_file():
                return path
        raise FileNotFoundError(f"Could not find training image {image!r} under {self.root}")


class _ParquetConversationDataset(IterableDataset):
    """Sequential, shard-aware reader for the staged 86 GB NeXT dataset.

    Images are embedded in the parquet ``image.bytes`` field.  Iterating row
    groups avoids extracting/duplicating the 51 GB tar distribution and avoids
    random one-row reads that would decompress a ~16 MB row group per sample.
    Shard order and a bounded row buffer provide deterministic mixing.
    """

    def __init__(
        self,
        files: Sequence[Path],
        *,
        limit: int | None,
        seed: int,
        shuffle_buffer: int,
        parquet_batch_size: int,
    ) -> None:
        super().__init__()
        self.files = tuple(Path(path) for path in files)
        if not self.files:
            raise ValueError("No LLaVA-NeXT parquet shards were supplied")
        self.limit = None if limit is None else int(limit)
        if self.limit is not None and self.limit < 1:
            raise ValueError("max_samples must be >= 1")
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.parquet_batch_size = int(parquet_batch_size)
        if self.shuffle_buffer < 1 or self.parquet_batch_size < 1:
            raise ValueError("shuffle_buffer and parquet_batch_size must be >= 1")

    def __len__(self) -> int:
        if self.limit is None:
            return sum(pq.ParquetFile(path).metadata.num_rows for path in self.files)
        return self.limit

    @staticmethod
    def _usable(record: Mapping[str, Any]) -> bool:
        image = record.get("image")
        if not isinstance(image, Mapping) or not image.get("bytes"):
            return False
        try:
            _question_and_answer(record)
        except ValueError:
            return False
        return True

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        files = list(self.files)
        random.Random(self.seed).shuffle(files)
        files = files[worker_id::workers]
        quota = None
        if self.limit is not None:
            quota = max(0, (self.limit + workers - 1 - worker_id) // workers)
            if quota == 0:
                return
        rng = random.Random(self.seed + worker_id * 1_000_003)
        buffer: list[Mapping[str, Any]] = []
        emitted = 0

        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=self.parquet_batch_size,
                columns=["id", "conversations", "data_source", "image"],
                use_threads=False,
            ):
                for record in batch.to_pylist():
                    if not isinstance(record, Mapping) or not self._usable(record):
                        continue
                    buffer.append(record)
                    if len(buffer) < self.shuffle_buffer:
                        continue
                    index = rng.randrange(len(buffer))
                    yield buffer.pop(index)
                    emitted += 1
                    if quota is not None and emitted >= quota:
                        return

        rng.shuffle(buffer)
        for record in buffer:
            if quota is not None and emitted >= quota:
                return
            yield record
            emitted += 1


def _speaker(turn: Mapping[str, Any]) -> str:
    return str(turn.get("from", turn.get("role", ""))).lower()


def _turn_text(turn: Mapping[str, Any]) -> str:
    value = turn.get("value", turn.get("content", ""))
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        parts = [
            str(part.get("text", ""))
            for part in value
            if isinstance(part, Mapping) and part.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return str(value)


def _question_and_answer(record: Mapping[str, Any]) -> tuple[str, str]:
    """Reduce a record to one ``(question, answer)`` pair, ``<image>`` markers stripped.

    The processor inserts its own placeholders, and ``[Q-CLS]`` is pooled over the
    prompt, so a literal ``<image>`` left in the text would put the placeholder
    embedding into the query -- which is exactly what the masking in
    ``_query_to_qcls`` removes.
    """
    conversations = record.get("conversations", record.get("messages"))
    if isinstance(conversations, Sequence) and not isinstance(conversations, (str, bytes)):
        turns = [turn for turn in conversations if isinstance(turn, Mapping)]
        answer_index = next(
            (
                index
                for index in range(len(turns) - 1, -1, -1)
                if _speaker(turns[index]) in {"gpt", "assistant"}
            ),
            None,
        )
        if answer_index is not None:
            answer = _turn_text(turns[answer_index]).strip()
            question = next(
                (
                    _turn_text(turns[index])
                    for index in range(answer_index - 1, -1, -1)
                    if _speaker(turns[index]) in {"human", "user"}
                ),
                "",
            )
            question = question.replace("<image>", "").strip()
            if question and answer:
                return question, answer

    question = str(record.get("question", record.get("prompt", ""))).replace("<image>", "").strip()
    answer = str(record.get("answer", record.get("response", record.get("caption", "")))).strip()
    if not question or not answer:
        raise ValueError("Training sample needs a question/prompt and an answer/response")
    return question, answer


def _open_training_image(
    dataset: _ConversationDataset | _ParquetConversationDataset,
    record: Mapping[str, Any],
) -> Image.Image:
    image = record.get("image", record.get("image_path"))
    if isinstance(image, Mapping):
        payload = image.get("bytes")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return Image.open(BytesIO(bytes(payload))).convert("RGB")
        image = image.get("path")
    if not image:
        raise ValueError("Training sample is missing image bytes/path")
    if not isinstance(dataset, _ConversationDataset):
        raise FileNotFoundError(
            f"Parquet record {record.get('id')!r} has no embedded image bytes "
            f"(path={image!r}); the staged raw tar archives are intentionally not extracted."
        )
    return Image.open(dataset.image_path(str(image))).convert("RGB")


def _make_batch(
    backbone: VLM,
    dataset: _ConversationDataset | _ParquetConversationDataset,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Tokenise one batch as prompt + answer, un-trimmed.

    Returns the full-token (teacher) inputs plus the per-row answer length. The
    student's inputs are the same tensors with the ``<image>`` runs trimmed, which
    the compressor does itself, so both branches stay token-for-token identical
    outside the visual span.
    """
    prompts: list[str] = []
    answers: list[str] = []
    images: list[Image.Image] = []
    for record in records:
        question, answer = _question_and_answer(record)
        images.append(_open_training_image(dataset, record))
        prompts.append(question)
        answers.append(answer)

    tokenizer = getattr(backbone.processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("QuietPrune training requires a processor with a tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    eos = tokenizer.eos_token or ""
    suffixes = [f" {answer}{eos}" for answer in answers]
    text = [prompt + suffix for prompt, suffix in zip(backbone.format_prompts(prompts), suffixes)]
    try:
        inputs = backbone.processor(
            images=backbone.format_images(images), text=text, return_tensors="pt", padding=True
        )
    finally:
        for image in images:
            image.close()

    answer_lengths = [
        len(tokenizer(suffix, add_special_tokens=False)["input_ids"]) for suffix in suffixes
    ]
    tensors = {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    return tensors, answer_lengths


def _answer_labels(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, answer_lengths: Sequence[int]
) -> torch.Tensor:
    """Supervise the answer tokens only; everything else is ``IGNORE_INDEX``.

    Positions are taken from the right end of each row's *active* span, so this
    holds for the teacher and the student alike even though trimming the visual
    placeholders shortens the student's sequence.
    """
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for row, length in enumerate(answer_lengths):
        active = attention_mask[row].nonzero(as_tuple=False).flatten()
        span = min(int(length), int(active.numel()))
        if span:
            positions = active[-span:]
            labels[row, positions] = input_ids[row, positions]
    return labels


# ----------------------------------------------------------------------- losses


def answer_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal-LM cross entropy over the answer positions only."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    if not bool((shift_labels != IGNORE_INDEX).any()):
        return shift_logits.sum() * 0.0
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def _answer_prediction_logits(logits: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
    """The logits that predict each supervised label, one ragged row per sample."""
    rows: list[torch.Tensor] = []
    for row in range(labels.shape[0]):
        positions = (labels[row] != IGNORE_INDEX).nonzero(as_tuple=False).flatten() - 1
        rows.append(logits[row].index_select(0, positions[positions >= 0]))
    return rows


def answer_kl_divergence_loss(
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """KL between the teacher's and the student's answer-token distributions.

    Answer positions are gathered per side rather than shared, because trimming
    the ``<image>`` placeholders makes the student's sequence shorter than the
    teacher's; the answer tokens themselves are identical, so the two ragged rows
    line up one-for-one.
    """
    student_rows = _answer_prediction_logits(student_logits, student_labels)
    teacher_rows = _answer_prediction_logits(teacher_logits, teacher_labels)
    student_parts: list[torch.Tensor] = []
    teacher_parts: list[torch.Tensor] = []
    for student_row, teacher_row in zip(student_rows, teacher_rows, strict=True):
        count = min(student_row.shape[0], teacher_row.shape[0])
        if count:
            student_parts.append(student_row[:count])
            teacher_parts.append(teacher_row[:count])
    if not student_parts:
        return student_logits.sum() * 0.0, 0

    student_answer = torch.cat(student_parts, dim=0).float()
    teacher_answer = torch.cat(teacher_parts, dim=0).float().to(student_answer.device)
    log_probabilities = F.log_softmax(student_answer / temperature, dim=-1)
    teacher_probabilities = F.softmax(teacher_answer / temperature, dim=-1)
    # The temperature**2 factor keeps the distillation gradient's scale comparable
    # to the CE term's as the temperature moves (Hinton et al.).
    loss = (
        F.kl_div(log_probabilities, teacher_probabilities, reduction="batchmean") * temperature**2
    )
    return loss, int(student_answer.shape[0])


# ---------------------------------------------------------------------- workers


def _require_flash_attention_2_package() -> str:
    """Require the pinned external FA2 wheel; never accept a kernels fallback."""

    try:
        installed = version("flash-attn")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "QuietPrune/LLaVA-NeXT training requires flash-attn=="
            f"{QUIETPRUNE_NEXT_FLASH_ATTN_VERSION}; install the pinned Conrad wheel."
        ) from error
    if installed != QUIETPRUNE_NEXT_FLASH_ATTN_VERSION:
        raise RuntimeError(
            "QuietPrune/LLaVA-NeXT training requires flash-attn=="
            f"{QUIETPRUNE_NEXT_FLASH_ATTN_VERSION}, got {installed}."
        )
    package = import_module("flash_attn")
    import_module("flash_attn_2_cuda")
    if not callable(getattr(package, "flash_attn_func", None)):
        raise RuntimeError("The pinned flash-attn package does not expose flash_attn_func")
    return installed


def _effective_next_attention_implementations(model: Any) -> tuple[str | None, str | None]:
    """Read the backends that Transformers actually installed on both towers."""

    inner = getattr(model, "model", None)
    language_model = getattr(inner, "language_model", None)
    vision_tower = getattr(inner, "vision_tower", None)
    text_config = getattr(language_model, "config", None)
    vision_config = getattr(vision_tower, "config", None)
    return (
        getattr(text_config, "_attn_implementation", None),
        getattr(vision_config, "_attn_implementation", None),
    )


def _require_llava_next_flash_attention_2(backbone: VLM) -> str | None:
    """Enforce FA2 for NeXT training while leaving other backbones unchanged."""

    if getattr(backbone, "compressor_family", None) != "llava_next":
        return None
    requested = getattr(backbone, "_attn_implementation", None)
    if requested != QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION:
        raise ValueError(
            "QuietPrune/LLaVA-NeXT training is FlashAttention-2-only. Set "
            "backbone_args.attn_implementation: flash_attention_2; SDPA remains "
            "the inference default."
        )
    if getattr(backbone, "_dtype", None) not in {torch.float16, torch.bfloat16}:
        raise ValueError("FlashAttention 2 training requires backbone dtype float16 or bfloat16")
    return _require_flash_attention_2_package()


def _require_effective_llava_next_flash_attention_2(model: Any) -> tuple[str, str]:
    """Fail if Transformers silently selected another backend for either tower."""

    text_attention, vision_attention = _effective_next_attention_implementations(model)
    expected = QUIETPRUNE_NEXT_TRAINING_ATTN_IMPLEMENTATION
    if text_attention != expected or vision_attention != expected:
        raise RuntimeError(
            "QuietPrune/LLaVA-NeXT training requires effective FlashAttention 2 "
            "for both Llama-3 and CLIP; got "
            f"text={text_attention!r}, vision={vision_attention!r}."
        )
    return text_attention, vision_attention


def _training_options(config: Config) -> dict[str, Any]:
    options = config.custom_args.get("training_args", {})
    if not isinstance(options, Mapping):
        raise TypeError("training_args must be a mapping")
    return dict(options)


def _build_compressor(config: Config, *, finetune: bool) -> QuietPruneCompressor:
    """Instantiate the compressor with the training-time defaults applied.

    The rate defaults to the paper's ``R = 25%`` because none of the evaluation
    budgets belong in training: the adapter is budget-agnostic, and distilling at
    an aggressive rate would fit it to one column of the sweep. The adapter is
    pinned to fp32 so the optimizer state and gradients stay there while autocast
    handles the forward.
    """
    compressor_args = config.custom_args.get("compressor_args", {})
    if not isinstance(compressor_args, Mapping):
        raise TypeError("compressor_args must be a mapping")
    compressor_args = dict(compressor_args)
    if not any(
        key in compressor_args
        for key in ("pruning_rate", "target_average_pruning_rate", "max_tokens")
    ):
        compressor_args["pruning_rate"] = TRAINING_PRUNING_RATE
    compressor_args.setdefault("adapter_dtype", "float32")
    if finetune and not compressor_args.get("checkpoint_path"):
        checkpoint_from_env = environ.get("QUIETPRUNE_CHECKPOINT_PATH")
        if checkpoint_from_env:
            compressor_args["checkpoint_path"] = checkpoint_from_env
    if finetune and not compressor_args.get("checkpoint_path"):
        raise ValueError(
            "QuietPrune `ft` continues from a trained adapter. Set compressor_args.checkpoint_path "
            "to the checkpoint `tr` wrote, or run `tr` to start from the inverse initialisation."
        )
    compressor = QuietPruneCompressor(**compressor_args)
    # Preserve the paper's canonical per-stage R=0.25 only inside tr/ft. Normal
    # NeXT evaluation must supply retain_ratio and derive its target per image.
    compressor._next_training_rate_mode = True
    return compressor


def _run_training(backbone: VLM, config: Config, *, finetune: bool) -> None:
    """Distil the adapter against the frozen full-token teacher, then checkpoint it."""
    options = _training_options(config)
    flash_attn_version = _require_llava_next_flash_attention_2(backbone)
    dataset_dir = Path(config.input_dir) / str(options.get("dataset_dir", DEFAULT_DATASET_DIR))
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing {dataset_dir.name} dataset under {config.input_dir}")
    data_json = options.get("data_json")
    data_format = str(options.get("data_format", "auto")).lower()
    if data_format not in {"auto", "json", "parquet"}:
        raise ValueError("training_args.data_format must be auto, json, or parquet")
    parquet_files = _find_parquet_files(dataset_dir)
    use_parquet = data_format == "parquet" or (
        data_format == "auto" and data_json is None and bool(parquet_files)
    )
    if use_parquet and not parquet_files:
        raise FileNotFoundError(f"No data/train-*.parquet shards under {dataset_dir}")
    annotations: Path | None = None
    image_root = Path(options["image_root"]) if options.get("image_root") else dataset_dir
    if not use_parquet:
        annotations = Path(data_json) if data_json else _find_annotations(dataset_dir)
        if not annotations.is_file():
            raise FileNotFoundError(f"QuietPrune training annotations not found: {annotations}")

    num_epochs = int(options.get("num_epochs", 1))
    batch_size = int(options.get("batch_size", 1))
    learning_rate = float(options.get("learning_rate", 1e-4))
    weight_decay = float(options.get("weight_decay", 0.0))
    accumulation = int(options.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(options.get("max_grad_norm", 1.0))
    num_workers = int(environ.get("QUIETPRUNE_NUM_WORKERS", options.get("num_workers", 0)))
    kl_weight = float(options.get("kl_weight", 1.0))
    ce_weight = float(options.get("ce_weight", 1.0))
    temperature = float(options.get("temperature", 1.0))
    seed = int(options.get("seed", 2026))
    if num_epochs < 1 or batch_size < 1 or accumulation < 1 or num_workers < 0:
        raise ValueError("num_epochs, batch_size, and gradient_accumulation_steps must be >= 1")
    if kl_weight < 0.0 or ce_weight < 0.0:
        raise ValueError("kl_weight and ce_weight must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if getattr(backbone, "compressor_family", None) == "llava_next" and batch_size != 1:
        raise NotImplementedError(
            "QuietPrune/LLaVA-NeXT training requires batch_size=1 because each "
            "image has a different AnyRes crop/packed sequence."
        )

    limit = config.dataset_limit
    option_limit = options.get("max_samples", DEFAULT_MAX_SAMPLES)
    if option_limit is not None:
        limit = int(option_limit) if limit is None else min(limit, int(option_limit))
    if use_parquet:
        dataset: _ConversationDataset | _ParquetConversationDataset = _ParquetConversationDataset(
            parquet_files,
            limit=limit,
            seed=seed,
            shuffle_buffer=int(options.get("shuffle_buffer", 1024)),
            parquet_batch_size=int(options.get("parquet_batch_size", 16)),
        )
        shuffle = False
    else:
        assert annotations is not None
        dataset = _ConversationDataset(image_root, annotations, limit=limit)
        shuffle = True
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=list,
        pin_memory=torch.cuda.is_available(),
    )

    output_dir = Path(config.output_dir) / "quietprune"
    output_dir.mkdir(parents=True, exist_ok=True)
    # The adapter is sized to the decoder it inverts -- 4096 -> 1024 on
    # LLaVA-1.5-7B, 5120 -> 1024 on the 13B, a different projector again on Qwen --
    # so the backbone belongs in the filename once it is not the released one. A
    # 13B adapter written as `quietprune.pt` would overwrite the 7B's and fail to
    # load anywhere. LLaVA-1.5-7B keeps the original name so existing checkpoints
    # still resolve.
    key = backbone_key(backbone)
    infix = "" if key in (None, "", "llava") else f"_{BACKBONE_SLUGS.get(key, key)}"
    stem = f"quietprune_finetuned{infix}" if finetune else f"quietprune{infix}"
    checkpoint_path = output_dir / str(options.get("checkpoint_name", f"{stem}.pt"))

    compressor = _build_compressor(config, finetune=finetune)
    backbone.validate_compressor(compressor)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = backbone.model
    effective_text_attention: str | None = None
    effective_vision_attention: str | None = None
    if getattr(backbone, "compressor_family", None) == "llava_next":
        effective_text_attention, effective_vision_attention = (
            _require_effective_llava_next_flash_attention_2(model)
        )
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(backbone._device)
    original_training = model.training
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    # The teacher and the student are the same weights; only the adapter learns, so
    # the VLM is frozen once and eval-mode for both branches.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    try:
        compressor.install(backbone)
        compressor.train()
        compressor._training_metadata = {
            "stage": "ft" if finetune else "tr",
            "recipe": "adapter_only_self_distillation_kl_plus_answer_ce",
            "dataset": str(dataset_dir),
            "data_format": "parquet" if use_parquet else "json",
            "max_samples": limit,
            "seed": seed,
            "epochs": num_epochs,
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": accumulation,
            "learning_rate": learning_rate,
            "kl_weight": kl_weight,
            "ce_weight": ce_weight,
            "temperature": temperature,
            "attention_implementation": getattr(backbone, "_attn_implementation", None),
            "effective_text_attention_implementation": effective_text_attention,
            "effective_vision_attention_implementation": effective_vision_attention,
            "flash_attn_version": flash_attn_version,
            "flash_attn_wheel": environ.get("QUIETPRUNE_FA2_WHEEL"),
        }
        optimizer = torch.optim.AdamW(
            compressor.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        updates_per_epoch = (len(loader) + accumulation - 1) // accumulation
        total_updates = max(1, updates_per_epoch * num_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
        use_autocast = model_device.type == "cuda" and backbone._dtype in {
            torch.float16,
            torch.bfloat16,
        }
        use_scaler = model_device.type == "cuda" and backbone._dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(num_epochs):
            stage = "ft" if finetune else "tr"
            progress = tqdm(loader, desc=f"QuietPrune {stage} epoch {epoch + 1}/{num_epochs}")
            for step, records in enumerate(progress):
                window_start = (step // accumulation) * accumulation
                window_size = min(accumulation, len(loader) - window_start)
                teacher_inputs, answer_lengths = _make_batch(backbone, dataset, records)
                teacher_inputs = {
                    key: value.to(model_device) for key, value in teacher_inputs.items()
                }
                if "pixel_values" in teacher_inputs:
                    teacher_inputs["pixel_values"] = teacher_inputs["pixel_values"].to(
                        backbone._dtype
                    )
                teacher_labels = _answer_labels(
                    teacher_inputs["input_ids"], teacher_inputs["attention_mask"], answer_lengths
                )
                autocast = (
                    torch.autocast(device_type="cuda", dtype=backbone._dtype)
                    if use_autocast
                    else nullcontext()
                )
                with autocast:
                    with torch.no_grad(), compressor.bypass():
                        teacher_logits = model(**teacher_inputs, use_cache=False).logits

                # Prepare the dynamic NeXT features before the model call, then
                # delete the matching placeholder/label/mask rows together. Q-CLS
                # sees teacher_labels here so ground-truth answer tokens are masked.
                student_inputs = compressor.prepare_training_inputs(teacher_inputs, teacher_labels)
                student_labels = student_inputs["labels"]
                student_model_inputs = {
                    key: value for key, value in student_inputs.items() if key != "labels"
                }

                autocast = (
                    torch.autocast(device_type="cuda", dtype=backbone._dtype)
                    if use_autocast
                    else nullcontext()
                )
                with autocast:
                    student_logits = model(**student_model_inputs, use_cache=False).logits
                    kl, answer_tokens = answer_kl_divergence_loss(
                        student_logits,
                        student_labels,
                        teacher_logits,
                        teacher_labels,
                        temperature=temperature,
                    )
                    cross_entropy = answer_cross_entropy_loss(student_logits, student_labels)
                    batch_loss = kl_weight * kl + ce_weight * cross_entropy
                    loss = batch_loss / window_size
                scaler.scale(loss).backward()

                last_batch = step + 1 == len(loader)
                if (step + 1) % accumulation == 0 or last_batch:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                progress.set_postfix(
                    loss=f"{float(batch_loss.detach()):.4f}",
                    kl=f"{float(kl.detach()):.4f}",
                    ce=f"{float(cross_entropy.detach()):.4f}",
                    answer_tokens=answer_tokens,
                )
            compressor.save_checkpoint(checkpoint_path)
    finally:
        compressor.uninstall()
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
        model.train(original_training)

    print(f"Saved QuietPrune adapter checkpoint to {checkpoint_path}", flush=True)


def train_quietprune(backbone: VLM, config: Config) -> None:
    """``tr``: distil the adapter from its inverse initialisation (Sec. 4.1 / 4.1)."""
    _run_training(backbone, config, finetune=False)


def finetune_quietprune(backbone: VLM, config: Config) -> None:
    """``ft``: continue distillation from a trained adapter checkpoint."""
    _run_training(backbone, config, finetune=True)
