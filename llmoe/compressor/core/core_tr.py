"""Two-stage CORE training (Sec. 3.3) and its LLMOE ``tr`` / ``ft`` entry points.

Stage 1 aligns the projector on the LLaVA-558K captions with ConvNeXt-L,
Mask2Former, and the decoder all frozen. Stage 2 tunes the projector plus LoRA
adapters, warm-started from stage 1; the Llama-3 port reads the supplied
LLaVA-NeXT-Data Parquet corpus while the existing decoder ports retain their
configured JSON route. Both optimise only the auto-regressive text loss -- the
OMG-Seg head stays frozen and contributes no segmentation loss -- and both run
on the full, uncompressed 1024-token grid, because a token budget is an
evaluation-time strategy.

The production path is one process per GPU under DDP, BF16 for the projector and
decoder, FP32 for the frozen vision stack, and FlashAttention 2 in the decoder.
Checkpoints carry everything needed to resume at the next batch: optimizer and
scheduler state plus every rank's RNG state.

The pieces, in order: checkpoint provenance, the LLaVA dataset and CORE
collator, the distributed training loop, and the ``tr``/``ft`` adapters that map
``custom_args`` onto :class:`TrainArgs`.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from datetime import timedelta
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version as distribution_version
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time
from typing import Any, Callable
import uuid

import numpy as np
import torch
import torch.distributed as dist
from mle.vars import ExpConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from llmoe.backbone.prototype import VLM
from llmoe.compressor.core.core import (
    CoreConfig,
    CoreModel,
    INFERENCE_ATTENTION_IMPLEMENTATION,
    SegmentationVision,
    TRAINING_ATTENTION_IMPLEMENTATION,
    TRAINING_RECIPES,
    asset_identity,
    build_image_processor,
    checkpoint_signature,
    load_tokenizer,
    model_attention_implementation,
    tokenize_image_prompt,
    validate_checkpoint_compatibility,
    VICUNA_SYSTEM_PROMPT,
)
from llmoe.compressor.core.core_data import (
    CoreParquetDataset,
    RowGroupDistributedSampler,
)
# The harness dataset layout is shared with the other trained baselines rather
# than restated here.
from llmoe.compressor.hidrop.data import dataset_dir_for_stage, find_annotations
from llmoe.config import Config


REQUIRED_FLASH_ATTN_VERSION = "2.8.3.post1"
CONRAD_FLASH_ATTN_WHEEL = (
    "/space/hall7/sitestore/eccc/mrd/rpnet/awa001/workspace/wheels/"
    "flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl"
)


def require_flash_attention_2(
    config: Any,
    *,
    verify_installation: bool = True,
) -> None:
    """Fail before model loading unless training can use the pinned FA2 build.

    CORE inference deliberately defaults to PyTorch SDPA, so the shared
    :class:`CoreConfig` cannot globally prohibit it. Every training entry point
    calls this guard instead. ``verify_installation=False`` is used by config
    construction and CPU tests to enforce the backend choice without importing
    a CUDA extension; the real :func:`train` path always verifies the installed
    distribution and compiled module.
    """

    requested = getattr(config, "attention_implementation", None)
    if requested != TRAINING_ATTENTION_IMPLEMENTATION:
        raise ValueError(
            "CORE training supports only attention_implementation="
            f"{TRAINING_ATTENTION_IMPLEMENTATION!r}; got {requested!r}. "
            "Inference uses 'sdpa', but training must use the pinned FA2 wheel."
        )
    if not verify_installation:
        return

    try:
        installed = distribution_version("flash-attn")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "CORE training requires flash-attn=="
            f"{REQUIRED_FLASH_ATTN_VERSION}. Install the pinned wheel with "
            "'python -m pip install --no-deps --force-reinstall "
            f"{CONRAD_FLASH_ATTN_WHEEL}'."
        ) from error
    if installed != REQUIRED_FLASH_ATTN_VERSION:
        raise RuntimeError(
            f"CORE training requires flash-attn=={REQUIRED_FLASH_ATTN_VERSION}, "
            f"but the active environment has {installed}. Reinstall the pinned wheel "
            "with 'python -m pip install --no-deps --force-reinstall "
            f"{CONRAD_FLASH_ATTN_WHEEL}'."
        )

    try:
        flash_attn = importlib.import_module("flash_attn")
        importlib.import_module("flash_attn_2_cuda")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "flash-attn metadata is present, but its compiled FA2 extension cannot "
            "be imported. Reinstall the pinned Conrad wheel in the varest environment."
        ) from error
    if not callable(getattr(flash_attn, "flash_attn_func", None)):
        raise RuntimeError("The installed flash_attn module does not expose flash_attn_func")


# ============================================================================
# Checkpoint provenance
# ============================================================================


def checkpoint_config_payload(config: Any) -> dict[str, Any]:
    payload = asdict(config)
    payload["_llm_asset_identity"] = asset_identity(config.llm_name_or_path)
    payload["_llm_family"] = config.llm_family
    signature = checkpoint_signature(config, include_lora=False)
    payload["_implementation"] = signature["implementation"]
    payload["_evaluation_budget_convention"] = signature[
        "evaluation_budget_convention"
    ]
    payload["_evaluation_structural_tokens"] = signature[
        "evaluation_structural_tokens"
    ]
    return payload


def training_provenance(config: Any, args: Any) -> dict[str, Any]:
    """Small, serializable manifest written beside every checkpoint."""

    family = config.llm_family
    if family == "llama3":
        prompt_recipe = ("llava_plain", "llama3_instruct")[args.stage - 1]
    elif family == "qwen":
        prompt_recipe = ("llava_plain", "qwen_chatml")[args.stage - 1]
    else:
        prompt_recipe = TRAINING_RECIPES[args.stage - 1]

    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path:
        root = Path(dataset_path).expanduser().resolve()
        shards = sorted((root / "data").glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No LLaVA-NeXT parquet shards under {root / 'data'}")
        stats = [path.stat() for path in shards]
        manifest = "\n".join(
            f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
            for path, stat in zip(shards, stats, strict=True)
        )
        dataset = {
            "name": "LLaVA-NeXT-Data",
            "format": "direct_parquet",
            "root": str(root),
            "parquet_shards": len(shards),
            "total_parquet_bytes": sum(stat.st_size for stat in stats),
            "shard_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
            "selection": "all rows including text-only rows",
        }
    else:
        data_path = Path(args.data_json).expanduser().resolve()
        image_root = Path(args.image_root).expanduser().resolve()
        stat = data_path.stat()
        is_pretrain = args.stage == 1
        dataset = {
            "name": "LLaVA-558K" if is_pretrain else "LLaVA-665K",
            "format": "llava_json_and_image_tree",
            "json": str(data_path),
            "json_size": stat.st_size,
            "json_mtime_ns": stat.st_mtime_ns,
            "image_root": str(image_root),
            "selection": "all caption pairs" if is_pretrain else "all instruction rows",
        }
    source_commits: dict[str, str] = {}
    if family == "llama3":
        from llmoe.compressor.core.core_llama3_lm import CORE_LLAMA3_SOURCE_COMMITS

        source_commits.update(CORE_LLAMA3_SOURCE_COMMITS)
    try:
        repository = Path(__file__).resolve().parents[3]
        implementation_commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        implementation_commit = os.environ.get("CORE_IMPLEMENTATION_COMMIT", "unknown")
    source_commits["varest_implementation"] = implementation_commit

    return {
        "version": 2,
        "variant": f"core-{family}",
        "language": checkpoint_signature(config, include_lora=args.stage == 2),
        "prompt_recipe": prompt_recipe,
        "dataset": dataset,
        "source_commits": source_commits,
        "vision": {
            "image_size": int(config.image_size),
            "feature_stride": int(config.feature_stride),
            "native_visual_tokens": int(config.num_visual_tokens),
            "vision_timm_model": config.vision_timm_model,
            "omgseg_checkpoint": config.omgseg_checkpoint,
            "omgseg_release": f"{config.omgseg_repo}/{config.omgseg_filename}",
        },
        "merging": {
            "training_rate": config.rate,
            "training_tokens": config.fixed_tokens,
            "evaluation_budget_semantics": (
                "max(1, round(requested_retention_ratio * "
                "original_content_tokens)) per source image"
            ),
            "source_tokens_per_image": int(config.num_visual_tokens),
            "views_per_image": 1,
            "structural_tokens_outside_budget": 0,
            "large_object_first": bool(config.large_object_first),
            "low_budget_overflow": "largest_k_minus_one_plus_one_catch_all",
        },
        "attention": {
            "training_backend": TRAINING_ATTENTION_IMPLEMENTATION,
            "required_flash_attn_version": REQUIRED_FLASH_ATTN_VERSION,
            "conrad_wheel": CONRAD_FLASH_ATTN_WHEEL,
            "inference_backend": INFERENCE_ATTENTION_IMPLEMENTATION,
        },
        "training_args": asdict(args),
    }


# ============================================================================
# LLaVA conversation data and the CORE collator
# ============================================================================


IGNORE_INDEX = -100


@dataclass
class Conversation:
    image_path: str | None
    turns: list[tuple[str, str]]
    image_bytes: bytes | None = None
    sample_id: str = ""
    data_source: str = ""

    @property
    def has_image(self) -> bool:
        return self.image_bytes is not None or self.image_path is not None


class LlavaJsonDataset(Dataset):
    """Read standard LLaVA JSON records without decoding images eagerly."""

    def __init__(self, json_path: str, image_root: str) -> None:
        with open(json_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        self.image_root = image_root
        self.records = [
            Conversation(
                image_path=entry.get("image"),
                turns=[
                    (str(turn["from"]), str(turn["value"]))
                    for turn in entry.get("conversations", [])
                ],
            )
            for entry in raw
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Conversation:
        return self.records[idx]

    def load_image(self, record: Conversation) -> Any:
        from PIL import Image

        if record.image_path is None:
            return None
        path = os.path.join(self.image_root, record.image_path)
        with Image.open(path) as image:
            return image.convert("RGB")


class CoreDataCollator:
    """Build pixel values, token IDs, labels, and modality flags for CORE."""

    def __init__(
        self,
        tokenizer: Any,
        image_processor: Callable[[list[Any]], torch.Tensor],
        image_root: str,
        *,
        image_token: str = "<image>",
        max_length: int = 4096,
        visual_tokens: int = 1,
        stage: int = 2,
        llm_family: str = "llama",
    ) -> None:
        if visual_tokens < 1:
            raise ValueError("visual_tokens must be >= 1")
        text_max_length = max_length - visual_tokens + 1
        if text_max_length < 1:
            raise ValueError(
                f"max_length={max_length} cannot hold {visual_tokens} visual tokens"
            )
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = image_root
        self.image_token = image_token
        self.max_length = max_length
        self.visual_tokens = visual_tokens
        self.text_max_length = text_max_length
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        self.stage = stage
        if llm_family not in ("llama", "llama3", "qwen"):
            raise ValueError(
                "llm_family must be 'llama', 'llama3', or 'qwen', got "
                f"{llm_family!r}"
            )
        #: Stage 2's chat template follows the decoder exactly.
        self.llm_family = llm_family

    def _tokenize(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=add_special_tokens)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("CORE expected one tokenized conversation")
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    def _tokenize_prompt(self, prompt: str, *, has_image: bool) -> list[int]:
        """HF-positive-ID equivalent of LLaVA ``tokenizer_image_token``.

        CORE keeps one placeholder and expands it to 1,024 embeddings inside the
        model. Splitting around the marker preserves SentencePiece boundaries
        exactly like upstream LLaVA instead of asking the tokenizer to parse a
        newly added special token embedded in arbitrary text.
        """

        if not has_image:
            return self._tokenize(prompt)
        # The complete multimodal prompt has exactly one marker, but some
        # masking recipes also tokenize each conversation round independently. Only
        # the first round contains <image>; later rounds must follow ordinary
        # tokenizer semantics just like LLaVA's tokenizer_image_token helper.
        if self.image_token not in prompt:
            return self._tokenize(prompt)
        return tokenize_image_prompt(self.tokenizer, prompt, self.image_token)

    def _canonicalize_turns(
        self,
        turns: list[tuple[str, str]],
        *,
        has_image: bool,
    ) -> list[tuple[str, str]]:
        canonical: list[tuple[str, str]] = []
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
        }
        for raw_role, raw_text in turns:
            if raw_role == "system":
                raise ValueError("LLaVA plain/v1 does not accept per-example system turns")
            try:
                role = role_map[raw_role]
            except KeyError as error:
                raise ValueError(f"Unsupported conversation role: {raw_role!r}") from error
            text = str(raw_text)
            if self.image_token in text:
                text = f"{self.image_token}\n{text.replace(self.image_token, '').strip()}".strip()
            canonical.append((role, text))

        if canonical and canonical[0][0] != "user":
            canonical = canonical[1:]
        if not canonical:
            raise ValueError("Training conversation contains no user turn")
        for index, (role, _) in enumerate(canonical):
            expected = "user" if index % 2 == 0 else "assistant"
            if role != expected:
                raise ValueError(
                    f"Expected alternating user/assistant turns; turn {index} is {role!r}"
                )
        marker_count = sum(text.count(self.image_token) for _, text in canonical)
        if has_image and marker_count == 0:
            role, text = canonical[0]
            canonical[0] = (role, f"{self.image_token}\n{text}".strip())
            marker_count = 1
        expected_markers = 1 if has_image else 0
        if marker_count != expected_markers:
            raise ValueError(
                f"Expected {expected_markers} image marker(s) for has_image={has_image}, "
                f"got {marker_count}"
            )
        return canonical

    def _render_ids_labels(
        self,
        turns: list[tuple[str, str]],
        has_image: bool | None = None,
    ) -> tuple[list[int], list[int]]:
        """Tokenize one conversation under the stage's recipe."""

        if has_image is None:
            has_image = any(self.image_token in text for _, text in turns)
        if self.stage == 1:
            # Stage 1 is caption text around the marker, with no chat template to
            # get wrong, so both decoders share it.
            ids, labels = self._render_llava_plain(turns, has_image=has_image)
        elif self.llm_family == "qwen":
            ids, labels = self._render_qwen_chatml(turns, has_image=has_image)
        elif self.llm_family == "llama3":
            ids, labels = self._render_llama3_instruct(turns, has_image=has_image)
        else:
            ids, labels = self._render_vicuna_v1(turns, has_image=has_image)
        limit = self.text_max_length if has_image else self.max_length
        return ids[:limit], labels[:limit]

    def _render_llava_plain(
        self,
        turns: list[tuple[str, str]],
        *,
        has_image: bool,
    ) -> tuple[list[int], list[int]]:
        canonical = self._canonicalize_turns(turns, has_image=has_image)
        if not has_image:
            raise ValueError("LLaVA plain projector pretraining requires an image")
        if len(canonical) != 2:
            raise ValueError(
                "LLaVA plain projector pretraining requires one user/assistant pair"
            )
        prompt = self.image_token + canonical[1][1] + "\n"
        ids = self._tokenize_prompt(prompt, has_image=True)
        labels = list(ids)
        prefix_length = len(self._tokenize_prompt(self.image_token, has_image=True))
        labels[:prefix_length] = [IGNORE_INDEX] * prefix_length
        return self._mask_image_labels(ids, labels)

    def _template_ids(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        value = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(value, Mapping):
            value = value["input_ids"]
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError("CORE expected one Llama-3 conversation")
            value = value[0]
        return [int(token) for token in value]

    def _render_llama3_instruct(
        self,
        turns: list[tuple[str, str]],
        *,
        has_image: bool,
    ) -> tuple[list[int], list[int]]:
        """Stage 2 with the pinned checkpoint's native Llama-3 template.

        Labels cover assistant content plus ``<|eot_id|>`` only. Prefix-based
        boundary discovery handles the one-token BPE change that can occur
        between an assistant header's final newline and the first answer token.
        """

        canonical = self._canonicalize_turns(turns, has_image=has_image)
        messages = [{"role": role, "content": text} for role, text in canonical]
        ids = self._template_ids(messages, add_generation_prompt=False)
        labels = [IGNORE_INDEX] * len(ids)
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            generation_prefix = self._template_ids(
                messages[:index], add_generation_prompt=True
            )
            completed = self._template_ids(
                messages[: index + 1], add_generation_prompt=False
            )
            if ids[: len(completed)] != completed:
                raise RuntimeError(
                    "Llama-3 completed turn is not a prefix of the conversation"
                )
            boundary = 0
            for generated_token, completed_token in zip(
                generation_prefix, completed, strict=False
            ):
                if generated_token != completed_token:
                    break
                boundary += 1
            if boundary < len(generation_prefix) - 1:
                raise RuntimeError(
                    "Llama-3 assistant header changed before its BPE boundary"
                )
            labels[boundary : len(completed)] = ids[boundary : len(completed)]
        return self._mask_image_labels(ids, labels)

    def _render_qwen_chatml(
        self,
        turns: list[tuple[str, str]],
        *,
        has_image: bool,
    ) -> tuple[list[int], list[int]]:
        """Stage 2 under Qwen's ChatML template.

        Much simpler than the Vicuna path, which reconstructs round boundaries by
        re-tokenizing prefixes and correcting for SentencePiece's word-boundary
        behaviour. Qwen's tokenizer is a BPE with explicit ``<|im_start|>`` and
        ``<|im_end|>`` tokens, so each segment can be tokenized on its own and the
        supervised span is exactly the assistant's content plus its terminator --
        no offset arithmetic, and nothing to drift when a round is empty.
        """
        from llmoe.compressor.core.core_qwen_lm import QWEN_SYSTEM_PROMPT

        canonical = self._canonicalize_turns(turns, has_image=has_image)
        segments: list[tuple[str, bool]] = [
            (f"<|im_start|>system\n{QWEN_SYSTEM_PROMPT}<|im_end|>\n", False)
        ]
        for role, text in canonical:
            if role == "user":
                segments.append((f"<|im_start|>user\n{text}<|im_end|>\n", False))
                continue
            segments.append(("<|im_start|>assistant\n", False))
            if text:
                # Supervised: the answer and the turn terminator that ends it, so
                # the model learns to stop.
                segments.append((f"{text}<|im_end|>\n", True))

        ids: list[int] = []
        labels: list[int] = []
        for text, supervised in segments:
            chunk = self._tokenize_prompt(text, has_image=has_image and
                                          self.image_token in text)
            ids.extend(chunk)
            labels.extend(chunk if supervised else [IGNORE_INDEX] * len(chunk))
        return self._mask_image_labels(ids, labels)

    def _render_vicuna_v1(
        self,
        turns: list[tuple[str, str]],
        *,
        has_image: bool,
    ) -> tuple[list[int], list[int]]:
        canonical = self._canonicalize_turns(turns, has_image=has_image)
        eos = self.tokenizer.eos_token or "</s>"
        prompt_parts = [VICUNA_SYSTEM_PROMPT, " "]
        for role, text in canonical:
            prompt_role = "USER" if role == "user" else "ASSISTANT"
            if text:
                prompt_parts.extend(
                    (prompt_role, ": ", text, " " if role == "user" else eos)
                )
            else:
                prompt_parts.extend((prompt_role, ":"))
        prompt = "".join(prompt_parts)
        ids = self._tokenize_prompt(prompt, has_image=has_image)
        labels = list(ids)

        separator = " ASSISTANT: "
        current_length = 1
        labels[:current_length] = [IGNORE_INDEX] * current_length
        for round_index, round_text in enumerate(prompt.split(eos)):
            if not round_text:
                break
            round_parts = round_text.split(separator)
            if len(round_parts) != 2:
                break
            instruction = round_parts[0] + separator
            round_length = len(
                self._tokenize_prompt(round_text, has_image=has_image)
            )
            instruction_length = len(
                self._tokenize_prompt(instruction, has_image=has_image)
            ) - 2
            if round_index != 0 and not bool(getattr(self.tokenizer, "legacy", True)):
                round_length -= 1
                instruction_length -= 1
            labels[
                current_length : current_length + instruction_length
            ] = [IGNORE_INDEX] * instruction_length
            current_length += round_length
        labels[current_length:] = [IGNORE_INDEX] * max(0, len(labels) - current_length)
        if current_length < self.max_length and current_length != len(ids):
            labels[:] = [IGNORE_INDEX] * len(labels)
            print(
                "WARNING: Vicuna-v1 tokenization mismatch while constructing CORE labels: "
                f"{current_length} vs. {len(ids)}. (ignored)",
                flush=True,
            )
        return self._mask_image_labels(ids, labels)

    def _mask_image_labels(
        self, ids: list[int], labels: list[int]
    ) -> tuple[list[int], list[int]]:
        image_id = int(self.tokenizer.convert_tokens_to_ids(self.image_token))
        return ids, [
            IGNORE_INDEX if token_id == image_id else label
            for token_id, label in zip(ids, labels, strict=True)
        ]

    def __call__(self, batch: list[Any]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("CORE collator received an empty batch")
        has_image = [bool(record.has_image) for record in batch]
        images = [self._open(record) for record in batch]
        pixel_values = self.image_processor(images)

        rows_ids: list[list[int]] = []
        rows_labels: list[list[int]] = []
        for record, row_has_image in zip(batch, has_image, strict=True):
            turns = list(record.turns)
            if self.stage == 1:
                turns = turns[:2]
            ids, labels = self._render_ids_labels(turns, row_has_image)
            if not any(label != IGNORE_INDEX for label in labels):
                sample_id = getattr(record, "sample_id", "")
                raise ValueError(
                    "Truncation removed every assistant target from CORE sample "
                    f"{sample_id!r}"
                )
            rows_ids.append(ids)
            rows_labels.append(labels)

        max_len = max(len(row) for row in rows_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id")
        input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row_index, (ids, row_labels) in enumerate(
            zip(rows_ids, rows_labels, strict=True)
        ):
            input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[row_index, : len(row_labels)] = torch.tensor(
                row_labels, dtype=torch.long
            )
            attention_mask[row_index, : len(ids)] = 1
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "has_image": torch.tensor(has_image, dtype=torch.bool),
        }

    def _open(self, record: Any) -> Any:
        from PIL import Image

        if getattr(record, "image_bytes", None) is not None:
            with Image.open(BytesIO(record.image_bytes)) as image:
                return image.convert("RGB")
        image_path = getattr(record, "image_path", None)
        if image_path is None:
            # A mean-color pixel becomes approximately zero after CLIP
            # normalization; its features are graph-tethered but never inserted.
            return Image.new("RGB", (1, 1), (123, 117, 104))
        path = Path(self.image_root) / image_path
        if not path.is_file():
            sample_id = getattr(record, "sample_id", "")
            raise FileNotFoundError(f"Image for sample {sample_id!r} not found: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")


# ============================================================================
# Distributed two-stage training loop
# ============================================================================


@dataclass
class TrainArgs:
    data_json: str
    image_root: str
    output_dir: str
    stage: int  # 1 (projector only) or 2 (projector + LoRA)
    # Stage 2 may read the supplied LLaVA-NeXT-Data Parquet corpus directly.
    # Stage 1 deliberately leaves this unset and uses LLaVA-558K JSON/images.
    dataset_path: str | None = None
    batch_size: int = 1  # per GPU
    grad_accum: int = 1
    lr: float = 1e-3  # stage-1 default; stage-2 overrides to 2e-4
    weight_decay: float = 0.0
    epochs: int = 1
    warmup_ratio: float = 0.03
    max_length: int = 4096
    num_workers: int = 4  # per GPU
    prefetch_factor: int = 2
    loader_timeout_seconds: int = 300
    log_every: int = 10
    save_every: int | None = None
    projector_checkpoint: str | None = None
    resume_from_checkpoint: str | None = None
    max_samples: int | None = None
    sample_indices_path: str | None = None
    max_steps: int | None = None
    # A controlled early exit used by the resume smoke test. Unlike max_steps,
    # this does not change the scheduler horizon stored in the checkpoint.
    exit_after_steps: int | None = None
    # Controlled scheduler-allocation rollover at an optimizer boundary.
    exit_after_seconds: int | None = None
    gradient_checkpointing: bool = True
    save_final: bool = True
    local_files_only: bool = True
    seed: int = 0
    expected_world_size: int | None = None
    expected_global_batch: int | None = None
    ddp_timeout_minutes: int = 60


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class StartBatchSampler(BatchSampler):
    """Batch sampler that can begin at an exact batch without replaying I/O.

    Checkpoints are written only at optimizer-step boundaries. On resume,
    skipping indices in the sampler avoids reopening and resizing every image
    that preceded the checkpoint while preserving the original epoch order and
    absolute batch numbering.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_batch = 0

    def __iter__(self):  # noqa: ANN204 - matches torch BatchSampler
        iterator = super().__iter__()
        for batch_index, batch in enumerate(iterator):
            if batch_index >= self.start_batch:
                yield batch

    def __len__(self) -> int:
        return max(0, super().__len__() - self.start_batch)


def build_model(
    config: CoreConfig,
    *,
    stage: int,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer_size: int | None = None,
    local_files_only: bool = True,
    gradient_checkpointing: bool = True,
) -> CoreModel:
    """Build CORE and expose only the trainable surface for one stage."""
    require_flash_attention_2(config, verify_installation=False)
    vision_dtype = getattr(torch, config.vision_dtype)
    segmentation = SegmentationVision(config).load(
        device=device, dtype=vision_dtype, local_files_only=local_files_only
    )
    model = CoreModel(config, segmentation=segmentation)
    model.load_llm(dtype=dtype, local_files_only=local_files_only)
    if (
        tokenizer_size is not None
        and tokenizer_size > model.llm.get_input_embeddings().weight.shape[0]
    ):
        # Resize before freezing/attaching PEFT. Otherwise Transformers creates
        # a fresh trainable embedding after the base LLM has been frozen.
        # The image-token row is replaced by projected visual features and is
        # never a supervised target. Avoid Transformers' costly full-vocabulary
        # covariance estimate for this single unused embedding row.
        model.llm.resize_token_embeddings(tokenizer_size, mean_resizing=False)
    model.freeze_base()
    if stage == 2 and config.use_lora:
        model.apply_lora()
    for param in model.projector.parameters():
        param.requires_grad_(True)

    # Keep the paper's frozen ConvNeXt/Mask2Former path in FP32. The projector
    # and language path use BF16, which is required by FA2 and native on H100.
    model.to(device=device)
    model.projector.to(dtype=dtype)
    model.llm.to(dtype=dtype)
    model.llm.config.use_cache = False
    if gradient_checkpointing:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def train(config: CoreConfig, args: TrainArgs) -> None:
    """Run one CORE training stage end to end on one or more CUDA ranks."""
    # Validate both the requested Transformers backend and the exact compiled
    # wheel before initializing distributed CUDA or materializing any model.
    require_flash_attention_2(config, verify_installation=True)
    # A scheduler-side SIGUSR1 now yields every Python thread's stack in the
    # listing.  This is especially useful for distinguishing a blocked input
    # worker from a rank waiting in a distributed collective.
    import faulthandler

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    trace_after = int(os.environ.get("CORE_TRACEBACK_LATER_SECONDS", "0"))
    if trace_after > 0:
        faulthandler.dump_traceback_later(trace_after, repeat=True)
    _validate_args(args)
    context = _initialise_distributed(args)
    try:
        actual_global_batch = args.batch_size * args.grad_accum * context.world_size
        if (
            args.expected_global_batch is not None
            and actual_global_batch != args.expected_global_batch
        ):
            raise ValueError(
                f"Expected global batch {args.expected_global_batch}, got "
                f"{args.batch_size} * {args.grad_accum} * {context.world_size} "
                f"= {actual_global_batch}"
            )
        _train(config, args, context)
    finally:
        if trace_after > 0:
            faulthandler.cancel_dump_traceback_later()
        if context.enabled and dist.is_initialized():
            dist.destroy_process_group()


def _train(config: CoreConfig, args: TrainArgs, context: DistributedContext) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _seed_everything(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    _barrier(context)
    resume_dir = _distributed_resume_path(args.resume_from_checkpoint, output_dir, context)

    tokenizer = load_tokenizer(config, local_files_only=args.local_files_only)
    if config.image_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [config.image_token]})

    model = build_model(
        config,
        stage=args.stage,
        device=context.device,
        dtype=torch.bfloat16,
        tokenizer_size=len(tokenizer),
        local_files_only=args.local_files_only,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if resume_dir is not None:
        validate_checkpoint_compatibility(
            config,
            resume_dir,
            include_lora=args.stage == 2,
            purpose=f"stage-{args.stage} resume checkpoint",
        )
        _load_trained_weights(model, resume_dir, args.stage)
        _print(context, f"resuming model weights from {resume_dir}")
    elif args.projector_checkpoint is not None:
        validate_checkpoint_compatibility(
            config,
            Path(args.projector_checkpoint).expanduser().resolve().parent,
            include_lora=False,
            purpose="stage-1 projector checkpoint",
        )
        state = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=True)
        model.projector.load_state_dict(state)
        _print(context, f"warm-started projector from {args.projector_checkpoint}")
    model.set_image_token_id(tokenizer.convert_tokens_to_ids(config.image_token))

    implementation = model_attention_implementation(model.llm)
    if implementation != TRAINING_ATTENTION_IMPLEMENTATION:
        raise RuntimeError(
            f"Training requires {TRAINING_ATTENTION_IMPLEMENTATION}, but the decoder reports "
            f"{implementation}"
        )

    image_processor = build_image_processor(config, args.local_files_only)
    collator = CoreDataCollator(
        tokenizer,
        image_processor,
        args.image_root,
        image_token=config.image_token,
        max_length=args.max_length,
        visual_tokens=(
            int(config.fixed_tokens)
            if config.rate == "fixed" and config.fixed_tokens is not None
            else int(config.num_visual_tokens)
        ),
        stage=args.stage,
        # Stage 2's chat template has to match the decoder that will read it.
        llm_family=config.llm_family,
    )
    full_parquet_dataset: CoreParquetDataset | None = None
    if args.dataset_path is not None:
        if args.stage == 1:
            raise ValueError(
                "CORE stage 1 must use LLaVA-558K JSON/images, not dataset_path"
            )
        full_parquet_dataset = CoreParquetDataset(
            args.dataset_path,
            multimodal_only=False,
        )
        dataset: Any = full_parquet_dataset
    else:
        dataset = LlavaJsonDataset(args.data_json, args.image_root)
    if args.sample_indices_path is not None:
        indices_path = Path(args.sample_indices_path).expanduser().resolve()
        indices = [
            int(line)
            for line in indices_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not indices:
            raise ValueError(f"No sample indices found in {indices_path}")
        invalid = [index for index in indices if index < 0 or index >= len(dataset)]
        if invalid:
            raise IndexError(f"Dataset indices out of range in {indices_path}: {invalid[:5]}")
        dataset = Subset(dataset, indices)
    elif args.max_samples is not None:
        subset_size = min(args.max_samples, len(dataset))
        indices = list(range(subset_size))
        # Exercise the legitimate text-only path during small stage-2 cluster
        # checks even if the first rows of the corpus are all multimodal.
        if full_parquet_dataset is not None and subset_size >= 2:
            first_text = full_parquet_dataset.first_text_only_index
            if first_text is not None and first_text not in indices:
                indices[-1] = first_text
        dataset = Subset(dataset, indices)
    if isinstance(dataset, CoreParquetDataset):
        sampler: Any = RowGroupDistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
    batch_sampler = StartBatchSampler(sampler, args.batch_size, drop_last=False)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "timeout": args.loader_timeout_seconds if args.num_workers > 0 else 0,
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)
    batches_per_epoch = len(loader)
    if batches_per_epoch == 0:
        raise ValueError("The training dataset is empty")

    # All ranks load the same pinned weights and use the same initialization
    # seed. Avoid broadcasting the multi-billion-parameter frozen base during DDP.
    training_model: CoreModel | DistributedDataParallel = model
    if context.enabled:
        training_model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            init_sync=False,
        )

    trainable = [param for param in training_model.parameters() if param.requires_grad]
    n_trainable = sum(param.numel() for param in trainable)
    _print(
        context,
        (
            f"stage={args.stage} ranks={context.world_size} per_gpu_batch={args.batch_size} "
            f"grad_accum={args.grad_accum} global_batch="
            f"{args.batch_size * args.grad_accum * context.world_size} "
            f"trainable={n_trainable / 1e6:.1f}M attention={implementation}"
        ),
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    natural_total_steps = steps_per_epoch * args.epochs
    total_steps = (
        min(natural_total_steps, args.max_steps)
        if args.max_steps is not None
        else natural_total_steps
    )
    scheduler = _cosine_with_warmup(optimizer, total_steps, args.warmup_ratio)

    global_step = 0
    start_epoch = 0
    start_batch = 0
    resume_rng: dict[str, Any] | None = None
    if resume_dir is not None:
        global_step, start_epoch, start_batch, resume_rng = _load_training_state(
            resume_dir,
            optimizer,
            scheduler,
            args,
            config,
            context,
            total_steps,
        )
        _print(
            context,
            f"resumed optimizer at step={global_step}, epoch={start_epoch}, batch={start_batch}",
        )

    # Model initialization must be identical because DDP init_sync is disabled;
    # stochastic training state should be rank-specific.
    _seed_everything(args.seed + context.rank)
    training_model.train()
    model.vision.eval()
    optimizer.zero_grad(set_to_none=True)
    last_log_time = time.perf_counter()
    run_started = last_log_time

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        batch_sampler.start_batch = epoch_start_batch
        progress = tqdm(
            enumerate(loader, start=epoch_start_batch),
            total=batches_per_epoch,
            initial=epoch_start_batch,
            desc=f"stage{args.stage}-epoch{epoch}",
            disable=not context.is_main,
        )
        for batch_index, batch in progress:
            if resume_rng is not None:
                # Iterator/worker creation consumes the CPU generator. Restore
                # the saved state immediately before the first resumed forward.
                _restore_rng_state(resume_rng, context.device)
                resume_rng = None

            accumulation_start = (batch_index // args.grad_accum) * args.grad_accum
            accumulation_size = min(
                args.grad_accum,
                batches_per_epoch - accumulation_start,
            )
            update_now = (
                (batch_index + 1) % args.grad_accum == 0
                or batch_index + 1 == batches_per_epoch
            )
            batch = {
                key: value.to(context.device, non_blocking=True)
                for key, value in batch.items()
            }
            sync_context = (
                training_model.no_sync()
                if context.enabled and not update_now
                else nullcontext()
            )
            with sync_context:
                output = training_model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    multimodal_max_length=args.max_length,
                    has_image=batch.get("has_image"),
                )
                _require_finite_loss(output.loss, context)
                (output.loss / accumulation_size).backward()

            if not update_now:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            next_epoch, next_batch = _next_position(
                epoch,
                batch_index,
                batches_per_epoch,
            )

            if global_step % args.log_every == 0 or global_step == 1:
                mean_loss = _mean_across_ranks(output.loss.detach(), context)
                torch.cuda.synchronize(context.device)
                now = time.perf_counter()
                metrics = {
                    "step": global_step,
                    "loss": float(mean_loss),
                    "lr": scheduler.get_last_lr()[0],
                    "seconds_since_log": now - last_log_time,
                    "gpu_allocated_gib": torch.cuda.memory_allocated(context.device) / 2**30,
                    "gpu_reserved_gib": torch.cuda.memory_reserved(context.device) / 2**30,
                    "gpu_peak_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
                }
                last_log_time = now
                if context.is_main:
                    progress.set_postfix(loss=metrics["loss"], lr=metrics["lr"])
                    print(f"[CORE metrics] {json.dumps(metrics, sort_keys=True)}", flush=True)

            saved_this_step = False
            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(
                    training_model,
                    output_dir,
                    tag=f"checkpoint-step{global_step:08d}",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    config=config,
                    context=context,
                    global_step=global_step,
                    next_epoch=next_epoch,
                    next_batch=next_batch,
                    total_steps=total_steps,
                )
                saved_this_step = True

            if args.exit_after_steps is not None and global_step >= args.exit_after_steps:
                if not saved_this_step:
                    save_checkpoint(
                        training_model,
                        output_dir,
                        tag=f"checkpoint-step{global_step:08d}",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        config=config,
                        context=context,
                        global_step=global_step,
                        next_epoch=next_epoch,
                        next_batch=next_batch,
                        total_steps=total_steps,
                    )
                _print(context, f"controlled smoke-test exit after step {global_step}")
                return
            rollover_due = False
            if args.exit_after_seconds is not None:
                # Rank initialization and batch completion times differ by a
                # few seconds. A local-only clock check can send one rank into
                # checkpoint collectives while its peers continue training.
                # Every rank participates in this reduction at every optimizer
                # boundary so the first expired clock rolls over all ranks.
                rollover_due = _any_across_ranks(
                    time.perf_counter() - run_started >= args.exit_after_seconds,
                    context,
                )
            if rollover_due:
                if not saved_this_step:
                    save_checkpoint(
                        training_model,
                        output_dir,
                        tag=f"checkpoint-step{global_step:08d}",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        config=config,
                        context=context,
                        global_step=global_step,
                        next_epoch=next_epoch,
                        next_batch=next_batch,
                        total_steps=total_steps,
                    )
                _print(context, f"controlled allocation rollover after step {global_step}")
                return
            if global_step >= total_steps:
                break
        start_batch = 0
        batch_sampler.start_batch = 0
        if global_step >= total_steps:
            break

    if args.save_final:
        save_checkpoint(
            training_model,
            output_dir,
            tag="final",
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            config=config,
            context=context,
            global_step=global_step,
            next_epoch=args.epochs,
            next_batch=0,
            total_steps=total_steps,
        )
    else:
        _print(context, f"completed {global_step} benchmark steps without a final checkpoint")


def _validate_args(args: TrainArgs) -> None:
    if args.stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch_size and grad_accum must be >= 1")
    if args.epochs < 1 or args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("epochs/prefetch_factor must be positive and num_workers non-negative")
    if args.loader_timeout_seconds < 1:
        raise ValueError("loader_timeout_seconds must be positive")
    if args.max_samples is not None and args.sample_indices_path is not None:
        raise ValueError("max_samples and sample_indices_path are mutually exclusive")
    if args.expected_global_batch is not None and args.expected_global_batch < 1:
        raise ValueError("expected_global_batch must be >= 1 when supplied")
    for name in (
        "save_every",
        "max_samples",
        "max_steps",
        "exit_after_steps",
        "exit_after_seconds",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1 when supplied")
    if args.exit_after_steps is not None and args.max_steps is not None:
        if args.exit_after_steps >= args.max_steps:
            raise ValueError("exit_after_steps must be lower than max_steps")


def _initialise_distributed(args: TrainArgs) -> DistributedContext:
    if not torch.cuda.is_available():
        raise RuntimeError("CORE training requires CUDA because FlashAttention 2 is mandatory")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.expected_world_size is not None and world_size != args.expected_world_size:
        raise RuntimeError(
            f"Expected {args.expected_world_size} training ranks, but torchrun supplied {world_size}"
        )
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=timedelta(minutes=args.ddp_timeout_minutes),
        )
    return DistributedContext(rank, local_rank, world_size, device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    warmup = max(1, int(total_steps * warmup_ratio))
    start_factor = 1e-5

    def lr_lambda(step: int) -> float:
        if step < warmup:
            # Match MMEngine LinearLR(start_factor=1e-5): the first optimizer
            # update is tiny but non-zero, then the LR reaches its base value
            # at the warmup boundary.
            return start_factor + (1.0 - start_factor) * (step / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    training_model: CoreModel | DistributedDataParallel,
    output_dir: Path,
    *,
    tag: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: TrainArgs,
    config: CoreConfig,
    context: DistributedContext,
    global_step: int,
    next_epoch: int,
    next_batch: int,
    total_steps: int,
) -> None:
    """Atomically persist model pieces and exact distributed training state."""
    rng_states = _gather_rng_states(context)
    _barrier(context)
    destination = output_dir / tag
    if context.is_main:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        temporary = output_dir / f".{tag}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True)
        model = _unwrap(training_model)
        torch.save(model.projector.state_dict(), temporary / "projector.pt")
        if args.stage == 2:
            if not hasattr(model.llm, "peft_config"):
                raise RuntimeError("Stage 2 checkpoint is missing its PEFT adapter")
            model.llm.save_pretrained(temporary, safe_serialization=True)
        _atomic_json_dump(checkpoint_config_payload(config), temporary / "core_config.json")
        _atomic_json_dump(asdict(args), temporary / "training_args.json")
        _atomic_json_dump(training_provenance(config, args), temporary / "provenance.json")
        trainer_state = {
            "version": 2,
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_states": rng_states,
            "world_size": context.world_size,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "total_steps": total_steps,
            "resume_signature": _resume_signature(
                args,
                config,
                context,
                total_steps,
            ),
        }
        torch.save(trainer_state, temporary / "trainer_state.pt")
        os.replace(temporary, destination)
        _atomic_json_dump(
            {"checkpoint": str(destination), "global_step": global_step},
            output_dir / "latest.json",
        )
        print(f"[CORE] saved resumable checkpoint -> {destination}", flush=True)
    _barrier(context)


def _load_trained_weights(model: CoreModel, checkpoint: Path, stage: int) -> None:
    projector_path = checkpoint / "projector.pt"
    if not projector_path.is_file():
        raise FileNotFoundError(f"Missing projector checkpoint: {projector_path}")
    model.projector.load_state_dict(
        torch.load(projector_path, map_location="cpu", weights_only=True)
    )
    if stage == 2:
        adapter_config = checkpoint / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(f"Missing LoRA adapter config: {adapter_config}")
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        adapter_state = load_peft_weights(str(checkpoint), device="cpu")
        result = set_peft_model_state_dict(model.llm, adapter_state, adapter_name="default")
        unexpected = getattr(result, "unexpected_keys", ())
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading LoRA checkpoint: {unexpected}")


def _load_training_state(
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: TrainArgs,
    config: CoreConfig,
    context: DistributedContext,
    total_steps: int,
) -> tuple[int, int, int, dict[str, Any]]:
    path = checkpoint / "trainer_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing resumable trainer state: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "world_size": context.world_size,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "total_steps": total_steps,
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume configuration differs from checkpoint: {mismatches}")
    if int(state.get("version", 1)) >= 2:
        expected_signature = _resume_signature(args, config, context, total_steps)
        actual_signature = state.get("resume_signature")
        if actual_signature != expected_signature:
            if isinstance(actual_signature, dict):
                differences = {
                    key: (actual_signature.get(key), expected_signature.get(key))
                    for key in sorted(set(actual_signature) | set(expected_signature))
                    if actual_signature.get(key) != expected_signature.get(key)
                }
            else:
                differences = {"resume_signature": (actual_signature, expected_signature)}
            raise ValueError(
                "Resume training recipe differs from checkpoint: "
                f"{differences}"
            )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng_states = state["rng_states"]
    if len(rng_states) != context.world_size:
        raise ValueError("Checkpoint does not contain one RNG state per rank")
    return (
        int(state["global_step"]),
        int(state["next_epoch"]),
        int(state["next_batch"]),
        rng_states[context.rank],
    )


def _resume_signature(
    args: TrainArgs,
    config: CoreConfig,
    context: DistributedContext,
    total_steps: int,
) -> dict[str, Any]:
    """Return the immutable recipe that must match across a resume boundary."""

    transient_args = {
        "output_dir",
        "resume_from_checkpoint",
        "exit_after_steps",
        "exit_after_seconds",
        "save_every",
        "save_final",
        "log_every",
    }
    training_args = {
        key: value
        for key, value in asdict(args).items()
        if key not in transient_args
    }
    if training_args.get("projector_checkpoint") is not None:
        training_args["projector_checkpoint"] = str(
            Path(training_args["projector_checkpoint"]).expanduser().resolve()
        )
    if training_args.get("data_json"):
        training_args["data_json"] = str(
            Path(training_args["data_json"]).expanduser().resolve()
        )
    if training_args.get("image_root"):
        training_args["image_root"] = str(
            Path(training_args["image_root"]).expanduser().resolve()
        )
    if training_args.get("dataset_path"):
        training_args["dataset_path"] = str(
            Path(training_args["dataset_path"]).expanduser().resolve()
        )
    return {
        "config": asdict(config),
        "training": training_args,
        "world_size": context.world_size,
        "total_steps": total_steps,
    }


def _distributed_resume_path(
    requested: str | None,
    output_dir: Path,
    context: DistributedContext,
) -> Path | None:
    resolved: str | None = None
    if context.is_main:
        path = _resolve_resume_path(requested, output_dir)
        resolved = str(path) if path is not None else None
    if context.enabled:
        payload = [resolved]
        dist.broadcast_object_list(payload, src=0)
        resolved = payload[0]
    return Path(resolved) if resolved is not None else None


def _resolve_resume_path(requested: str | None, output_dir: Path) -> Path | None:
    if requested is None:
        return None
    if requested != "auto":
        path = Path(requested).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint is not a directory: {path}")
        return path
    latest_path = output_dir / "latest.json"
    if latest_path.is_file():
        with latest_path.open(encoding="utf-8") as handle:
            path = Path(json.load(handle)["checkpoint"]).expanduser().resolve()
        if path.is_dir():
            return path
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(r"checkpoint-step(\d+)$")
    for path in output_dir.glob("checkpoint-step*"):
        match = pattern.fullmatch(path.name)
        if match and (path / "trainer_state.pt").is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No resumable checkpoints found under {output_dir}")
    return max(candidates)[1].resolve()


def _gather_rng_states(context: DistributedContext) -> list[dict[str, Any]]:
    local = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(context.device),
    }
    if not context.enabled:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("Failed to gather RNG state from every training rank")
    return [item for item in gathered if item is not None]


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device=device)


def _require_finite_loss(loss: torch.Tensor, context: DistributedContext) -> None:
    finite = torch.tensor(
        int(bool(torch.isfinite(loss.detach()).item())),
        device=context.device,
        dtype=torch.int32,
    )
    if context.enabled:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError("Non-finite loss detected on at least one rank")


def _mean_across_ranks(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.float().clone()
    if context.enabled:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= context.world_size
    return result


def _any_across_ranks(value: bool, context: DistributedContext) -> bool:
    """Return whether ``value`` is true on any rank, identically on all ranks."""
    if not context.enabled:
        return value
    flag = torch.tensor(int(value), dtype=torch.uint8, device=context.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _next_position(epoch: int, batch: int, loader_length: int) -> tuple[int, int]:
    if batch + 1 == loader_length:
        return epoch + 1, 0
    return epoch, batch + 1


def _unwrap(model: CoreModel | DistributedDataParallel) -> CoreModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def _print(context: DistributedContext, message: str) -> None:
    if context.is_main:
        print(f"[CORE stage] {message}", flush=True)


# ============================================================================
# LLMOE tr / ft entry points
# ============================================================================


# Stage defaults from the released recipe (`train/stage1_pretrain.py` and
# `train/stage2_finetune.py`): stage 1 aligns the projector at 1e-3, stage 2
# tunes projector + LoRA at 2e-4 over longer instruction sequences.
LR_BY_STAGE = {1: 1e-3, 2: 2e-4}


MAX_LENGTH_BY_STAGE = {1: 2048, 2: 4096}


# The original Vicuna/Qwen path keeps the repository's two JSON mixtures.  The
# Llama-3 port keeps LLaVA-558K for stage 1 and explicitly selects the supplied
# LLaVA-NeXT-Data Parquet corpus for stage 2 in its configuration.
HIDROP_STAGE = {1: "pretrain", 2: "finetune"}


def _resolve_training_data(config: Config, stage: int, training_args: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve ``(data_json, image_root)`` for a stage, honouring explicit overrides."""
    dataset_dir = dataset_dir_for_stage(config.input_dir, HIDROP_STAGE[stage])
    data_json = training_args.get("data_json")
    if data_json:
        resolved_data = Path(data_json)
    elif stage == 1 and (dataset_dir / "blip_laion_cc_sbu_558k.json").is_file():
        # Do not recursively walk the 558K image tree just to find the JSON that
        # is specified by the released dataset layout and lives at its root.
        resolved_data = dataset_dir / "blip_laion_cc_sbu_558k.json"
    else:
        resolved_data = find_annotations(dataset_dir, HIDROP_STAGE[stage])
    if not resolved_data.is_file():
        raise FileNotFoundError(f"CORE stage-{stage} annotations not found: {resolved_data}")
    image_root = training_args.get("image_root")
    if image_root:
        resolved_images = Path(image_root)
    else:
        # 558K ships its images under `images/`; the instruction mixtures already
        # prefix each path with its source dataset, so the root is the folder.
        candidate = dataset_dir / "images"
        resolved_images = candidate if stage == 1 and candidate.is_dir() else dataset_dir
    if not resolved_images.is_dir():
        raise FileNotFoundError(f"CORE stage-{stage} image root not found: {resolved_images}")
    return resolved_data, resolved_images


def _dataset_check(config: ExpConfig, stage: int) -> str:
    raw = getattr(config, "custom_args", {}).get("training_args", {})
    if isinstance(raw, dict) and raw.get("dataset_path"):
        root = Path(raw["dataset_path"]).expanduser()
        shards = sorted((root / "data").glob("*.parquet")) if root.is_dir() else []
        if not shards:
            return f"Missing LLaVA-NeXT parquet shards under {root / 'data'}"
        return f"OK with LLaVA-NeXT-Data ({len(shards)} parquet shards)"
    dataset_dir = dataset_dir_for_stage(config.input_dir, HIDROP_STAGE[stage])
    if not dataset_dir.is_dir():
        return f"Missing {dataset_dir.name} dataset under {config.input_dir}"
    try:
        direct = dataset_dir / "blip_laion_cc_sbu_558k.json"
        annotations = (
            direct
            if stage == 1 and direct.is_file()
            else find_annotations(dataset_dir, HIDROP_STAGE[stage])
        )
    except FileNotFoundError:
        return f"Missing {dataset_dir.name} JSON annotations under {dataset_dir}"
    return f"OK with {dataset_dir.name} ({annotations.name})"


def core_tr_fn_cd(config: ExpConfig) -> str:
    return _dataset_check(config, 1)


def core_ft_fn_cd(config: ExpConfig) -> str:
    return _dataset_check(config, 2)


def _core_config_for_training(backbone: VLM, config: Config, stage: int) -> CoreConfig:
    """Take the backbone's CORE hyper-parameters and pin them to the training recipe."""
    base = getattr(backbone, "core_config", None)
    if base is None:
        raise TypeError(
            f"CORE training needs the CORE backbone; got {type(backbone).__name__}. "
            "Pass --backbone core."
        )
    compressor_args = config.custom_args.get("compressor_args", {})
    if not isinstance(compressor_args, dict):
        raise TypeError("custom_args.compressor_args must be a JSON/YAML object")
    core_config = deepcopy(base)
    require_flash_attention_2(core_config, verify_installation=False)
    for key in (
        "merge_mode",
        "rate",
        "fixed_tokens",
        "retain_ratio",
        "mask_threshold",
        "large_object_first",
    ):
        if key in compressor_args:
            setattr(core_config, key, compressor_args[key])
    # Training always sees the full F3 grid; a compressed budget here would bake
    # an evaluation-time strategy into the projector.
    full_grid = core_config.num_visual_tokens
    if core_config.rate == "fixed" and core_config.fixed_tokens not in (None, full_grid):
        raise ValueError(
            f"CORE trains on the full {full_grid}-token grid, but this run asks for "
            f"fixed_tokens={core_config.fixed_tokens}. Token budgets belong to evaluation."
        )
    if core_config.retain_ratio not in (None, 1.0):
        raise ValueError(
            f"CORE trains on the full {full_grid}-token grid, but this run asks for "
            f"retain_ratio={core_config.retain_ratio}. Token budgets belong to evaluation."
        )
    core_config.rate = "fixed"
    core_config.fixed_tokens = full_grid
    core_config.retain_ratio = None
    core_config.merge_mode = "hard"
    core_config.use_lora = stage == 2
    core_config.validate()
    return core_config


def _train_args(config: Config, stage: int) -> TrainArgs:
    """Map ``custom_args.training_args`` onto the stage's :class:`TrainArgs`."""
    raw = config.custom_args.get("training_args", {})
    if not isinstance(raw, dict):
        raise TypeError("custom_args.training_args must be a JSON/YAML object")
    training_args = dict(raw)
    valid = {field.name for field in fields(TrainArgs)}
    unknown = sorted(set(training_args) - valid - {"data_json", "image_root", "output_dir"})
    if unknown:
        raise ValueError(f"Unknown CORE training_args fields: {unknown}")

    dataset_path = training_args.get("dataset_path")
    if dataset_path:
        if stage == 1:
            raise ValueError(
                "CORE stage 1 is projector pretraining on LLaVA-558K; "
                "do not set training_args.dataset_path"
            )
        dataset_root = Path(dataset_path).expanduser().resolve()
        if not sorted((dataset_root / "data").glob("*.parquet")):
            raise FileNotFoundError(
                f"No LLaVA-NeXT parquet shards under {dataset_root / 'data'}"
            )
        data_json = Path("")
        image_root = dataset_root
    else:
        data_json, image_root = _resolve_training_data(config, stage, training_args)
    output_dir = training_args.get("output_dir") or f"{config.output_dir}/core/stage{stage}"
    if (
        stage == 2
        and not training_args.get("projector_checkpoint")
        and not training_args.get("resume_from_checkpoint")
    ):
        raise ValueError(
            "CORE stage 2 warm-starts from stage 1. Set training_args.projector_checkpoint "
            "to the projector.pt written by `tr`."
        )
    defaults = {
        "lr": LR_BY_STAGE[stage],
        "max_length": MAX_LENGTH_BY_STAGE[stage],
        "local_files_only": True,
    }
    overrides = {
        key: value
        for key, value in training_args.items()
        if key not in ("data_json", "image_root", "output_dir")
    }
    return TrainArgs(
        data_json="" if dataset_path else str(data_json),
        image_root=str(image_root),
        output_dir=str(output_dir),
        stage=stage,
        **{**defaults, **overrides},
    )


def _run_stage(backbone: VLM, config: Config, stage: int) -> None:
    core_config = _core_config_for_training(backbone, config, stage)
    args = _train_args(config, stage)
    if core_config.llm_family == "llama3":
        if stage == 1 and args.dataset_path is not None:
            raise ValueError("Llama-3 CORE stage 1 must use LLaVA-558K")
        if stage == 2 and args.dataset_path is None:
            raise ValueError(
                "Llama-3 CORE stage 2 requires training_args.dataset_path for "
                "LLaVA-NeXT-Data"
            )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(core_config, args)


def train_core(backbone: VLM, config: Config) -> None:
    """Stage 1: projector alignment on the caption mixture (everything else frozen)."""
    _run_stage(backbone, config, 1)


def finetune_core(backbone: VLM, config: Config) -> None:
    """Stage 2: projector + LoRA instruction tuning from the stage-1 projector."""
    _run_stage(backbone, config, 2)
