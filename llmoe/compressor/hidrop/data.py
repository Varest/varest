"""Datasets and LLaVA/Qwen collators for the two HiDrop training stages.

Both stages read the released LLaVA JSON format -- a list of rows with an optional
``image`` and a ``conversations`` list -- and differ only in which file:

* ``tr`` (projector pretraining): ``LLaVA-Pretrain/blip_laion_cc_sbu_558k.json``
  with its ``images`` tree;
* ``ft`` (full finetuning): ``LLaVA-Tuning/llava_v1_5_mix665k.json`` with the
  dataset root as the image folder (its ``image`` paths are already prefixed).

Directory names follow the harness convention (``LLaVA-Pretrain`` is what
``soft_topk_tr.py`` uses); ``training_args.data_path`` / ``training_args.image_folder``
override the lookup. Rows keep their modality length so the released
modality-length-grouped sampler can order them.

Ported from the sidecar's ``train/data.py``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import warnings

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from llmoe.compressor.hidrop.hidrop_llama import IGNORE_INDEX


DATASET_DIR_BY_STAGE = {"pretrain": "LLaVA-Pretrain", "finetune": "LLaVA-Tuning"}
ANNOTATIONS_BY_STAGE = {
    "pretrain": ("blip_laion_cc_sbu_558k.json",),
    "finetune": ("llava_v1_5_mix665k.json",),
}
# Visual tokens LLaVA-1.5 adds per image; used only for length bookkeeping.
VISUAL_TOKENS = 576


def dataset_dir_for_stage(input_dir: str | Path, stage: str) -> Path:
    if stage not in DATASET_DIR_BY_STAGE:
        raise ValueError(f"Unknown HiDrop training stage {stage!r}.")
    return Path(input_dir) / DATASET_DIR_BY_STAGE[stage]


def find_annotations(root: Path, stage: str) -> Path:
    """Locate the stage's released JSON, falling back to any JSON under ``root``."""
    for name in ANNOTATIONS_BY_STAGE[stage]:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    matches = sorted(root.rglob("*.json"))
    if not matches:
        raise FileNotFoundError(f"No JSON annotations found under {root}")
    return matches[0]


def resolve_training_data(input_dir: str | Path, stage: str, *, data_path: str | None = None,
                          image_folder: str | None = None) -> tuple[Path, Path]:
    """Resolve ``(data_path, image_folder)`` for a stage, honouring explicit overrides."""
    dataset_dir = dataset_dir_for_stage(input_dir, stage)
    resolved_data = Path(data_path) if data_path else find_annotations(dataset_dir, stage)
    if not resolved_data.is_file():
        raise FileNotFoundError(f"HiDrop {stage} annotations not found: {resolved_data}")
    if image_folder:
        resolved_images = Path(image_folder)
    else:
        # 558K ships its images under `images/`; 665K paths are already prefixed
        # with their source dataset, so the dataset root is the image folder.
        candidate = dataset_dir / "images"
        resolved_images = candidate if stage == "pretrain" and candidate.is_dir() else dataset_dir
    if not resolved_images.is_dir():
        raise FileNotFoundError(f"HiDrop {stage} image folder not found: {resolved_images}")
    return resolved_data, resolved_images


@dataclass
class LlavaConversationExample:
    image: torch.Tensor | None
    conversations: list[dict[str, str]]
    has_image: bool


class LlavaJsonDataset(Dataset):
    def __init__(self, data_path: str | Path, image_folder: str | Path, dataset_limit: int | None = None) -> None:
        self.data_path = Path(data_path)
        self.image_folder = Path(image_folder)
        rows = json.loads(self.data_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected {self.data_path} to contain a JSON list.")
        self.rows = rows[: int(dataset_limit)] if dataset_limit is not None else rows
        self.has_images = [bool(row.get("image")) for row in self.rows]
        conversation_lengths = [self._conversation_length(row) for row in self.rows]
        self.lengths = [
            length + (VISUAL_TOKENS if has_image else 0)
            for length, has_image in zip(conversation_lengths, self.has_images, strict=True)
        ]
        # Sign carries the modality: the released sampler groups image and
        # text-only rows into separate megabatches.
        self.modality_lengths = [
            length if has_image else -length
            for length, has_image in zip(conversation_lengths, self.has_images, strict=True)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        conversations = self._normalize_conversations(row.get("conversations") or [])
        image_name = row.get("image")
        image = None
        if image_name:
            image = read_image(str(self.image_folder / str(image_name)), mode=ImageReadMode.RGB)
        return {"image": image, "conversations": conversations, "has_image": image is not None}

    def _normalize_conversations(self, conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized = []
        for turn in conversations:
            role = turn.get("from") or turn.get("role")
            value = str(turn.get("value") or turn.get("content") or "")
            if role in {"human", "user"}:
                normalized.append({"role": "user", "value": value})
            elif role in {"gpt", "assistant"}:
                normalized.append({"role": "assistant", "value": value})
            elif role == "system":
                normalized.append({"role": "system", "value": value})
        if not any(turn["role"] == "assistant" for turn in normalized):
            raise ValueError("LLaVA training row must contain at least one assistant turn.")
        return normalized

    def _conversation_length(self, row: dict[str, Any]) -> int:
        conversations = row.get("conversations") or []
        length = 0
        for turn in conversations:
            value = str(turn.get("value") or turn.get("content") or "")
            length += len(value.split())
        return max(1, length)


# ---------------------------------------------------------------------------
# LLaVA and Qwen training collators
# ---------------------------------------------------------------------------


LLAVA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _encoded_input_ids(encoded: Any) -> list[int]:
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("HiDrop expected a single tokenized prompt.")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def tokenize_prompt_with_image_span(prompt: str, tokenizer: Any, image_token_id: int, *,
                                    image_token: str = "<image>", visual_tokens: int = 576) -> list[int]:
    """HF-positive-ID equivalent of upstream LLaVA ``tokenizer_image_token``."""
    chunks = [_encoded_input_ids(tokenizer(chunk)) for chunk in prompt.split(image_token)]
    offset = 0
    input_ids: list[int] = []
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if chunks and chunks[0] and bos_token_id is not None and chunks[0][0] == int(bos_token_id):
        offset = 1
        input_ids.append(chunks[0][0])

    separator = [int(image_token_id)] * (offset + int(visual_tokens))
    interleaved: list[list[int]] = []
    for index, chunk in enumerate(chunks):
        interleaved.append(chunk)
        if index + 1 < len(chunks):
            interleaved.append(separator)
    for chunk in interleaved:
        input_ids.extend(chunk[offset:])
    return input_ids


def canonicalize_multimodal_conversations(conversations: list[dict[str, str]],
                                          image_token: str = "<image>") -> list[dict[str, str]]:
    """Match LLaVA ``preprocess_multimodal``: move the marker to the sentence start."""
    canonicalized: list[dict[str, str]] = []
    for turn in conversations:
        value = str(turn["value"])
        if image_token in value:
            value = f"{image_token}\n{value.replace(image_token, '').strip()}".strip()
        canonicalized.append({"role": str(turn["role"]), "value": value})
    return canonicalized


@dataclass
class LlavaHiDropCollator:
    processor: Any
    model_max_length: int = 2048
    visual_tokens: int = 576
    stage: str = "finetune"

    def __post_init__(self) -> None:
        if self.stage not in {"pretrain", "finetune"}:
            raise ValueError(f"Unknown HiDrop training stage {self.stage!r}.")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("HiDrop collator received an empty batch.")

        rows: list[tuple[list[int], list[int]]] = []
        image_token = getattr(self.processor, "image_token", "<image>")
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None or int(image_token_id) < 0:
            raise ValueError(f"HiDrop tokenizer does not define the {image_token!r} token.")
        for item in batch:
            has_image = bool(item["has_image"])
            input_ids, labels = self._build_tokenized_conversation(item["conversations"], has_image)
            if len(input_ids) > self.model_max_length:
                input_ids = input_ids[: self.model_max_length]
                labels = labels[: self.model_max_length]
            count = sum(1 for token_id in input_ids if token_id == int(image_token_id))
            expected_count = self.visual_tokens if has_image else 0
            if count != expected_count:
                raise ValueError(
                    f"HiDrop expected {expected_count} image placeholder tokens after truncation, got {count}."
                )
            rows.append((input_ids, labels))

        tokenizer = self.processor.tokenizer
        pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        max_len = max(len(input_ids) for input_ids, _ in rows)
        padded_ids = []
        padded_labels = []
        attention_masks = []
        for input_ids, labels in rows:
            pad = max_len - len(input_ids)
            padded_ids.append(input_ids + [pad_token_id] * pad)
            padded_labels.append(labels + [IGNORE_INDEX] * pad)
            attention_masks.append([1] * len(input_ids) + [0] * pad)

        result: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        # Upstream LLaVA always includes an image tensor for multimodal
        # finetuning, using a zero image for text-only rows. This keeps the
        # vision/projector path in the distributed autograd graph. The HF
        # wrapper consumes one feature row per example and attaches an empty
        # feature slice for rows without an image placeholder.
        image_inputs_by_key: dict[str, list[torch.Tensor]] = {}
        for item in batch:
            if item["image"] is None:
                image_inputs = {"pixel_values": self._dummy_pixel_values()}
            else:
                image_inputs = self.processor.image_processor(item["image"], return_tensors="pt")
            for key, value in image_inputs.items():
                if torch.is_tensor(value):
                    image_inputs_by_key.setdefault(key, []).append(value)
        for key, values in image_inputs_by_key.items():
            if len(values) != len(batch):
                # LlavaImageProcessor currently emits only pixel_values. Fail
                # closed if a future processor emits an input for only part of
                # a mixed-modality batch.
                raise ValueError(f"Image processor key {key!r} is missing for part of the training batch.")
            result[key] = torch.cat(values, dim=0) if values and values[0].shape[:1] == (1,) \
                else torch.stack(values, dim=0)
        return result

    def _dummy_pixel_values(self) -> torch.Tensor:
        image_processor = self.processor.image_processor
        crop_size = getattr(image_processor, "crop_size", None)
        if isinstance(crop_size, dict):
            height = int(crop_size.get("height", crop_size.get("shortest_edge", 0)))
            width = int(crop_size.get("width", crop_size.get("shortest_edge", 0)))
        elif isinstance(crop_size, int):
            height = width = int(crop_size)
        else:
            height = int(getattr(crop_size, "height", 0) or 0)
            width = int(getattr(crop_size, "width", 0) or 0)
        if height < 1 or width < 1:
            raise ValueError("HiDrop could not determine the vision crop size for a text-only dummy image.")
        channels = len(getattr(image_processor, "image_mean", (0.0, 0.0, 0.0)))
        return torch.zeros((1, channels, height, width), dtype=torch.float32)

    def _tokenize_prompt(self, prompt: str, has_image: bool) -> list[int]:
        tokenizer = self.processor.tokenizer
        image_token = getattr(self.processor, "image_token", "<image>")
        if not has_image:
            return _encoded_input_ids(tokenizer(prompt))
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        return tokenize_prompt_with_image_span(
            prompt,
            tokenizer,
            int(image_token_id),
            image_token=image_token,
            visual_tokens=self.visual_tokens,
        )

    def _validate_and_canonicalize(self, conversations: list[dict[str, str]],
                                   has_image: bool) -> list[dict[str, str]]:
        image_token = getattr(self.processor, "image_token", "<image>")
        turns = canonicalize_multimodal_conversations(conversations, image_token)
        if any(turn["role"] == "system" for turn in turns):
            raise ValueError("The paper's LLaVA plain/v1 preprocessing does not accept per-example system turns.")
        if turns and turns[0]["role"] != "user":
            turns = turns[1:]
        if not turns:
            raise ValueError("HiDrop training conversation contains no user turn.")
        for index, turn in enumerate(turns):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if turn["role"] != expected_role:
                raise ValueError(
                    f"HiDrop expected alternating user/assistant turns; turn {index} is {turn['role']!r}."
                )
        marker_count = sum(turn["value"].count(image_token) for turn in turns)
        expected_markers = 1 if has_image else 0
        if marker_count != expected_markers:
            raise ValueError(
                f"HiDrop expected {expected_markers} image marker(s) for has_image={has_image}, got {marker_count}."
            )
        return turns

    def _build_plain(self, turns: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        image_token = getattr(self.processor, "image_token", "<image>")
        if len(turns) != 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
            raise ValueError("LLaVA plain projector pretraining requires exactly one user/assistant pair.")
        if image_token not in turns[0]["value"]:
            raise ValueError("LLaVA plain projector pretraining requires the image marker in the user turn.")
        prompt = image_token + turns[1]["value"] + "\n"
        input_ids = self._tokenize_prompt(prompt, True)
        labels = list(input_ids)
        image_prefix_length = len(self._tokenize_prompt(image_token, True))
        labels[:image_prefix_length] = [IGNORE_INDEX] * image_prefix_length
        return input_ids, labels

    def _build_vicuna_v1(self, turns: list[dict[str, str]], has_image: bool) -> tuple[list[int], list[int]]:
        tokenizer = self.processor.tokenizer
        eos = tokenizer.eos_token or "</s>"
        prompt_parts = [LLAVA_SYSTEM_PROMPT, " "]
        for turn in turns:
            role = "USER" if turn["role"] == "user" else "ASSISTANT"
            if turn["value"]:
                prompt_parts.extend((role, ": ", turn["value"], " " if role == "USER" else eos))
            else:
                prompt_parts.extend((role, ":"))
        prompt = "".join(prompt_parts)
        input_ids = self._tokenize_prompt(prompt, has_image)
        labels = list(input_ids)

        separator = " ASSISTANT: "
        rounds = prompt.split(eos)
        current_length = 1
        labels[:current_length] = [IGNORE_INDEX] * current_length
        for round_index, round_text in enumerate(rounds):
            if not round_text:
                break
            parts = round_text.split(separator)
            if len(parts) != 2:
                break
            instruction = parts[0] + separator
            round_length = len(self._tokenize_prompt(round_text, has_image))
            instruction_length = len(self._tokenize_prompt(instruction, has_image)) - 2
            if round_index != 0 and not bool(getattr(tokenizer, "legacy", True)):
                round_length -= 1
                instruction_length -= 1
            labels[current_length:current_length + instruction_length] = [IGNORE_INDEX] * instruction_length
            current_length += round_length
        labels[current_length:] = [IGNORE_INDEX] * max(0, len(labels) - current_length)
        if current_length < self.model_max_length and current_length != len(input_ids):
            # Match the official LLaVA/HiDrop preprocess_v1 behavior: malformed
            # or tokenizer-mismatched examples remain in the batch but make no
            # contribution to the language-model loss.
            labels[:] = [IGNORE_INDEX] * len(labels)
            print(
                "WARNING: Vicuna-v1 tokenization mismatch while constructing HiDrop labels: "
                f"{current_length} vs. {len(input_ids)}. (ignored)"
            )
        return input_ids, labels

    def _build_tokenized_conversation(self, conversations: list[dict[str, str]],
                                      has_image: bool) -> tuple[list[int], list[int]]:
        turns = self._validate_and_canonicalize(conversations, has_image)
        if self.stage == "pretrain":
            if not has_image:
                raise ValueError("LLaVA plain projector pretraining requires an image for every row.")
            input_ids, labels = self._build_plain(turns)
        else:
            input_ids, labels = self._build_vicuna_v1(turns, has_image)
        image_token = getattr(self.processor, "image_token", "<image>")
        image_token_id = int(self.processor.tokenizer.convert_tokens_to_ids(image_token))
        labels = [
            IGNORE_INDEX if token_id == image_token_id else label
            for token_id, label in zip(input_ids, labels, strict=True)
        ]
        return input_ids, labels


# ---------------------------------------------------------------------------
# Qwen2.5-VL preprocessing
# ---------------------------------------------------------------------------

# Qwen2.5-VL's released assistant system prompt.
QWEN_SYSTEM_PROMPT = "You are a helpful assistant."
IMAGE_MARKER = "<image>"


#: Largest image one row may build, in visual tokens; `--max_pixels` is this many
#: times (patch * merge)^2. 2048 is the common setting in the Qwen finetuning
#: literature (1280 is the other one often quoted, and is what Qwen's own scripts
#: default to). NOTE this only binds if `model_max_length` leaves room for it:
#: at the released 2048-token context an image may occupy the whole sequence, so
#: the effective cap falls back to the text headroom and `_clamp_image_budget`
#: says so. Raise `model_max_length` to spend the full budget on an image.
DEFAULT_QWEN_IMAGE_TOKENS = 2048


@dataclass
class QwenHiDropCollator:
    """Build Qwen batches for HiDrop projector pretraining and full finetuning.

    Qwen's dynamic image grid makes the placeholder count image-specific. Labels are
    built from supervised/unsupervised ChatML segments, while visual patches are packed
    in the flat NaViT representation expected by Qwen2.5-VL. Text-only rows retain a
    zero-weighted dummy image so distributed training keeps the vision path in-graph.
    """

    processor: Any
    model_max_length: int = 2048
    stage: str = "finetune"
    system_prompt: str = QWEN_SYSTEM_PROMPT
    #: Cap on what one image may cost in visual tokens. ``None`` takes
    #: :data:`DEFAULT_QWEN_IMAGE_TOKENS`, clamped to the room ``model_max_length``
    #: leaves; set an integer to pin it. See :meth:`_clamp_image_budget`.
    image_max_tokens: int | None = None
    #: Context reserved for text when the cap is derived.
    text_token_reserve: int = 512

    def __post_init__(self) -> None:
        if self.stage not in {"pretrain", "finetune"}:
            raise ValueError(f"Unknown HiDrop training stage {self.stage!r}.")
        self.image_token: str = getattr(self.processor, "image_token", "<|image_pad|>")
        tokenizer = self.processor.tokenizer
        self.image_token_id: int = int(tokenizer.convert_tokens_to_ids(self.image_token))
        if self.image_token_id < 0:
            raise ValueError(f"Qwen tokenizer does not define {self.image_token!r}.")
        self.merge_size: int = int(getattr(self.processor.image_processor, "merge_size", 2))
        self._clamp_image_budget()

    def _clamp_image_budget(self) -> None:
        """Bound what one image may cost, in visual tokens.

        Qwen sizes each image dynamically, so a large or extreme-aspect one can be
        worth thousands of tokens -- the 558K set holds images costing over 5000,
        which overruns ``model_max_length`` and makes truncation cut into the image
        span. Raising ``model_max_length`` to fit the outliers would inflate memory
        for every row; capping the image is the cheaper half of the trade the
        truncation error offers.

        The knob is in *tokens*, converted to pixels the way upstream DUET's own
        ``qwen2_5_vl/eval_benchmarks.py`` does (its ``--max_pixels`` is multiplied
        by ``28 * 28``). The factor is read from the processor rather than
        hard-coded, since it is ``(patch_size * merge_size) ** 2``.

        Left unset it is :data:`DEFAULT_QWEN_IMAGE_TOKENS`, clamped to whatever
        ``model_max_length`` actually leaves for text -- with a warning when the
        clamp bites, since a run that quietly spends less than its stated budget
        is worse than one that refuses. It only binds on images larger than the
        cap, so ordinary rows are untouched: across a 4000-row sample the median
        image is 192 tokens in the 558K pretrain set and 345 in mix665k, and the
        largest single image in mix665k is 1380.

        Upstream leaves the budget open because it only ever *evaluates* on Qwen,
        where one long row is harmless; training has to bound it, or truncation
        eventually cuts into an image span and the collator refuses the batch.
        """
        headroom = max(1, int(self.model_max_length) - int(self.text_token_reserve))
        if self.image_max_tokens is not None:
            budget = int(self.image_max_tokens)
        else:
            budget = min(DEFAULT_QWEN_IMAGE_TOKENS, headroom)
            if DEFAULT_QWEN_IMAGE_TOKENS > headroom:
                # Silently spending less than the configured budget is how a run
                # ends up not matching the setting it claims to use.
                warnings.warn(
                    f"Qwen image budget clamped to {budget} tokens: the default "
                    f"{DEFAULT_QWEN_IMAGE_TOKENS} does not fit "
                    f"model_max_length={self.model_max_length} with "
                    f"{self.text_token_reserve} reserved for text. Raise "
                    f"model_max_length to at least "
                    f"{DEFAULT_QWEN_IMAGE_TOKENS + self.text_token_reserve} to use "
                    f"the full budget.",
                    stacklevel=3,
                )
        if budget < 1:
            raise ValueError(f"image_max_tokens must be >= 1, got {budget}")
        if budget > headroom:
            # An explicit budget still cannot be allowed to overrun the context;
            # that is the failure this cap exists to prevent.
            raise ValueError(
                f"image_max_tokens={budget} leaves no room for text in "
                f"model_max_length={self.model_max_length} (reserve "
                f"{self.text_token_reserve}); the image span would be truncated."
            )
        image_processor = self.processor.image_processor
        per_token = (int(image_processor.patch_size) * self.merge_size) ** 2
        max_pixels = budget * per_token
        setattr(image_processor, "max_pixels", max_pixels)
        # transformers 5.x carries the same bound inside `size` as `longest_edge`;
        # set both so the cap holds whichever the installed version reads.
        size = getattr(image_processor, "size", None)
        shortest = per_token
        if isinstance(size, dict):
            shortest = int(size.get("shortest_edge") or per_token)
        elif size is not None and getattr(size, "shortest_edge", None):
            shortest = int(size.shortest_edge)
        image_processor.size = {"shortest_edge": min(shortest, max_pixels),
                                "longest_edge": max_pixels}

    # ------------------------------------------------------------------ tokenizing

    def _encode(self, text: str) -> list[int]:
        """Tokenize a rendered segment without letting the tokenizer add its own specials."""
        return [
            int(token_id)
            for token_id in self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        ]

    def _segments_to_tokens(self, segments: list[tuple[str, bool]], image_tokens: int
                            ) -> tuple[list[int], list[int]]:
        """Tokenize ``(text, supervised)`` segments, splicing the image span in place.

        The image marker is replaced by ``image_tokens`` copies of the placeholder id, so
        the span is contiguous and every placeholder is masked out of the labels.
        """
        input_ids: list[int] = []
        labels: list[int] = []
        for text, supervised in segments:
            if IMAGE_MARKER in text:
                if image_tokens <= 0:
                    raise ValueError("A row without an image cannot contain the image marker.")
                before, _, after = text.partition(IMAGE_MARKER)
                pieces = [
                    self._encode(before),
                    [self.image_token_id] * int(image_tokens),
                    self._encode(after),
                ]
            else:
                pieces = [self._encode(text)]
            for piece in pieces:
                input_ids.extend(piece)
                labels.extend(piece if supervised else [IGNORE_INDEX] * len(piece))
        # Placeholders are never a prediction target, whatever segment they landed in.
        labels = [
            IGNORE_INDEX if token_id == self.image_token_id else label
            for token_id, label in zip(input_ids, labels, strict=True)
        ]
        return input_ids, labels

    # ------------------------------------------------------------------- templates

    def _normalize_turns(self, conversations: list[dict[str, str]], has_image: bool
                         ) -> list[dict[str, str]]:
        turns = []
        for turn in conversations:
            value = str(turn["value"])
            if IMAGE_MARKER in value:
                value = f"{IMAGE_MARKER}\n{value.replace(IMAGE_MARKER, '').strip()}".strip()
            turns.append({"role": str(turn["role"]), "value": value})
        if any(turn["role"] == "system" for turn in turns):
            raise ValueError("HiDrop's Qwen preprocessing does not accept per-example system turns.")
        if turns and turns[0]["role"] != "user":
            turns = turns[1:]
        if not turns:
            raise ValueError("HiDrop training conversation contains no user turn.")
        for index, turn in enumerate(turns):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if turn["role"] != expected_role:
                raise ValueError(
                    f"HiDrop expected alternating user/assistant turns; turn {index} is "
                    f"{turn['role']!r}."
                )
        marker_count = sum(turn["value"].count(IMAGE_MARKER) for turn in turns)
        expected_markers = 1 if has_image else 0
        if marker_count != expected_markers:
            raise ValueError(
                f"HiDrop expected {expected_markers} image marker(s) for has_image={has_image}, "
                f"got {marker_count}."
            )
        return turns

    def _vision_wrapped(self, text: str) -> str:
        """Put the image marker inside Qwen's ``<|vision_start|>``/``<|vision_end|>`` pair."""
        return text.replace(IMAGE_MARKER, f"<|vision_start|>{IMAGE_MARKER}<|vision_end|>")

    def _build_plain(self, turns: list[dict[str, str]]) -> list[tuple[str, bool]]:
        """Stage 1, mirroring LLaVA ``plain``: drop the instruction, supervise the caption."""
        if len(turns) != 2:
            raise ValueError("HiDrop stage-1 pretraining requires exactly one user/assistant pair.")
        if IMAGE_MARKER not in turns[0]["value"]:
            raise ValueError("HiDrop stage-1 pretraining requires the image marker in the user turn.")
        return [
            (self._vision_wrapped(IMAGE_MARKER), False),
            (turns[1]["value"] + "<|im_end|>", True),
        ]

    def _build_chatml(self, turns: list[dict[str, str]]) -> list[tuple[str, bool]]:
        """Stage 2: Qwen ChatML rounds, supervising only assistant content."""
        segments: list[tuple[str, bool]] = [
            (f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n", False)
        ]
        for turn in turns:
            if turn["role"] == "user":
                segments.append(
                    (f"<|im_start|>user\n{self._vision_wrapped(turn['value'])}<|im_end|>\n", False)
                )
            else:
                # The generation prefix is not supervised; the answer and its terminator are.
                segments.append(("<|im_start|>assistant\n", False))
                segments.append((f"{turn['value']}<|im_end|>\n", True))
        return segments

    def _build_row(self, conversations: list[dict[str, str]], has_image: bool, image_tokens: int
                   ) -> tuple[list[int], list[int]]:
        turns = self._normalize_turns(conversations, has_image)
        if self.stage == "pretrain":
            if not has_image:
                raise ValueError("HiDrop stage-1 pretraining requires an image for every row.")
            segments = self._build_plain(turns)
        else:
            segments = self._build_chatml(turns)
        return self._segments_to_tokens(segments, image_tokens)

    # ---------------------------------------------------------------------- images

    def _process_image(self, image: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
        """``(pixel_values, image_grid_thw, llm_token_count)`` for one image."""
        processed = self.processor.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        grid = processed["image_grid_thw"]
        tokens = int(grid.prod(-1).sum().item()) // (self.merge_size ** 2)
        return pixel_values, grid, tokens

    def _dummy_image(self) -> np.ndarray:
        """The smallest image the processor will accept, for text-only rows."""
        side = int(self.processor.image_processor.patch_size) * self.merge_size
        return np.zeros((side, side, 3), dtype=np.uint8)

    # ------------------------------------------------------------------- collating

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("HiDrop collator received an empty batch.")
        rows: list[tuple[list[int], list[int]]] = []
        pixel_chunks: list[torch.Tensor] = []
        grid_chunks: list[torch.Tensor] = []

        for item in batch:
            has_image = bool(item["has_image"])
            image = item["image"] if has_image else self._dummy_image()
            pixel_values, grid, image_tokens = self._process_image(self._as_processor_image(image))
            pixel_chunks.append(pixel_values)
            grid_chunks.append(grid)
            input_ids, labels = self._build_row(
                item["conversations"], has_image, image_tokens if has_image else 0
            )
            if len(input_ids) > self.model_max_length:
                placeholders_before = sum(
                    1 for token_id in input_ids if token_id == self.image_token_id
                )
                input_ids = input_ids[: self.model_max_length]
                labels = labels[: self.model_max_length]
                placeholders_after = sum(
                    1 for token_id in input_ids if token_id == self.image_token_id
                )
                if placeholders_after != placeholders_before:
                    raise ValueError(
                        f"Truncating to model_max_length={self.model_max_length} cut into the "
                        f"image span ({placeholders_before} -> {placeholders_after} "
                        f"placeholders). Raise model_max_length, or lower the image processor's "
                        f"max_pixels so each image costs fewer tokens."
                    )
            expected = image_tokens if has_image else 0
            actual = sum(1 for token_id in input_ids if token_id == self.image_token_id)
            if actual != expected:
                raise ValueError(
                    f"HiDrop expected {expected} image placeholders, got {actual}."
                )
            rows.append((input_ids, labels))

        tokenizer = self.processor.tokenizer
        pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        max_len = max(len(input_ids) for input_ids, _ in rows)
        padded_ids, padded_labels, attention_masks = [], [], []
        for input_ids, labels in rows:
            pad = max_len - len(input_ids)
            padded_ids.append(input_ids + [pad_token_id] * pad)
            padded_labels.append(labels + [IGNORE_INDEX] * pad)
            attention_masks.append([1] * len(input_ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            # NaViT packing: patches concatenate along dim 0, one grid row per image.
            "pixel_values": torch.cat(pixel_chunks, dim=0),
            "image_grid_thw": torch.cat(grid_chunks, dim=0),
        }

    @staticmethod
    def _as_processor_image(image: Any) -> Any:
        """Qwen's image processor takes HWC arrays/PIL; the dataset yields CHW uint8 tensors."""
        if torch.is_tensor(image):
            if image.ndim != 3:
                raise ValueError(f"Expected a CHW image tensor, got {tuple(image.shape)}.")
            return image.permute(1, 2, 0).cpu().numpy()
        return image
