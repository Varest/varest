"""Training entry point for VisionZip Efficient Tuning.

LLaVA keeps its vision encoder and language model frozen, trains only
multi_modal_projector after VisionZip reduction, and saves that projector state.
Qwen follows the same ownership rule: its visual.merger combines 2-by-2 patches
and projects them into decoder width, so it is the projector-equivalent module
and the only trainable/saved state.

The Qwen forward lives in vision_zip_qwen_ft.py because its reduction is
structurally different: it runs after the merger, preserves full-sequence
M-RoPE before pruning, and reduces each variable-length NaViT row independently.
Both backbones accept an earlier module artifact through
compressor_args.checkpoint_path and emit a small module-only checkpoint.
"""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from json import load
from pathlib import Path
from random import Random
from typing import Any

import torch
from mle.vars import ExpConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llmoe.backbone import VLM
from llmoe.weights import BACKBONE_SLUGS, backbone_key
from llmoe.config import Config
from llmoe.compressor.vision_zip.vision_zip import VisionZipCompressor, resolve_image_token_id


# Directory under `config.input_dir` holding llava_v1_5_mix665k.json plus the
# image tree its relative "image" paths resolve against. Both spellings are in use
# across the harness (`hidrop/data.py` reads the same file from `LLaVA-Tuning`).
DATASET_DIR_NAMES = ("LLaVA-Instruct", "LLaVA-Tuning")
DATASET_DIR_NAME = DATASET_DIR_NAMES[0]


def _resolve_dataset_dir(input_dir: str | Path) -> Path | None:
    for name in DATASET_DIR_NAMES:
        candidate = Path(input_dir) / name
        if candidate.is_dir():
            return candidate
    return None

# System prompt and role layout match LLaVA-1.5 (vicuna-v1) and the inference
# template in `llmoe/backbone/llava.py`.
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
_ROLE_USER = "USER"
_ROLE_ASSISTANT = "ASSISTANT"
_IMAGE_PLACEHOLDER = "<image>"
_IGNORE_INDEX = -100


def vision_zip_ft_fn_cd(config: ExpConfig) -> str:
    dataset_dir = _resolve_dataset_dir(config.input_dir)
    if dataset_dir is None:
        names = " or ".join(DATASET_DIR_NAMES)
        return f"Missing {names} dataset under {config.input_dir}"
    if not any(dataset_dir.rglob("*.json")):
        return f"Missing JSON annotations under {dataset_dir}"
    return f"OK with {dataset_dir.name}"


# --------------------------------------------------------------------------- io

def _find_annotations(root: Path) -> Path:
    preferred = ("llava_v1_5_mix665k.json", "llava_instruct_150k.json", "train.json")
    for name in preferred:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    matches = sorted(root.rglob("*.json"))
    if not matches:
        raise FileNotFoundError(f"No JSON annotations found under {root}")
    return matches[0]


def load_image_samples(data_path: str | Path, *, subset_frac: float = 1.0, seed: int = 42,
                       limit: int | None = None) -> list[dict[str, Any]]:
    """Load the instruction set, keep image-bearing conversations, take a subset.

    The paper tunes on ~1/10 of the data, so ``subset_frac=0.1`` reproduces it.
    ``limit`` applies after sampling, for quick runs.
    """
    with open(data_path, encoding="utf-8") as handle:
        raw = load(handle)
    samples = [item for item in raw if item.get("image")]
    if not 0.0 < subset_frac <= 1.0:
        raise ValueError(f"subset_frac must be in (0, 1], got {subset_frac}")
    if subset_frac < 1.0:
        count = max(1, int(round(len(samples) * subset_frac)))
        samples = Random(seed).sample(samples, count)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No image-bearing training samples found in {data_path}")
    return samples


# ------------------------------------------------------------------ formatting

def _normalize_conversations(conversations: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Return ``[(role, text)]`` with exactly one ``<image>`` on the first user turn."""
    turns: list[tuple[str, str]] = []
    for turn in conversations:
        role = _ROLE_USER if turn.get("from") == "human" else _ROLE_ASSISTANT
        turns.append((role, str(turn.get("value", "")).replace(_IMAGE_PLACEHOLDER, "").strip()))
    if not turns or turns[0][0] != _ROLE_USER:
        raise ValueError("Conversation must start with a human/user turn")
    turns[0] = (turns[0][0], f"{_IMAGE_PLACEHOLDER}\n{turns[0][1]}")
    return turns


def _build_prompt_and_spans(turns: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    """Build the vicuna-v1 conversation and, per assistant turn, the
    ``(prefix_ending_at_"ASSISTANT:", prefix_plus_answer)`` pair locating its span.

    Layout (SeparatorStyle.TWO): ``{system} USER: {q} ASSISTANT: {a}</s>USER: ...``
    — space after a user turn, ``</s>`` after an assistant turn.
    """
    prompt = SYSTEM_PROMPT + " "
    spans: list[tuple[str, str]] = []
    for role, text in turns:
        if role == _ROLE_USER:
            prompt += f"{_ROLE_USER}: {text} "
        else:
            prefix = prompt + f"{_ROLE_ASSISTANT}:"
            prefix_plus_answer = f"{prefix} {text}"
            spans.append((prefix, prefix_plus_answer))
            prompt = prefix_plus_answer + "</s>"
    return prompt, spans


def build_example(sample: Mapping[str, Any], tokenizer: Any, *, image_token_id: int,
                  num_image_tokens: int, max_length: int = 2048) -> tuple[list[int], list[int]]:
    """Tokenize one conversation into ``(input_ids, labels)``.

    Answer spans are located by prefix length — tokenizing up to ``"ASSISTANT:"``
    and up to the end of the answer — rather than by splitting rounds: boundaries
    fall at the ``:`` token, so the prefix tokenization is a true prefix and the
    span is exact. ``<image>`` stays a single token while masking is computed,
    then expands to ``num_image_tokens``; truncation is tail-only so the image
    tokens near the start are never cut.
    """
    prompt, spans = _build_prompt_and_spans(_normalize_conversations(sample["conversations"]))

    input_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    labels = [_IGNORE_INDEX] * len(input_ids)
    eos_id = tokenizer.eos_token_id
    for prefix, prefix_plus_answer in spans:
        start = len(tokenizer(prefix, add_special_tokens=True).input_ids)
        end = len(tokenizer(prefix_plus_answer, add_special_tokens=True).input_ids)
        for index in range(start, min(end, len(input_ids))):
            labels[index] = input_ids[index]
        # Unmask the `</s>` after the answer so the model learns to stop.
        if end < len(input_ids) and input_ids[end] == eos_id:
            labels[end] = eos_id

    expanded_ids: list[int] = []
    expanded_labels: list[int] = []
    for token_id, label in zip(input_ids, labels):
        if token_id == image_token_id:
            expanded_ids.extend([image_token_id] * num_image_tokens)
            expanded_labels.extend([_IGNORE_INDEX] * num_image_tokens)
        else:
            expanded_ids.append(token_id)
            expanded_labels.append(label)
    return expanded_ids[:max_length], expanded_labels[:max_length]


# ---------------------------------------------------------------------- dataset

class _LlavaInstructDataset(Dataset):
    """Instruction samples with ``<image>`` pre-expanded to the compressed count.

    Expanding directly to ``num_image_tokens`` (rather than emitting the full 576
    and trimming) guarantees input/label alignment. ``pixel_values`` come from the
    backbone's own image processor, so there is no train/inference preprocessing
    skew.
    """

    def __init__(self, samples: list[dict[str, Any]], *, image_root: Path, tokenizer: Any,
                 image_processor: Any, image_token_id: int, num_image_tokens: int,
                 max_length: int = 2048) -> None:
        self._samples = samples
        self._image_root = image_root
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._image_token_id = image_token_id
        self._num_image_tokens = num_image_tokens
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._samples)

    def _image_path(self, image: str) -> Path:
        candidate = Path(image)
        for path in (candidate, self._image_root / candidate,
                     self._image_root / "images" / candidate,
                     self._image_root / "images" / candidate.name):
            if path.is_file():
                return path
        raise FileNotFoundError(f"Could not find training image {image!r} under {self._image_root}")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self._samples[index]
        input_ids, labels = build_example(
            sample, self._tokenizer, image_token_id=self._image_token_id,
            num_image_tokens=self._num_image_tokens, max_length=self._max_length,
        )
        with Image.open(self._image_path(str(sample["image"]))) as image:
            pixel_values = self._image_processor(
                images=[image.convert("RGB")], return_tensors="pt"
            )["pixel_values"][0]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class _DataCollator:
    """Right-pad the batch (correct for teacher-forced loss; left padding is only
    needed for generation) and stack ``pixel_values``."""

    def __init__(self, pad_token_id: int) -> None:
        self._pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []
        for item in batch:
            ids, label = item["input_ids"], item["labels"]
            pad = max_len - ids.shape[0]
            if pad:
                ids = torch.cat([ids, torch.full((pad,), self._pad_token_id, dtype=ids.dtype)])
                label = torch.cat([label, torch.full((pad,), _IGNORE_INDEX, dtype=label.dtype)])
            input_ids.append(ids)
            labels.append(label)
            attention_mask.append(torch.cat([
                torch.ones(max_len - pad, dtype=torch.long), torch.zeros(pad, dtype=torch.long),
            ]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        }



# ---------------------------------------------------------------------- trainer

def _training_options(config: Config) -> dict[str, Any]:
    options = config.custom_args.get("training_args", {})
    if not isinstance(options, Mapping):
        raise TypeError("training_args must be a mapping")
    return dict(options)


def finetune_vision_zip(backbone: VLM, config: Config) -> None:
    """Tune the projector-equivalent module, with the rest of the VLM frozen.

    ``config.custom_args`` accepts ``compressor_args`` plus ``training_args``:
    ``num_epochs``, ``batch_size``, ``learning_rate``, ``weight_decay``,
    ``gradient_accumulation_steps``, ``max_grad_norm``, ``num_workers``,
    ``warmup_ratio``, ``subset_frac``, ``max_length``, ``seed``, ``max_samples``,
    ``max_steps``, and ``checkpoint_name``.
    """
    dataset_dir = _resolve_dataset_dir(config.input_dir)
    if dataset_dir is None:
        names = " or ".join(DATASET_DIR_NAMES)
        raise FileNotFoundError(f"Missing {names} dataset under {config.input_dir}")

    compressor_args = config.custom_args.get("compressor_args", {})
    if not isinstance(compressor_args, Mapping):
        raise TypeError("compressor_args must be a mapping")
    compressor = VisionZipCompressor(**dict(compressor_args))
    if compressor._retain_ratio is not None:
        from llmoe.compressor.vision_zip.vision_zip_qwen_ft import (
            finetune_vision_zip_qwen,
        )

        finetune_vision_zip_qwen(
            backbone, config, compressor, dataset_dir
        )
        return
    options = _training_options(config)

    num_epochs = int(options.get("num_epochs", 1))
    batch_size = int(options.get("batch_size", 8))
    learning_rate = float(options.get("learning_rate", 2e-5))
    weight_decay = float(options.get("weight_decay", 0.0))
    accumulation = int(options.get("gradient_accumulation_steps", 16))
    max_grad_norm = float(options.get("max_grad_norm", 1.0))
    num_workers = int(options.get("num_workers", 0))
    warmup_ratio = float(options.get("warmup_ratio", 0.03))
    subset_frac = float(options.get("subset_frac", 0.1))
    max_length = int(options.get("max_length", 2048))
    seed = int(options.get("seed", 42))
    if num_epochs < 1 or batch_size < 1 or accumulation < 1 or num_workers < 0:
        raise ValueError("num_epochs, batch_size, and gradient accumulation must be >= 1")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")

    output_dir = Path(config.output_dir) / "vision_zip"
    output_dir.mkdir(parents=True, exist_ok=True)
    # The backbone belongs in the name once it is not the released one: the
    # projector is sized to the decoder's hidden width, so a 13B projector saved
    # under the 7B's filename would both overwrite it and fail to load anywhere.
    # LLaVA-1.5-7B keeps the original name so existing checkpoints still resolve.
    key = backbone_key(backbone)
    infix = "" if key in (None, "", "llava") else f"_{BACKBONE_SLUGS.get(key, key)}"
    checkpoint_name = str(options.get(
        "checkpoint_name", f"vision_zip_projector{infix}_k{compressor._max_tokens}.pt"
    ))
    checkpoint_path = output_dir / checkpoint_name

    limit = config.dataset_limit
    if options.get("max_samples") is not None:
        option_limit = int(options["max_samples"])
        limit = option_limit if limit is None else min(limit, option_limit)
    samples = load_image_samples(
        _find_annotations(dataset_dir), subset_frac=subset_frac, seed=seed, limit=limit
    )

    model = backbone.model
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(backbone._device)
    original_training = model.training
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    original_use_cache = getattr(model.config, "use_cache", None)

    # Install before reading the image token id so the forced "full" feature-select
    # strategy and the encoder patch are already in place for every training step.
    compressor.to(device=model_device, dtype=backbone._dtype)
    compressor.install(backbone)
    try:
        image_token_id = resolve_image_token_id(model.config, backbone.processor)
        tokenizer = getattr(backbone.processor, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("VisionZip training requires a processor with a tokenizer")

        dataset = _LlavaInstructDataset(
            samples, image_root=dataset_dir, tokenizer=tokenizer,
            image_processor=backbone.processor.image_processor, image_token_id=image_token_id,
            num_image_tokens=compressor._max_tokens, max_length=max_length,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
            collate_fn=_DataCollator(int(tokenizer.pad_token_id or 0)),
            pin_memory=torch.cuda.is_available(),
        )
        if len(loader) == 0:
            raise RuntimeError(
                "The training dataloader has no batches. Lower batch_size, or raise max_samples "
                "or subset_frac."
            )

        # Freeze everything, then re-enable the projector alone. It is upcast to
        # fp32 so AdamW keeps fp32 master weights while the frozen model stays
        # half precision under autocast.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        projector = model.model.multi_modal_projector
        projector.to(torch.float32)
        for parameter in projector.parameters():
            parameter.requires_grad_(True)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        projector_params = sum(p.numel() for p in projector.parameters())
        if trainable != projector_params:
            raise RuntimeError(
                f"Trainable params ({trainable}) != projector params ({projector_params}); "
                "freezing is wrong."
            )
        print(f"[VisionZip] tuning {trainable:,} projector params on {len(samples):,} samples "
              f"(budget={compressor._max_tokens}, subset_frac={subset_frac})", flush=True)

        model.train()
        model.config.use_cache = False
        # Checkpoint the frozen LLM so backward activations fit; the projector
        # still receives gradients because inputs_embeds carry grad. The vision
        # tower stays un-checkpointed so VisionZip's hook is not recomputed.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            vision_tower = model.model.vision_tower
            if hasattr(vision_tower, "gradient_checkpointing_disable"):
                vision_tower.gradient_checkpointing_disable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        optimizer = torch.optim.AdamW(
            projector.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        updates_per_epoch = (len(loader) + accumulation - 1) // accumulation
        total_updates = max(1, updates_per_epoch * num_epochs)
        if options.get("max_steps") is not None:
            total_updates = min(total_updates, max(1, int(options["max_steps"])))
        warmup_updates = int(warmup_ratio * total_updates)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-8, total_iters=max(1, warmup_updates)
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_updates - warmup_updates)
                ),
            ],
            milestones=[max(1, warmup_updates)],
        ) if warmup_updates > 0 else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates
        )

        use_autocast = model_device.type == "cuda" and backbone._dtype in {
            torch.float16, torch.bfloat16,
        }
        use_scaler = model_device.type == "cuda" and backbone._dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        optimizer.zero_grad(set_to_none=True)
        update = 0

        for epoch in range(num_epochs):
            progress = tqdm(loader, desc=f"VisionZip epoch {epoch + 1}/{num_epochs}")
            for step, batch in enumerate(progress):
                inputs = {key: value.to(model_device) for key, value in batch.items()}
                inputs["pixel_values"] = inputs["pixel_values"].to(backbone._dtype)
                labels = inputs.pop("labels")

                window_start = (step // accumulation) * accumulation
                window_size = min(accumulation, len(loader) - window_start)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=backbone._dtype)
                    if use_autocast else nullcontext()
                )
                with autocast:
                    outputs = model(**inputs, labels=labels, use_cache=False)
                    batch_loss = outputs.loss
                    loss = batch_loss / window_size
                scaler.scale(loss).backward()

                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    update += 1
                progress.set_postfix(
                    loss=f"{float(batch_loss.detach()):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                if update >= total_updates:
                    break
            _save_tuned_module(
                projector, checkpoint_path, {"max_tokens": compressor._max_tokens}, backbone
            )
            if update >= total_updates:
                break
    finally:
        compressor.uninstall()
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        model.train(original_training)

    print(f"Saved VisionZip projector checkpoint to {checkpoint_path}", flush=True)



def _save_tuned_module(module: Any, path: Path, budget: Mapping[str, Any], backbone: VLM) -> None:
    """Write the tuned projector/merger in fp32; eval downcasts on load.

    ``budget`` records what the module was tuned for -- ``max_tokens`` on LLaVA,
    ``retain_ratio`` on Qwen -- and ``_load_projector`` refuses a mismatch.
    """
    torch.save(
        {
            "format_version": 1,
            "compressor": VisionZipCompressor.__name__,
            "config": {**dict(budget), "base_model": backbone._model_id},
            "state_dict": {
                key: value.detach().to(torch.float32).cpu().contiguous()
                for key, value in module.state_dict().items()
            },
        },
        path,
    )
