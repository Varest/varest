"""Training entry point for :class:`SoftTopkCompressor`."""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from json import load
from os.path import exists
from pathlib import Path
from typing import Any

import torch
from mle.vars import ExpConfig
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.compressor.soft_topk.soft_topk import SoftTopkCompressor


def soft_topk_tr_fn_cd(config: ExpConfig) -> str:
    dataset_dir = Path(config.input_dir) / "LLaVA-Pretrain"
    if not dataset_dir.is_dir():
        return f"Missing LLaVA-Pretrain dataset under {config.input_dir}"
    if not any(dataset_dir.rglob("*.json")):
        return f"Missing LLaVA-Pretrain JSON annotations under {dataset_dir}"
    return "OK with LLaVA-Pretrain"


class _LLaVAPretrainDataset(Dataset):
    def __init__(self, root: Path, *, limit: int | None = None) -> None:
        self.root = root
        annotations = self._find_annotations(root)
        with annotations.open(encoding="utf-8") as file:
            records = load(file)
        if isinstance(records, Mapping):
            for key in ("data", "annotations", "samples"):
                if isinstance(records.get(key), list):
                    records = records[key]
                    break
        if not isinstance(records, list):
            raise TypeError(f"Expected a JSON list in {annotations}")
        self.records = records[:limit] if limit is not None else records
        if not self.records:
            raise ValueError(f"No training samples found in {annotations}")

    @staticmethod
    def _find_annotations(root: Path) -> Path:
        preferred = (
            "blip_laion_cc_sbu_558k.json",
            "llava_pretrain.json",
            "train.json",
        )
        for name in preferred:
            matches = list(root.rglob(name))
            if matches:
                return matches[0]
        matches = sorted(root.rglob("*.json"))
        if not matches:
            raise FileNotFoundError(f"No JSON annotations found under {root}")
        return matches[0]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        record = self.records[index]
        if not isinstance(record, Mapping):
            raise TypeError(f"Training record {index} is not an object")
        return record

    def image_path(self, image: str) -> Path:
        candidate = Path(image)
        candidates = (
            candidate,
            self.root / candidate,
            self.root / "images" / candidate,
            self.root / "images" / candidate.name,
        )
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"Could not find training image {image!r} under {self.root}")


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
        raise ValueError("Training sample needs a question/prompt and answer/response")
    return question, answer


def _training_options(config: Config) -> dict[str, Any]:
    custom = config.custom_args
    options = custom.get("training_args", custom.get("soft_topk_training_args", {}))
    if not isinstance(options, Mapping):
        raise TypeError("training_args must be a mapping")
    return dict(options)


def _make_batch(
    backbone: VLM,
    compressor: SoftTopkCompressor,
    dataset: _LLaVAPretrainDataset,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prompts: list[str] = []
    answers: list[str] = []
    images: list[Image.Image] = []
    for record in records:
        question, answer = _question_and_answer(record)
        image_name = record.get("image", record.get("image_path"))
        if not image_name:
            raise ValueError("Training sample is missing its image path")
        image = Image.open(dataset.image_path(str(image_name))).convert("RGB")
        images.append(image)
        prompts.append(question)
        answers.append(answer)

    tokenizer = getattr(backbone.processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("SoftTopk training requires a processor with a tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    formatted_prompts = backbone.format_prompts(prompts)
    eos = tokenizer.eos_token or ""
    answer_suffixes = [f" {answer}{eos}" for answer in answers]
    full_text = [prompt + suffix for prompt, suffix in zip(formatted_prompts, answer_suffixes)]
    try:
        inputs = backbone.processor(
            images=backbone.format_images(images),
            text=full_text,
            return_tensors="pt",
            padding=True,
        )
    finally:
        for image in images:
            image.close()

    image_token_id = compressor._image_token_id(backbone)
    if image_token_id is not None and compressor._retain_ratio is None:
        compressor.trim_placeholders(inputs, image_token_id, int(tokenizer.pad_token_id))

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    labels = torch.full_like(input_ids, -100)
    for index, suffix in enumerate(answer_suffixes):
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        suffix_length = min(len(suffix_ids), int(attention_mask[index].sum()))
        if suffix_length:
            active = attention_mask[index].nonzero(as_tuple=False).flatten()
            labels[index, active[-suffix_length:]] = input_ids[index, active[-suffix_length:]]

    tensor_inputs = {
        key: value
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    return tensor_inputs, labels


def _qwen_pruned_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    keep: torch.Tensor | None,
) -> torch.Tensor:
    """Compute causal loss after gathering labels onto Qwen survivors."""
    if keep is not None:
        if keep.ndim != 1 or keep.numel() != labels.shape[1]:
            raise RuntimeError(
                f"SoftTopk/Qwen got keep mask {tuple(keep.shape)} for "
                f"labels {tuple(labels.shape)}."
            )
        labels = labels[:, keep.to(labels.device)]
    if logits.shape[:2] != labels.shape:
        raise RuntimeError(
            f"SoftTopk/Qwen got logits {tuple(logits.shape[:2])} for "
            f"labels {tuple(labels.shape)}."
        )
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1).to(logits.device),
        ignore_index=-100,
    )


def train_soft_topk(backbone: VLM, config: Config) -> None:
    """Train and save a SoftTopk scorer while keeping the VLM frozen.

    ``config.custom_args`` accepts ``compressor_args`` plus ``training_args``:
    ``num_epochs``, ``batch_size``, ``learning_rate``, ``weight_decay``,
    ``gradient_accumulation_steps``, ``max_grad_norm``, ``num_workers``,
    ``temperature_final``, and ``checkpoint_name``.
    """
    dataset_dir = Path(config.input_dir) / "LLaVA-Pretrain"
    if not exists(dataset_dir):
        raise FileNotFoundError(f"Missing LLaVA-Pretrain dataset under {config.input_dir}")

    compressor_args = config.custom_args.get("compressor_args", {})
    if not isinstance(compressor_args, Mapping):
        raise TypeError("compressor_args must be a mapping")
    compressor = SoftTopkCompressor(**dict(compressor_args))
    options = _training_options(config)

    num_epochs = int(options.get("num_epochs", 1))
    batch_size = int(options.get("batch_size", 1))
    learning_rate = float(options.get("learning_rate", 1e-4))
    weight_decay = float(options.get("weight_decay", 0.01))
    accumulation = int(options.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(options.get("max_grad_norm", 1.0))
    num_workers = int(options.get("num_workers", 0))
    if num_epochs < 1 or batch_size < 1 or accumulation < 1 or num_workers < 0:
        raise ValueError("num_epochs, batch_size, and gradient accumulation must be >= 1")
    initial_temperature = compressor.temperature
    final_temperature = float(options.get("temperature_final", initial_temperature))
    if final_temperature <= 0:
        raise ValueError("temperature_final must be > 0")
    output_dir = Path(config.output_dir) / "soft_topk"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = str(options.get("checkpoint_name", "soft_topk.pt"))
    checkpoint_path = output_dir / checkpoint_name

    limit = config.dataset_limit
    if options.get("max_samples") is not None:
        option_limit = int(options["max_samples"])
        limit = option_limit if limit is None else min(limit, option_limit)
    dataset = _LLaVAPretrainDataset(dataset_dir, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list,
        pin_memory=torch.cuda.is_available(),
    )

    model = backbone.model
    inner = getattr(model, "model", None)
    is_qwen = getattr(inner, "visual", None) is not None
    if is_qwen and batch_size != 1:
        raise ValueError(
            "SoftTopk training on Qwen2.5-VL requires batch_size=1: "
            "NaViT gives each image a different visual-token count."
        )
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(backbone._device)
    original_training = model.training
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    try:
        # Keep optimizer parameters and gradients in FP32.  Autocast converts
        # scorer operations as needed while avoiding unsupported FP16 gradient
        # unscaling and low-precision AdamW state.
        compressor.to(device=model_device, dtype=torch.float32)
        compressor.train()
        compressor._for_training = is_qwen
        compressor.install(backbone)
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
        update = 0

        for epoch in range(num_epochs):
            progress = tqdm(loader, desc=f"SoftTopk epoch {epoch + 1}/{num_epochs}")
            for step, records in enumerate(progress):
                window_start = (step // accumulation) * accumulation
                window_size = min(accumulation, len(loader) - window_start)
                inputs, labels = _make_batch(backbone, compressor, dataset, records)
                inputs = {
                    key: value.to(model_device)
                    for key, value in inputs.items()
                }
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(backbone._dtype)
                labels = labels.to(model_device)

                autocast = (
                    torch.autocast(device_type="cuda", dtype=backbone._dtype)
                    if use_autocast
                    else nullcontext()
                )
                with autocast:
                    if is_qwen:
                        outputs = model(**inputs, use_cache=False)
                        batch_loss = _qwen_pruned_loss(
                            outputs.logits,
                            labels,
                            compressor._qwen.last_keep_mask,
                        )
                    else:
                        outputs = model(
                            **inputs,
                            labels=labels,
                            use_cache=False,
                        )
                        batch_loss = outputs.loss
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
                    update += 1
                    fraction = min(1.0, update / total_updates)
                    compressor.temperature = (
                        initial_temperature * (1.0 - fraction) + final_temperature * fraction
                    )
                progress.set_postfix(
                    loss=f"{float(batch_loss.detach()):.4f}",
                    temperature=f"{compressor.temperature:.3f}",
                )
            compressor.save_checkpoint(checkpoint_path)
    finally:
        compressor.uninstall()
        compressor._for_training = False
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
        model.train(original_training)

    print(f"Saved SoftTopk checkpoint to {checkpoint_path}", flush=True)
