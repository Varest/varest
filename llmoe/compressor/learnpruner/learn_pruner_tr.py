"""End-to-end LPM-only training on 10% of the staged LLaVA 665K mixture.

The paper specifies the STE graph mask and L_ntp + L_prune (tau=0, weight=1).
It does not publish an optimizer, LR, seed, epoch count, or exact MLP activation;
those are explicit run settings here and are recorded with the checkpoint.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import os

import torch
from mle.vars import ExpConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from llmoe.backbone import VLM
from llmoe.compressor.hidrop.data import (
    LlavaHiDropCollator, LlavaJsonDataset, resolve_training_data,
)
from llmoe.compressor.hidrop.hidrop_llama import IGNORE_INDEX
from llmoe.compressor.learnpruner.learn_pruner import LearnPrunerCompressor
from llmoe.config import Config
from llmoe.weights import BACKBONE_SLUGS, backbone_key


def _options(config: Config) -> dict:
    options = config.custom_args.get("training_args", {})
    if not isinstance(options, Mapping):
        raise TypeError("training_args must be a mapping")
    return dict(options)


def _check_data(config: ExpConfig) -> str:
    custom = getattr(config, "custom_args", None)
    options = custom.get("training_args", {}) if isinstance(custom, Mapping) else {}
    try:
        annotations, images = resolve_training_data(
            config.input_dir, "finetune",
            data_path=options.get("data_path"), image_folder=options.get("image_folder"))
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)
    return f"OK: {annotations.name}, images at {images}"


def learn_pruner_tr_fn_cd(config: ExpConfig) -> str:
    return _check_data(config)


def learn_pruner_ft_fn_cd(config: ExpConfig) -> str:
    return _check_data(config)


def _selected_indices(dataset: LlavaJsonDataset, fraction: float, seed: int,
                      limit: int | None) -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError("subset_frac must be in (0,1]")
    if len(dataset) < 1:
        raise ValueError("LLaVA mixture is empty")
    generator = torch.Generator().manual_seed(seed)
    count = max(1, round(len(dataset) * fraction))
    sampled = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    # The paper samples the whole 665K mixture. Text-only members cannot update
    # the LPM when the base VLM is frozen, so only the sampled image rows enter
    # this LPM-only loader; this does not enlarge the 10% sample.
    selected = [index for index in sampled if dataset.has_images[index]]
    if not selected:
        raise ValueError("The sampled 10% LLaVA subset contains no image-bearing examples")
    return selected[:limit] if limit is not None else selected


def _assistant_labels(input_ids: torch.Tensor, tokenizer: Any,
                      image_token_id: int) -> torch.Tensor:
    """Mask Llama-3 headers/users and supervise assistant content plus EOT."""
    if input_ids.ndim != 1:
        raise ValueError(f"Expected one token row, got {tuple(input_ids.shape)}")
    start_id = int(tokenizer.convert_tokens_to_ids("<|start_header_id|>"))
    end_id = int(tokenizer.convert_tokens_to_ids("<|end_header_id|>"))
    eot_id = int(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    assistant_ids = [
        int(value) for value in tokenizer.encode("assistant", add_special_tokens=False)
    ]
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    values = [int(value) for value in input_ids.tolist()]
    cursor = 0
    assistant_regions = 0
    while cursor < len(values):
        try:
            header_start = values.index(start_id, cursor)
            header_end = values.index(end_id, header_start + 1)
        except ValueError:
            break
        role_ids = values[header_start + 1:header_end]
        try:
            content_end = values.index(eot_id, header_end + 1)
            stop = content_end + 1
        except ValueError:
            stop = len(values)
        if role_ids == assistant_ids:
            labels[header_end + 1:stop] = input_ids[header_end + 1:stop]
            assistant_regions += 1
        cursor = max(stop, header_end + 1)
    labels[input_ids == int(image_token_id)] = IGNORE_INDEX
    if assistant_regions == 0 or not bool((labels != IGNORE_INDEX).any().item()):
        raise ValueError(
            "Could not locate supervised assistant content in the Llama-3 chat template"
        )
    return labels


@dataclass
class LlavaNextLearnPrunerCollator:
    """Build a native, unpruned AnyRes training row from the same 665K data."""

    processor: Any
    model_max_length: int = 4096

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(batch) != 1:
            raise NotImplementedError(
                "LearnPruner/LLaVA-NeXT training requires per-device batch size 1 "
                "for ragged AnyRes spans"
            )
        item = batch[0]
        image_token = getattr(self.processor, "image_token", "<image>")
        has_image = item.get("image") is not None
        if not has_image:
            raise ValueError("LearnPruner's sampled LLaVA-NeXT rows must contain an image")
        marker_count = sum(
            str(turn["value"]).count(image_token) for turn in item["conversations"]
        )
        if marker_count != 1:
            raise ValueError(
                f"LLaVA-NeXT training needs exactly one {image_token!r} marker, "
                f"got {marker_count}"
            )

        messages: list[dict[str, Any]] = []
        image_message_index: int | None = None
        for turn in item["conversations"]:
            parts = str(turn["value"]).split(image_token)
            content: list[dict[str, str]] = []
            for part_index, part in enumerate(parts):
                text_value = part.strip()
                if text_value:
                    content.append({"type": "text", "text": text_value})
                if part_index < len(parts) - 1:
                    if turn["role"] != "user":
                        raise ValueError("The LLaVA-NeXT image marker must occur in a user turn")
                    content.append({"type": "image"})
                    image_message_index = len(messages)
            if not content:
                content.append({"type": "text", "text": ""})
            messages.append({"role": turn["role"], "content": content})
        if image_message_index is None:
            raise RuntimeError("The validated LLaVA-NeXT image marker was not inserted")
        if not any(
            message["role"] == "assistant"
            for message in messages[image_message_index + 1:]
        ):
            raise ValueError(
                "LLaVA-NeXT training requires an assistant response after the image turn"
            )

        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        encoded = self.processor(
            images=item["image"], text=rendered, return_tensors="pt",
            truncation=True, max_length=int(self.model_max_length), padding=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise RuntimeError(
                f"LLaVA-NeXT processor returned input_ids {tuple(input_ids.shape)}"
            )
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None or int(image_token_id) < 0:
            raise ValueError(f"LLaVA-NeXT tokenizer does not define {image_token!r}")
        positions = torch.where(input_ids[0] == int(image_token_id))[0]
        if positions.numel() == 0:
            raise ValueError("LLaVA-NeXT truncation removed the image placeholder span")
        expected = torch.arange(
            int(positions[0]), int(positions[-1]) + 1, device=positions.device,
        )
        if not torch.equal(positions, expected):
            raise ValueError(
                "LearnPruner/LLaVA-NeXT supports one contiguous image placeholder run"
            )
        result = {
            key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)
        }
        result["labels"] = _assistant_labels(
            input_ids[0], self.processor.tokenizer, int(image_token_id),
        ).unsqueeze(0)
        if "attention_mask" not in result:
            result["attention_mask"] = torch.ones_like(input_ids)
        if "pixel_values" not in result or "image_sizes" not in result:
            raise RuntimeError("LLaVA-NeXT processor omitted pixel_values or image_sizes")
        return result


def _checkpoint_name(backbone: VLM, options: Mapping[str, Any]) -> str:
    configured = options.get("checkpoint_name")
    if configured is not None:
        return str(configured)
    key = backbone_key(backbone)
    if key in (None, "llava"):
        return "learnpruner_lpm.pt"
    slug = BACKBONE_SLUGS.get(key, str(key))
    return f"learnpruner_lpm_{slug}.pt"


def _train(backbone: VLM, config: Config, *, continuation: bool) -> None:
    options = _options(config)
    key = backbone_key(backbone)
    if key not in {"llava", "llava-13b", "llava-next-8b"}:
        raise NotImplementedError(
            "LearnPruner training supports llava, llava-13b, and llava-next-8b; "
            f"got {key!r}"
        )
    llava_next = key == "llava-next-8b"
    output = Path(options.get("output_dir", Path(config.output_dir) / "learnpruner"))
    checkpoint = output / _checkpoint_name(backbone, options)

    seed = int(options.get("seed", 2026))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    compressor_options = config.custom_args.get("compressor_args", {})
    if not isinstance(compressor_options, Mapping):
        raise TypeError("compressor_args must be a mapping")
    compressor_options = dict(compressor_options)
    start = compressor_options.pop("checkpoint_path", None)
    if continuation and start is None:
        start = checkpoint
    if continuation and not Path(start).is_file():
        raise FileNotFoundError(f"LearnPruner ft needs a tr checkpoint: {start}")
    if not continuation and start is not None:
        raise ValueError("LearnPruner tr starts from a fresh LPM; use ft to continue")
    compressor = LearnPrunerCompressor(
        **compressor_options, checkpoint_path=start, for_training=True,
    )
    if llava_next:
        # Training dispatch initially validates by registry key; this second
        # pass checks the actual ratio-valued compressor configuration.
        backbone.validate_compressor(compressor)
    elif compressor.retain_ratio is not None:
        raise ValueError(
            "LearnPruner on LLaVA-1.5 7B/13B uses max_tokens; "
            "retain_ratio is only for LLaVA-NeXT"
        )

    annotations, image_folder = resolve_training_data(
        config.input_dir, "finetune", data_path=options.get("data_path"),
        image_folder=options.get("image_folder"),
    )
    source = LlavaJsonDataset(annotations, image_folder)
    limit = config.dataset_limit
    if options.get("dataset_limit") is not None:
        own_limit = int(options["dataset_limit"])
        limit = own_limit if limit is None else min(limit, own_limit)
    subset_frac = float(options.get("subset_frac", 0.10))
    selected = _selected_indices(source, subset_frac, seed, limit)
    dataset = Subset(source, selected)

    batch_size = int(options.get("per_device_train_batch_size", 1))
    accumulation = int(options.get("gradient_accumulation_steps", 8))
    epochs = int(options.get("num_epochs", 1))
    workers = int(options.get("num_workers", 0))
    if min(batch_size, accumulation, epochs) < 1 or workers < 0:
        raise ValueError(
            "Batch size, accumulation, and num_epochs must be positive; "
            "num_workers must be non-negative"
        )
    if llava_next and batch_size != 1:
        raise ValueError(
            "LearnPruner/LLaVA-NeXT training requires per_device_train_batch_size=1 "
            "because packed AnyRes spans are ragged"
        )
    processor = backbone.processor
    if llava_next:
        collator = LlavaNextLearnPrunerCollator(
            processor, model_max_length=int(options.get("model_max_length", 4096)),
        )
    else:
        collator = LlavaHiDropCollator(
            processor, model_max_length=int(options.get("model_max_length", 2048)),
            stage="finetune",
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=collator, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )

    model = backbone.model
    model_device = next(model.parameters()).device
    original_training = model.training
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    learning_rate = float(options.get("learning_rate", 1e-4))
    weight_decay = float(options.get("weight_decay", 0.0))
    max_norm = float(options.get("max_grad_norm", 1.0))
    if learning_rate <= 0 or weight_decay < 0 or max_norm <= 0:
        raise ValueError(
            "learning_rate and max_grad_norm must be positive; weight_decay must be non-negative"
        )
    use_autocast = (
        model_device.type == "cuda"
        and backbone._dtype in (torch.float16, torch.bfloat16)
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    settings_name = (
        "learnpruner_training.json"
        if key == "llava" else f"{checkpoint.stem}_training.json"
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    try:
        # NeXT materializes a 4096-wide post-projector LPM here, whereas both
        # fixed-grid backbones retain the paper's 1024-wide CLIP-feature LPM.
        # The optimizer must therefore be created only after installation.
        compressor.install(backbone)
        compressor.train()
        trainable = tuple(compressor.trainable_parameters)
        if not trainable:
            raise RuntimeError("LearnPruner produced no trainable LPM parameters")
        optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_autocast and backbone._dtype == torch.float16,
        )
        optimizer.zero_grad(set_to_none=True)
        # Mid-epoch checkpoint/resume, off unless save_every_steps is set so the
        # LLaVA-1.5 path is unchanged. A LLaVA-NeXT epoch (~62k steps at ~1.36
        # it/s on 4 sharded H100s) outlasts a 12h wall, and the LPM is otherwise
        # written only after the epoch loop, so a walled run would keep nothing.
        save_every = int(options.get("save_every_steps") or 0)
        stop_after = int(options.get("stop_after_steps") or 0)
        if save_every < 0 or stop_after < 0:
            raise ValueError("save_every_steps and stop_after_steps must be non-negative")
        if save_every and save_every % accumulation:
            raise ValueError(
                "save_every_steps must be a multiple of gradient_accumulation_steps, "
                "so every resume point falls on an optimizer step"
            )
        if stop_after and not save_every:
            raise ValueError("stop_after_steps needs save_every_steps, or the stopped run keeps nothing")
        resume_path = output / f"{checkpoint.stem}_resume.pt"
        start_epoch, start_step, stopped = 0, 0, False
        if save_every and resume_path.is_file():
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            if state.get("seed") != seed or state.get("examples") != len(selected):
                raise RuntimeError(
                    f"{resume_path} was written for a different data selection; "
                    "remove it to restart from scratch"
                )
            compressor.lpm.load_state_dict(state["lpm"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, start_step = int(state["epoch"]), int(state["step"])
            print(f"Resuming LearnPruner from {resume_path}: "
                  f"epoch {start_epoch + 1}, step {start_step}", flush=True)

        def save_resume(epoch_index: int, next_step: int) -> None:
            payload = {
                "epoch": epoch_index, "step": next_step,
                "seed": seed, "examples": len(selected),
                "lpm": compressor.lpm.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
            partial = resume_path.with_name(resume_path.name + ".tmp")
            torch.save(payload, partial)
            os.replace(partial, resume_path)  # atomic: a kill mid-save keeps the old one

        for epoch in range(start_epoch, epochs):
            first = start_step if epoch == start_epoch else 0
            if save_every:
                # An explicit per-epoch order, so a resumed segment skips the batches
                # it already trained on without decoding their images.
                order = torch.randperm(
                    len(dataset), generator=torch.Generator().manual_seed(seed + epoch),
                ).tolist()
                epoch_loader = DataLoader(
                    Subset(dataset, order[first * batch_size:]), batch_size=batch_size,
                    shuffle=False, num_workers=workers, collate_fn=collator,
                    pin_memory=torch.cuda.is_available(),
                )
                n_steps = math.ceil(len(dataset) / batch_size)
            else:
                epoch_loader, n_steps = loader, len(loader)
            progress = tqdm(epoch_loader, desc=f"LearnPruner LPM epoch {epoch + 1}/{epochs}",
                            initial=first, total=n_steps)
            for offset, batch in enumerate(progress):
                step = first + offset
                batch = {name: value.to(model_device) for name, value in batch.items()}
                batch["pixel_values"] = batch["pixel_values"].to(backbone._dtype)
                if not bool((batch["labels"] != IGNORE_INDEX).any()):
                    continue
                autocast = (
                    torch.autocast("cuda", dtype=backbone._dtype)
                    if use_autocast else nullcontext()
                )
                with autocast:
                    outputs = model(**batch, use_cache=False)
                    ntp = outputs.loss
                    pruning = compressor.pruning_loss()
                    # Under a sharded device_map the LM head output and the LPM masks
                    # live on different GPUs, and adding them raises. Same device in
                    # the single-GPU case, where .to() is a no-op.
                    total = ntp + pruning.to(ntp.device)  # paper Eq. 8: lambda=1
                    window = min(
                        accumulation,
                        n_steps - (step // accumulation) * accumulation,
                    )
                    loss = total / window
                scaler.scale(loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == n_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if save_every and step + 1 < n_steps and (step + 1) % save_every == 0:
                        save_resume(epoch, step + 1)
                        if stop_after and step + 1 >= stop_after:
                            stopped = True
                progress.set_postfix(
                    ntp=f"{float(ntp.detach()):.3f}",
                    prune=f"{float(pruning.detach()):.3f}",
                )
                if stopped:
                    break

            start_step = 0
            if stopped:
                break
            if save_every and epoch + 1 < epochs:
                save_resume(epoch + 1, 0)
            compressor.save_checkpoint(checkpoint)
            layers = model.model.language_model.layers
            settings = {
                "paper": "arXiv:2604.23950v1",
                "dataset": str(annotations),
                "image_folder": str(image_folder),
                "subset_frac": subset_frac,
                "seed": seed,
                "sampled_rows": max(1, round(len(source) * subset_frac)),
                "training_image_examples": len(selected),
                "epoch": epoch + 1,
                "optimizer": "AdamW",
                "learning_rate": learning_rate,
                "lpm": f"{compressor.feature_dim}->{compressor.hidden_dim}->2 GELU",
                "feature_space": compressor.feature_space,
                "backbone": key,
                "decoder_layers": len(layers),
                "pruning_boundary": compressor.resolved_pruning_layer,
                "budget_parameter": "retain_ratio" if llava_next else "max_tokens",
                "budget": compressor.budget,
                "loss": "L_ntp + L_prune; tau=0; lambda=1",
                "base_model": str(getattr(backbone, "_model_id", "unknown")),
            }
            (output / settings_name).write_text(
                json.dumps(settings, indent=2), encoding="utf-8",
            )
    finally:
        compressor.uninstall()
        for parameter, original in zip(
            model.parameters(), original_requires_grad, strict=True,
        ):
            parameter.requires_grad_(original)
        model.train(original_training)
    if stopped:
        print(f"Stopped LearnPruner after the requested steps; resume state at {resume_path}",
              flush=True)
        return
    if save_every and resume_path.is_file():
        resume_path.unlink()  # the final LPM supersedes it
    print(f"Saved LearnPruner LPM checkpoint at {checkpoint}", flush=True)

def train_learn_pruner(backbone: VLM, config: Config) -> None:
    _train(backbone, config, continuation=False)


def finetune_learn_pruner(backbone: VLM, config: Config) -> None:
    _train(backbone, config, continuation=True)
