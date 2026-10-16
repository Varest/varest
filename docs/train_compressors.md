# Train compressors

Most compressors here are training-free — install them and run `bm`. The
registered learned methods train through two subcommands:

| Compressor | `tr` (pretrain) | `ft` (finetune) | What gets trained |
| --- | --- | --- | --- |
| SoftTopk | yes | yes | the compressor's own parameters |
| VisionZip | — | yes | the projector (LLaVA) / patch merger (Qwen) |
| HiDrop | yes | yes | stage 1: adapter; stage 2: the whole model |
| DUET | yes | yes | projector/merger, then adapter + language model |
| CORE | yes | yes | CORE projector, then projector + LoRA |
| QuietPrune | yes | yes | LLaVA text-to-vision Q-CLS adapter |
| Varest (W5RELJOINT) | yes | yes | LoRA (vision tower + LLM + lm_head), projector, position tag, [REL] head |
| VarestW5 | yes | yes | LoRA (vision tower + LLM + lm_head), projector, position tag |

VisionZip has no pretraining stage, so it is registered under `ft` only.
QuietPrune's trainer is fixed-grid LLaVA; Varest trains on LLaVA-1.5 7B/13B and LLaVA-NeXT-8B. DUET and HiDrop
also provide experimental Qwen training adaptations; neither claims a released
Qwen recipe or accepts a LLaVA stage-1 artifact.

Both take the same global options as `bm` (see
[run_benchmarks.md](run_benchmarks.md)), with `training_args` alongside
`compressor_args` in the custom-args file:

```console
python -m llmoe --config erbium --backbone qwen-vl --compressor VisionZip \
  --input_dir /path/to/datasets --output_dir /path/to/results \
  --custom_args train.yaml ft
```

## Datasets

Under `--input_dir`:

| Directory | Contents | Used by |
| --- | --- | --- |
| `LLaVA-Pretrain` | `blip_laion_cc_sbu_558k.json` + `images/` | HiDrop/DUET/CORE `tr` |
| `LLaVA-Tuning` (or `LLaVA-Instruct`) | `llava_v1_5_mix665k.json` + its image tree | HiDrop/DUET `ft`, VisionZip `ft` |
| `LLaVA-NeXT-Data` | `data/*.parquet` with conversations and embedded/path-backed images | CORE `ft` on `core-llama3-8b` |

CORE's Llama-3 port intentionally uses the original 558K plain-caption corpus
for projector alignment, then the supplied LLaVA-NeXT instruction corpus for
projector + LoRA tuning. See [the implementation guide](core_llava_next_8b.md).

## VisionZip

The paper's "Efficient Tuning" (VisionZip-double-dagger) freezes the vision
encoder and LLM and adapts only the module that feeds visual tokens into the
decoder. On LLaVA that is `multi_modal_projector`. Qwen has no separate
projector: `visual.merger` both merges 2-by-2 patches and projects the result
into decoder width, so it is the sole trainable and saved module.

```yaml
backbone_args:
  model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct

compressor_args:
  retain_ratio: 0.5

training_args:
  subset_frac: 0.1
  per_device_train_batch_size: 4
  global_batch_size: 128
  learning_rate: 2.0e-5
```

The Qwen runner uses the released VisionZip ordering: compute full-sequence
M-RoPE, obtain merger outputs and last-ViT attention/key signals, reduce the
decoder-side visual span, and gather positions, masks, and labels through the
same keep set. Upstream limits Qwen to batch size 1 because NaViT image lengths
vary. This implementation reduces and repads each row independently, so
`per_device_train_batch_size` may be greater than one.

Also accepted on both paths: `num_epochs`, `weight_decay`,
`max_grad_norm`, `warmup_ratio`, `num_workers`, `max_length`,
`seed`, `max_samples`, `max_steps`, and `checkpoint_name`.
The Qwen path additionally accepts `global_batch_size`,
`per_device_train_batch_size`, `deepspeed_stage`, `bf16`/`fp16`,
and `attn_implementation`; `batch_size` remains an alias for the
per-device value. DeepSpeed is disabled by default because only the small
merger owns optimizer state; set `deepspeed_stage: 2` or `3` when
that runtime is installed and sharding is useful.

The final artifact lands in `<output_dir>/vision_zip/` as
`vision_zip_projector_k<budget>.pt` or
`vision_zip_merger_r<ratio>.pt`. The Qwen file contains only
`model.visual.merger` weights plus metadata, never the ViT or LLM. Pass it
back as `compressor_args.checkpoint_path` to benchmark or to initialize
another tuning run; loading rejects a checkpoint tuned at a different ratio.

The official Qwen release provides the compression-aware model forward and
points users to LLaMA-Factory, but does not publish a Qwen-specific Efficient
Tuning freeze/optimizer recipe. The local parameter scope is therefore the
projector-equivalent adaptation of the paper's LLaVA recipe, while its token
selection and positional geometry follow the released Qwen implementation.

## HiDrop

HiDrop is trained *per schedule*: the pruning schedule is applied during
training, so a checkpoint is only valid at the budget it was trained for. The two
stages chain — `ft` starts from `tr`'s adapter for the same budget.

```console
python -m llmoe ... --compressor HiDrop --custom_args train.yaml tr   # stage 1
python -m llmoe ... --compressor HiDrop --custom_args train.yaml ft   # stage 2
```

```yaml
compressor_args:
  retain_ratio: 0.25              # `average_retained_tokens: 64` on LLaVA

training_args:
  per_device_train_batch_size: 1  # raise to fit; global batch is fixed
```

Stage 1 writes `<output_dir>/hidrop/<model>-hidrop-<budget>-flash_attention_2-pretrain/`
containing `mm_projector.bin` (LLaVA) or `mm_merger.bin` (Qwen). Stage 2 writes
the full model to the same path without `-pretrain`; benchmark either with
`compressor_args.checkpoint_path` pointing at the directory.

The released recipe is pinned and deviations are rejected rather than accepted
silently: global batch 256 (`tr`) / 128 (`ft`), DeepSpeed ZeRO‑2 / ZeRO‑3,
`model_max_length` 2048, FlashAttention‑2, and `adamw_torch`. Tune only the
micro-batch and gradient accumulation. Training therefore needs `deepspeed` and
`flash-attn` installed, on the pinned runtime (torch 2.11.0, transformers 5.5.4,
tokenizers 0.22.2, accelerate 1.13.0); the environment is checked before any
weights load.

Two backbone differences are worth knowing:

- LLaVA's stage 1 trains a **randomly initialised** projector, as upstream does.
  Qwen's merger ships already aligned as part of the instruct weights, so its
  stage 1 **warm-starts** from it instead; pass `reset_stage1: true` to force a
  re-initialisation.
- Upstream publishes LLaVA-1.5 checkpoints and no Qwen recipe. The Qwen path is
  our adaptation of the same recipe, and both the run manifest and an
  install-time warning record that its numbers are not the paper's.

Every artifact carries a `run_manifest.json` fingerprinting the base model,
dataset, schedule, batching and software versions. Reuse fails closed: `ft`
refuses an adapter from another schedule, and `bm` refuses a checkpoint that
never finished stage 2. Pass `compressor_args.validate_provenance: false` for
hand-assembled checkpoints, such as the released LLaVA ones.

## DUET

DUET uses the same two-stage dataset and optimizer schedule as HiDrop:

```console
python -m llmoe ... --backbone qwen-vl --compressor DUET \
  --custom_args duet_qwen.yaml tr
python -m llmoe ... --backbone qwen-vl --compressor DUET \
  --custom_args duet_qwen.yaml ft
```

```yaml
backbone_args:
  model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct

compressor_args:
  retain_ratio: 0.4813            # layer-averaged, = upstream's 0.65 + 0.05 Stage-1 split
  contextual_ratio: 0.05
  # Defaults to Qwen's published inference schedule:
  # pdrop_layers: [14, 21]
  # pdrop_keep_ratios: [0.5, 0.25]
  # `retain_ratio` is the budget averaged over the decoder's layers, so Stage 1
  # starts wider: this schedule averages 0.6875, giving Stage 1 0.4813 / 0.6875
  # = 0.7. Values above 0.6875 are unreachable and raise at install().

training_args:
  per_device_train_batch_size: 4
  # per-device batch x world size x accumulation must be 256 for tr, 128 for ft
```

Qwen training is not restricted to micro-batch 1. The collator packs each
image's dynamic-resolution patches, and Stage 1 plus every PyramidDrop boundary
prunes each row to its own count before right-padding the batch again. Labels
and original M-RoPE grid coordinates travel through the same row-wise gathers.
Text-only instruction rows receive HiDrop's zero-weight dummy image so all
distributed ranks retain the vision path in their autograd graph.

Stage 1 warm-starts and trains Qwen's pretrained `visual.merger`, writing
`mm_merger.bin` under
`<output_dir>/duet/qwen2.5-vl-7b-duet-<schedule>-pretrain/`. Stage 2 loads that
merger, freezes the rest of the ViT, and trains the merger plus language model.
It writes a stock Hugging Face Qwen checkpoint to the same path without
`-pretrain`.

The official DUET repository publishes Qwen inference and evaluation code but
no Qwen compression-aware training recipe or trained checkpoint. These runs are
therefore adaptations: their `duet_config.json` records
`released_training_recipe: false`, and the launcher emits a warning before
weights load.

## Varest

`Varest` (W5RELJOINT) and `VarestW5` fine-tune LLaVA-1.5 7B/13B or LLaVA-NeXT-8B to read W5 visual tokens: one pass
over the backbone's own instruction-tuning set — the image rows of `LLaVA-Tuning/llava_v1_5_mix665k.json` for
LLaVA-1.5 (624,609 steps), `LLaVA-NeXT-Data/llava_next_raw_format/` for LLaVA-NeXT-8B (738,590 steps) — batch 1, LR 2e-5, a fresh budget
K ~ U[32, 576] every step, LoRA r=32 on the vision tower, the language model and `lm_head`, the projector, and the
position tag; W5RELJOINT also trains the [REL] head from scratch (LR 1e-3). The recipe and every head/label/cache step
behind it live in the `varest` package — see [varest/README.md](../varest/README.md).

The trainer reads precomputed field caches (text field, boundary field and k_min per image), written once by
`python -m varest.train.dump_fields --family llava|llava_next ...`. Point `training_args.fields` at them:

```yaml
training_args:
  fields: /path/to/fields/llava_fields_*of20.pkl     # llava_next_fields_* on llava-next-8b
  wall_seconds: 41400                                # optional: stop and save before a 12 h walltime
  package: [text_head.pt, object_head.pt, varest_w5reljoint.pt]   # optional: SWA5 + release file at the end
```

```console
python -m llmoe --backbone llava --compressor Varest --input_dir /path/to/datasets \
  --output_dir /path/to/results --custom_args train.yaml tr
```

Checkpoints land in `<output_dir>/varest/varest_<variant>_<backbone>[_step<N>].pt`; rerunning (`tr` or `ft`) resumes
from the newest snapshot. The released weights are the SWA of the last four 32k snapshots and the final step
(512k–608k + 624,609 on LLaVA-1.5, 640k–736k + 738,590 on LLaVA-NeXT).

