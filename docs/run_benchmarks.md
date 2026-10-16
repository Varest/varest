# Run benchmarks

LLMOE can evaluate a full-token model or a registered compressor on one or more
benchmarks. The same model is reused when several benchmarks are listed.

## Quick start

See the available choices:

```console
python -c "from llmoe import BACKBONES; print(*BACKBONES, sep='\n')"
python -c "from llmoe import COMPRESSORS; print(*COMPRESSORS, sep='\n')"
python -c "from llmoe import BENCHMARKS; print(*BENCHMARKS, sep='\n')"
```

Run two benchmarks with a compressor:

```console
python -m llmoe \
  --config erbium \
  --backbone llava \
  --compressor SoftTopk \
  --input_dir /path/to/datasets \
  --output_dir /path/to/results \
  --custom_args benchmark.yaml \
  bm textvqa gqa
```

Use the named LLaVA-1.5-13B backbone with the same baseline interface:

```console
python -m llmoe \
  --config erbium \
  --backbone llava-13b \
  --compressor SoftTopk \
  --input_dir /path/to/datasets \
  --output_dir /path/to/results \
  --custom_args benchmark.yaml \
  bm textvqa gqa
```

`llava-13b` defaults to `llava-hf/llava-1.5-13b-hf`. All registered baselines
that support LLaVA use the same compressor path with either `llava` or
`llava-13b`; select them through `--compressor` as usual. `CORE` is the
exception: it is a complete VLM with its own `core` (Vicuna) or
`core-llama3-8b` backbone, not a compressor for native LLaVA/LLaVA-NeXT vision
towers.

### Budgets on the 13B decoder

Vicuna-13B has 40 decoder layers where the 7B has 32, and every decoder-stage
baseline here states its budget as a **layer average** — `sum(visual tokens held
by each decoder layer) / depth`. A schedule published for 32 layers therefore
spends a different budget on 40 unless it is re-solved, so each baseline resolves
its schedule against the loaded decoder's own depth at `install()`:

* boundaries move to the same *fraction* of the decoder — FastV's cut at layer 3
  of 32 becomes layer 4 of 40, ApET's and Nuwa's mid-decoder cut at 16 becomes
  20, VFlowOpt's even thirds become 13/13/14;
* the per-stage keep counts are then re-solved so the layer average returns to
  the budget label.

At 32 layers the re-solve is the identity, so no LLaVA-1.5-7B number changes.
Each baseline prints the schedule it resolved at install. A schedule you pin
yourself (`layer_list`, `pruning_layers`, `k`, …) is always taken literally.

Released checkpoints are a different matter and do **not** transfer: every
learned projector or adapter here is sized to the decoder's hidden width (4096
on 7B, 5120 on 13B), and HiDrop's and DUET's released artifacts are whole
finetuned 7B models that would replace the backbone outright. No 13B artifacts
are published, so `pretrained: true` (and the trained-only default for HiDrop and
QuietPrune) fails on `--backbone llava-13b`, naming what does exist, rather than
silently evaluating the 7B. Run those baselines training-free with
`compressor_args: {pretrained: false}`, or train them for this backbone.

GlimpsePrune is the one baseline with no depth-independent schedule to re-solve:
its layers are tied to per-layer trained parameters, so it needs a config and
checkpoint built for 40 layers.

### Training for the 13B backbone

`tr` and `ft` take `--backbone llava-13b` like `bm` does. The trainers read the
decoder depth from the checkpoint's config before any weights load, so the
schedule they train under is the same one `install()` resolves at evaluation
time — training and evaluation cannot drift apart.

Checkpoints are written under a tag naming the model they were trained on, so a
13B run cannot overwrite or be mistaken for a 7B one:

| baseline | 7B (unchanged) | 13B |
| --- | --- | --- |
| HiDrop | `hidrop/llava-v1.5-7b-hidrop-64-…` | `hidrop/llava-1.5-13b-hidrop-64-…` |
| DUET | `duet/llava-v1.5-7b-duet-64` | `duet/llava-1.5-13b-duet-64` |
| VisionZip | `vision_zip/vision_zip_projector_k64.pt` | `vision_zip/vision_zip_projector_llava-1.5-13b_k64.pt` |

Point `compressor_args.checkpoint_path` at the resulting directory to benchmark
it. A HiDrop checkpoint records the depth it was trained at and refuses to be
re-solved for another one, so a 13B checkpoint cannot be run on the 7B or vice
versa.

All global options must appear before `bm`. To evaluate the uncompressed model,
omit `--compressor` or pass `--compressor none`.

For DUET on the variable-length `llava-next-8b` backbone, use batch size one and
one of `configs/duet/llava_next_8b_benchmark_{640,320,160}.yaml`. Its AnyRes
layout, budget accounting, checkpoint validation, and unsupported modes are
documented in [DUET on LLaVA-NeXT](duet_llava_next.md).

Configuration backends are `erbium` (the default), `slurm`, and `local`. SLURM
also requires `--suser`. Useful overrides include:

- `--experiment_name`: name stored with the run;
- `--input_dir`: parent directory containing benchmark datasets;
- `--output_dir`: destination for configs and results;
- `--dataset-limit`: maximum samples per benchmark;
- `--device`: device override such as `cuda:0` or `cpu`.

If `--output_dir` is omitted, LLMOE uses the backend's output directory with
`LLMOE` appended. An explicit `--output_dir` is used as-is.

## Custom arguments

`--custom_args` accepts a `.json` or `.yaml` file. Its contents are stored in
`config.custom_args` and routed to the relevant component.

```yaml
backbone_args:
  model_name_or_path: llava-hf/llava-1.5-7b-hf
  local_files_only: true

compressor_args:
  max_tokens: 64
  checkpoint_path: /path/to/soft_topk.pt

batch_size: 4
max_new_tokens: 128
do_sample: false
num_beams: 1
use_cache: true
temperature: 0.0
top_p: null
dataset_split: test

# Profiling is enabled by default.
performance_profile:
  enabled: true
```

Common keys are:

| Key | Used for |
| --- | --- |
| `backbone_args` | Keyword arguments passed to the backbone constructor. |
| `compressor_args` | Keyword arguments passed to the compressor constructor. |
| `batch_size` | Samples generated together; defaults to 16. |
| `max_new_tokens` | Generation limit; defaults to 16 for CountBench/VStar, 1024 for MMBench, and 128 otherwise. |
| `do_sample` | Enables sampling. When false, `temperature` and `top_p` are not passed to generation. |
| `num_beams` | Beam count; defaults to 1. |
| `use_cache` | Enables the generation KV cache; defaults to true. |
| `temperature`, `top_p` | Sampling controls. |
| `dataset_split` | Dataset split for benchmarks that support it. Defaults vary by benchmark. |
| `performance_profile` | `false` disables profiling; a mapping may use `enabled: false`, `flops: false` (skip FLOP counting) and `schedule` (the traced-batch window). |
| `saliency_args` | Controls `--visualize-saliency`; see `llmoe/viz.py` for its knobs. |

Some benchmarks accept additional keys, such as `refcoco_datasets`,
`refcoco_splits`, `refcoco_padding_off`, `vstar_category`, and the
`varest_bench_*` keys described [below](#runtime-and-flops-varest_bench).

CLI overrides such as `--dataset-limit` and `--device` are applied after the
custom-args file is loaded. For generation settings, a concrete field supplied
by the selected backend config takes precedence; otherwise the runner uses the
matching custom argument and then its default.

## LLaVA-NeXT 8B backbone

The full-token `llava-hf/llama3-llava-next-8b-hf` checkpoint is registered as
`--backbone llava-next-8b`. It uses the checkpoint's Llama 3 chat template and
native AnyRes image processor.

The canonical training-free baselines are `DivPrune`, `OTPrune`, `FastV`,
`PyramidDrop`, `VisionZip`, `Nuwa`, `ApET`, `SparseVLM`, `ToMe`, `MustDrop`,
and `VFlowOpt`. Each accepts exactly one NeXT compression argument:
`compressor_args.retain_ratio`.

For all eleven, `retain_ratio` is the fraction of the original visual span seen
on average across decoder layers. Static selectors, `VisionZip`, and `ToMe`
emit that exact per-image fraction before the decoder. The progressive methods
derive their fixed method-specific schedules from the layer-average target.

```yaml
compressor_args:
  retain_ratio: 0.25

batch_size: 1
num_beams: 1
use_cache: true
```

```console
python -m llmoe \
  --config erbium \
  --backbone llava-next-8b \
  --compressor DivPrune \
  --custom_args /path/to/llava_next.yaml \
  --input_dir /path/to/datasets \
  --output_dir /path/to/results \
  bm textvqa
```

Change only `--compressor` among the eleven names to compare the supported
baselines under the same ratio.

`FastV`, `PyramidDrop`, `Nuwa`, `ApET`, `SparseVLM`, `MustDrop`, and
`VFlowOpt` physically shorten the decoder sequence, so their canonical path
requires `batch_size: 1`, `num_beams: 1`, and exactly one contiguous image
span. Nuwa uses `retain_ratio <= 0.55`; ApET uses `retain_ratio <= 2/3`; and
SparseVLM reports the image-specific reachable range for its fixed v1.5
profile. MustDrop fixes its 3x3/440-of-576/8% policy and four decoder cuts;
VFlowOpt fixes the released OneVision 25% importance/recycling profile and two
decoder cuts. All accept `1.0` for identity. VisionZip, ToMe, ApET, MustDrop,
and VFlowOpt run their vision policies before the projector; NeXT-specific tuned
projector weights are not accepted. ToMe captures the selected CLIP layer's
head-averaged keys and applies its balanced bipartite, size-weighted merge
independently to the base crop and valid local tile regions. Merged centroids
restore raster positions and newline rows remain protected. This preserves the
released matching rule but adapts its placement from progressive in-ViT merging
to NeXT's stock AnyRes packing boundary. Other compressors are rejected until
their NeXT-specific adapter is implemented.

See [LLaVA-NeXT 8B baseline support](llava_next_8b.md) for the architectural
boundary, implementation status, and remaining work.

## Varest (W5 / W5RELJOINT)

`Varest` is W5RELJOINT (the W5 variable-resolution tokenizer plus [REL] pruning inside the LLM); `VarestW5` is the
tokenizer alone. Both are trained methods: by default `install` fetches the released checkpoint for the loaded backbone
from `methods/varest/<backbone>/{w5reljoint,w5}/` (pass `checkpoint_path` for your own). On LLaVA-1.5 7B/13B the
budget is `max_tokens` (the layer-averaged count; standard 32/64/128/224). On LLaVA-NeXT it is `retain_ratio`
(standard 0.0556/0.1111/0.2222/0.3889, i.e. b/576), applied per image to NeXT's native AnyRes length.

```yaml
compressor_args:
  max_tokens: 64            # retain_ratio: 0.1111 on --backbone llava-next-8b
```

Varest builds its visual tokens from the image and decodes the prompt itself, one row at a time and greedily, exactly
as the checkpoints were trained; sampling or beam search is refused. The released checkpoints were trained and
evaluated in bfloat16, while the harness loads LLaVA in float16; on full TextVQA (7B W5RELJOINT, B=128) that changes
3% of answers and the score by -0.11 (see `varest/README.md`).
W5RELJOINT's pruned decoding runs the decoder layers directly, so generation-latency profiling does not see it.

## New baseline support on Qwen2.5-VL

Qwen's NaViT encoder emits a different number of post-merger visual tokens for
different images. The newly merged baselines therefore use `retain_ratio`, not
a fixed `max_tokens`, and resolve counts from the image's concatenated visual
span. Static reducers report a final ratio; progressive decoder methods report
the ratio averaged over decoder layers.

| Compressor | Qwen budget | Qwen implementation |
| --- | --- | --- |
| `DivPrune` | final `retain_ratio` | Released max-min diversity selector after concatenation. |
| `PyramidDrop` | layer-average `retain_ratio` | Last-prompt-token boundary rank with a depth-derived geometric schedule and M-RoPE-aware physical cuts. |
| `QuietPrune` | final `retain_ratio` | Spatial group pruning plus redundant-token aggregation, ranked by VisionZip-style final-ViT received attention. This is not the released Q-CLS path, and LLaVA adapter checkpoints are rejected. |
| `VFlowOpt` | layer-average `retain_ratio` | VisionZip-style final-ViT importance, rectangular cell recycling, and two M-RoPE-aware decoder cuts. The fixed-grid LLaVA calibration and entropy term are not reused. |
| `CORE` | unavailable | CORE is a complete ConvNeXt-L/Mask2Former-backed VLM (`--backbone core`), not a compressor that can be installed on Qwen. |

Example:

```yaml
backbone_args:
  model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct

compressor_args:
  retain_ratio: 0.25

batch_size: 1
num_beams: 1
use_cache: true
```

These adapters currently support one still image in one contiguous visual span,
batch size 1, cached greedy or sampling generation, and no beam expansion. They
preserve Qwen's original M-RoPE grid coordinates when physically gathering
tokens. The runtime fails closed on video, multiple image spans, incompatible
LLaVA checkpoints, or an unsupported VAREST ablation.

## Training-free baseline support on InternVL

InternVL dynamically turns one source image into a batch of tiles. Every baseline
in this section first restores those source-image boundaries and concatenates the
tile features in processor order. Its selector, merger, or decoder schedule then
runs once over that complete source-image span; `max_tokens: 64` never means 64
tokens independently for every tile.

The shared architecture plumbing stays in `llmoe/compressor/internvl.py`. Each
baseline's method-specific policy and lifecycle live beside its normal and Qwen
implementations as `<method>/<method>_internvl.py` (for example,
`nuwa/nuwa_internvl.py` and `vision_zip/vision_zip_internvl.py`).

| Compressor | InternVL budget | InternVL implementation |
| --- | --- | --- |
| `VisionZip` | final `max_tokens` | Training-free CLS-attention/key selection after InternVL's pixel shuffle. Projector checkpoints are rejected. |
| `ToMe` | final `max_tokens` | Post-projector token merging over the concatenated source span. |
| `DivPrune` | final `max_tokens` or `subset_ratio` | Released max-min diversity selection over the concatenated source span. |
| `OTPrune` | final `max_tokens` | Released DPP selector over the concatenated source span. |
| `ApET` | layer-average `budget` | Stage A runs over the concatenated source span; the decoder cut stays at the loaded model's midpoint. |
| `FastV` | layer-average `max_tokens` | One decoder attention ranking and one global keep, back-solved for the source's tile count and decoder depth. |
| `SparseVLM` | published `max_tokens` profile | The published count schedule and recycling stages apply once to the complete visual span. |
| `PyramidDrop` | layer-average `max_tokens` | The Qwen branch's depth-aware schedule solver is applied to the complete visual span using InternVL's Qwen3 decoder geometry. |
| `MustDrop` | published `max_tokens` profile | Windows remain local to a tile, but redundant windows are ranked and merged under one source-level budget; decoder filtering sees one span. |
| `Nuwa` | layer-average `max_tokens` | Exact tiled grouping from the upstream LLaVA-NeXT path: thumbnail separately, all local tiles as one stitched mosaic, then concatenate both outputs. |
| `VFlowOpt` | layer-average `max_tokens` | One global ViT importance/recycling stage followed by two source-level decoder cuts. |

Nuwa's row is intentionally more specific than the common rule. Upstream
[`CLIPVisionTower_Nuwa_Next.forward`](https://github.com/Man-PaperRejected/Nuwa/blob/175b7d57a7890b9818dd26d82f2d91d9868b0d80/nuwa/clip_encoder.py#L431-L494)
assigns one tile's share to the global thumbnail, stitches all local tiles before
calling `prune_vision_tokens` once with the remaining proportional budget, and
finally concatenates the two outputs. The InternVL adapter preserves that
two-group recipe; it does not invent a single thumbnail-plus-local call.

The other methods have no released InternVL recipe. Their ports retain each
method's selection policy and reuse the Qwen implementation's dynamic-span,
Qwen3 attention, physical sequence-cut, and cache-handling infrastructure where
applicable. This is an architecture adaptation, not an InternVL-specific set of
claimed paper hyperparameters.

Example:

```yaml
backbone_args:
  model_name_or_path: OpenGVLab/InternVL3-8B-hf

compressor_args:
  max_tokens: 64

batch_size: 1
num_beams: 1
use_cache: true
```

Use `budget` instead of `max_tokens` for ApET. Progressive decoder methods
currently require one source image, batch size 1, cached generation, and no beam
expansion. Static feature reducers support one independently budgeted visual run
per source and preserve left-padded batches.

This training-free scope excludes finetuned VisionZip, CORE, QuietPrune, DUET,
and HiDrop as requested. It also does not port SoftTopk or GlimpsePrune, which
depend on trained components, or VAREST, which is this repository's proposed
method rather than an external baseline.

## What the pipeline does

Before inference, LLMOE:

1. builds the selected Erbium, SLURM, or local config;
2. applies CLI overrides and loads custom arguments;
3. writes the resolved config to `<output_dir>/config.yaml`;
4. checks that every requested dataset directory exists;
5. loads one backbone and, when selected, installs one compressor.

The config snapshot is YAML generated from the resolved config, not a verbatim
copy of the custom-args file. It includes `custom_args`, device, paths, dataset
limit, and visualization flags. It is written before dataset validation, so it
may exist even if the run stops during its environment checks. Unlike result
files, `config.yaml` is not timestamped and is replaced when the same output
directory is reused.

For each benchmark, the runner makes an in-memory config copy, sets the
benchmark name and required dataset, then:

1. validates benchmark-specific settings;
2. loads rows and groups them into batches;
3. formats images and prompts and calls model generation;
4. parses each raw response into a benchmark prediction;
5. records per-sample correctness where available;
6. computes benchmark metrics and timing;
7. writes the result artifacts.

Multiple benchmarks run sequentially with the same loaded model and compressor.
After the last benchmark, compressor hooks are removed when `uninstall()` is
available, the model is released, and the CUDA cache is cleared.

## Output files

Normal benchmark results use this layout:

```text
<output_dir>/
├── config.yaml
├── textvqa/
│   └── SoftTopk-b64/
│       ├── textvqa_SoftTopk-b64_2026-07-31_12-30-00.json
│       └── textvqa_SoftTopk-b64_2026-07-31_12-30-00_profile_summary.txt
└── gqa/
    └── SoftTopk-b64/
        ├── gqa_SoftTopk-b64_2026-07-31_12-45-00.json
        └── gqa_SoftTopk-b64_2026-07-31_12-45-00_profile_summary.txt
```

The method directory is `compressor.name`. Without a compressor it is
`full_token`. Timestamps prevent normal runs from overwriting earlier results.

### The method directory

`compressor.name` normally uses the format defined by
[`TokenCompressor`](../llmoe/compressor/prototype.py):

```text
<Method>-b<budget>
```

`<Method>` is the class name without the `Compressor` suffix, and `<budget>` is
the `budget` property — `max_tokens` unless the compressor overrides it. So a
32/64/128/224 sweep gives `MustDrop-b32 … MustDrop-b224`, directly comparable
against `SparseVLM-b32 … SparseVLM-b224` by path alone. The punctuation-preserving
trained alias is the one exception: `VisionZipPlusPlusCompressor` reports
`VisionZip++-b<budget>`.

Two consequences worth knowing:

- **Ratio-driven methods report the ratio**, since they have no token count to
  report: `GlimpsePrune-b0.25`, `DivPrune-b0.098`, `FastV-b0.25`.
- **The label carries the budget and nothing else.** Ablations of one method at
  one budget — SparseVLM without recycling, ToMe without proportional
  attention, DUET stage-1 only — all share a directory and are separated only by
  timestamp. Every knob is still recorded in each result's `config.custom_args`,
  so read that to tell such runs apart.

Each result is UTF-8, indented JSON with four main sections:

```json
{
  "textvqa": {"metrics": {}},
  "overall": {
    "total_examples": 100,
    "total_inference_time_seconds": 12.34,
    "avg_inference_time_per_example_seconds": 0.1234,
    "evaluation_date": "2026-07-31T12:30:00",
    "output_path": "/path/to/result.json"
  },
  "config": {
    "benchmark": "textvqa",
    "backbone": "llava",
    "compressor": "SoftTopk",
    "custom_args": {}
  },
  "predictions": []
}
```

The benchmark section contains its metrics. `config` records resolved generation
settings and custom arguments. Each prediction record contains identifiers,
question and answer data, the raw model response, parsed prediction, and a
`correct` value when the benchmark supports per-row scoring. Numeric values are
rounded to four decimal places.

Performance profiling is on by default. The adjacent
`*_profile_summary.txt` contains generation timing, prefill/decode timing,
measured FLOPs, profiler metadata, and an operator table. Its structured
summary is also embedded in `overall.performance_profile`. Set
`performance_profile: false` to skip it; basic total-inference timing remains in
the JSON.

Generation timing covers the whole run. The operator table and
`compute.flops` cover only the batches `torch.profiler` traces (by default the
3rd to 5th), because tracing and FLOP counting both slow the batches they
watch. `compute.flops` reports measured FLOPs per example, split into prefill,
decode, work outside `generate`, and per top-level module. For clean
per-query latency and FLOPs, use `varest_bench`.

## Runtime and FLOPs (`varest_bench`)

`varest_bench` measures efficiency rather than accuracy. It has 180 short-answer
VQA cases from DocVQA, ChartQA, TextVQA, GQA and AI2D in two sweeps:
`input_text_tokens` (prompt length 57–74 LLaVA-1.5 tokens at a fixed answer
length) and `reference_answer_tokens` (gold answer length 1–21 tokens at a fixed
prompt length). The benchmark folder goes under the input directory as
`Varest_Benchmark/`, holding `varest_benchmark.json` and `images/`.

```console
python -m llmoe --backbone llava --compressor VisionZip \
  --custom_args visionzip64.yaml --input_dir /workspace/input \
  --output_dir /path/to/results bm varest_bench
```

Preprocessing matches the other benchmarks. Images load through
`coerce_image_tensor`, and the backbone's processor and prompt template are used
unchanged. Each case's question gets the backbone's short-answer suffix. Through
LLaVA-1.5's template this reproduces every case's `formatted_text_tokens`, which
is why TextVQA cases carry no OCR tokens and AI2D cases keep their inline
options instead of lettered ones. Doing otherwise would move each case off its
level. Responses are recorded in `predictions` but not scored; the result
holds profiler metrics only.

How it measures:

- Every case runs alone at batch size 1, whatever `batch_size` says, and the
  generic profiler is off.
- The first case runs `varest_bench_warmup` times (default 3) untimed.
- Each case is timed `varest_bench_repeats` times (default 3). Only CUDA events
  around each forward pass are active, and CUDA is synchronized around each call.
  The run with the median end-to-end latency supplies the breakdown. A forward pass
  is one call of the model's `forward`, or, for decoders driven layer by layer
  without it (Varest's [REL] loop), one run from the first decoder block to the
  output head.
- A separate pass counts FLOPs with `llmoe.flops.FlopCounter` (disable with
  `varest_bench_flops: false`). It counts every matmul, convolution and
  attention kernel that actually runs, at its real shape: compressor hooks, the
  vision tower, the prefill, and each decode step at its real KV length. It uses
  2 FLOPs per MAC, ignores elementwise work, and prices attention at the full
  (non-causal) score matrix. Matmul- or attention-like ops that no formula covers
  are listed in `measurement.uncounted_ops`, so they are never silently dropped.
- The same pass records the theoretical TFLOPs that token-pruning papers report,
  described below.
- On CUDA, a last untimed pass samples allocated GPU memory
  (`torch.cuda.memory_allocated`) from a background thread every
  `varest_bench_memory_interval_ms` (default 1) and reports its peak and
  time-weighted mean, also relative to what was allocated before the pass (the
  weights and anything the model keeps between queries). Disable it with
  `varest_bench_memory: false`. It runs separately so the polling thread cannot
  slow the timed runs. The counter moves when tensors are created or freed on the
  host, so the mean follows the host-side allocation timeline.

### Theoretical TFLOPs

`theoretical_tflops` uses the per-layer formula of HiDrop
([arXiv:2602.23699](https://arxiv.org/abs/2602.23699)), summed over decoder layers:

```text
FLOPs = sum_l (4 n_l d^2 + 2 n_l^2 d + 3 n_l d m)
```

`n_l` is the number of vision tokens entering layer `l` during prefill, `d` the
hidden size and `m` the FFN intermediate size. `n_l` is measured with hooks on
the decoder layers, so it reflects pruning before and inside the LLM. It is
available for the HF LLaVA backbones and CORE. A backbone whose compressor feeds the
decoder its own embeddings, so no image token reaches it, reports `n_l` through
`VLM.vision_tokens_per_layer()`; Varest does (all `K` visual tokens before its
[REL] layer, `K'` from it on, none after an adaptive halt; the [REL] registers are
not counted).

Unlike the measured `flops`, it covers only the vision tokens in the LLM prefill
and counts one FLOP per multiply-accumulate.

Each prediction record carries an `efficiency` block:

| Field | Meaning |
| --- | --- |
| `latency_seconds`, `latency_runs_seconds` | End-to-end backbone call (preprocessing, generation, decoding to text): the median run and all runs. |
| `prefill_seconds`, `decode_seconds`, `decode_ms_per_token` | First forward, the remaining forwards, and their mean per decode step. |
| `generate_seconds`, `outside_generate_seconds` | `model.generate` wall time and the rest of the call. |
| `generated_tokens` | Tokens generated (forward passes). |
| `peak_memory_mib` | Peak allocated CUDA memory in the median timed run, weights included. |
| `memory_mean_mib`, `memory_peak_mib` | Time-weighted mean and peak allocated CUDA memory over the memory pass, weights included. |
| `memory_baseline_mib` | Allocated CUDA memory before the memory pass (weights and persistent state). |
| `memory_mean_above_baseline_mib`, `memory_peak_above_baseline_mib` | The mean and peak minus the baseline: what the query itself allocates. |
| `memory_samples`, `memory_pass_matches_output` | Samples taken, and whether the memory pass generated the same text as the timed runs. |
| `theoretical_tflops` | Theoretical prefill TFLOPs (see above); `null` when vision tokens cannot be traced. |
| `vision_tokens_per_layer` | Vision tokens each decoder layer took in during prefill, the `n_l` in the formula. |
| `flops`, `prefill_flops`, `decode_flops`, `outside_generate_flops` | Measured FLOPs. The last is work outside `generate`, such as a `prepare_inputs` hook or CORE's vision stack. |
| `flops_by_module` | FLOPs per top-level module (`model.vision_tower`, `model.multi_modal_projector`, `model.language_model`, `lm_head`). |
| `flop_pass_matches_output` | Whether the FLOP pass generated the same text as the timed runs. |

The `varest_bench` section of the result JSON aggregates these fields. `metrics`
holds overall means, medians and the p90 latency, including
`theoretical_tflops_mean`, `peak_memory_mib_max` / `peak_memory_mib_mean` and the
means of the memory-pass fields (`memory_mean_mib_mean`, `memory_peak_mib_mean`,
`memory_mean_above_baseline_mib_mean`, `memory_peak_above_baseline_mib_mean`). `per_level` breaks them down
by sweep and level, `per_dataset` by source dataset, and
`tflops_by_module_mean` gives the per-module means. `measurement` records the
settings, device and versions.

Timing is only meaningful on an otherwise idle GPU. Check `nvidia-smi` first.

## Visualization modes

These modes replace normal evaluation, so they do not write benchmark JSON or
profile summaries:

```console
# Compression snapshots; requires a compressor.
python -m llmoe --compressor SoftTopk --dataset-limit 2 \
  bm --visualize textvqa

# Per-word attention saliency; works with or without a compressor.
python -m llmoe --compressor none --dataset-limit 2 \
  bm --visualize-saliency textvqa
```

They write PNGs below `<output_dir>/<compressor-name>/image_<index>_<id>/`
(for example `<output_dir>/SoftTopk-b64/image_0_.../`) — note there is no
benchmark level here, unlike the result layout above. Full-token saliency uses
`<output_dir>/full_token/`. Visualization currently
targets the LLaVA/CLIP path and processes at most four samples when no dataset
limit is given.

## Common issues

- Dataset folders must use the names expected by the registry, such as
  `TextVQA`'s `textvqa` and GQA's `GQA`, under `input_dir`.
- Paired benchmarks require an even `--dataset-limit`.
- Reduce `batch_size` if generation runs out of GPU memory.
- Use `.yaml`, not `.yml`, for custom arguments.
- With the default `local_files_only: true`, the selected model must already be
  available in the Hugging Face cache.
