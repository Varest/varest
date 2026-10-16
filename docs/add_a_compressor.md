# Add a compressor

A compressor takes visual tokens, keeps or merges a smaller set, and returns
them to the language model.

Adding one takes four steps:

1. Implement `TokenCompressor`.
2. Install it on the backbone.
3. Register it.
4. Test it on tensors and a real model.

This example adds a simple `EnergyCompressor` that keeps the tokens with the
largest L2 norms.

## 1. Implement the compressor

Create `llmoe/compressor/energy.py`:

```python
from typing import Any, override

import torch

from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor


class EnergyCompressor(TokenCompressor):
    """Keep the highest-energy visual tokens in spatial order."""

    def __init__(self, max_tokens: int = 64) -> None:
        super().__init__(int(max_tokens))
        if self._max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @override
    def compress(
        self,
        features: torch.Tensor,
        *,
        attention: torch.Tensor | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> CompressedFeatures:
        del attention, layer_idx, kwargs

        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(
                f"Expected [B, N, D] or [N, D], got {tuple(features.shape)}"
            )
        if features.shape[1] == 0:
            raise ValueError("Cannot compress an empty sequence")

        batch, n_tokens, hidden_dim = features.shape
        k = min(self._max_tokens, n_tokens)

        scores = features.float().square().sum(dim=-1)
        indices = scores.topk(k, dim=1).indices.sort(dim=1).values
        tokens = features.gather(
            1,
            indices.unsqueeze(-1).expand(batch, k, hidden_dim),
        )

        # Used by `llmoe ... bm --visualize`.
        identity = torch.eye(
            n_tokens,
            device=features.device,
            dtype=features.dtype,
        )
        source = features.new_zeros((batch, k, n_tokens))
        source.scatter_(2, indices.unsqueeze(-1), 1.0)
        self._source_history = [identity, source[0]]

        if unbatched:
            tokens = tokens.squeeze(0)
        return CompressedFeatures(tokens=tokens)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        return self._source_history
```

The main contract is:

- Input is `[B, N, D]` or `[N, D]`.
- Output keeps the same rank, dtype, and device.
- Every image in a batch must produce the same number of tokens.
- Source-history entries have shape `[current_tokens, original_tokens]`.
  Pruning uses one-hot rows; merging uses the actual merge weights.

`EnergyCompressor` defines no `name`. It does not need one: `TokenCompressor`
labels every run `<Method>-b<budget>`, so this one is `Energy-b64` and its
results land under `<output_dir>/mme/Energy-b64/`. **Do not override `name`** —
one format is what makes methods comparable by path across a budget sweep.

Override the `budget` property instead, and only when the budget you report is
not the `max_tokens` you passed to `super().__init__()`:

```python
@property
@override
def budget(self) -> int:
    # `max_tokens` holds the derived stage-1 count, not the reported budget.
    return self._budget
```

`ApETCompressor` and `VFlowOptCompressor` do this because `max_tokens` carries an
internal stage count for them (64 reports as `_max_tokens` 32 and 89
respectively). Compressors driven by a retention ratio rather than a count
report the ratio — `GlimpsePrune-b0.25`, `DivPrune-b0.098`.

See [`prototype.py`](../llmoe/compressor/prototype.py) for the full interface.

## 2. Install it on the backbone

The inherited `install()` wraps `vlm.model.model.get_image_features` and calls
`compress()`. That is enough only when the backbone does not separately track
the original image-token count.

### Important: LLaVA placeholders

LLaVA requires the number of `<image>` placeholders in `input_ids` to match the
number of image features. If compression changes `N` features to `K`, your
installation must also:

1. wrap `vlm.prepare_inputs`;
2. shorten each image-placeholder run to `K`;
3. update `attention_mask` and padding;
4. restore the original methods in `uninstall()`.

Follow
[`SoftTopkCompressor.install()`](../llmoe/compressor/soft_topk/soft_topk.py)
for the working lifecycle. Keep installation idempotent so running several
benchmarks does not stack wrappers.

If the output count depends on image content, it cannot be known when
placeholders are prepared. Use masking or another backbone-specific strategy
instead. `MustDropCompressor` is an example of this approach.

For language-layer compression, keep every hook handle and remove it during
`uninstall()`:

```python
handle = layer.register_forward_pre_hook(fn, with_kwargs=True)
self._hook_handles.append(handle)

# In uninstall():
for handle in self._hook_handles:
    handle.remove()
self._hook_handles.clear()
```

## 3. Register it

Update `llmoe/compressor/__init__.py`:

```python
from llmoe.compressor.energy import EnergyCompressor


COMPRESSORS: dict[str, type[TokenCompressor]] = {
    "SoftTopk": SoftTopkCompressor,
    "MustDrop": MustDropCompressor,
    "Energy": EnergyCompressor,
}

type CompressorLiteral = _Literal["SoftTopk", "MustDrop", "Energy"]
```

The registry key is the value passed to `--compressor`. Keep registration here;
the compressor module should not import and mutate `COMPRESSORS` while the
package is initializing.

For an external compressor, register it from a launcher instead:

```python
from llmoe import __entry__
from my_compressors import EnergyCompressor


if __name__ == "__main__":
    __entry__(custom_compressors={"Energy": EnergyCompressor})
```

## 4. Configure and run it

Create `energy.yaml`:

```yaml
compressor_args:
  max_tokens: 64
```

Run a small benchmark:

```console
llmoe --config local \
  --compressor Energy \
  --custom_args energy.yaml \
  --dataset-limit 2 \
  bm mme
```

`compressor_args` is passed directly to `EnergyCompressor(**compressor_args)`.
Put global options before the `bm` subcommand.

### Standard budgets

Benchmarks sweep the same four budgets across every compressor: **32, 64, 128 and
224**. `max_tokens` has to accept all four.

What the budget *means* is per-compressor, and belongs in the module docstring:

- a **final token count**, as in `EnergyCompressor` above and `divprune.py`; or
- the count **averaged over the decoder's layers**, which is what most
  decoder-side papers report — `must_drop.py`, `sparse_vlm.py`, `duet.py`,
  `hidrop.py`.

A layer-average budget needs a profile table keyed by budget, with the source of
each row recorded — the paper, or an extrapolation from it:

```python
MUSTDROP_PROFILES: dict[int, dict[str, object]] = {
    64: {
        "global_thr": 0.011,
        "individual_thr": 0.01,
        "pruning_layers": (2, 6, 10, 14),
        "key_rate": 0.08,
        "source": "paper",
    },
    ...
}
```

Three rules for those tables:

- **Never fall back to another budget's profile.** `PRESETS.get(budget,
  PRESETS[64])` silently benchmarks 64 tokens under a `32` label. Raise for an
  unlisted budget instead, and name the knobs a caller can pass to explore one.
- **Check the budget is reachable.** A schedule that holds every token through
  its first `L` layers, or that protects a key set from pruning, has a hard floor
  on its layer-average that no threshold or ratio can beat. Compute that floor.
  Give a low budget its own schedule if it lands under one — MustDrop's 32
  profile prunes a layer earlier and protects a smaller key set for exactly this
  reason, since the paper's own schedule floors at 61.25 — and warn when a
  custom configuration still cannot reach its label.
- **Record which depth the profile is stated at, and re-solve for the loaded
  one.** A layer average divides by the decoder depth, so a table written for
  Vicuna-7B's 32 layers describes a different budget on the 13B's 40. Do not
  hand-write a second table: build a
  [`LayerSchedule`](../llmoe/utils/depth.py) from the published boundaries and
  keeps at `REFERENCE_DEPTH`, and call `rescaled_to(decoder_depth(vlm))` in
  `install()`. That moves each boundary to the same fraction of the decoder and
  re-solves the keeps so the label still holds; at depth 32 it returns the
  schedule unchanged. `LayerSchedule` is also where `layer_average`, `floor()`
  and `counts_by_layer()` live, so the arithmetic above has one implementation
  rather than one per compressor. Keep the configured schedule and restore it in
  `uninstall()`, and take a caller-supplied schedule literally.

Print the achieved average per run so the label can be checked against it.

## 5. Test it

Start without loading a VLM:

```python
import torch

from llmoe.compressor.energy import EnergyCompressor


compressor = EnergyCompressor(max_tokens=3)
features = torch.randn(2, 8, 16)
result = compressor.compress(features)

assert result.tokens.shape == (2, 3, 16)
assert result.tokens.dtype == features.dtype
assert [x.shape for x in compressor.get_source_history()] == [(8, 8), (3, 8)]
```

Then run one real sample and check:

- generation completes;
- placeholder and feature counts match;
- the token budget is correct;
- batch size 1 and the intended larger batch both work;
- repeated `install()` calls do not add duplicate wrappers;
- `uninstall()` restores methods and removes hooks;
- inference is deterministic.

To check source tracking visually:

```console
llmoe --config local \
  --compressor Energy \
  --custom_args energy.yaml \
  --dataset-limit 1 \
  bm --visualize mme
```

Visualization is currently LLaVA/CLIP-specific.

## 6. Add training or fine-tuning

The `Energy` example is training-free. A compressor with learned parameters,
such as a token-scoring MLP, needs a trainer.

Use:

- `tr` to learn a new compressor from its initial weights;
- `ft` to continue from a checkpoint or adapt it to new data or a backbone.

These commands use separate registries. Add only the workflows the compressor
supports. The class must also be registered in `COMPRESSORS` and
`CompressorLiteral` as shown earlier.

```python
COMPRESSOR_TRAINERS["LearnedEnergy"] = (
    learned_energy_fn_cd,
    train_learned_energy,
)
COMPRESSOR_TUNERS["LearnedEnergy"] = (
    learned_energy_fn_cd,
    finetune_learned_energy,
)
```

Each value is `(environment_check, worker)`:

```python
def learned_energy_fn_cd(config: ExpConfig) -> str:
    dataset_dir = Path(config.input_dir) / "MyTrainingData"
    return "OK" if dataset_dir.is_dir() else f"Missing dataset: {dataset_dir}"


def train_learned_energy(vlm: VLM, config: Config) -> None:
    ...
```

The worker should:

1. read `compressor_args` and `training_args` from `config.custom_args`;
2. create the compressor and move it to the model's device;
3. freeze the VLM if only the compressor should learn;
4. install the compressor and switch it to training mode;
5. compute a differentiable task loss and update the intended parameters;
6. save the compressor config and `state_dict`;
7. uninstall it and restore the VLM's mode and `requires_grad` flags.

Hard `topk` indices do not provide gradients to a learned scorer. Use a
straight-through estimator or another differentiable relaxation during
training. `SoftTopkCompressor` shows this pattern while keeping true hard-top-k
values in the forward pass.

### Training configuration

For a new training run:

```yaml
compressor_args:
  max_tokens: 64
  scorer_hidden_dim: 256

training_args:
  num_epochs: 3
  batch_size: 2
  learning_rate: 0.0001
  weight_decay: 0.01
  gradient_accumulation_steps: 4
  checkpoint_name: learned_energy.pt
```

Run:

```console
llmoe --config local \
  --compressor LearnedEnergy \
  --custom_args train_energy.yaml \
  tr
```

### Fine-tuning configuration

Fine-tuning normally starts from a saved checkpoint:

```yaml
compressor_args:
  max_tokens: 64
  scorer_hidden_dim: 256
  checkpoint_path: /path/to/learned_energy.pt

training_args:
  num_epochs: 1
  learning_rate: 0.00001
  checkpoint_name: learned_energy_finetuned.pt
```

Run:

```console
llmoe --config local \
  --compressor LearnedEnergy \
  --custom_args finetune_energy.yaml \
  ft
```

The codebase does not otherwise define how `tr` and `ft` differ; that behavior
belongs in the registered worker functions. They may share one implementation
when loading `checkpoint_path` is the only difference, as SoftTopk currently
does.

Checkpoint loading should validate architecture-sensitive settings such as
`max_tokens` and hidden width. For benchmarking, pass the trained
`checkpoint_path` under `compressor_args`.

See
[`soft_topk_tr.py`](../llmoe/compressor/soft_topk/soft_topk_tr.py) for a complete
trainer. It freezes the VLM, supervises the compressor with language-model loss,
supports mixed precision and gradient accumulation, and writes checkpoints.

## Final checklist

- `compress()` handles batched and unbatched inputs.
- Output rank, dtype, device, and token count are correct.
- Source history accurately describes pruning or merging.
- Backbone placeholders, masks, and caches remain consistent.
- `install()` and `uninstall()` are safe to call repeatedly.
- The compressor is registered and accepts `compressor_args`.
- Every standard budget (32/64/128/224) is accepted, and none silently reuses
  another budget's settings.
- `name` is not overridden, and `budget` reports the label the run should carry.
- Tensor tests, one-sample inference, batching, and visualization pass.
- Training restores model state and produces a loadable checkpoint.
- Fine-tuning loads and validates its starting checkpoint.
