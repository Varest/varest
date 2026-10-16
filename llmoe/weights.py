"""Locating, checking, and fetching the released LLMOE method weights.

Every trained artifact this repo consumes -- HiDrop and DUET checkpoints, the
VisionZip projectors, CORE's adapters, VFlowOpt's calibrated strategies -- lives
in one bundle whose layout mirrors the public Hugging Face bucket
``ProjectNeura/LLMOE``::

    <HF_HOME>/llmoe_weights/
        methods/hidrop/llava-1.5-7b/budget_64/
        methods/duet/llava-1.5-7b/budget_64/
        methods/visionzip/llava-1.5-7b/mm_projector_vz64.safetensors
        methods/core/vicuna-7b-v1.5/budget_1024/
        methods/vflowopt/strategies/strategy_b64.json

``HF_HOME`` is the Hugging Face cache root the rest of the stack already uses
(`llmoe.bootstrap` points it at ``<root_dir>/output/hf_cache``), so method weights
sit beside the base-model cache instead of in a hand-managed directory.
``LLMOE_WEIGHTS_DIR`` overrides the bundle root outright.

Nothing is downloaded behind the user's back: when a requested artifact is
missing, :func:`ensure` prints what it wants and asks. ``LLMOE_DOWNLOAD_WEIGHTS``
answers ahead of time (``1``/``yes`` to always download, ``0``/``no`` to never),
which is what non-interactive runs need -- without it, a missing artifact raises
rather than blocking on a prompt nobody can see.
"""

import sys
import warnings
from dataclasses import dataclass
from os import PathLike, environ, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

__all__ = [
    "WEIGHTS_BUCKET", "WEIGHTS_DIR_NAME", "BACKBONE_SLUGS", "Artifact", "ARTIFACTS",
    "artifact_path", "backbones_for", "hf_home", "weights_root",
    "artifact_for", "supported_budgets", "ensure", "ensure_relative", "resolve_weights",
    "redirect_backbone", "check_weights",
]

# Public, anonymous-readable. `token=False` on every call keeps a stale or
# unrelated HF_TOKEN from turning a public read into a 401.
WEIGHTS_BUCKET = "ProjectNeura/varest_weights"
WEIGHTS_DIR_NAME = "llmoe_weights"


@dataclass(frozen=True)
class Artifact:
    """One released artifact: where it lives in the bundle and what it is.

    Attributes:
        method: Compressor key as it appears in ``COMPRESSORS``.
        path: Location inside the bundle, relative and POSIX-style. Identical in
            the local bundle and in the bucket, so one string addresses both.
        is_directory: Whether ``path`` names a directory to fetch whole.
        budget: Visual-token budget the artifact was trained/calibrated for, or
            ``None`` when one release serves every budget.
        backbone: Backbone the weights belong to; they are not interchangeable.
        kind: Short human label used in prompts and warnings.
        trained_only: ``True`` when the method has no meaningful untrained form,
            so :func:`resolve_weights` reaches for the released weights even
            without an explicit ``pretrained`` flag (warning as it does).
    """

    method: str
    path: str
    is_directory: bool
    budget: int | None
    backbone: str
    kind: str
    trained_only: bool

    @property
    def label(self) -> str:
        budget = "any budget" if self.budget is None else f"budget {self.budget}"
        return f"{self.method} {self.backbone} {budget} ({self.kind})"


#: Canonical directory segment per ``--backbone``. Every artifact lives under
#: ``methods/<method>/<backbone>/…`` so the same method can ship weights for more
#: than one backbone; ``hidrop`` and ``visionzip`` already followed this, while
#: ``quietprune`` and ``vflowopt`` were flat and could hold only one.
#: Keyed by ``VLM.weights_key`` (equivalently the ``--backbone`` value). LLaVA-1.5
#: 7B and 13B are separate keys because their artifacts are not interchangeable:
#: every learned projector or adapter here is sized to the decoder's hidden width,
#: 4096 on 7B and 5120 on 13B.
BACKBONE_SLUGS: dict[str, str] = {
    "llava": "llava-1.5-7b",
    "llava-next-8b": "llama3-llava-next-8b",
    "llava-13b": "llava-1.5-13b",
    "qwen-vl": "qwen2.5-vl-7b",
    "internvl": "internvl3.5-8b",
    "core": "vicuna-7b-v1.5",
    "core-13b": "vicuna-13b-v1.5",
    # No artifact is registered below: this key exists to keep future trained
    # Llama-3 CORE adapters distinct from the released Vicuna adapter.
    "core-llama3-8b": "core-convnext-omgseg-llama3-8b",
}


def backbone_key(vlm: Any) -> str | None:
    """The artifact key for a loaded backbone, for :func:`resolve_weights`.

    Every backbone in this repo is a :class:`~llmoe.backbone.prototype.VLM` and
    declares ``weights_key``. Anything else -- a test double, a third-party
    wrapper -- returns ``None``, which means "unspecified" and leaves the old
    behaviour of falling back to a method's single released backbone. A ``VLM``
    subclass that forgets to set one keeps the base class's empty string and
    fails by name instead, which is what we want from a real backbone.
    """
    return getattr(vlm, "weights_key", None)


def artifact_path(method: str, backbone: str, leaf: str) -> str:
    """Bucket path for one artifact: ``methods/<method>/<backbone>/<leaf>``.

    ``backbone`` is a ``--backbone`` key (``qwen-vl``) or an already-resolved slug
    (``qwen2.5-vl-7b``); both are accepted so callers need not care which they hold.
    """
    slug = BACKBONE_SLUGS.get(backbone, backbone)
    return f"methods/{method.lower()}/{slug}/{leaf}".rstrip("/")


def _quietprune(backbone: str) -> Artifact:
    return Artifact("QuietPrune", artifact_path("quietprune", backbone, "adapter.pt"),
                    False, None, BACKBONE_SLUGS.get(backbone, backbone),
                    "text-to-vision adapter", True)


def _varest(backbone: str, variant: str, corpus: str = "") -> Artifact:
    """One self-contained file per (backbone, variant): LoRA + projector + position tag (+ [REL]) + both heads.

    ``corpus`` tags a release trained on a corpus other than LLaVA-1.5's 665K mix (``"nextdata"``: LLaVA-NeXT-Data).
    It is part of the directory name, so a cached copy of an older release at the old path can never be taken for it
    (the cache only checks that the file exists).
    """
    leaf = f"{variant}-{corpus}" if corpus else variant
    return Artifact("Varest" if variant == "w5reljoint" else "VarestW5",
                    artifact_path("varest", backbone, f"{leaf}/varest_{variant}.pt"), False, None,
                    BACKBONE_SLUGS.get(backbone, backbone), f"{variant.upper()} checkpoint", True)


def _learnpruner(backbone: str, leaf: str) -> Artifact:
    """One released LPM. Named for the budget it trained under, which it ignores.

    The LLaVA-1.5 filenames carry a ``b64`` because that is what their training
    run was configured with. It does not restrict them: the training mask is an
    argmax over the LPM's own scores, so the retention budget never enters the
    loss and is applied only at inference. The released ``*_training.json`` says
    so, and the four 7B runs, four 13B runs and three NeXT runs at different
    budgets each produced bitwise-identical weights. Hence one artifact per
    backbone, keyed at budget ``None`` like QuietPrune's adapter.
    """
    return Artifact("LearnPruner", artifact_path("learnpruner", backbone, leaf),
                    False, None, BACKBONE_SLUGS.get(backbone, backbone),
                    "learned pruning module", True)


def _hidrop(budget: int, backbone: str = "llava") -> Artifact:
    slug = BACKBONE_SLUGS.get(backbone, backbone)
    return Artifact("HiDrop", f"methods/hidrop/{slug}/budget_{budget}", True, budget,
                    slug, "finetuned checkpoint", True)


def _duet(budget: int, backbone: str = "llava") -> Artifact:
    slug = BACKBONE_SLUGS.get(backbone, backbone)
    return Artifact("DUET", f"methods/duet/{slug}/budget_{budget}", True, budget,
                    slug, "compression-aware finetuned checkpoint", False)


def _vision_zip(budget: int) -> Artifact:
    return Artifact("VisionZip", f"methods/visionzip/llava-1.5-7b/mm_projector_vz{budget}.safetensors",
                    False, budget, "llava-1.5-7b", "efficient-tuning projector", False)


def _vision_zip_13b(budget: int) -> Artifact:
    """The 13B efficient-tuning projector.

    Named differently from the 7B release -- ``vision_zip_projector_<slug>_k<b>.pt``
    rather than ``mm_projector_vz<b>.safetensors`` -- and a torch pickle rather
    than safetensors, so it needs its own template instead of reusing
    :func:`_vision_zip`.
    """
    return Artifact("VisionZip",
                    f"methods/visionzip/llava-1.5-13b/vision_zip_projector_llava-1.5-13b_k{budget}.pt",
                    False, budget, "llava-1.5-13b", "efficient-tuning projector", False)


def _vflowopt_13b(budget: int) -> Artifact:
    """A 13B strategy, under the backbone-scoped path `artifact_path` builds.

    The 7B files predate that convention and stay at
    `methods/vflowopt/strategies/`, so they keep their own template.
    """
    return Artifact("VFlowOpt", artifact_path("vflowopt", "llava-13b", f"strategy_b{budget}.json"),
                    False, budget, "llava-1.5-13b", "calibrated strategy", False)


#: DUET's LLaVA-NeXT releases are named for their *layer-average* token count
#: over the 2880-token maximum AnyRes span, the same convention every
#: layer-scheduled method reports. They line up with the sweep's budgets --
#: 160/2880 = 0.0553 against b32's 0.0556, and so on -- but the ratio each
#: checkpoint was trained at is its Stage-1 content retention, which is roughly
#: 1.6x larger because two pdrop cuts decay it before the average is taken.
#: Sweep budget -> published profile, for callers that think in sweep budgets.
DUET_NEXT_PROFILE = {32: 160, 64: 320, 128: 640, 224: 1120}


def _hidrop_next(budget: int) -> Artifact:
    """One released LLaVA-NeXT HiDrop checkpoint.

    The *key* is the checkpoint's ``average_retained_tokens`` -- 32/64/128/224 --
    because that is what the compressor resolves with: on this backbone
    ``_weight_budget`` takes that value, not the ratio. The *directory* is named
    for what the budget actually spends on NeXT, where a full five-view image is
    2880 content tokens against LLaVA-1.5's 576, so ``0.0556`` is 160 tokens.

    That split is deliberate: ``average_retained_tokens`` is not a token count
    here, it is the denominator that rescales the stored per-stage curve
    (``keep = base_keep * target / profile_budget``), so it must stay 32 while the
    directory says 160.
    """
    profile = DUET_NEXT_PROFILE[budget]
    return Artifact("HiDrop", f"methods/hidrop/llama3-llava-next-8b/budget_{profile}",
                    True, budget, "llama3-llava-next-8b", "finetuned checkpoint", True)


def _duet_next(profile: int) -> Artifact:
    """One released LLaVA-NeXT DUET checkpoint, keyed by its published profile.

    The key is the profile, not the sweep budget, because that is what the
    compressor resolves with: `llava_next_profile` sets `_max_tokens` to the
    profile, and `install` looks the checkpoint up by it.
    """
    return Artifact("DUET", f"methods/duet/llama3-llava-next-8b/budget_{profile}",
                    True, profile, "llama3-llava-next-8b",
                    "compression-aware finetuned checkpoint", False)


def _vision_zip_next(budget: int) -> Artifact:
    """The LLaVA-NeXT VisionZip++ projector for one sweep budget.

    Keyed under ``VisionZip++`` rather than ``VisionZip``: on this backbone the
    two are different compressors, not one with a flag. Plain ``VisionZip`` is
    training-free on NeXT, while ``VisionZipPlusPlusCompressor`` swaps the
    multimodal projector and takes an explicit ``checkpoint_path``.

    Named by retention ratio rather than token count, because NeXT's AnyRes span
    is variable and every adapter on that backbone budgets a fraction of it. An
    ``r0.25`` projector is also published; it is not registered because it
    matches none of the sweep's operating points.
    """
    ratio = f"{budget / 576:.4f}"
    return Artifact("VisionZip",
                    f"methods/visionzip/llama3-llava-next-8b/"
                    f"vision_zip_pp_llava_next_r{ratio}.pt",
                    False, budget, "llama3-llava-next-8b",
                    "efficient-tuning projector", False)


def _vflowopt(budget: int) -> Artifact:
    return Artifact("VFlowOpt", f"methods/vflowopt/strategies/strategy_b{budget}.json", False, budget,
                    "llava-1.5-7b", "calibrated strategy", False)


# Keyed by ``(method, backbone, budget)``. The backbone belongs in the key: the
# same method ships different weights per backbone -- QuietPrune's adapter inverts
# a different projector on each -- and a two-part key could only ever hold one of
# them. ``backbone`` is the ``--backbone`` value, not the model slug.
#
# A method absent from here has no released weights: Nuwa, FastV, ToMe, SparseVLM,
# PyramidDrop, DivPrune, OTPrune, ApET and MustDrop are training-free, and no
# completed checkpoint exists for SoftTopk or GlimpsePrune.
ARTIFACTS: dict[tuple[str, str, int | None], Artifact] = {
    **{("HiDrop", "llava", budget): _hidrop(budget) for budget in (32, 64, 128, 224)},
    **{("DUET", "llava", budget): _duet(budget) for budget in (32, 64, 128, 224)},
    **{("VisionZip", "llava", budget): _vision_zip(budget) for budget in (32, 64, 128, 224)},
    **{("VFlowOpt", "llava", budget): _vflowopt(budget)
       for budget in (32, 58, 64, 128, 144, 224, 288)},
    ("CORE", "core", 1024): Artifact("CORE", "methods/core/vicuna-7b-v1.5/budget_1024", True,
                                     1024, "vicuna-7b-v1.5", "projector + LoRA adapters", True),
    # Budget-independent: QuietPrune's only trained part is the text-to-vision
    # adapter, and its geometry knob `R` is chosen at evaluation time, so one
    # checkpoint per backbone serves every budget.
    ("QuietPrune", "llava", None): _quietprune("llava"),
    ("QuietPrune", "qwen-vl", None): _quietprune("qwen-vl"),
    # Budget-independent for a different reason: LearnPruner's LPM is trained
    # against its own argmax mask, so the budget only ever acts at inference.
    ("LearnPruner", "llava", None): _learnpruner("llava", "learnpruner_lpm_7b_b64.pt"),
    # Budget-independent: Varest trains with the budget sampled per step (K ~ U[32, 576]), so one checkpoint
    # serves every budget. `Varest` is W5RELJOINT, `VarestW5` the tokenizer without [REL].
    ("Varest", "llava", None): _varest("llava", "w5reljoint"),
    ("VarestW5", "llava", None): _varest("llava", "w5"),
    ("Varest", "llava-13b", None): _varest("llava-13b", "w5reljoint"),
    ("VarestW5", "llava-13b", None): _varest("llava-13b", "w5"),
    # LLaVA-NeXT-8B releases are trained on LLaVA-NeXT-Data (NeXT's own tuning set), not the 665K mix.
    ("Varest", "llava-next-8b", None): _varest("llava-next-8b", "w5reljoint", corpus="nextdata"),
    ("VarestW5", "llava-next-8b", None): _varest("llava-next-8b", "w5", corpus="nextdata"),
    # -- LLaVA-1.5-13B releases -------------------------------------------
    # Same methods, retrained on the 40-layer/5120-wide decoder. The 7B
    # artifacts are not substitutable: HiDrop and QuietPrune ship weights
    # shaped to the decoder width, and CORE's LoRA targets its own decoder.
    **{("HiDrop", "llava-13b", budget): _hidrop(budget, "llava-13b")
       for budget in (32, 64, 128, 224)},
    ("QuietPrune", "llava-13b", None): _quietprune("llava-13b"),
    # Retrained on the 40-layer decoder: the LPM reads 1024-wide CLIP features on
    # both LLaVA-1.5 sizes, but the 13B's pruning boundary sits at layer 15, not
    # 12, so the run that produced it is not the 7B's.
    ("LearnPruner", "llava-13b", None): _learnpruner("llava-13b",
                                                     "learnpruner_lpm_13b_b64.pt"),
    # DUET's 13B release is compression-aware finetuned on the 40-layer decoder:
    # its duet_config records decoder_layers 40 and pdrop_layers [20, 30], the
    # depth-scaled form of the 7B's [16, 24] over 32 layers.
    **{("DUET", "llava-13b", budget): _duet(budget, "llava-13b")
       for budget in (32, 64, 128, 224)},
    # VisionZip++ (the double-dagger variant) = VisionZip + this projector.
    **{("VisionZip", "llava-13b", budget): _vision_zip_13b(budget)
       for budget in (32, 64, 128, 224)},
    # CORE carries its own ConvNeXt-L/OMG-Seg vision stack; the 13B release
    # swaps only the Vicuna decoder, so it is keyed by its own backbone name.
    ("CORE", "core-13b", 1024): Artifact("CORE", "methods/core/vicuna-13b-v1.5/budget_1024",
                                         True, 1024, "vicuna-13b-v1.5",
                                         "projector + LoRA adapters", True),
    # Re-calibrated on the 13B decoder rather than reusing the 7B strategies:
    # VFlowOpt's Eq. 5 defines the budget as a layer average over the three
    # pruning stages, so ratios solved for 32 layers (stages 10/10/12) describe
    # a different budget on 40 (13/13/14). The 7B files are also stamped
    # `model: llava-hf/llava-1.5-7b-hf`. Produced by re-running the paper's
    # Algorithm 1; each JSON carries a `provenance` field saying so.
    **{("VFlowOpt", "llava-13b", budget): _vflowopt_13b(budget)
       for budget in (32, 64, 128, 224)},
    # -- LLaVA-NeXT-8B releases -------------------------------------------
    # One adapter serves every budget here too: the trained part inverts the
    # projector, and the geometry knob is chosen at evaluation time.
    ("QuietPrune", "llava-next-8b", None): _quietprune("llava-next-8b"),
    # The NeXT LPM is a different input space, not just a retrain: it scores the
    # 4096-wide packed AnyRes sequence the decoder consumes, after projection,
    # unpadding and newline insertion, where the LLaVA-1.5 LPMs score 1024-wide
    # raw CLIP features. Its checkpoint records `packed_projected_anyres`.
    ("LearnPruner", "llava-next-8b", None): _learnpruner("llava-next-8b",
                                                         "learnpruner_lpm_next.pt"),
    # CORE keeps its own ConvNeXt-L/OMG-Seg stack and its native 1024-token
    # grid on this port -- it never builds LlavaNextForConditionalGeneration and
    # never consumes AnyRes crops -- so only the decoder is Llama-3, and the
    # release is keyed by its own backbone name like the other CORE ports.
    ("CORE", "core-llama3-8b", 1024): Artifact(
        "CORE", "methods/core/llama3-llava-next-8b/budget_1024", True, 1024,
        "llama3-llava-next-8b", "projector + LoRA adapters", True),
    # VisionZip++ on LLaVA-NeXT is its own compressor and its own key: the
    # projectors are named for the ratio they were tuned at, not a token count,
    # because every NeXT adapter budgets a fraction of the image's AnyRes span.
    # The b<N> key keeps the sweep's rows lined up; N/576 selects the file.
    **{("VisionZip++", "llava-next-8b", budget): _vision_zip_next(budget)
       for budget in (32, 64, 128, 224)},
    **{("DUET", "llava-next-8b", profile): _duet_next(profile)
       for profile in (160, 320, 640, 1120)},
    **{("HiDrop", "llava-next-8b", budget): _hidrop_next(budget)
       for budget in (32, 64, 128, 224)},
}


#: Methods with no meaningful untrained form, derived from the artifacts themselves
#: so it cannot drift from them. Consulted when a method has releases but none for
#: the selected backbone: falling through to "untrained" would then be an ablation
#: wearing the paper's label.
_TRAINED_ONLY_METHODS: dict[str, bool] = {}
for _name, _, _ in ARTIFACTS:
    _TRAINED_ONLY_METHODS[_name] = any(
        artifact.trained_only for (name, _, _), artifact in ARTIFACTS.items() if name == _name
    )


def hf_home() -> Path:
    """The Hugging Face cache root, read from the environment at call time.

    ``huggingface_hub.constants.HF_HOME`` is frozen at import, but the pipeline
    sets ``HF_HOME`` while starting up, so the constant is routinely stale here.
    The fallback mirrors Hugging Face's own default.
    """
    override = environ.get("HF_HOME")
    if override:
        return Path(override).expanduser()
    return Path(environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "huggingface"


def weights_root() -> Path:
    """Bundle root: ``$LLMOE_WEIGHTS_DIR``, else ``<HF_HOME>/llmoe_weights``."""
    override = environ.get("LLMOE_WEIGHTS_DIR")
    return Path(override).expanduser() if override else hf_home() / WEIGHTS_DIR_NAME


def backbones_for(method: str) -> list[str]:
    """Backbones `method` has released weights for."""
    return sorted({backbone for name, backbone, _ in ARTIFACTS if name == method})


def supported_budgets(method: str, backbone: str | None = None) -> list[int]:
    """Budgets `method` has released weights for, ascending.

    Empty for a method whose single release covers every budget; use
    :func:`artifact_for`, which resolves that case.
    """
    return sorted(
        budget for name, bone, budget in ARTIFACTS
        if name == method and budget is not None and (backbone is None or bone == backbone)
    )


def artifact_for(method: str, budget: int, backbone: str | None = None) -> Artifact:
    """The released artifact for one method, backbone and budget.

    ``backbone`` may be omitted only while the method has weights for exactly one
    backbone -- otherwise the choice is genuinely ambiguous and picking silently
    would hand back another backbone's weights, which for QuietPrune means an
    adapter that inverts the wrong projector.

    Raises:
        KeyError: No such release, or the backbone is ambiguous. The message lists
            what does exist, because an unlisted budget means "train it yourself",
            not "close enough".
    """
    available = backbones_for(method)
    if backbone is None:
        if len(available) == 1:
            backbone = available[0]
        elif available:
            raise KeyError(
                f"{method} has released weights for {', '.join(available)}; pass "
                f"backbone= to say which. They are not interchangeable."
            )
    artifact = (ARTIFACTS.get((method, backbone, int(budget)))
                or ARTIFACTS.get((method, backbone, None)))
    if artifact is not None:
        return artifact
    if not available:
        raise KeyError(
            f"No released {method} weights. Released methods: "
            f"{', '.join(sorted({name for name, _, _ in ARTIFACTS}))}."
        )
    budgets = supported_budgets(method, backbone)
    if not budgets:
        raise KeyError(
            f"No released {method} weights for backbone {backbone!r}. Released "
            f"backbones: {', '.join(available)}."
        )
    raise KeyError(
        f"No released {method} weights for backbone {backbone!r} at budget {budget}. "
        f"Released budgets: {', '.join(str(b) for b in budgets)}."
    )


# ---------------------------------------------------------------------------
# Presence and download
# ---------------------------------------------------------------------------

def _api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def _bucket_files(prefix: str) -> list[Any]:
    """Every file at or under `prefix` in the bucket, as `BucketFile` entries."""
    entries = _api().list_bucket_tree(WEIGHTS_BUCKET, prefix, recursive=True, token=False)
    files = [entry for entry in entries if getattr(entry, "type", None) == "file"]
    if not files:
        raise FileNotFoundError(
            f"hf://buckets/{WEIGHTS_BUCKET}/{prefix} holds no files. The bucket layout may have "
            f"changed; check https://huggingface.co/buckets/{WEIGHTS_BUCKET}."
        )
    return files


def _is_present(destination: Path, size: int) -> bool:
    return destination.is_file() and destination.stat().st_size == size


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB"):
        if size < 1024:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _has_partial_files(destination: Path) -> bool:
    """Whether an interrupted download left `.part` files behind."""
    if destination.is_file():
        return False
    return any(destination.rglob("*.part"))


def _consent(description: str, prefix: str, destination: Path, missing: Sequence[Any], *,
             assume_yes: bool = False) -> bool:
    """Ask before downloading; `LLMOE_DOWNLOAD_WEIGHTS` answers in advance.

    `assume_yes` is for the caller who was asked already -- ``llmoe weights
    --download`` is itself the answer, so re-prompting there is noise.
    """
    total = sum(int(entry.size) for entry in missing)
    preset = "yes" if assume_yes else (environ.get("LLMOE_DOWNLOAD_WEIGHTS") or "").strip().lower()
    summary = (
        f"\nLLMOE needs weights that are not in the local bundle: {description}\n"
        f"  bucket : hf://buckets/{WEIGHTS_BUCKET}/{prefix}\n"
        f"  local  : {destination}\n"
        f"  missing: {len(missing)} file(s), {_format_size(total)}\n"
    )
    if preset in ("1", "y", "yes", "true", "always"):
        reason = "requested" if assume_yes else f"LLMOE_DOWNLOAD_WEIGHTS={preset}"
        print(f"{summary}Downloading ({reason}).", flush=True)
        return True
    if preset in ("0", "n", "no", "false", "never"):
        return False
    if not sys.stdin.isatty():
        raise FileNotFoundError(
            f"Missing LLMOE weights: {description} (expected at {destination}). This run is not "
            f"interactive, so it cannot ask for permission to download them. Set "
            f"LLMOE_DOWNLOAD_WEIGHTS=1 to allow downloads without a prompt, or fetch them ahead "
            f"of time from an interactive shell with `python -m llmoe weights --download`."
        )
    print(summary, flush=True)
    try:
        answer = input("Download now? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _download(files: Sequence[Any], root: Path, prefix: str) -> None:
    """Fetch `files` into the bundle, mirroring their bucket paths.

    Each file lands on a ``.part`` sibling first and is renamed once the whole
    batch is in, so an interrupted download never leaves behind a short file
    that the next presence check would accept.
    """
    root.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Any, Path]] = []
    for entry in files:
        final = root / PurePosixPath(entry.path)
        final.parent.mkdir(parents=True, exist_ok=True)
        pairs.append((entry, final.with_name(final.name + ".part")))
    print(f"Downloading {len(pairs)} file(s) from hf://buckets/{WEIGHTS_BUCKET}/{prefix} ...", flush=True)
    _api().download_bucket_files(
        WEIGHTS_BUCKET, [(entry, str(part)) for entry, part in pairs],
        raise_on_missing_files=True, token=False,
    )
    for entry, part in pairs:
        actual = part.stat().st_size
        if actual != int(entry.size):
            part.unlink(missing_ok=True)
            raise OSError(
                f"Truncated download for {entry.path}: got {actual} bytes, expected {entry.size}."
            )
        replace(part, part.with_name(part.name.removesuffix(".part")))
    print(f"Weights ready under {root}", flush=True)


def ensure_relative(relative: str, *, description: str | None = None,
                    assume_yes: bool = False) -> Path:
    """Local path for a bundle-relative artifact, downloading it if needed.

    Args:
        relative: POSIX path inside the bundle, e.g.
            ``methods/hidrop/llava-1.5-7b/budget_64``.
        description: Human label for the prompt; defaults to `relative`.
        assume_yes: Skip the prompt because the caller already has consent.

    Returns:
        The absolute local path.

    Raises:
        FileNotFoundError: The artifact is missing and the download was declined
            or impossible to ask about.
    """
    prefix = relative.strip("/")
    root = weights_root()
    destination = root / PurePosixPath(prefix)
    label = description or prefix
    # An artifact already on disk is taken at face value: re-listing the bucket on
    # every compressor construction is not worth the round trip. `.part` files are
    # the one local sign of an interrupted fetch, so those force a re-check.
    if destination.exists() and not _has_partial_files(destination):
        return destination
    files = _bucket_files(prefix)
    missing = [entry for entry in files if not _is_present(root / PurePosixPath(entry.path), int(entry.size))]
    if not missing:
        return destination
    if not _consent(label, prefix, destination, missing, assume_yes=assume_yes):
        raise FileNotFoundError(
            f"Missing LLMOE weights: {label} (expected at {destination}). Download them with "
            f"`python -m llmoe weights --download`, or point the knob at your own checkpoint."
        )
    _download(missing, root, prefix)
    if not destination.exists():
        raise FileNotFoundError(f"Download finished but {destination} is still missing.")
    return destination


def ensure(artifact: Artifact, *, assume_yes: bool = False) -> Path:
    """Local path for a registered artifact, downloading it if needed."""
    return ensure_relative(artifact.path, description=artifact.label, assume_yes=assume_yes)


def resolve_weights(method: str, budget: int, *, checkpoint_path: Any = None,
                    pretrained: bool | str | PathLike[str] | None = None,
                    backbone: str | None = None) -> Path | str | None:
    """Turn a compressor's weight knobs into something loadable.

    The precedence a caller can rely on:

    1. An explicit ``checkpoint_path`` (or a path-valued ``pretrained``) wins.
       Absolute paths and Hugging Face repo ids pass through untouched; a
       relative path is read as bundle-relative and fetched if absent, so
       ``checkpoint_path: methods/hidrop/llava-1.5-7b/budget_64`` works from
       anywhere.
    2. ``pretrained: true`` selects the released artifact for `budget`.
    3. ``pretrained: false`` forces the untrained/training-free form.
    4. Neither given: methods with no untrained form (``trained_only``) still get
       the released artifact, with a warning naming it, because the alternative
       is an ablation masquerading as the paper's setting. Everything else stays
       untrained, which is what it has always done.

    Returns:
        A path (or repo id) to load, or ``None`` for "no checkpoint".
    """
    explicit = checkpoint_path
    if explicit is None and isinstance(pretrained, (str, PathLike)):
        explicit = pretrained
    if explicit is not None and not isinstance(explicit, bool):
        return _resolve_explicit(explicit)
    if isinstance(explicit, bool):
        # `checkpoint_path: true` in a custom-args file reads as the same request.
        pretrained = explicit
    if pretrained is True:
        return ensure(artifact_for(method, budget, backbone))
    if pretrained is False:
        return None
    try:
        artifact = artifact_for(method, int(budget), backbone)
    except KeyError:
        # "This method has no release at all" is the training-free case and falls
        # through to the untrained form. "This method has releases, just not for
        # this backbone" is not: for a method with no untrained form that would
        # quietly downgrade the run to an ablation under the paper's label, so it
        # raises with the message naming what does exist.
        if backbones_for(method) and _TRAINED_ONLY_METHODS.get(method):
            raise
        artifact = None
    if artifact is None or not artifact.trained_only:
        return None
    warnings.warn(
        f"{method} has no training-free form, so this run uses the released {artifact.label} "
        f"from hf://buckets/{WEIGHTS_BUCKET}/{artifact.path}. Pass `pretrained: false` in "
        f"compressor_args for the untrained ablation, or `checkpoint_path` for your own weights.",
        stacklevel=3,
    )
    return ensure(artifact)


def _resolve_explicit(value: str | PathLike[str]) -> Path | str:
    """An explicit knob value: local path, bundle-relative path, or repo id."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    text = str(value).strip("/")
    # Anything else that is not addressable inside the bundle is a Hugging Face
    # repo id; leave it for the caller's own `snapshot_download`.
    if text.startswith("methods/") or (weights_root() / PurePosixPath(text)).exists():
        return ensure_relative(text)
    return str(value)


def redirect_backbone(vlm: Any, directory: Path) -> str | None:
    """Point a lazy backbone at `directory` and drop anything already built.

    Returns the previous model id so an ``uninstall`` can put it back, or
    ``None`` when the backbone was already loading from `directory`.
    """
    previous = getattr(vlm, "_model_id", None)
    if previous:
        current = Path(str(previous)).expanduser()
        if current.is_dir() and current.resolve() == directory.resolve():
            return None
    vlm._model_id = str(directory)
    if getattr(vlm, "_model", None) is not None or getattr(vlm, "_processor", None) is not None:
        vlm._model = None
        vlm._processor = None
    print(f"[llmoe.weights] loading backbone weights from {directory}", flush=True)
    return previous


# ---------------------------------------------------------------------------
# `python -m llmoe weights`
# ---------------------------------------------------------------------------

def check_weights(methods: Iterable[str] | None = None, *, download: bool = False) -> int:
    """Report which released artifacts are present, optionally fetching them.

    Args:
        methods: Restrict to these compressor keys; ``None`` covers every
            released method.
        download: Fetch what is missing (still asking, unless
            ``LLMOE_DOWNLOAD_WEIGHTS`` says otherwise).

    Returns:
        Process exit status: ``0`` when nothing is missing afterwards.
    """
    selected = set(methods) if methods else {name for name, _, _ in ARTIFACTS}
    unknown = selected - {name for name, _, _ in ARTIFACTS}
    if unknown:
        raise KeyError(
            f"No released weights for {', '.join(sorted(unknown))}. Released methods: "
            f"{', '.join(sorted({name for name, _, _ in ARTIFACTS}))}."
        )
    root = weights_root()
    print(f"LLMOE weights bundle: {root}")
    print(f"Bucket: hf://buckets/{WEIGHTS_BUCKET}\n")
    missing: list[Artifact] = []
    for key in sorted(ARTIFACTS, key=lambda item: (item[0], -1 if item[1] is None else item[1])):
        artifact = ARTIFACTS[key]
        if artifact.method not in selected:
            continue
        present = (root / PurePosixPath(artifact.path)).exists()
        print(f"  [{'x' if present else ' '}] {artifact.label:<64} {artifact.path}")
        if not present:
            missing.append(artifact)
    if not missing:
        print("\nAll selected weights are present.")
        return 0
    if not download:
        print(f"\n{len(missing)} artifact(s) missing. Re-run with --download to fetch them.")
        return 1
    for artifact in missing:
        # `--download` is the consent; asking again for each of twenty artifacts
        # would be a worse experience than the flag already promised.
        ensure(artifact, assume_yes=True)
    return 0
