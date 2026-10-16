from typing import Literal as _Literal, Callable as _Callable

from mle.vars import ExpConfig as _ExpConfig

# Before any compressor module: CORE's model is a VLM that `llmoe.backbone`
# imports back, so loading the backbone package first settles that cycle.
from llmoe.backbone import VLM
from llmoe.compressor.prototype import TokenCompressor
from llmoe.compressor.apet import ApETCompressor
from llmoe.compressor.core import CORECompressor, core_tr_fn_cd, core_ft_fn_cd, train_core, \
    finetune_core
from llmoe.compressor.divprune import DivPruneCompressor
from llmoe.compressor.duet import duet_tr_fn_cd, duet_ft_fn_cd, train_duet, finetune_duet, \
    DUETCompressor
from llmoe.compressor.fastv import FastVCompressor
from llmoe.compressor.glimpse_prune import GlimpsePruneCompressor
from llmoe.compressor.hidrop import hidrop_tr_fn_cd, hidrop_ft_fn_cd, train_hidrop, finetune_hidrop, \
    HiDropCompressor
from llmoe.compressor.learnpruner import learn_pruner_tr_fn_cd, learn_pruner_ft_fn_cd, \
    train_learn_pruner, finetune_learn_pruner, LearnPrunerCompressor
from llmoe.compressor.must_drop import MustDropCompressor
from llmoe.compressor.nuwa import NuwaCompressor
from llmoe.compressor.otprune import OTPruneCompressor
from llmoe.compressor.pyramiddrop import PyramidDropCompressor
from llmoe.compressor.quietprune import quietprune_tr_fn_cd, quietprune_ft_fn_cd, train_quietprune, \
    finetune_quietprune, QuietPruneCompressor
from llmoe.compressor.soft_topk import soft_topk_tr_fn_cd, train_soft_topk, SoftTopkCompressor
from llmoe.compressor.sparse_vlm import SparseVLMCompressor
from llmoe.compressor.tome import ToMeCompressor
from llmoe.compressor.varest import (VarestCompressor, VarestW5Compressor, train_varest, train_varest_w5,
                                     varest_tr_fn_cd)
from llmoe.compressor.vflowopt import VFlowOptCompressor
from llmoe.compressor.vision_zip import (
    vision_zip_ft_fn_cd, finetune_vision_zip, VisionZipCompressor,
    vision_zip_pp_ft_fn_cd, finetune_vision_zip_llava_next, VisionZipPlusPlusCompressor,
)
from llmoe.config import Config


COMPRESSORS: dict[str, type[TokenCompressor]] = {
    # Fixed max_tokens on LLaVA; dynamic retain_ratio on Qwen. Qwen training
    # gathers labels through the same concatenated-prompt keep mask.
    "SoftTopk": SoftTopkCompressor,
    # Paper threshold presets on LLaVA/InternVL; exact final retain_ratio on Qwen.
    # InternVL applies the vision merge and decoder thresholds to one globally
    # concatenated source-image span.
    "MustDrop": MustDropCompressor,
    # LLaVA and InternVL use an absolute layer-averaged `budget`; Qwen2.5-VL uses
    # `retain_ratio` plus a ratio schedule. InternVL groups tiles before stage A.
    "ApET": ApETCompressor,
    # In-CLIP fixed max_tokens on LLaVA; post-projector fixed max_tokens over each
    # concatenated InternVL source; exact-ratio post-concatenation merging on Qwen.
    "ToMe": ToMeCompressor,
    # Fixed max_tokens on LLaVA/InternVL and retain_ratio on Qwen/NeXT. InternVL
    # and NeXT support only the training-free path; architecture-specific tuned
    # projector checkpoints are rejected.
    "VisionZip": VisionZipCompressor,
    # Checkpoint-required, native-unpadded per-image ratio tuning path for NeXT.
    "VisionZip++": VisionZipPlusPlusCompressor,
    # Qwen and LLaVA-NeXT take `retain_ratio`; LLaVA/InternVL take an absolute
    # layer-average budget. NeXT fixes the paper's k; InternVL back-solves once
    # over all source tiles.
    "FastV": FastVCompressor,
    # Batch_size=1. The published count schedule applies once to LLaVA/InternVL
    # (one concatenated source span), and as dynamic ratios on Qwen2.5-VL.
    "SparseVLM": SparseVLMCompressor,
    # Detects LLaVA-1.5, NeXT, Qwen2.5-VL, and InternVL at install(). LLaVA/InternVL
    # take max_tokens; Qwen/NeXT take retain_ratio. NeXT fixes every other released
    # method choice internally.
    "Nuwa": NuwaCompressor,
    # Detects LLaVA-1.5 vs Qwen2.5-VL at install(); the two upstream releases are
    # parameterised differently (LLaVA `max_tokens`, Qwen `retain_ratio`).
    # Its LLaVA compression-aware `tr`/`ft` recipe is also registered.
    "DUET": DUETCompressor,
    # The released DPP selector: fixed max_tokens on LLaVA/concatenated InternVL,
    # and per-image retain_ratio over Qwen or NeXT's complete packed visual span.
    "OTPrune": OTPruneCompressor,
    # LLaVA uses fixed layer-average profiles. Qwen and NeXT use one
    # layer-averaged ratio and a hidden depth-derived schedule; InternVL reuses
    # that solver with an absolute global budget.
    "PyramidDrop": PyramidDropCompressor,
    # Released max-min selector: fixed max_tokens/subset_ratio on LLaVA and
    # concatenated InternVL; per-image retain_ratio on Qwen or NeXT's packed span.
    "DivPrune": DivPruneCompressor,
    # Released calibrated fixed-grid strategy on LLaVA. Qwen uses a ratio-driven
    # adaptation; InternVL applies the absolute strategy once to its concatenated
    # tile span and resolves decoder keeps for the loaded depth.
    "VFlowOpt": VFlowOptCompressor,
    # Detects LLaVA-1.5 vs Qwen2.5-VL at install(); needs the matching released
    # glimpse-token + VIP weights, which are not interchangeable.
    "GlimpsePrune": GlimpsePruneCompressor,
    # Detects LLaVA-1.5 vs Qwen2.5-VL at install(); batch_size=1 only either way.
    # LLaVA takes the absolute `average_retained_tokens` and is trained per schedule
    # (`tr` then `ft` produce the checkpoint `compressor_args.checkpoint_path` loads);
    # Qwen2.5-VL takes `retain_ratio`; `tr`/`ft` support an in-repo adaptation of the
    # LLaVA recipe, but upstream publishes no Qwen HiDrop recipe or checkpoint.
    "HiDrop": HiDropCompressor,
    # Fixed max_tokens on LLaVA-1.5 7B/13B; per-image layer-average
    # retain_ratio over NeXT's complete packed AnyRes sequence.
    "LearnPruner": LearnPrunerCompressor,
    # Installs only on CORE's own ConvNeXt-L + OMG-Seg/Mask2Former backbone.
    # `core` retains Vicuna; `core-llama3-8b` changes only the decoder/tokenizer.
    # Llama-3 uses a per-image `retain_ratio` over the actual pre-merge F3
    # content length; legacy Vicuna/Qwen keep absolute `max_tokens` support.
    # `rate: adaptive` remains the paper's data-dependent ablation.
    "CORE": CORECompressor,
    # Released LLaVA path prunes inside CLIP under [Q-CLS] guidance and can train
    # its adapter with tr/ft. Qwen uses a training-free final-ViT
    # received-attention adaptation and a per-image retain_ratio; LLaVA adapter
    # checkpoints are rejected there.
    "QuietPrune": QuietPruneCompressor,
    # W5 variable-resolution tokenizer (text-first W partition from the TEXT and OBJECT heads) on
    # LLaVA-1.5 7B/13B and LLaVA-NeXT-8B. `Varest` is W5RELJOINT (+ [REL] pruning inside the LLM,
    # trained jointly); `VarestW5` is the tokenizer alone. Both are trained methods (released weights).
    "Varest": VarestCompressor,
    "VarestW5": VarestW5Compressor,
}
type CompressorLiteral = _Literal[
    "SoftTopk", "MustDrop", "ApET", "ToMe", "VisionZip", "VisionZip++", "FastV", "SparseVLM", "Nuwa", "DUET",
    "OTPrune", "PyramidDrop", "DivPrune", "VFlowOpt", "GlimpsePrune", "HiDrop", "CORE",
    "QuietPrune", "Varest", "VarestW5", "LearnPruner"
]


COMPRESSOR_TRAINERS: dict[str, tuple[_Callable[[_ExpConfig], str], _Callable[[VLM, Config], None]]] = {
    "SoftTopk": (soft_topk_tr_fn_cd, train_soft_topk),
    # Stage 1 of the released LLaVA-1.5 recipe: this budget's projector on 558K.
    "HiDrop": (hidrop_tr_fn_cd, train_hidrop),
    "LearnPruner": (learn_pruner_tr_fn_cd, train_learn_pruner),
    # Stage 1: projector alignment on 558K, everything else frozen.
    "CORE": (core_tr_fn_cd, train_core),
    # Adapter-only self-distillation against the frozen full-token teacher.
    "QuietPrune": (quietprune_tr_fn_cd, train_quietprune),
    # Stage 1: this budget's projector on 558K, with the DUET schedule active.
    "DUET": (duet_tr_fn_cd, train_duet),
    # LoRA (vision tower + LLM + lm_head) + projector + position tag on 665K, from the base model.
    "Varest": (varest_tr_fn_cd, train_varest),
    "VarestW5": (varest_tr_fn_cd, train_varest_w5),
}
COMPRESSOR_TUNERS: dict[str, tuple[_Callable[[_ExpConfig], str], _Callable[[VLM, Config], None]]] = {
    "SoftTopk": (soft_topk_tr_fn_cd, train_soft_topk),
    "VisionZip": (vision_zip_ft_fn_cd, finetune_vision_zip),
    "VisionZip++": (vision_zip_pp_ft_fn_cd, finetune_vision_zip_llava_next),
    # Stage 2: full finetune on 665K from the matching `tr` projector.
    "HiDrop": (hidrop_ft_fn_cd, finetune_hidrop),
    "LearnPruner": (learn_pruner_ft_fn_cd, finetune_learn_pruner),
    # Stage 2: projector + LoRA on 665K from the stage-1 projector.
    "CORE": (core_ft_fn_cd, finetune_core),
    # Same loop as `tr`, continued from the checkpoint it wrote.
    "QuietPrune": (quietprune_ft_fn_cd, finetune_quietprune),
    # Stage 2: full finetune on 665K from the matching budget's `tr` projector.
    "DUET": (duet_ft_fn_cd, finetune_duet),
    # Same loop, continued from `checkpoint_path` (or the trainer's own last snapshot).
    "Varest": (varest_tr_fn_cd, train_varest),
    "VarestW5": (varest_tr_fn_cd, train_varest_w5),
}
