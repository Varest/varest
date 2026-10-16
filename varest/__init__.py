"""Varest: variable-resolution visual tokens for LLaVA (W5 and W5RELJOINT).

A budget of K visual tokens is spent unevenly: a TEXT head and an OBJECT head read the image's pixels, and the W5
tokenizer (:mod:`varest.tokenizer`) partitions the CLIP cell grid text-first -- text pinned at native resolution when
the budget allows, the rest as object regions -- emitting one mean-pooled, position-tagged token per region.
W5RELJOINT adds learned [REL] registers that prune the visual tokens inside the language model (:mod:`varest.rel`).

  varest.heads         the TEXT head (v5n) and the OBJECT head (w96h288)
  varest.fields        per-image text / boundary fields for LLaVA-1.5 and both LLaVA-NeXT views
  varest.tokenizer     the W5 partition and region pooling
  varest.next          LLaVA-NeXT AnyRes geometry and retain-ratio budgets
  varest.postag        the position tag
  varest.rel           [REL]: scoring, keep-masked training attention, pruned decoding
  varest.checkpoint    the release format (one file per backbone and variant)
  varest.runtime       attach a checkpoint to a Hugging Face model; build visual tokens
  varest.engine        end-to-end generation (image, prompt, budget) -> answer
  varest.swa           the SWA5 average of the released weights
  varest.train         OCR labels, head training, DINOv3 targets, field caches, LLaVA fine-tuning
"""
from varest.checkpoint import VarestCheckpoint
from varest.engine import VarestEngine
from varest.rel import budget_split
from varest.tokenizer import build_partition, pin_threshold, pool_regions

__all__ = ["VarestCheckpoint", "VarestEngine", "budget_split", "build_partition", "pin_threshold", "pool_regions"]
