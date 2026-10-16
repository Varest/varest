# Varest

## What is Varest?

Varest is a variable-resolution tokenizer, guided by semantic cues and complemented by learned in-LLM relevance pruning.
This repository is the official implementation of [*Just Respect the Semantics: Image-Token Compression via Pre-LLM
Semantic Variable-Resolution Tokenization and In-LLM Relevance Pruning*](https://arxiv.org/abs/2610.18690).

### Beyond a Varest

We have noticed that the field of visual token compression has a lot of confounded evaluation protocols, including
mismatched token accounting (final token count versus layer-averaged token count versus pre-LLM token count), privileged
capabilities (usually by replacing the vision tower or the language model with a more powerful alternative but comparing
with baselines that do not have such advantages), and privileged exposure to unseen data (training on a data
distribution that the base model or baselines have not seen). To address these integrity issues, we present Varest not
just as a new compressor but together with LLMOE, a fair and transparent benchmarking framework across 15+ methods. It
is also easy to extend LLMOE to many other compression methods.

It contains two packages: `llmoe` and `varest`. `varest` defines a series of variable-resolution tokenization methods
proposed and tested in our paper. `llmoe` provides an executable pipeline for training, fine-tuning, and benchmarking
(inference + evaluation).

## Getting Started

This codebase is huge. To get started, you can read the documentation and examples in [docs/](docs/), but the best way is
to throw it to your AI agent.