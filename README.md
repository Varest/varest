<div align="center">

<h1>Varest</h1>

<p><strong>Just Respect the Semantics: Image-Token Compression via Pre-LLM Semantic Variable-Resolution Tokenization and In-LLM Relevance Pruning</strong></p>

<p style="width: 80%">
  Tianhao&nbsp;Fu<sup>†,&nbsp;1,&nbsp;2,&nbsp;</sup>* · Austin&nbsp;Wang<sup>†,&nbsp;3</sup> · Charles&nbsp;Chen<sup>†,&nbsp;1,&nbsp;4</sup>, Roby&nbsp;Aldave-Garza<sup>5</sup>, Victor&nbsp;Xie<sup>3</sup>
</p>
<p><sup>†</sup> Equal contribution. &nbsp; <sup>†</sup>Project lead with correspondence.</p>
<p style="width: 80%"><span><sup>1</sup>&nbsp;University of Toronto</span>&nbsp; <span><sup>2</sup>&nbsp;Vector Institute</span>&nbsp; <span><sup>3</sup>&nbsp;McGill University</span>&nbsp; <span><sup>4</sup>&nbsp;University Health Network</span>&nbsp; <span><sup>5</sup>&nbsp;University of Waterloo</span></p>
<p>
  <a href="https://arxiv.org/abs/2610.18690"><img src="https://img.shields.io/badge/arXiv-2610.18690-b31b1b.svg?style=flat-square" alt="arXiv: 2610.18690"></a>
  <a href="https://varest.github.io"><img src="https://img.shields.io/badge/Project%20Website-black.svg?style=flat-square&amp;logo=github&amp;logoColor=white" alt="Project Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-478A68.svg?style=flat-square" alt="License: MIT"></a>
</p>

</div>

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