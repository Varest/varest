"""Llama-3-8B-Instruct as the decoder of the unchanged CORE vision model.

CORE is a complete VLM.  Its frozen OpenCLIP ConvNeXt-L supplies both the
``F_3`` tokens consumed by the language model and the feature pyramid consumed
by the frozen OMG-Seg Mask2Former head.  Replacing the decoder must not replace
that vision stack with LLaVA-NeXT's CLIP-ViT.  This module therefore extracts
only the Llama-3 language weights and tokenizer from the pinned converted
LLaVA-NeXT checkpoint; the checkpoint's vision tower, projector, AnyRes packing,
and image-newline parameter are deliberately not loaded.

The checkpoint stores the decoder under ``language_model.*``.  Transformers'
``key_mapping`` streams those tensors into a plain ``LlamaForCausalLM`` while
leaving the unrelated multimodal tensors on disk.  No converted copy of the 8B
weights is required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


DEFAULT_CORE_LLAMA3_MODEL = "llava-hf/llama3-llava-next-8b-hf"
DEFAULT_CORE_LLAMA3_REVISION = "b041c0d0ea0dd0196d147206c210c8d1752fc2da"
CORE_LLAMA3_IMPLEMENTATION = "core-convnext-omgseg-llama3-8b-v3"
LLAMA3_LLM_HIDDEN_SIZE = 4096
LLAMA3_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LLAMA3_TRAINING_RECIPES = ("llava_plain", "llama3_instruct")
CORE_LLAMA3_SOURCE_COMMITS = {
    "core_paper": "arXiv:2511.14072",
    "official_core": "1a86030f0051ae85eb093536a8bd83fd985dddfc",
    "projectneura_core": "f86af1848f1fbc2e079007faf2d2cf6a4f966fe3",
    "projectneura_trained_reference": "254edd76a76c87da4f1bc408d57097b72da2fe1f",
    "official_nuwa": "175b7d57a7890b9818dd26d82f2d91d9868b0d80",
    "varest_integration_base": "475fc2d768132d39dd8974a41e4da3df1e572ef6",
    "target_model": DEFAULT_CORE_LLAMA3_REVISION,
    "transformers": "75d3bdcd4b3ba70cee21287218d9764f33da41f0",
    "conrad_launcher_reference": "1d17b95",
}

_EXPECTED_TEXT = {
    "model_type": "llama",
    "vocab_size": 128320,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 8192,
    "bos_token_id": 128000,
    "eos_token_id": 128009,
}


def resolve_llama3_source_path(
    name_or_path: str,
    revision: str | None,
) -> str:
    """Resolve the repository's flat HF cache without downloading anything.

    Varest sets ``HF_HOME`` to ``<root>/output/hf_cache`` and the supplied cache
    stores ``models--...`` directly there (rather than below an extra ``hub/``
    directory).  Hugging Face normally derives ``HF_HUB_CACHE=$HF_HOME/hub``.
    Resolve the immutable snapshot explicitly so ordinary CLI runs find the
    provided files even when a launcher did not also export ``HF_HUB_CACHE``.
    """

    direct = Path(str(name_or_path)).expanduser()
    if direct.exists() or name_or_path != DEFAULT_CORE_LLAMA3_MODEL or not revision:
        return str(direct) if direct.exists() else str(name_or_path)
    repository = "models--llava-hf--llama3-llava-next-8b-hf"
    roots = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME")]
    for raw_root in roots:
        if not raw_root:
            continue
        root = Path(raw_root).expanduser()
        for cache_root in (root, root / "hub"):
            candidate = cache_root / repository / "snapshots" / str(revision)
            if (candidate / "config.json").is_file():
                return str(candidate)
    return str(name_or_path)


def _source_kwargs(
    config: Any,
    *,
    local_files_only: bool,
    source_path: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": local_files_only,
    }
    source = Path(source_path or str(config.llm_name_or_path)).expanduser()
    revision = getattr(config, "llm_revision", None)
    if revision and not source.exists():
        kwargs["revision"] = str(revision)
    return kwargs


def source_config(config: Any, *, local_files_only: bool) -> Any:
    """Load metadata only for the decoder source."""

    from transformers import AutoConfig

    source_path = resolve_llama3_source_path(
        str(config.llm_name_or_path), getattr(config, "llm_revision", None)
    )
    return AutoConfig.from_pretrained(
        source_path,
        **_source_kwargs(
            config, local_files_only=local_files_only, source_path=source_path
        ),
    )


def text_config(source: Any) -> Any:
    """Return a wrapper checkpoint's language config or a plain config itself."""

    return getattr(source, "text_config", None) or source


def looks_like_llama3(text: Any) -> bool:
    """Distinguish Llama-3 from Vicuna without relying on a directory name."""

    return (
        getattr(text, "model_type", None) == "llama"
        and int(getattr(text, "vocab_size", 0)) == 128320
        and int(getattr(text, "bos_token_id", -1)) == 128000
        and int(getattr(text, "eos_token_id", -1)) == 128009
    )


def validate_llama3_source(source: Any) -> Any:
    """Fail closed unless ``source`` contains the assigned Llama-3-8B decoder."""

    text = text_config(source)
    differences = {
        name: (getattr(text, name, None), expected)
        for name, expected in _EXPECTED_TEXT.items()
        if getattr(text, name, None) != expected
    }
    rope_parameters = getattr(text, "rope_parameters", None) or {}
    rope_theta = float(
        getattr(text, "rope_theta", None)
        or rope_parameters.get("rope_theta", 0.0)
        or 0.0
    )
    if rope_theta != 500000.0:
        differences["rope_theta"] = (rope_theta, 500000.0)
    rms_norm_eps = float(getattr(text, "rms_norm_eps", 0.0) or 0.0)
    if rms_norm_eps != 1e-5:
        differences["rms_norm_eps"] = (rms_norm_eps, 1e-5)
    if differences:
        raise ValueError(f"CORE requires the pinned Llama-3-8B-Instruct decoder: {differences}")
    return text


def validate_llama3_tokenizer(tokenizer: Any) -> None:
    """Validate the exact Llama-3 turn and image-token vocabulary."""

    expected = {
        "bos_token_id": 128000,
        "eos_token_id": 128009,
        "pad_token_id": 128257,
    }
    differences = {
        name: (getattr(tokenizer, name, None), value)
        for name, value in expected.items()
        if getattr(tokenizer, name, None) != value
    }
    image_id = tokenizer.convert_tokens_to_ids("<image>")
    if image_id != 128256:
        differences["image_token_id"] = (image_id, 128256)
    special_ids = {
        "<|start_header_id|>": 128006,
        "<|end_header_id|>": 128007,
        "<|eot_id|>": 128009,
    }
    for token, expected_id in special_ids.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != expected_id:
            differences[token] = (token_id, expected_id)
    if not getattr(tokenizer, "chat_template", None):
        differences["chat_template"] = (None, "Llama-3 Instruct template")
    if differences:
        raise ValueError(f"Incompatible Llama-3 tokenizer: {differences}")


def load_llama3_tokenizer(config: Any, *, local_files_only: bool) -> Any:
    """Load the tokenizer shipped with the pinned Llama-3 LLaVA-NeXT source."""

    from transformers import AutoTokenizer

    source = source_config(config, local_files_only=local_files_only)
    validate_llama3_source(source)
    source_path = resolve_llama3_source_path(
        str(config.llm_name_or_path), getattr(config, "llm_revision", None)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source_path,
        use_fast=True,
        **_source_kwargs(
            config, local_files_only=local_files_only, source_path=source_path
        ),
    )
    validate_llama3_tokenizer(tokenizer)
    return tokenizer


def format_llama3_inference_prompt(
    tokenizer: Any,
    image_token: str,
    prompt: str,
) -> str:
    """Render one image/user turn with the source checkpoint's chat template."""

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{image_token}\n{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered.count(image_token) != 1:
        raise RuntimeError("Llama-3 chat template did not preserve exactly one image marker")
    return str(rendered)


def tokenize_llama3_image_prompt(
    tokenizer: Any,
    prompt: str,
    image_token: str,
    *,
    repetitions: int,
) -> list[int]:
    """Tokenize the already-templated prompt and expand its image marker.

    ``apply_chat_template(tokenize=False)`` writes ``<|begin_of_text|>`` into
    the returned string. Calling the tokenizer with its normal
    ``add_special_tokens=True`` would prepend a second BOS. Tokenize the complete
    rendered template without automatic specials, then replace its one registered
    image-token ID; this also preserves BPE boundaries around the marker.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    encoded = tokenizer(prompt, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("CORE expected one Llama-3 inference prompt")
        ids = ids[0]
    ids = [int(token) for token in ids]
    image_id = int(tokenizer.convert_tokens_to_ids(image_token))
    positions = [index for index, token in enumerate(ids) if token == image_id]
    if len(positions) != 1:
        raise ValueError(
            f"CORE expects exactly one {image_token!r} marker, got {len(positions)}"
        )
    position = positions[0]
    return ids[:position] + [image_id] * repetitions + ids[position + 1 :]


def _load_prefixed_llama(
    source_path: str,
    source: Any,
    *,
    dtype: torch.dtype | None,
    local_files_only: bool,
    attention_implementation: str,
    revision: str | None,
) -> Any:
    """Stream only ``language_model.*`` into a plain Llama causal LM."""

    from transformers import LlamaConfig, LlamaForCausalLM

    nested = validate_llama3_source(source)
    lm_config = LlamaConfig.from_dict(nested.to_dict())
    lm_config._attn_implementation = attention_implementation
    kwargs: dict[str, Any] = {
        "config": lm_config,
        "trust_remote_code": False,
        "dtype": dtype,
        "attn_implementation": attention_implementation,
        "local_files_only": local_files_only,
        "output_loading_info": True,
    }
    if revision and not Path(source_path).expanduser().exists():
        kwargs["revision"] = revision
    if getattr(source, "model_type", None) != "llama":
        kwargs["key_mapping"] = {r"^language_model\.": ""}
    model, loading = LlamaForCausalLM.from_pretrained(source_path, **kwargs)

    # A wrapper source contains only these non-language tensors. Anything still
    # missing from the Llama model, or any unexpected language tensor, means the
    # prefix mapping or source revision no longer matches this implementation.
    missing = [
        key
        for key in loading.get("missing_keys", ())
        if not key.endswith("rotary_emb.inv_freq")
    ]
    unexpected_language = [
        key
        for key in loading.get("unexpected_keys", ())
        if key.startswith("language_model.")
    ]
    mismatched = list(loading.get("mismatched_keys", ()))
    if missing or unexpected_language or mismatched:
        raise RuntimeError(
            "Could not extract the Llama-3 decoder from the multimodal checkpoint: "
            f"missing={missing}, unexpected_language={unexpected_language}, "
            f"mismatched={mismatched}"
        )
    return model


def load_llama3_causal_lm(
    config: Any,
    *,
    dtype: torch.dtype | None,
    local_files_only: bool,
    attention_implementation: str,
) -> Any:
    """Load only Llama-3-8B from the pinned source, never its CLIP components."""

    from llmoe.compressor.core.core import model_attention_implementation

    source_path = resolve_llama3_source_path(
        str(config.llm_name_or_path), getattr(config, "llm_revision", None)
    )
    source = source_config(config, local_files_only=local_files_only)
    model = _load_prefixed_llama(
        source_path,
        source,
        dtype=dtype,
        local_files_only=local_files_only,
        attention_implementation=attention_implementation,
        revision=getattr(config, "llm_revision", None),
    )
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size != int(config.llm_hidden_size):
        raise RuntimeError(
            f"Llama-3 hidden size {hidden_size} does not match CORE projector output "
            f"{config.llm_hidden_size}"
        )
    actual = model_attention_implementation(model)
    if actual != attention_implementation:
        raise RuntimeError(
            "Llama-3 did not activate the requested attention backend: "
            f"requested {attention_implementation!r}, got {actual!r}"
        )
    return model


__all__ = [
    "CORE_LLAMA3_IMPLEMENTATION",
    "DEFAULT_CORE_LLAMA3_MODEL",
    "DEFAULT_CORE_LLAMA3_REVISION",
    "LLAMA3_LLM_HIDDEN_SIZE",
    "LLAMA3_LORA_TARGETS",
    "LLAMA3_TRAINING_RECIPES",
    "format_llama3_inference_prompt",
    "load_llama3_causal_lm",
    "load_llama3_tokenizer",
    "looks_like_llama3",
    "resolve_llama3_source_path",
    "source_config",
    "text_config",
    "validate_llama3_source",
    "validate_llama3_tokenizer",
]
