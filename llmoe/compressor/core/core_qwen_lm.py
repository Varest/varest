"""CORE with Qwen's language model in place of Vicuna.

CORE (arXiv:2511.14072) is a whole VLM: a frozen ConvNeXt-L produces ``F_3``, a
frozen OMG-Seg Mask2Former head produces object masks over the same grid, Eq. (1)
merges ``F_3`` per object, a projector lifts the merged tokens into the decoder's
embedding space, and a Vicuna-7B decoder answers. Porting it to Qwen means
swapping *that decoder*, not the vision stack -- the object-centric merge is the
method and stays exactly as published.

Which is why the port still has to train. The projector's whole job is to map
ConvNeXt's 1536-d features into the decoder's embedding space; change the decoder
and that mapping is wrong, so stage 1 (projector alignment) and stage 2
(projector + LoRA instruction tuning) are as necessary for Qwen as they were for
Vicuna. Both stages run on the full ``F_3`` grid -- CORE never trains under a
token budget, because the budget is an evaluation-time strategy -- and that is
unchanged here.

Which Qwen LM
-------------
The decoder is the language half of ``Qwen2.5-VL-7B-Instruct``, not the separate
``Qwen2.5-7B-Instruct`` text model, so CORE's decoder weights are literally the
ones the other four baselines run on and the sweep compares vision front-ends
rather than two different LLMs. The checkpoint stores that half under the plain
``model.layers`` / ``model.embed_tokens`` / ``model.norm`` / ``lm_head`` names a
``Qwen2ForCausalLM`` expects, so it loads directly and the ``visual.*`` tensors
are simply left behind.

Loading it as a *text* causal LM also drops M-RoPE for ordinary 1-D RoPE, which
is correct rather than merely convenient: M-RoPE reduces to 1-D RoPE whenever its
three channels carry the same position, and CORE emits a flat, centroid-sorted
sequence of object tokens with no 2-D grid to address. That flat sequence over
1-D positions is exactly what the published method feeds Vicuna.
"""

from pathlib import Path
from typing import Any

import torch

__all__ = [
    "QWEN_LLM_HIDDEN_SIZE",
    "QWEN_LM_MODEL_TYPES",
    "llm_family",
    "load_qwen_causal_lm",
    "load_qwen_tokenizer",
    "validate_qwen_tokenizer",
    "format_qwen_inference_prompt",
]

#: Hidden width of Qwen2.5-VL-7B's decoder, and so of CORE's projector output.
QWEN_LLM_HIDDEN_SIZE = 3584

#: `model_type` values that mean "this is a Qwen decoder".
QWEN_LM_MODEL_TYPES = frozenset(
    {"qwen2", "qwen2_vl", "qwen2_5_vl", "qwen2_vl_text", "qwen2_5_vl_text", "qwen3"}
)

#: Qwen's ChatML system turn. CORE's Vicuna prompt has no counterpart here; the
#: decoder was tuned under ChatML and answers poorly outside it.
QWEN_SYSTEM_PROMPT = "You are a helpful assistant."


def llm_family(
    name_or_path: str,
    *,
    local_files_only: bool = True,
    revision: str | None = None,
) -> str:
    """Return ``"llama"`` (Vicuna), ``"llama3"``, or ``"qwen"``.

    Read from the checkpoint's own ``model_type`` rather than the path, so a
    renamed local directory cannot silently select the wrong recipe.
    """
    from transformers import AutoConfig

    from llmoe.compressor.core.core_llama3_lm import (
        looks_like_llama3,
        resolve_llama3_source_path,
        text_config,
    )

    kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": local_files_only,
    }
    resolved_source = resolve_llama3_source_path(name_or_path, revision)
    if revision and not Path(resolved_source).expanduser().exists():
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(resolved_source, **kwargs)
    model_type = str(getattr(config, "model_type", "") or "")
    nested = text_config(config)
    if looks_like_llama3(nested):
        return "llama3"
    if model_type == "llama":
        return "llama"
    if model_type in QWEN_LM_MODEL_TYPES:
        return "qwen"
    raise TypeError(
        f"CORE's decoder must be Vicuna, Llama-3, or Qwen; "
        f"{name_or_path!r} has model_type={model_type!r}."
    )


def _text_config(name_or_path: str, *, local_files_only: bool) -> Any:
    """The decoder half of a config, whether or not it is wrapped in a VL one."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        name_or_path, trust_remote_code=False, local_files_only=local_files_only
    )
    return getattr(config, "text_config", None) or config


def load_qwen_causal_lm(config: Any, *, dtype: torch.dtype | None,
                        local_files_only: bool,
                        attention_implementation: str) -> Any:
    """Load Qwen's decoder as a plain causal LM, dropping the vision tower."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    source = str(config.llm_name_or_path)
    text = _text_config(source, local_files_only=local_files_only)

    # Rebuilt rather than reused: the VL text config carries `rope_scaling` with
    # M-RoPE sections, and this decoder runs on 1-D positions (see the module
    # docstring). Passing that through would configure a rotary embedding the
    # flat sequence never exercises.
    lm_config = Qwen2Config(
        vocab_size=int(text.vocab_size),
        hidden_size=int(text.hidden_size),
        intermediate_size=int(text.intermediate_size),
        num_hidden_layers=int(text.num_hidden_layers),
        num_attention_heads=int(text.num_attention_heads),
        num_key_value_heads=int(text.num_key_value_heads),
        max_position_embeddings=int(getattr(text, "max_position_embeddings", 32768)),
        rms_norm_eps=float(getattr(text, "rms_norm_eps", 1e-6)),
        rope_theta=float(getattr(text, "rope_theta", 1000000.0)),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        attention_dropout=float(getattr(text, "attention_dropout", 0.0)),
    )
    lm_config._attn_implementation = attention_implementation

    # The VL checkpoint's `visual.*` tensors have no counterpart here and are
    # reported as unexpected; that is the point, not a problem.
    model = Qwen2ForCausalLM.from_pretrained(
        source,
        config=lm_config,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation=attention_implementation,
        local_files_only=local_files_only,
    )

    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size != int(config.llm_hidden_size):
        raise RuntimeError(
            f"Qwen decoder hidden size {hidden_size} does not match CORE's projector "
            f"output {config.llm_hidden_size}; set llm_hidden_size="
            f"{hidden_size} (QWEN_LLM_HIDDEN_SIZE)."
        )
    return model


def load_qwen_tokenizer(config: Any, *, local_files_only: bool) -> Any:
    """Qwen's tokenizer, which unlike Vicuna's has no slow SentencePiece form."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.llm_name_or_path),
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    validate_qwen_tokenizer(tokenizer)
    return tokenizer


def validate_qwen_tokenizer(tokenizer: Any) -> None:
    """Fail closed when a tokenizer is not the ChatML one CORE/Qwen assumes.

    Vicuna's check requires ``</s>``; Qwen ends a turn with ``<|im_end|>`` and has
    no BOS at all, so the Vicuna validator rejects a perfectly good tokenizer.
    """
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("Qwen tokenizer is missing eos_token_id")
    end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_of_turn is None or int(end_of_turn) < 0:
        raise ValueError(
            "CORE/Qwen expects a ChatML tokenizer defining <|im_end|>; this one "
            "does not, so turn boundaries would not be maskable."
        )


def format_qwen_inference_prompt(image_token: str, prompt: str) -> str:
    """The single-image ChatML prompt used for benchmark inference."""
    return (
        f"<|im_start|>system\n{QWEN_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{image_token}\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
