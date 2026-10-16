"""CORE with Qwen's decoder in place of Vicuna.

The port swaps the *decoder*, not the vision stack: ConvNeXt-L, the OMG-Seg head
and Eq. (1) are unchanged, and the projector that lifts F_3 into the decoder's
embedding space is what has to be retrained. These cover the three places the
decoder's identity leaks into the code -- how it is loaded, how its tokenizer is
validated, and which chat template stage 2 renders.
"""

import pytest
import torch

from llmoe.compressor.core.core import CoreConfig, format_inference_prompt
from llmoe.compressor.core.core_qwen_lm import (
    QWEN_LLM_HIDDEN_SIZE,
    format_qwen_inference_prompt,
    llm_family,
    validate_qwen_tokenizer,
)

QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
HUB = ("/space/hall7/sitestore/eccc/mrd/rpnet/awa001/workspace/output/"
       "model_weights/hf_cache/hub")
VICUNA = ("/space/hall7/sitestore/eccc/mrd/rpnet/awa001/workspace/input/"
          "datasets/vicuna-7b-v1.5")


def _qwen_tokenizer(register_image_token: bool = True):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True,
                                                  cache_dir=HUB)
    except Exception as exc:  # noqa: BLE001 - a cache miss is the only story
        pytest.skip(f"Qwen tokenizer not cached: {type(exc).__name__}")
    # CORE registers its sentinel and resizes the embedding before tokenizing
    # anything (core.py ~1869, core_tr.py ~564). Qwen's vocabulary has no
    # <image>, so a test that skips this step is testing the wrong object.
    if register_image_token and "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    return tokenizer


# ------------------------------------------------------------------ family


def test_family_is_read_from_the_checkpoint_not_the_path(tmp_path, monkeypatch):
    """A renamed directory must not select the wrong recipe."""
    import transformers

    class _Cfg:
        model_type = "qwen2_5_vl"

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _Cfg()))
    assert llm_family("/somewhere/called-vicuna") == "qwen"


def test_llama_checkpoints_stay_on_the_vicuna_path(monkeypatch):
    import transformers

    class _Cfg:
        model_type = "llama"

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _Cfg()))
    assert llm_family("/anything") == "llama"


def test_an_unsupported_decoder_is_refused(monkeypatch):
    import transformers

    class _Cfg:
        model_type = "gpt2"

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _Cfg()))
    with pytest.raises(TypeError, match="Vicuna, Llama-3, or Qwen"):
        llm_family("/anything")


# --------------------------------------------------------------- tokenizer


def test_qwen_tokenizer_passes_its_own_validator():
    """The Vicuna validator demands </s>, which Qwen does not have."""
    from llmoe.compressor.core.core import validate_tokenizer

    tokenizer = _qwen_tokenizer(register_image_token=False)
    validate_qwen_tokenizer(tokenizer)          # the right check passes...
    with pytest.raises(ValueError):             # ...and the Vicuna one would not
        validate_tokenizer(tokenizer)


def test_a_tokenizer_without_chatml_is_refused():
    class _NoChatML:
        eos_token_id = 2

        def convert_tokens_to_ids(self, token):
            return -1

    with pytest.raises(ValueError, match="<\\|im_end\\|>"):
        validate_qwen_tokenizer(_NoChatML())


# ------------------------------------------------------------------ prompt


def test_inference_prompt_follows_the_decoder():
    vicuna = format_inference_prompt("<image>", "What is this?")
    qwen = format_inference_prompt("<image>", "What is this?", "qwen")
    assert "USER:" in vicuna and "ASSISTANT:" in vicuna
    assert "<|im_start|>" in qwen and "<|im_end|>" in qwen
    assert qwen == format_qwen_inference_prompt("<image>", "What is this?")
    # The default must stay Vicuna so existing callers are untouched.
    assert format_inference_prompt("<image>", "What is this?") == vicuna


def test_qwen_prompt_ends_ready_for_the_answer():
    prompt = format_qwen_inference_prompt("<image>", "Describe it.")
    assert prompt.endswith("<|im_start|>assistant\n")
    assert prompt.count("<image>") == 1


# ------------------------------------------------------- stage-2 rendering


def _collator(family: str, tokenizer):
    from llmoe.compressor.core.core_tr import CoreDataCollator

    return CoreDataCollator(
        tokenizer, lambda images: torch.zeros(len(images), 3, 8, 8), image_root="/tmp",
        image_token="<image>", max_length=512, visual_tokens=4, stage=2,
        llm_family=family)


def test_stage2_renders_chatml_for_a_qwen_decoder():
    tokenizer = _qwen_tokenizer()
    collator = _collator("qwen", tokenizer)
    turns = [("user", "<image>\nWhat colour?"), ("assistant", "Red.")]

    ids, labels = collator._render_ids_labels(turns, has_image=True)

    text = tokenizer.decode(ids)
    assert "<|im_start|>user" in text and "<|im_start|>assistant" in text
    assert "USER:" not in text
    assert len(ids) == len(labels)
    # Only the answer is supervised.
    supervised = tokenizer.decode([i for i, l in zip(ids, labels) if l != -100])
    assert "Red." in supervised
    assert "What colour?" not in supervised


def test_image_placeholders_are_never_supervised():
    tokenizer = _qwen_tokenizer()
    collator = _collator("qwen", tokenizer)
    turns = [("user", "<image>\nWhat colour?"), ("assistant", "Red.")]

    ids, labels = collator._render_ids_labels(turns, has_image=True)

    image_id = int(tokenizer.convert_tokens_to_ids("<image>"))
    assert all(label == -100 for token, label in zip(ids, labels)
               if token == image_id)


def test_collator_rejects_an_unknown_family():
    tokenizer = _qwen_tokenizer()
    with pytest.raises(ValueError, match="llm_family"):
        _collator("mistral", tokenizer)


# ------------------------------------------------------------------- config


def test_projector_width_must_match_the_decoder():
    """CoreConfig's default is Vicuna's 4096; Qwen's decoder is 3584."""
    assert QWEN_LLM_HIDDEN_SIZE == 3584
    assert CoreConfig().llm_hidden_size == 4096
    config = CoreConfig(llm_name_or_path=QWEN, llm_hidden_size=QWEN_LLM_HIDDEN_SIZE)
    assert config.llm_hidden_size == 3584


def test_the_sentinel_must_be_registered_before_rendering():
    """Qwen's vocabulary has no <image>; CORE adds it and resizes by one row.

    Pinned because forgetting it fails deep inside tokenization with a bare
    `int(None)` rather than anything that names the cause.
    """
    tokenizer = _qwen_tokenizer(register_image_token=False)
    assert "<image>" not in tokenizer.get_vocab()
    collator = _collator("qwen", tokenizer)
    with pytest.raises(TypeError):
        collator._render_ids_labels([("user", "<image>\nq"), ("assistant", "a")],
                                    has_image=True)
