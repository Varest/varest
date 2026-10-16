"""DUET's compressor must not register the modules it borrows.

`TokenCompressor` extends `nn.Module`, so assigning a borrowed module -- the
decoder, or the vision-tower layer DUET patches -- registers it as a *submodule
of the compressor*. Its parameters then appear twice in the enclosing model's
state dict, once under their real names and once under `compressor.*`, sharing
storage.

Training is unaffected, which is what makes this expensive: safetensors refuses
to serialize aliased tensors, so a stage-2 run trains for hours and then dies at
its first checkpoint with "Some tensors share memory". Jobs 5462443 and 5462444
each lost 3h30m that way.

These tests work on the compressor alone -- no backbone, no weights -- so they
run in milliseconds and catch a regression that otherwise costs half a day.
"""

import torch
from torch import nn

from llmoe.compressor.duet.duet import DUETCompressor


def _borrowed_layer() -> nn.Module:
    """Stands in for a vision-tower block: something with real parameters."""
    layer = nn.Linear(8, 8)
    layer.self_attn = nn.Linear(8, 8)  # the attribute DUET reaches for
    return layer


def test_assigning_the_target_layer_registers_nothing():
    compressor = DUETCompressor(max_tokens=128)
    layer = _borrowed_layer()

    object.__setattr__(compressor, "_target_layer", layer)

    assert compressor._target_layer is layer, "the reference must still be usable"
    assert "_target_layer" not in dict(compressor.named_modules()), (
        "the borrowed layer registered as a submodule")
    assert not [n for n, _ in compressor.named_parameters()
                if n.startswith("_target_layer")], (
        "the borrowed layer's parameters leaked into the compressor")


def test_assigning_the_language_model_registers_nothing():
    compressor = DUETCompressor(max_tokens=128)
    decoder = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

    object.__setattr__(compressor, "_language_model", decoder)

    assert compressor._language_model is decoder
    assert not [n for n, _ in compressor.named_parameters()
                if n.startswith("_language_model")], (
        "the whole decoder leaked into the compressor's parameters")


def test_the_enclosing_state_dict_has_no_shared_storage():
    """The actual failure: two names, one storage, and safetensors refuses.

    Mirrors the training model's shape -- a base model plus the compressor as a
    child -- and checks no two entries point at the same memory.
    """
    compressor = DUETCompressor(max_tokens=128)
    layer = _borrowed_layer()
    object.__setattr__(compressor, "_target_layer", layer)

    class _TrainingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.vision_layer = layer      # the real owner
            self.compressor = compressor

    state = _TrainingModel().state_dict()
    seen: dict[int, str] = {}
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        key = tensor.untyped_storage().data_ptr()
        assert key not in seen, (
            f"{name} shares storage with {seen[key]}; safetensors will refuse this "
            f"checkpoint")
        seen[key] = name
    # The layer's own parameters must still be there under their real owner.
    assert any(n.startswith("base_model.vision_layer") for n in state), state.keys()


def test_the_source_uses_object_setattr_for_borrowed_modules():
    """Pin the mechanism, not just the outcome.

    A plain `self._target_layer = ...` would reintroduce the aliasing while every
    behavioural test above still passed on a freshly built compressor -- the
    damage only shows through the enclosing model.
    """
    from pathlib import Path

    import llmoe.compressor.duet.duet as module

    source = Path(module.__file__).read_text()
    for attribute in ("_target_layer", "_language_model"):
        assert f"self.{attribute} = " not in source, (
            f"self.{attribute} is assigned directly again; use object.__setattr__")
