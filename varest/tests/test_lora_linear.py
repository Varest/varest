"""varest.runtime.LoraLinear is bit-identical to peft's lora.Linear (the module the released numbers ran through)."""
import pytest
import torch
from torch import nn

from varest.runtime import LoraLinear


class _Toy(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.q_proj = nn.Linear(64, 48, bias=False).to(dtype)
        self.o_proj = nn.Linear(48, 64, bias=True).to(dtype)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


DEV = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEV)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1, 64), (2, 7, 64)])
@pytest.mark.parametrize("r,alpha", [(4, 8), (4, 6)])         # scaling 2 (folded into B) and 1.5 (not folded)
def test_matches_peft_bitwise(device, dtype, shape, r, alpha):
    from peft import LoraConfig, inject_adapter_in_model
    torch.manual_seed(0)
    ref = _Toy(dtype)
    mine = _Toy(dtype)
    mine.load_state_dict(ref.state_dict())
    inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=["q_proj", "o_proj"]), ref)
    for name in ("q_proj", "o_proj"):
        layer = getattr(ref, name)
        a = torch.randn(r, layer.base_layer.in_features).bfloat16().float()
        b = torch.randn(layer.base_layer.out_features, r).bfloat16().float()
        layer.lora_A["default"].weight.data = a.clone()          # float32 adapter, as runtime.attach_lora used to
        layer.lora_B["default"].weight.data = b.clone()
        setattr(mine, name, LoraLinear(getattr(mine, name), a.clone(), b.clone(), alpha / r))
    assert mine.q_proj.folded == (alpha / r == 2.0)
    ref.to(device); mine.to(device)
    x = torch.randn(*shape).to(dtype).to(device)
    with torch.no_grad():
        want, got = ref(x), mine(x)
    assert got.dtype == want.dtype == dtype
    assert torch.equal(got, want)


def test_detach_restores_the_base_modules():
    from varest.runtime import LoraAttachment, detach_lora
    toy = _Toy(torch.float32)
    base = toy.q_proj
    toy.q_proj = LoraLinear(base, torch.zeros(2, 64), torch.zeros(48, 2), 1.0)
    detach_lora(LoraAttachment(toy, ["q_proj"]))
    assert toy.q_proj is base


@pytest.mark.parametrize("device", DEV)
def test_shared_cast_of_one_input_matches_separate_casts(device):
    from varest.runtime import _SharedCast
    torch.manual_seed(1)
    bases = [nn.Linear(64, 32, bias=False).half().to(device) for _ in range(3)]
    ab = [(torch.randn(4, 64).bfloat16().float().to(device), torch.randn(32, 4).bfloat16().float().to(device)) for _ in range(3)]
    shared = _SharedCast(3)
    alone = [LoraLinear(bases[i], *ab[i], 2.0) for i in range(3)]
    together = [LoraLinear(bases[i], *ab[i], 2.0, shared) for i in range(3)]
    for _ in range(2):                                   # the copy is dropped after the third reader and rebuilt
        x = torch.randn(1, 5, 64).half().to(device)
        with torch.no_grad():
            for m1, m2 in zip(alone, together):
                assert torch.equal(m1(x), m2(x))
        assert shared.src is None and shared.x32 is None


@pytest.mark.parametrize("device", DEV)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [1, 4096, 32064, 700 * 4096 + 3])
def test_inplace_mixed_add_rounds_like_add_then_cast(device, dtype, n):
    """The single-kernel ``result.add_(delta)`` LoraLinear uses equals ``(result + delta).to(result.dtype)``."""
    torch.manual_seed(2)
    base = (torch.randn(n, device=device) * 30).to(dtype)
    delta = torch.randn(n, device=device) * 1e-2
    delta[::7] = torch.randn(len(delta[::7]), device=device) * 1e4      # large updates too
    want = (base + delta).to(dtype)
    assert torch.equal(base.clone().add_(delta), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
def test_graphed_call_mixes_inference_mode_and_no_grad():
    """A graph first captured under inference_mode must still accept calls outside it (and vice versa)."""
    from varest.graphs import GraphedCall
    lin = nn.Linear(16, 8).cuda()
    g = GraphedCall(lambda x: lin(x) * 2)
    x = torch.randn(4, 16, device="cuda")
    with torch.inference_mode():
        a = g(x)
    with torch.no_grad():
        b = g(x)
        want = lin(x) * 2
    assert torch.equal(a, want) and torch.equal(b, want)
