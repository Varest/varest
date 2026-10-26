"""GraphDecoder sizes its static KV cache to the request (CUDA + flash-attn only)."""
import pytest
import torch

pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("GraphDecoder needs CUDA", allow_module_level=True)

from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM  # noqa: E402

from varest.graphdecode import GraphDecoder  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    return LlamaForCausalLM(cfg).to("cuda", torch.float16).eval()


def _decode(model, dec, t, max_new=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (1, t), generator=g).cuda()
    cache = DynamicCache()
    with torch.no_grad():
        h = model.model(input_ids=ids, past_key_values=cache, use_cache=True).last_hidden_state
        first = int(model.lm_head(h[:, -1:])[0, -1].argmax())
    assert dec.fits(cache, max_new)
    return dec.decode(cache, first, t, max_new, eos_id=-1)


def _decoder(model, max_capacity=4096):
    return GraphDecoder(model.model, model.lm_head, model.get_input_embeddings(), max_capacity, block=256)


def test_capacity_follows_the_request(model):
    dec = _decoder(model)
    assert dec.capacity == 0 and dec.k == []                     # nothing allocated before the first request
    _decode(model, dec, 100)                                     # needs 120 -> one block
    assert dec.capacity == 256 and dec.k[0].shape[1] == 256 and len(dec.k) == 2
    _decode(model, dec, 300)                                     # needs 320 -> grows
    assert dec.capacity == 512
    g = dec.graph
    _decode(model, dec, 400)                                     # needs 420, more than half of 512 -> reused
    assert dec.capacity == 512 and dec.graph is g
    _decode(model, dec, 100)                                     # needs 120, at most half -> shrinks
    assert dec.capacity == 256 and dec.graph is not g


def test_capacity_never_exceeds_the_cap(model):
    dec = _decoder(model, max_capacity=300)
    _decode(model, dec, 200)                                     # needs 220 -> 256
    assert dec.capacity == 256
    _decode(model, dec, 260)                                     # needs 280 -> capped at 300, not 512
    assert dec.capacity == 300


def test_reused_decoder_matches_a_fresh_one(model):
    reused = _decoder(model)
    _decode(model, reused, 900, seed=2)                          # grow to 1024 first
    a = _decode(model, reused, 100, seed=3)                      # then shrink to 256
    b = _decode(model, _decoder(model), 100, seed=3)
    assert reused.capacity == 256 and a == b and len(a) == 20
