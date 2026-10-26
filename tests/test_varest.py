"""Varest / VarestW5 in the harness: registry, budgets, weight resolution, and a CPU end-to-end install -> generate ->
uninstall on a tiny random LLaVA (the backbone's parameters must come back bit-identical)."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from llmoe.compressor import COMPRESSORS, COMPRESSOR_TRAINERS, COMPRESSOR_TUNERS
from llmoe.compressor.varest import VarestCompressor, VarestW5Compressor
from llmoe.weights import ARTIFACTS

from varest.checkpoint import VarestCheckpoint
from varest.heads.object import ObjectHead
from varest.heads.text import TextHead
from varest.postag import PosTag
from varest.rel import RelHead


class TrainingCorpusTest(unittest.TestCase):
    """Each backbone fine-tunes on its own instruction-tuning set; SWA5 takes the last four snapshots of that run."""

    def test_swa_steps(self) -> None:
        from varest.swa import SWA_STEPS, swa_steps
        self.assertEqual(swa_steps(624609), (512000, 544000, 576000, 608000))     # LLaVA-1.5, 665K mix
        self.assertEqual(swa_steps(738590), (640000, 672000, 704000, 736000))     # LLaVA-NeXT, LLaVA-NeXT-Data
        self.assertEqual(swa_steps(640000), (512000, 544000, 576000, 608000))     # a final step on a snapshot is excluded
        self.assertEqual(SWA_STEPS, swa_steps(624609))
        with self.assertRaises(ValueError):
            swa_steps(100000)

    def test_corpus_per_backbone(self) -> None:
        from llmoe.compressor.varest.varest_tr import corpus
        from varest.train.dump_fields import MIX_JSON
        self.assertEqual(corpus("/in", "llava-1.5-7b"), ("/in/LLaVA-Tuning", MIX_JSON, 624609))
        self.assertEqual(corpus("/in", "llava-1.5-13b"), ("/in/LLaVA-Tuning", MIX_JSON, 624609))
        self.assertEqual(corpus("/in", "llama3-llava-next-8b"),
                         ("/in/LLaVA-NeXT-Data/llava_next_raw_format/images",
                          "/in/LLaVA-NeXT-Data/llava_next_raw_format/llava_next_raw_format_processed.json", 738590))


class RegistryTest(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(COMPRESSORS["Varest"], VarestCompressor)
        self.assertIs(COMPRESSORS["VarestW5"], VarestW5Compressor)
        for name in ("Varest", "VarestW5"):
            self.assertIn(name, COMPRESSOR_TRAINERS)
            self.assertIn(name, COMPRESSOR_TUNERS)
        a = ARTIFACTS[("Varest", "llava", None)]
        self.assertEqual(a.path, "methods/varest/llava-1.5-7b/w5reljoint/varest_w5reljoint.pt")
        self.assertTrue(a.trained_only)
        self.assertEqual(ARTIFACTS[("VarestW5", "llava", None)].path, "methods/varest/llava-1.5-7b/w5/varest_w5.pt")
        self.assertEqual(ARTIFACTS[("Varest", "llava-13b", None)].path,
                         "methods/varest/llava-1.5-13b/w5reljoint/varest_w5reljoint.pt")
        self.assertEqual(ARTIFACTS[("VarestW5", "llava-13b", None)].path,
                         "methods/varest/llava-1.5-13b/w5/varest_w5.pt")
        self.assertEqual(ARTIFACTS[("Varest", "llava-next-8b", None)].path,
                         "methods/varest/llama3-llava-next-8b/w5reljoint-nextdata/varest_w5reljoint.pt")
        self.assertEqual(ARTIFACTS[("VarestW5", "llava-next-8b", None)].path,
                         "methods/varest/llama3-llava-next-8b/w5-nextdata/varest_w5.pt")

    def test_names_and_budgets(self) -> None:
        self.assertEqual(VarestCompressor(max_tokens=64).name, "Varest-b64")
        self.assertEqual(VarestW5Compressor(max_tokens=224).name, "VarestW5-b224")
        self.assertEqual(VarestCompressor(retain_ratio=0.1111).name, "Varest-b0.1111")
        VarestCompressor(retain_ratio=0.2222).validate_llava_next()
        with self.assertRaises(ValueError):
            VarestCompressor(max_tokens=64).validate_llava_next()
        with self.assertRaises(ValueError):
            VarestCompressor(retain_ratio=1.5)


# ------------------------------------------------------------------------------------------------ end to end
def _tokenizer_available() -> bool:
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", local_files_only=True)
        return True
    except Exception:
        return False


def _tiny_llava():
    from transformers import (AutoProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                              LlavaForConditionalGeneration, LlavaImageProcessor)
    proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", local_files_only=True)
    ip = proc.image_processor.to_dict()
    ip.pop("do_pad", None)
    ip["image_processor_type"] = "LlavaImageProcessor"
    proc.image_processor = LlavaImageProcessor(**ip, do_pad=True)
    vis = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=336, patch_size=14, projection_dim=32)
    txt = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, vocab_size=len(proc.tokenizer))
    cfg = LlavaConfig(vision_config=vis, text_config=txt, image_token_index=proc.tokenizer.convert_tokens_to_ids("<image>"))
    torch.manual_seed(0)
    return proc, LlavaForConditionalGeneration(cfg).eval()


def _tiny_checkpoint(model, variant: str) -> VarestCheckpoint:
    import copy
    from peft import LoraConfig, get_peft_model
    peft = get_peft_model(copy.deepcopy(model), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj",
                                                                                                 "v_proj", "o_proj",
                                                                                                 "lm_head"]))
    lora = {k: torch.randn_like(v) * 0.01 for k, v in peft.state_dict().items() if "lora" in k}
    proj = {f"model.multi_modal_projector.{k}": v + 0.01 for k, v in
            model.model.multi_modal_projector.state_dict().items()}
    heads = {"text_head": {"sd": TextHead(width=8, hid=16).state_dict(), "width": 8, "hid": 16},
             "object_head": {"sd": ObjectHead(width=8, hid=16).state_dict(), "width": 8, "hid": 16}}
    rel = RelHead(d=64, nreg=4, rank=8).state_dict() if variant == "w5reljoint" else None
    return VarestCheckpoint(variant=variant, backbone="llava-1.5-7b", lora=lora, projector=proj,
                            postag=PosTag(64).state_dict(), relhead=rel, rel_layer=1 if rel else None,
                            lora_rank=4, lora_alpha=8, **heads)


class FakeVLM:
    weights_key = "llava"

    def __init__(self, model, processor) -> None:
        self.model, self.processor = model, processor
        self._model_id = "llava-hf/llava-1.5-7b-hf"

    def prepare_inputs(self, images, prompts, **kwargs):
        from transformers import BatchFeature
        ids = self.processor.tokenizer([p for p in prompts], return_tensors="pt", padding=True).input_ids
        return BatchFeature({"input_ids": ids})

    def generate(self, inputs, **kwargs):
        raise AssertionError("the backbone's own generate must be replaced while Varest is installed")


class FormattingVLM(FakeVLM):
    """A backbone that exposes ``format_prompts`` (as llmoe's VLMs do): Varest then tokenizes the prompts itself."""

    def format_prompts(self, prompts):
        return [f"USER: <image>\n{p} ASSISTANT:" for p in prompts]

    def device(self):                               # llmoe's HasDevice.device is a METHOD, not a property
        return torch.device("cpu")


@pytest.mark.skipif(not _tokenizer_available(), reason="llava-1.5-7b processor not in the local HF cache")
class EndToEndTest(unittest.TestCase):
    def test_prompt_only_prepare_inputs(self) -> None:
        proc, model = _tiny_llava()
        img = Image.fromarray((np.random.default_rng(1).random((120, 90, 3)) * 255).astype(np.uint8))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "varest_w5reljoint.pt")
            _tiny_checkpoint(model, "w5reljoint").save(path)
            vlm = FormattingVLM(model, proc)
            comp = VarestCompressor(max_tokens=32, checkpoint_path=path)
            comp.install(vlm)
            inputs = vlm.prepare_inputs([img], ["What is shown?"])
            self.assertNotIn("pixel_values", inputs)
            want = proc.tokenizer(vlm.format_prompts(["What is shown?"]), return_tensors="pt", padding=True).input_ids
            self.assertTrue(torch.equal(inputs["input_ids"], want))
            out = vlm.generate(inputs, max_new_tokens=4)
            self.assertTrue(torch.equal(out[:, :want.shape[1]], want))
            comp.uninstall()

    def test_install_generate_uninstall(self) -> None:
        proc, model = _tiny_llava()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        img = Image.fromarray((np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8))
        for cls, variant in ((VarestCompressor, "w5reljoint"), (VarestW5Compressor, "w5")):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, f"varest_{variant}.pt")
                _tiny_checkpoint(model, variant).save(path)
                vlm = FakeVLM(model, proc)
                comp = cls(max_tokens=32, checkpoint_path=path)
                comp.install(vlm)
                inputs = vlm.prepare_inputs([img], ["What is shown?"])
                out = vlm.generate(inputs, max_new_tokens=4)
                self.assertEqual(out.shape[0], 1)
                self.assertGreaterEqual(out.shape[1], inputs["input_ids"].shape[1])
                meta = comp.get_benchmark_record_metadata(1)[0]
                self.assertEqual(meta["visual_tokens"], 51 if variant == "w5reljoint" else 32)
                if variant == "w5reljoint":
                    self.assertEqual(meta["rel_kept"], 13)
                with self.assertRaises(ValueError):
                    vlm.generate(inputs, max_new_tokens=4, do_sample=True)
                comp.uninstall()
                after = model.state_dict()
                self.assertEqual(set(after), set(before))
                for k, v in before.items():
                    self.assertTrue(torch.equal(v, after[k]), k)

    def test_checkpoint_for_another_backbone_is_refused(self) -> None:
        proc, model = _tiny_llava()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.pt")
            ck = _tiny_checkpoint(model, "w5")
            ck.backbone = "llava-1.5-13b"
            ck.save(path)
            vlm = FakeVLM(model, proc)
            with self.assertRaises(ValueError):
                VarestW5Compressor(max_tokens=32, checkpoint_path=path).install(vlm)

    def test_pretrained_resolution(self) -> None:
        proc, model = _tiny_llava()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.pt")
            _tiny_checkpoint(model, "w5").save(path)
            import llmoe.weights as W
            seen = {}
            orig = W.resolve_weights

            def fake(method, budget, **kw):
                seen.update(method=method, budget=budget, **kw)
                return path
            W.resolve_weights = fake
            try:
                comp = VarestW5Compressor(max_tokens=64)
                comp.install(FakeVLM(model, proc))
                comp.uninstall()
            finally:
                W.resolve_weights = orig
            self.assertEqual((seen["method"], seen["backbone"]), ("VarestW5", "llava"))

class AdaptiveHaltArgsTest(unittest.TestCase):
    """The harness's compressor_args reach varest.halt.HaltConfig; halting is off unless asked for."""

    def test_default_off(self) -> None:
        self.assertIsNone(VarestCompressor(max_tokens=64).halt)
        self.assertIsNone(VarestW5Compressor(max_tokens=64).halt)

    def test_entropy_gate_defaults(self) -> None:
        from varest.halt import HaltConfig
        self.assertEqual(VarestCompressor(max_tokens=64, adaptive_halt=True).halt,
                         HaltConfig(min_layer=8, window=3, entropy=1.0, maxp=None))

    def test_maxp_gate_replaces_entropy(self) -> None:
        h = VarestCompressor(max_tokens=64, adaptive_halt=True, halt_min_layer=7, halt_maxp=0.7).halt
        self.assertEqual((h.min_layer, h.entropy, h.maxp), (7, None, 0.7))


if __name__ == "__main__":
    unittest.main()
