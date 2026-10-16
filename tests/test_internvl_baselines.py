import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import (
    BatchFeature,
    InternVLConfig,
    InternVLForConditionalGeneration,
    InternVLVisionConfig,
    Qwen3Config,
)

from llmoe.backbone.internvl import InternVL
from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import decoder_attention_rows
from llmoe.compressor.apet import ApETCompressor
from llmoe.compressor.divprune import DivPruneCompressor
from llmoe.compressor.fastv import FastVCompressor
from llmoe.compressor.internvl import (
    compress_image_sequences,
    image_sequences,
    pool_tile_signal,
    trim_image_placeholders,
)
from llmoe.compressor.must_drop import MustDropCompressor
from llmoe.compressor.otprune import OTPruneCompressor
from llmoe.compressor.pyramiddrop import PyramidDropCompressor
from llmoe.compressor.sparse_vlm import SparseVLMCompressor
from llmoe.compressor.tome import ToMeCompressor
from llmoe.compressor.vflowopt import VFlowOptCompressor
from llmoe.compressor.vision_zip import VisionZipCompressor


class InternVLAdapterLayoutTest(unittest.TestCase):
    def test_each_baseline_has_a_method_local_adapter_module(self) -> None:
        adapters = {
            "llmoe.compressor.apet.apet_internvl": "InternVLApETAdapter",
            "llmoe.compressor.divprune.divprune_internvl": "InternVLDivPruneAdapter",
            "llmoe.compressor.fastv.fastv_internvl": "InternVLFastVAdapter",
            "llmoe.compressor.must_drop.must_drop_internvl": "InternVLMustDropAdapter",
            "llmoe.compressor.nuwa.nuwa_internvl": "InternVLNuwaAdapter",
            "llmoe.compressor.otprune.otprune_internvl": "InternVLOTPruneAdapter",
            "llmoe.compressor.pyramiddrop.pyramiddrop_internvl": "InternVLPyramidDropAdapter",
            "llmoe.compressor.sparse_vlm.sparse_vlm_internvl": "InternVLSparseVLMAdapter",
            "llmoe.compressor.tome.tome_internvl": "InternVLToMeAdapter",
            "llmoe.compressor.vflowopt.vflowopt_internvl": "InternVLVFlowOptAdapter",
            "llmoe.compressor.vision_zip.vision_zip_internvl": "InternVLVisionZipAdapter",
        }
        for module_name, class_name in adapters.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, class_name))


class InternVLSequenceContractTest(unittest.TestCase):
    def test_tiles_are_concatenated_once_per_source_before_compression(self) -> None:
        outputs = SimpleNamespace(
            pooler_output=torch.arange(20, dtype=torch.float32).view(5, 4, 1)
        )
        received: list[torch.Tensor] = []

        def compress_one(tokens: torch.Tensor, image_index: int) -> torch.Tensor:
            received.append(tokens.clone())
            return tokens[: image_index + 2]

        result = compress_image_sequences(
            outputs,
            compress_one,
            tile_counts=(2, 3),
            output_tokens_per_tile=4,
            flatten_output=True,
        )

        self.assertEqual([tuple(tokens.shape) for tokens in received], [(8, 1), (12, 1)])
        torch.testing.assert_close(received[0].flatten(), torch.arange(8).float())
        torch.testing.assert_close(received[1].flatten(), torch.arange(8, 20).float())
        self.assertEqual(tuple(result.pooler_output.shape), (1, 5, 1))
        torch.testing.assert_close(
            result.pooler_output.flatten(),
            torch.tensor([0.0, 1.0, 8.0, 9.0, 10.0]),
        )

    def test_ambiguous_tile_batch_fails_closed_without_boundaries(self) -> None:
        outputs = SimpleNamespace(pooler_output=torch.zeros(2, 4, 3))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            image_sequences(outputs)

    def test_vit_signal_pooling_matches_pixel_shuffle_blocks(self) -> None:
        signal = torch.cat(
            [torch.tensor([[-1.0]]), torch.arange(16).float().unsqueeze(1)],
            dim=0,
        ).unsqueeze(0)

        pooled = pool_tile_signal(signal, 4, has_cls=True)

        torch.testing.assert_close(
            pooled[0, :, 0],
            torch.tensor([2.5, 4.5, 10.5, 12.5]),
        )

    def test_placeholder_trim_budgets_each_source_run_not_each_tile(self) -> None:
        inputs = BatchFeature(
            {
                "input_ids": torch.tensor(
                    [[1, 99, 99, 99, 99, 2, 99, 99, 99, 3]]
                ),
                "attention_mask": torch.ones((1, 10), dtype=torch.long),
            }
        )

        trimmed = trim_image_placeholders(
            inputs,
            token_id=99,
            keep_for_run=lambda _run_length: 2,
            pad_token_id=0,
        )

        torch.testing.assert_close(
            trimmed["input_ids"],
            torch.tensor([[1, 99, 99, 2, 99, 99, 3]]),
        )
        torch.testing.assert_close(
            trimmed["attention_mask"],
            torch.ones((1, 7), dtype=torch.long),
        )


class InternVLBackboneMetadataTest(unittest.TestCase):
    def test_prepare_inputs_records_source_tile_boundaries_before_trimming(self) -> None:
        first = [0, 0, 0, 0, 1, *([99] * 8), 2]
        second = [1, *([99] * 12), 2]
        prepared = BatchFeature(
            {
                "input_ids": torch.tensor([first, second]),
                "attention_mask": torch.tensor(
                    [
                        [0, 0, 0, 0, *([1] * 10)],
                        [*([1] * 14)],
                    ]
                ),
            }
        )
        backbone = InternVL.__new__(InternVL)
        backbone._model = SimpleNamespace(
            config=SimpleNamespace(image_token_id=99, image_seq_length=4)
        )
        backbone._processor = SimpleNamespace(image_seq_length=4)
        backbone._tile_counts = []

        with patch.object(VLM, "prepare_inputs", return_value=prepared):
            result = InternVL.prepare_inputs(
                backbone,
                torch.zeros(2, 3, 8, 8),
                ["first", "second"],
            )

        self.assertIs(result, prepared)
        self.assertEqual(backbone.tile_counts, (2, 3))


class _ShapeRecorder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shape: tuple[int, ...] | None = None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.shape = tuple(values.shape)
        return values


class InternVLDecoderGeometryTest(unittest.TestCase):
    def test_qk_norm_runs_on_qwen3_head_geometry(self) -> None:
        q_norm = _ShapeRecorder()
        k_norm = _ShapeRecorder()
        attention = SimpleNamespace(
            config=SimpleNamespace(
                num_attention_heads=2,
                num_key_value_heads=2,
            ),
            head_dim=2,
            scaling=2**-0.5,
            q_proj=torch.nn.Identity(),
            k_proj=torch.nn.Identity(),
            q_norm=q_norm,
            k_norm=k_norm,
        )
        layer = SimpleNamespace(
            input_layernorm=torch.nn.Identity(),
            self_attn=attention,
        )

        weights = decoder_attention_rows(
            layer,
            torch.randn(1, 3, 4),
            None,
            None,
            torch.tensor([[2]]),
            apply_rope=None,
            label="InternVL test",
        )

        self.assertEqual(q_norm.shape, (1, 3, 2, 2))
        self.assertEqual(k_norm.shape, (1, 3, 2, 2))
        self.assertEqual(tuple(weights.shape), (1, 1, 3))


class InternVLCompressorIntegrationTest(unittest.TestCase):
    @staticmethod
    def _tiny_vlm(*, tiles: int = 20, text_layers: int = 4) -> SimpleNamespace:
        vision = InternVLVisionConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=32,
            image_size=(8, 8),
            patch_size=(2, 2),
            use_qk_norm=True,
        )
        text = Qwen3Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=text_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=max(128, tiles * 4 + 16),
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config = InternVLConfig(
            vision_config=vision,
            text_config=text,
            image_token_id=99,
            image_seq_length=4,
            downsample_ratio=0.5,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )
        model = InternVLForConditionalGeneration(config).eval()
        visual_tokens = tiles * config.image_seq_length

        def prepare_inputs(*_args, **_kwargs) -> BatchFeature:
            input_ids = torch.tensor([[1, *([99] * visual_tokens), 3]])
            return BatchFeature(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
            )

        return SimpleNamespace(
            model=model,
            processor=SimpleNamespace(
                tokenizer=SimpleNamespace(pad_token_id=0),
                image_processor=SimpleNamespace(),
            ),
            tile_counts=(tiles,),
            prepare_inputs=prepare_inputs,
        )

    def test_training_free_path_compresses_one_global_multitile_span(self) -> None:
        torch.manual_seed(0)
        vlm = self._tiny_vlm()
        compressor = VisionZipCompressor(max_tokens=32)
        try:
            compressor.install(vlm)
            inputs = vlm.prepare_inputs(None, None)
            self.assertEqual(int((inputs["input_ids"] == 99).sum()), 32)

            outputs = vlm.model(
                **inputs,
                pixel_values=torch.randn(20, 3, 8, 8),
                use_cache=False,
            )

            self.assertEqual(tuple(outputs.image_hidden_states.shape), (1, 32, 16))
            self.assertEqual(tuple(outputs.logits.shape), (1, 34, 128))
        finally:
            compressor.uninstall()

    def _forward(
        self,
        compressor,
        *,
        tiles: int,
        text_layers: int = 4,
        use_cache: bool = False,
    ) -> tuple[int, object]:
        torch.manual_seed(0)
        vlm = self._tiny_vlm(tiles=tiles, text_layers=text_layers)
        try:
            compressor.install(vlm)
            inputs = vlm.prepare_inputs(None, None)
            placeholders = int((inputs["input_ids"] == 99).sum())
            outputs = vlm.model(
                **inputs,
                pixel_values=torch.randn(tiles, 3, 8, 8),
                use_cache=use_cache,
            )
            return placeholders, outputs
        finally:
            compressor.uninstall()

    def test_static_reducers_share_one_absolute_budget_across_tiles(self) -> None:
        cases = {
            "DivPrune": DivPruneCompressor(max_tokens=2),
            "OTPrune": OTPruneCompressor(max_tokens=2),
            "ToMe": ToMeCompressor(max_tokens=2),
        }
        for name, compressor in cases.items():
            with self.subTest(compressor=name):
                placeholders, outputs = self._forward(compressor, tiles=5)
                self.assertEqual(placeholders, 2)
                self.assertEqual(
                    tuple(outputs.image_hidden_states.shape),
                    (1, 2, 16),
                )
                self.assertEqual(tuple(outputs.logits.shape), (1, 4, 128))

    def test_apet_runs_both_stages_on_one_global_visual_span(self) -> None:
        placeholders, outputs = self._forward(
            ApETCompressor(budget=32),
            tiles=25,
            use_cache=True,
        )

        self.assertEqual(placeholders, 48)
        self.assertEqual(tuple(outputs.image_hidden_states.shape), (1, 48, 16))
        self.assertEqual(tuple(outputs.logits.shape), (1, 18, 128))

    def test_fastv_backsolves_one_global_keep_for_the_loaded_depth(self) -> None:
        placeholders, outputs = self._forward(
            FastVCompressor(max_tokens=64),
            tiles=20,
            use_cache=True,
        )

        self.assertEqual(placeholders, 80)
        self.assertEqual(tuple(outputs.image_hidden_states.shape), (20, 4, 16))
        self.assertEqual(tuple(outputs.logits.shape), (1, 18, 128))

    def test_sparsevlm_prunes_the_complete_visual_span_once(self) -> None:
        placeholders, outputs = self._forward(
            SparseVLMCompressor(
                max_tokens=5,
                schedule=(5,),
                pruning_layers=(0,),
                recycling=False,
            ),
            tiles=5,
            use_cache=True,
        )

        self.assertEqual(placeholders, 20)
        self.assertEqual(tuple(outputs.image_hidden_states.shape), (5, 4, 16))
        self.assertEqual(tuple(outputs.logits.shape), (1, 7, 128))

    def test_pyramiddrop_derives_one_global_layer_average_schedule(self) -> None:
        placeholders, outputs = self._forward(
            PyramidDropCompressor(max_tokens=32),
            tiles=20,
            use_cache=True,
        )

        self.assertEqual(placeholders, 80)
        self.assertEqual(tuple(outputs.image_hidden_states.shape), (20, 4, 16))
        self.assertEqual(tuple(outputs.logits.shape), (1, 6, 128))

    def test_mustdrop_selects_local_windows_under_one_source_budget(self) -> None:
        placeholders, outputs = self._forward(
            MustDropCompressor(
                max_tokens=5,
                prefill_stage=False,
                window_size=2,
            ),
            tiles=5,
        )

        self.assertEqual(placeholders, 5)
        self.assertEqual(tuple(outputs.image_hidden_states.shape), (1, 5, 16))
        self.assertEqual(tuple(outputs.logits.shape), (1, 7, 128))

    def test_vflowopt_spends_one_budget_across_vision_and_decoder_stages(self) -> None:
        compressor = VFlowOptCompressor(max_tokens=32, verbose=False)
        placeholders, outputs = self._forward(
            compressor,
            tiles=20,
            use_cache=True,
        )

        self.assertEqual(placeholders, 46)
        self.assertEqual(tuple(outputs.image_hidden_states.shape), (1, 46, 16))
        self.assertEqual(tuple(outputs.logits.shape), (1, 25, 128))
        self.assertEqual(compressor.mean_layer_average(), 32.0)

if __name__ == "__main__":
    unittest.main()
