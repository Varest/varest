"""CPU-only contracts for LLaVA-NeXT VisionZip/VisionZip++."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from llmoe.backbone import BACKBONES
from llmoe.backbone.llava_next import (
    LLaVANext8B,
    LLaVANextImagePreprocessor,
)
from llmoe.compressor import COMPRESSORS, COMPRESSOR_TUNERS
from llmoe.compressor.vision_zip.vision_zip import (
    VisionZipCompressor,
    VisionZipPlusPlusCompressor,
)
from llmoe.compressor.vision_zip.vision_zip_llava_next_pp import (
    FLASH_ATTN_VERSION,
    FINAL_TRAINING_RECIPE,
    TARGET_IMAGE_GRID_PINPOINTS,
    TARGET_MODEL_ID,
    TARGET_MODEL_REVISION,
    TRAINING_ATTN_IMPLEMENTATION,
    CropReduction,
    LlavaNextBudgetReport,
    LlavaNextVisionZipLayout,
    _native_geometry,
    anyres_grid_shape,
    checkpoint_metadata,
    crop_count,
    load_projector_checkpoint,
    pack_official_order,
    reduce_crop,
    retention_target,
    replace_image_placeholder_span,
)
from llmoe.compressor.vision_zip.vision_zip_llava_next_ft import (
    IGNORE_INDEX,
    LlavaNextParquetDataset,
    LlavaNextVisionZipCollator,
    assistant_only_labels,
    build_llava_next_training_example,
    llava_next_messages,
    projectneura_warmup_steps,
    processor_native_token_counts,
    require_effective_flash_attention,
    require_training_attention_backend,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="llava_next",
        architectures=["LlavaNextForConditionalGeneration"],
        image_token_id=99,
        image_token_index=99,
        image_grid_pinpoints=[list(point) for point in TARGET_IMAGE_GRID_PINPOINTS],
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        image_seq_length=576,
        use_image_newline_parameter=True,
        vision_config=SimpleNamespace(
            image_size=336,
            patch_size=14,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
        ),
        text_config=SimpleNamespace(
            model_type="llama",
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            max_position_embeddings=8192,
            vocab_size=128320,
            bos_token_id=128000,
            eos_token_id=128009,
            rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
        ),
    )


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 1
        self.head_dim = 4
        self.scale = 0.5
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))
            self.k_proj.weight.copy_(torch.eye(4))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.self_attn(hidden_states=hidden_states)
        return hidden_states


class _FakeVisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])
        self.encoder = SimpleNamespace(layers=self.layers)

    def forward(self, pixel_values: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        del kwargs
        views = int(pixel_values.shape[0])
        hidden = torch.zeros(views, 577, 4, device=pixel_values.device)
        patch_values = torch.arange(576, dtype=hidden.dtype, device=hidden.device)
        for view in range(views):
            hidden[view, 0, 0] = float(view + 1)
            hidden[view, 1:, 0] = patch_values + view * 1000
            hidden[view, :, 1] = 1.0
        for layer in self.layers:
            hidden = layer(hidden_states=hidden)
        return SimpleNamespace(pooler_output=None, hidden_states=None)


class _FakeInner:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.vision_tower = _FakeVisionTower()
        self.multi_modal_projector = nn.Identity()
        self.image_newline = nn.Parameter(torch.tensor([999.0, 0.0, 0.0, 0.0]))

    def get_image_features(self, *args: object, **kwargs: object) -> str:
        del args, kwargs
        return "native"


class _FakeVLM:
    compressor_family = "llava_next"

    def __init__(self) -> None:
        config = _config()
        self._model_id = TARGET_MODEL_ID
        self._model_revision = TARGET_MODEL_REVISION
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.model = SimpleNamespace(config=config, model=_FakeInner(config))
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

    def prepare_inputs(self) -> dict[str, torch.Tensor]:
        native_visual_length = 2880 + 48
        return {
            "input_ids": torch.tensor([[1] + [99] * native_visual_length + [2]]),
            "attention_mask": torch.ones(1, native_visual_length + 2, dtype=torch.long),
            "position_ids": torch.arange(native_visual_length + 2).unsqueeze(0),
            "pixel_values": torch.zeros(1, 5, 3, 2, 2),
            "image_sizes": torch.tensor([[672, 672]]),
        }

    def generate(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {"args": args, "kwargs": kwargs}


class VisionZipPolicyTests(unittest.TestCase):
    def test_visualization_source_drops_every_view_cls_and_single_newline(self) -> None:
        from llmoe.viz import _llava_next_patch_source, _llava_next_view_strip

        pixels = torch.stack((torch.zeros(3, 2, 2), torch.ones(3, 2, 2)))
        canvas = _llava_next_view_strip(
            pixels,
            SimpleNamespace(image_mean=(0.0, 0.0, 0.0), image_std=(1.0, 1.0, 1.0)),
        )
        self.assertEqual(tuple(canvas.shape), (3, 2, 4))
        self.assertTrue((canvas[:, :, :2] == 0).all())
        self.assertTrue((canvas[:, :, 2:] == 1).all())

        source = _llava_next_patch_source(
            torch.eye(6),
            view_count=2,
            patches_per_view=2,
        )
        self.assertEqual(tuple(source.shape), (6, 4))
        self.assertEqual(source[0].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(source[1].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(source[3].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(source[4].tolist(), [0.0, 0.0, 1.0, 0.0])

        with_structural = _llava_next_patch_source(
            torch.eye(9),
            view_count=2,
            patches_per_view=2,
        )
        self.assertEqual(tuple(with_structural.shape), (9, 4))
        self.assertTrue((with_structural[6:] == 0).all())

        single = _llava_next_patch_source(
            torch.eye(4),
            view_count=1,
            patches_per_view=2,
        )
        self.assertEqual(tuple(single.shape), (4, 2))
        self.assertEqual(single[-1].tolist(), [0.0, 0.0])

    def test_crop_reduction_preserves_dominant_order_and_context_merge(self) -> None:
        features = torch.tensor([[0.0], [10.0], [20.0], [30.0], [40.0], [50.0]])
        attention = torch.tensor([0.1, 0.9, 0.2, 0.3, 1.0])
        keys = torch.ones(5, 2)

        result = reduce_crop(
            features,
            attention,
            keys,
            dominant=3,
            contextual=1,
            track_source=True,
        )

        self.assertEqual(result.dominant_patch_indices.tolist(), [1, 4])
        torch.testing.assert_close(result.tokens[:, 0], torch.tensor([0.0, 20.0, 50.0, 45.0]))
        self.assertEqual(tuple(result.source.shape), (4, 6))
        torch.testing.assert_close(
            result.source[3],
            torch.tensor([0.0, 1.0, 0.0, 0.5, 0.5, 0.0]),
        )

    def test_official_multiview_pack_order(self) -> None:
        projected = torch.tensor(
            [
                [[0.0], [1.0], [2.0]],
                [[10.0], [11.0], [12.0]],
                [[20.0], [21.0], [22.0]],
                [[30.0], [31.0], [32.0]],
                [[40.0], [41.0], [42.0]],
            ]
        )
        indices = (
            torch.tensor([0]),
            torch.tensor([575]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
        reductions = [
            CropReduction(projected[index], indices[index], 1, None) for index in range(5)
        ]

        packed, order = pack_official_order(
            projected,
            reductions,
            tile_rows=2,
            tile_columns=2,
            patch_rows=24,
            patch_columns=24,
        )

        expected = [0, 1, 2, 21, 11, 31, 41, 10, 20, 30, 40, 12, 22, 32, 42]
        self.assertEqual(packed[:, 0].tolist(), expected)
        self.assertEqual(order.tolist(), [0, 1, 2, 7, 4, 10, 13, 3, 6, 9, 12, 5, 8, 11, 14])

    def test_single_view_appends_and_accounts_for_newline(self) -> None:
        projected = torch.arange(3.0).reshape(1, 3, 1)
        reduction = CropReduction(projected[0], torch.tensor([0]), 1, None)
        packed, order = pack_official_order(
            projected,
            [reduction],
            tile_rows=0,
            tile_columns=0,
            patch_rows=24,
            patch_columns=24,
            image_newline=torch.tensor([9.0]),
        )
        self.assertEqual(packed[:, 0].tolist(), [0.0, 1.0, 2.0, 9.0])
        self.assertEqual(order.tolist(), [0, 1, 2, -1])
        report = LlavaNextBudgetReport(
            (LlavaNextVisionZipLayout(1, 0, 0, 0, 576, 0.5, 288, 288, 1, (576,), (288,), 0, 0),),
        )
        self.assertEqual(report.target_content_tokens, 288)
        self.assertEqual(report.realized_total, 289)
        self.assertEqual(
            set(report.to_dict()),
            {
                "budget_semantics",
                "original_content_tokens",
                "requested_retention_ratio",
                "target_content_tokens",
                "realized_content_tokens",
                "realized_retention_ratio",
                "structural_tokens",
                "realized_total",
                "total_views",
                "view_policy",
                "output_order",
                "newline_policy",
                "rounding_policy",
                "layouts",
            },
        )

    def test_view_counts_and_ratio_rounding(self) -> None:
        config = _config()
        cases = [
            ((336, 672), 3, (1, 2)),
            ((1008, 336), 4, (3, 1)),
            ((672, 672), 5, (2, 2)),
        ]
        for size, views, grid in cases:
            with self.subTest(size=size):
                self.assertEqual(crop_count(config, size), views)
                self.assertEqual(anyres_grid_shape(config, size), grid)
        self.assertEqual(retention_target(1152, 0.25), 288)
        self.assertEqual(retention_target(5, 0.5), 3)

    def test_native_geometry_excludes_newlines_and_padding(self) -> None:
        inner = _FakeInner(_config())
        cases = {
            (336, 672): (1728, 24, (576, 576, 576)),
            (1008, 336): (2304, 72, (576, 576, 576, 576)),
            (672, 672): (2880, 48, (576, 576, 576, 576, 576)),
            (400, 1000): (2016, 24, (576, 432, 576, 432)),
        }
        for size, expected in cases.items():
            with self.subTest(size=size):
                geometry = _native_geometry(inner, size, crop_count(inner.config, size))
                self.assertEqual(
                    (
                        geometry.original_content_tokens,
                        geometry.structural_tokens,
                        geometry.capacities,
                    ),
                    expected,
                )


class SequenceBookkeepingTests(unittest.TestCase):
    def test_compact_prefill_and_decode_grow_every_llama_cache_layer(self) -> None:
        """Exercise native Llama masks, RoPE positions, and DynamicCache bookkeeping."""

        inputs = {
            "input_ids": torch.tensor([[1, 99, 99, 99, 99, 2]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "position_ids": torch.arange(6, dtype=torch.long).unsqueeze(0),
        }
        replace_image_placeholder_span(inputs, image_token_id=99, target_length=3)
        self.assertEqual(inputs["position_ids"].tolist(), [[0, 1, 2, 3, 4]])

        decoder = LlamaModel(
            LlamaConfig(
                vocab_size=128,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                rope_theta=500000.0,
                rms_norm_eps=1e-5,
            )
        ).eval()
        with torch.no_grad():
            prefill = decoder(
                **inputs,
                use_cache=True,
                return_dict=True,
            )
        cache = prefill.past_key_values
        self.assertEqual(cache.get_seq_length(), 5)
        self.assertEqual(len(cache.layers), 2)
        for layer in cache.layers:
            self.assertEqual(tuple(layer.keys.shape), (1, 2, 5, 4))
            self.assertEqual(tuple(layer.values.shape), (1, 2, 5, 4))

        with torch.no_grad():
            decode = decoder(
                input_ids=torch.tensor([[3]]),
                attention_mask=torch.ones(1, 6, dtype=torch.long),
                position_ids=torch.tensor([[5]]),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        self.assertEqual(decode.past_key_values.get_seq_length(), 6)
        for layer in decode.past_key_values.layers:
            self.assertEqual(tuple(layer.keys.shape), (1, 2, 6, 4))
            self.assertEqual(tuple(layer.values.shape), (1, 2, 6, 4))

    def test_placeholder_compaction_updates_labels_masks_positions_and_cache(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[0, 5, 99, 99, 99, 99, 99, 99, 7]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, 7]]),
            "position_ids": torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6, 7]]),
            "cache_position": torch.arange(9),
        }
        result = replace_image_placeholder_span(inputs, image_token_id=99, target_length=3)
        self.assertEqual(result["input_ids"].tolist(), [[0, 5, 99, 99, 99, 7]])
        self.assertEqual(result["attention_mask"].tolist(), [[0, 1, 1, 1, 1, 1]])
        self.assertEqual(result["labels"].tolist(), [[-100, -100, -100, -100, -100, 7]])
        self.assertEqual(result["position_ids"].tolist(), [[0, 0, 1, 2, 3, 4]])
        self.assertEqual(result["cache_position"].tolist(), list(range(6)))

    def test_placeholder_expansion_masks_inserted_labels(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[1, 99, 99, 2]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, 2]]),
        }
        result = replace_image_placeholder_span(inputs, image_token_id=99, target_length=4)
        self.assertEqual(result["input_ids"].tolist(), [[1, 99, 99, 99, 99, 2]])
        self.assertEqual(result["labels"].tolist(), [[-100, -100, -100, -100, -100, 2]])

    def test_two_dimensional_cache_position_is_rebuilt_contiguously(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[1, 99, 99, 99, 2]]),
            "cache_position": torch.arange(5).unsqueeze(0),
        }
        result = replace_image_placeholder_span(
            inputs,
            image_token_id=99,
            target_length=2,
        )
        self.assertEqual(result["cache_position"].tolist(), [[0, 1, 2, 3]])

    def test_multiple_image_runs_fail_closed(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "exactly one image"):
            replace_image_placeholder_span(
                {"input_ids": torch.tensor([[99, 99, 1, 99, 99]])},
                image_token_id=99,
                target_length=2,
            )


class AdapterLifecycleTests(unittest.TestCase):
    def test_install_forward_source_and_uninstall(self) -> None:
        vlm = _FakeVLM()
        original_get = vlm.model.model.get_image_features
        original_prepare = vlm.prepare_inputs
        original_generate = vlm.generate
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projector.pt"
            metadata = checkpoint_metadata(
                model_id=TARGET_MODEL_ID,
                model_revision=TARGET_MODEL_REVISION,
                retain_ratio=0.25,
                image_grid_pinpoints=TARGET_IMAGE_GRID_PINPOINTS,
                training_schedule=FINAL_TRAINING_RECIPE,
                artifact_role="final_evaluation",
                training_complete=True,
                global_step=123,
            )
            torch.save(
                {
                    "format_version": metadata["format_version"],
                    "compressor": "VisionZipPlusPlusCompressor",
                    "config": metadata,
                    "state_dict": {},
                },
                path,
            )
            compressor = VisionZipPlusPlusCompressor(
                retain_ratio=0.25, checkpoint_path=path, track_source=True
            )
            compressor.install(vlm)
            compressor.install(vlm)
            prepared = vlm.prepare_inputs()
            self.assertEqual(int((prepared["input_ids"] == 99).sum()), 768)
            self.assertEqual(prepared["input_ids"].shape[1], 770)
            output = vlm.model.model.get_image_features(
                torch.zeros(1, 5, 3, 2, 2),
                torch.tensor([[672, 672]]),
                vision_feature_layer=-2,
                vision_feature_select_strategy="default",
            )
            self.assertEqual(tuple(output.pooler_output[0].shape), (768, 4))
            report = compressor.budget_report.to_dict()
            self.assertEqual(report["original_content_tokens"], 2880)
            self.assertEqual(report["target_content_tokens"], 720)
            self.assertEqual(report["realized_content_tokens"], 720)
            self.assertEqual(report["structural_tokens"], 48)
            self.assertEqual(report["realized_total"], 768)
            self.assertEqual(int((output.pooler_output[0][:, 0] == 999).sum()), 48)
            source = compressor.get_source_history()
            self.assertEqual(tuple(source[0].shape), (2933, 2933))
            self.assertEqual(tuple(source[1].shape), (768, 2933))
            with self.assertRaisesRegex(ValueError, "exactly 5 views"):
                vlm.model.model.get_image_features(
                    torch.zeros(1, 6, 3, 2, 2),
                    torch.tensor([[672, 672]]),
                )
            self.assertEqual(vlm.generate(use_cache=True)["kwargs"]["use_cache"], True)
            with self.assertRaisesRegex(NotImplementedError, "beam search"):
                vlm.generate(num_beams=2)
            with self.assertRaisesRegex(NotImplementedError, "externally supplied"):
                vlm.generate(past_key_values=object())
            compressor.uninstall()

        self.assertEqual(vlm.model.model.get_image_features, original_get)
        self.assertEqual(vlm.prepare_inputs, original_prepare)
        self.assertEqual(vlm.generate, original_generate)

    def test_plus_plus_requires_checkpoint_and_untrained_next_requires_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_path"):
            VisionZipPlusPlusCompressor(retain_ratio=0.25)
        with self.assertRaisesRegex(ValueError, "retain_ratio"):
            VisionZipCompressor(max_tokens=64).install(_FakeVLM())


class CheckpointTests(unittest.TestCase):
    def _metadata(self, **overrides: object) -> dict[str, object]:
        values = checkpoint_metadata(
            model_id=TARGET_MODEL_ID,
            model_revision=TARGET_MODEL_REVISION,
            retain_ratio=0.25,
            image_grid_pinpoints=TARGET_IMAGE_GRID_PINPOINTS,
            training_schedule=FINAL_TRAINING_RECIPE,
            artifact_role="final_evaluation",
            training_complete=True,
            global_step=123,
        )
        values.update(overrides)
        return values

    def test_metadata_checkpoint_loads_strictly(self) -> None:
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projector.pt"
            metadata = self._metadata()
            torch.save(
                {
                    "format_version": metadata["format_version"],
                    "compressor": "VisionZipPlusPlusCompressor",
                    "config": metadata,
                    "state_dict": source.state_dict(),
                },
                path,
            )
            load_projector_checkpoint(
                path,
                target,
                retain_ratio=0.25,
                model_id=TARGET_MODEL_ID,
                model_revision=TARGET_MODEL_REVISION,
                image_grid_pinpoints=TARGET_IMAGE_GRID_PINPOINTS,
            )
        torch.testing.assert_close(target.weight, source.weight)

    def test_incomplete_and_wrong_budget_checkpoints_are_rejected(self) -> None:
        projector = nn.Linear(3, 2)
        for metadata, message in (
            (
                self._metadata(artifact_role="incomplete_run", training_complete=False),
                "not a completed",
            ),
            (self._metadata(requested_retention_ratio=0.5), "requested_retention_ratio"),
            (
                self._metadata(
                    training_schedule={
                        **FINAL_TRAINING_RECIPE,
                        "learning_rate": 1e-4,
                    }
                ),
                "training_schedule.learning_rate",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "projector.pt"
                torch.save(
                    {
                        "format_version": metadata["format_version"],
                        "compressor": "VisionZipPlusPlusCompressor",
                        "config": metadata,
                        "state_dict": projector.state_dict(),
                    },
                    path,
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_projector_checkpoint(
                        path,
                        nn.Linear(3, 2),
                        retain_ratio=0.25,
                        model_id=TARGET_MODEL_ID,
                        model_revision=TARGET_MODEL_REVISION,
                        image_grid_pinpoints=TARGET_IMAGE_GRID_PINPOINTS,
                    )


class TrainingAndBackboneTests(unittest.TestCase):
    def test_parquet_dataset_filters_null_images_before_sampling(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        conversations = [
            {"from": "human", "value": "<image>\nWhat?"},
            {"from": "gpt", "value": "An image."},
        ]
        rows = [
            {
                "id": "image-0",
                "conversations": conversations,
                "data_source": "test",
                "image": {"bytes": b"first", "path": "first.jpg"},
            },
            {
                "id": "text-only",
                "conversations": conversations,
                "data_source": "test",
                "image": None,
            },
            {
                "id": "image-1",
                "conversations": conversations,
                "data_source": "test",
                "image": {"bytes": b"second", "path": "second.jpg"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            pq.write_table(
                pa.Table.from_pylist(rows),
                root / "data" / "train-00000-of-00001.parquet",
                row_group_size=2,
            )
            dataset = LlavaNextParquetDataset(
                root,
                processor=SimpleNamespace(),
                retain_ratio=0.25,
                model_max_length=2048,
                subset_frac=1.0,
                seed=42,
                limit=None,
            )
            self.assertEqual(dataset.total_rows, 3)
            self.assertEqual(dataset.image_rows, 2)
            self.assertEqual(len(dataset), 2)
            selected = [dataset._row(location)["id"] for location in dataset.locations]
            self.assertEqual(set(selected), {"image-0", "image-1"})

    def test_registration_revision_and_image_boundaries(self) -> None:
        self.assertIs(BACKBONES["llava-next-8b"], LLaVANext8B)
        self.assertIs(COMPRESSORS["VisionZip++"], VisionZipPlusPlusCompressor)
        self.assertEqual(
            COMPRESSOR_TUNERS["VisionZip++"][1].__name__,
            "finetune_vision_zip_llava_next",
        )
        backbone = LLaVANext8B(dtype="bfloat16")
        self.assertEqual(backbone._model_revision, TARGET_MODEL_REVISION)
        self.assertIs(backbone._dtype, torch.bfloat16)
        images = torch.randn(2, 3, 4, 5)
        split = LLaVANextImagePreprocessor().forward(SimpleNamespace(), images)
        self.assertEqual(len(split), 2)
        torch.testing.assert_close(split[1], images[1])
        backbone.validate_compressor("VisionZip++")
        with self.assertRaises(NotImplementedError):
            backbone.validate_compressor("Nuwa")

    def test_loader_passes_revision_to_native_model_and_processor(self) -> None:
        from transformers import PreTrainedModel, PretrainedConfig, ProcessorMixin
        import llmoe.backbone.llava_next as module

        processor = object.__new__(ProcessorMixin)
        processor.patch_size = None
        processor.vision_feature_select_strategy = None
        processor.num_additional_image_tokens = None
        processor.image_token = "<image>"
        processor.chat_template = "<|start_header_id|><|end_header_id|><|eot_id|><image>"
        processor.image_processor = SimpleNamespace(
            image_grid_pinpoints=[list(point) for point in TARGET_IMAGE_GRID_PINPOINTS],
            size={"shortest_edge": 336},
            crop_size={"height": 336, "width": 336},
            do_pad=True,
        )
        processor.tokenizer = SimpleNamespace(
            bos_token_id=128000,
            eos_token_id=128009,
            pad_token_id=128257,
            convert_tokens_to_ids=lambda token: 128256 if token == "<image>" else None,
        )
        model = PreTrainedModel(PretrainedConfig())
        model.config = _config()
        model.config.image_token_id = 128256
        calls: dict[str, tuple[str, dict[str, object]]] = {}

        def load_processor(model_id: str, **kwargs: object) -> ProcessorMixin:
            calls["processor"] = (model_id, kwargs)
            return processor

        class _Loader:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs: object) -> PreTrainedModel:
                calls["model"] = (model_id, kwargs)
                return model

        with (
            patch.object(module.AutoProcessor, "from_pretrained", load_processor),
            patch("transformers.LlavaNextForConditionalGeneration", _Loader),
        ):
            loaded_processor, loaded_model = LLaVANext8B(
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).load_model()

        self.assertIs(loaded_processor, processor)
        self.assertIs(loaded_model, model)
        self.assertEqual(calls["processor"][1]["revision"], TARGET_MODEL_REVISION)
        self.assertEqual(calls["model"][1]["revision"], TARGET_MODEL_REVISION)
        self.assertEqual(calls["model"][1]["attn_implementation"], "sdpa")
        self.assertEqual(processor.patch_size, 14)
        self.assertEqual(processor.vision_feature_select_strategy, "default")
        self.assertEqual(processor.num_additional_image_tokens, 1)

    def test_training_requires_fa2_but_inference_defaults_to_sdpa(self) -> None:
        self.assertEqual(TRAINING_ATTN_IMPLEMENTATION, "flash_attention_2")
        self.assertEqual(FLASH_ATTN_VERSION, "2.8.3.post1")
        self.assertEqual(
            require_training_attention_backend("flash_attention_2"),
            "flash_attention_2",
        )
        for unsupported in ("sdpa", "eager", None):
            with self.subTest(unsupported=unsupported), self.assertRaises(ValueError):
                require_training_attention_backend(unsupported)
        self.assertEqual(LLaVANext8B()._attn_implementation, "sdpa")

        model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    config=SimpleNamespace(_attn_implementation="flash_attention_2")
                ),
                vision_tower=SimpleNamespace(
                    config=SimpleNamespace(_attn_implementation="flash_attention_2")
                ),
            )
        )
        require_effective_flash_attention(model)
        model.model.vision_tower.config._attn_implementation = "sdpa"
        with self.assertRaisesRegex(RuntimeError, "vision tower='sdpa'"):
            require_effective_flash_attention(model)

    def test_training_collator_positions_labels_and_budget(self) -> None:
        target, structural = 288, 24
        example = {
            "input_ids": torch.tensor([1] + [99] * (target + structural) + [2, 3]),
            "labels": torch.tensor([IGNORE_INDEX] * (target + structural + 2) + [3]),
            "pixel_values": torch.zeros(5, 3, 2, 2),
            "image_sizes": torch.tensor([672, 672]),
            "view_count": torch.tensor(5),
            "original_content_tokens": torch.tensor(1152),
            "target_content_tokens": torch.tensor(target),
            "structural_tokens": torch.tensor(structural),
        }
        batch = LlavaNextVisionZipCollator(
            image_token_id=99,
            retain_ratio=0.25,
        )([example])
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 315))
        self.assertEqual(batch["position_ids"].tolist(), [list(range(315))])
        self.assertTrue((batch["labels"][0, 1:313] == IGNORE_INDEX).all())
        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 5, 3, 2, 2))
        self.assertEqual(projectneura_warmup_steps(0.03, 101), 3)

    def test_cached_tokenizer_builds_assistant_only_compact_example(self) -> None:
        snapshot = Path(
            "/workspace/output/hf_cache/models--llava-hf--llama3-llava-next-8b-hf/"
            "snapshots/b041c0d0ea0dd0196d147206c210c8d1752fc2da"
        )
        if not snapshot.is_dir():
            self.skipTest("Pinned local processor snapshot is unavailable")
        from transformers import AutoConfig, AutoProcessor

        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        sample = {
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "A bus."},
                {"from": "human", "value": "What color?"},
                {"from": "gpt", "value": "Red."},
            ]
        }
        original_content, structural = processor_native_token_counts(processor, (672, 672))
        self.assertEqual((original_content, structural), (2880, 48))
        ids, labels = build_llava_next_training_example(
            sample,
            processor,
            original_content_tokens=original_content,
            structural_tokens=structural,
            retain_ratio=0.25,
            model_max_length=2048,
        )
        self.assertEqual(ids[0], processor.tokenizer.bos_token_id)
        image_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        self.assertEqual(ids.count(image_id), 768)
        self.assertTrue(
            all(label == IGNORE_INDEX for token, label in zip(ids, labels) if token == image_id)
        )
        self.assertGreater(sum(label != IGNORE_INDEX for label in labels), 2)

        # Apart from the intentionally shorter visual run, training must use
        # the exact IDs produced by LlavaNextProcessor (including Llama-3 BOS).
        from PIL import Image

        prompt = processor.apply_chat_template(
            llava_next_messages(sample["conversations"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        native = processor(
            images=[Image.new("RGB", (672, 672))],
            text=[prompt],
            return_tensors="pt",
        )
        self.assertEqual(int(native["pixel_values"].shape[1]), 5)
        replace_image_placeholder_span(
            native,
            image_token_id=image_id,
            target_length=768,
        )
        self.assertEqual(native["input_ids"][0].tolist(), ids)

        direct = assistant_only_labels(
            processor.tokenizer(
                processor.apply_chat_template(
                    [
                        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                ),
                add_special_tokens=False,
            ).input_ids,
            processor.tokenizer,
        )
        self.assertGreater(sum(label != IGNORE_INDEX for label in direct), 1)

        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        self.assertEqual(config.architectures, ["LlavaNextForConditionalGeneration"])
        self.assertEqual(config.text_config.num_hidden_layers, 32)
        self.assertEqual(config.text_config.num_key_value_heads, 8)
        self.assertEqual(config.text_config.rope_parameters["rope_theta"], 500000.0)
        self.assertEqual(
            config.image_grid_pinpoints,
            [list(point) for point in TARGET_IMAGE_GRID_PINPOINTS],
        )


if __name__ == "__main__":
    unittest.main()
