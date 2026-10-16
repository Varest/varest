import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from llmoe.compressor.vision_zip.vision_zip import (
    VisionZipCompressor,
    contextual_merge,
)
from llmoe.compressor.vision_zip.vision_zip_qwen import QwenVisionZipAdapter
from llmoe.compressor.vision_zip.vision_zip_qwen_ft import (
    VisionZipTrainingQwenForConditionalGeneration,
    _freeze_for_merger_tuning,
    _save_merger_artifact,
    build_qwen_training_args,
)
from llmoe.utils.qwen_training import qwen_visual_spans


class VisionZipPolicyTest(unittest.TestCase):
    def test_contextual_merge_keeps_targets_and_averages_assignments(self) -> None:
        hidden = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]]]
        )
        metric = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
        )

        merged = contextual_merge(hidden, metric, contextual_num=2)

        torch.testing.assert_close(merged.target_indices, torch.tensor([0, 2]))
        torch.testing.assert_close(
            merged.tokens,
            torch.tensor([[[1.0, 2.0], [2.0, 0.0]]]),
        )


class VisionZipQwenAdapterTest(unittest.TestCase):
    def test_prefill_uses_shared_geometry_and_tracks_context_sources(self) -> None:
        adapter = QwenVisionZipAdapter(
            dominant_ratio=0.25,
            contextual_ratio=0.25,
            attention_chunk=1,
            track_source=True,
        )
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor([[7, 99, 99, 99, 99, 8]])
        adapter._image_grid_thw = torch.tensor([[1, 4, 4]])
        adapter._received = torch.ones(1)
        adapter._keys = torch.ones(1)
        adapter._merged_token_signals = lambda: (
            torch.tensor([0.1, 0.9, 0.2, 0.3]),
            torch.eye(4),
        )

        hidden = torch.arange(12).view(1, 6, 2).float()
        position_ids = torch.stack(
            [
                torch.arange(6),
                torch.arange(10, 16),
                torch.arange(20, 26),
                torch.arange(30, 36),
            ]
        ).unsqueeze(1)
        kwargs = {
            "inputs_embeds": hidden,
            "position_ids": position_ids,
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "cache_position": torch.arange(6),
        }

        result = adapter._prefill((), kwargs, hidden)

        self.assertIsNotNone(result)
        expected = hidden[:, [0, 1, 2, 5]].clone()
        expected[:, 1] = torch.tensor([9.0, 11.0])
        torch.testing.assert_close(kwargs["inputs_embeds"], expected)
        torch.testing.assert_close(
            adapter.last_keep_mask,
            torch.tensor([True, True, True, False, False, True]),
        )
        torch.testing.assert_close(
            adapter.get_source_history()[-1],
            torch.tensor(
                [[1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]]
            ),
        )
        self.assertEqual(adapter._prefill_state["kv_len"], 4)
        torch.testing.assert_close(
            kwargs["position_ids"][0], torch.arange(4).unsqueeze(0)
        )

    def test_new_prompt_clears_previous_source_attribution(self) -> None:
        adapter = QwenVisionZipAdapter(
            dominant_ratio=0.25,
            contextual_ratio=0.25,
            track_source=True,
        )
        adapter._source_history = [torch.eye(2)]

        adapter._capture_model_inputs(
            None,
            (),
            {
                "input_ids": torch.tensor([[1, 2]]),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
        )

        self.assertEqual(adapter.get_source_history(), [])


class VisionZipQwenTrainingTest(unittest.TestCase):
    @staticmethod
    def _wrapper() -> VisionZipTrainingQwenForConditionalGeneration:
        wrapper = object.__new__(
            VisionZipTrainingQwenForConditionalGeneration
        )
        nn.Module.__init__(wrapper)
        wrapper.compressor = SimpleNamespace(
            _retain_ratio=0.5,
            _contextual_ratio=0.25,
        )
        wrapper.image_token_id = 99
        wrapper.pad_token_id = 0
        return wrapper

    def test_mrope_ignores_text_only_dummy_grids(self) -> None:
        wrapper = self._wrapper()

        class RopeModel:
            def __init__(self) -> None:
                self.grids = None

            def get_rope_index(
                self,
                input_ids,
                token_types,
                *,
                image_grid_thw,
                attention_mask,
            ):
                self.grids = image_grid_thw.clone()
                positions = torch.zeros(
                    3,
                    input_ids.shape[0],
                    input_ids.shape[1],
                    dtype=torch.long,
                )
                return positions, torch.zeros(input_ids.shape[0], 1)

        rope_model = RopeModel()
        wrapper.base_model = SimpleNamespace(model=rope_model)
        input_ids = torch.tensor(
            [[10, 99, 20], [11, 12, 0], [13, 99, 21]]
        )
        attention_mask = torch.tensor(
            [[1, 1, 1], [1, 1, 0], [1, 1, 1]],
            dtype=torch.bool,
        )
        grids = torch.tensor(
            [[1, 4, 4], [1, 2, 2], [1, 6, 6]]
        )

        wrapper._position_ids(
            input_ids, attention_mask, grids, None
        )

        torch.testing.assert_close(
            rope_model.grids,
            torch.tensor([[1, 4, 4], [1, 6, 6]]),
        )

    def test_batched_image_sizes_preserve_labels_and_gradients(
        self,
    ) -> None:
        wrapper = self._wrapper()
        input_ids = torch.tensor(
            [
                [10, 99, 99, 99, 99, 20, 21, 0, 0, 0, 0],
                [11, 99, 99, 99, 99, 99, 99, 99, 99, 22, 23],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        labels = torch.full_like(input_ids, -100)
        labels[0, 5:7] = torch.tensor([30, 31])
        labels[1, 9:11] = torch.tensor([40, 41])
        inputs_embeds = torch.randn(2, 11, 4, requires_grad=True)
        mrope_ids = (
            torch.arange(11)
            .view(1, 1, 11)
            .expand(3, 2, -1)
            .clone()
        )
        feature_rows = [
            torch.randn(4, 4, requires_grad=True),
            torch.randn(8, 4, requires_grad=True),
        ]
        score_rows = [
            torch.tensor([0.1, 0.9, 0.8, 0.2]),
            torch.arange(8, dtype=torch.float32),
        ]
        metric_rows = [torch.eye(4), torch.eye(8)]

        hidden, positions, ids, mask, updated_labels = (
            wrapper._apply_vision_zip(
                inputs_embeds,
                input_ids,
                mrope_ids,
                attention_mask,
                labels,
                feature_rows,
                score_rows,
                metric_rows,
            )
        )

        self.assertEqual(tuple(hidden.shape), (2, 7, 4))
        self.assertEqual(mask.sum(dim=1).tolist(), [5, 7])
        self.assertEqual(
            [
                span.count
                for span in qwen_visual_spans(ids, 99, mask)
                if span is not None
            ],
            [2, 4],
        )
        torch.testing.assert_close(
            updated_labels[0, 3:5], torch.tensor([30, 31])
        )
        torch.testing.assert_close(
            updated_labels[1, 5:7], torch.tensor([40, 41])
        )
        self.assertEqual(int(positions[0, 0, 3]), 5)
        self.assertEqual(int(positions[0, 1, 5]), 9)

        hidden.sum().backward()
        self.assertGreater(
            float(feature_rows[0].grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(feature_rows[1].grad.abs().sum()), 0.0
        )

    def test_freezing_enables_only_visual_merger(self) -> None:
        base_model = nn.Module()
        base_model.model = nn.Module()
        base_model.model.visual = nn.Module()
        base_model.model.visual.encoder = nn.Linear(4, 4)
        base_model.model.visual.merger = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, 4),
        )
        base_model.model.language_model = nn.Linear(4, 4)

        merger = _freeze_for_merger_tuning(base_model)

        self.assertIs(merger, base_model.model.visual.merger)
        trainable = {
            name
            for name, parameter in base_model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("model.visual.merger.")
                for name in trainable
            )
        )

    def test_args_allow_microbatch_and_artifact_is_merger_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "train.json"
            data.write_text("[]", encoding="utf-8")
            output = root / "results"
            config = SimpleNamespace(
                custom_args={
                    "training_args": {
                        "per_device_train_batch_size": 4,
                        "attn_implementation": "eager",
                        "bf16": False,
                    }
                },
                output_dir=str(output),
                dataset_limit=None,
            )
            backbone = SimpleNamespace(
                _model_id="Qwen/Qwen2.5-VL-7B-Instruct"
            )
            compressor = VisionZipCompressor(
                retain_ratio=0.5,
                contextual_ratio=0.25,
            )
            with patch.dict(
                "os.environ", {"WORLD_SIZE": "1"}
            ):
                args = build_qwen_training_args(
                    backbone, config, compressor, root
                )

            self.assertEqual(args.per_device_train_batch_size, 4)
            self.assertEqual(args.gradient_accumulation_steps, 32)
            self.assertEqual(args.subset_frac, 0.1)
            self.assertEqual(args.deepspeed_stage, 0)
            self.assertEqual(
                Path(args.output_dir), output / "vision_zip"
            )
            self.assertEqual(
                args.checkpoint_name,
                "vision_zip_merger_r0.5.pt",
            )

            wrapper = self._wrapper()
            artifact = Path(args.output_dir) / args.checkpoint_name
            merger_state = {
                "linear.weight": torch.randn(4, 4),
                "linear.bias": torch.randn(4),
            }
            _save_merger_artifact(
                artifact, merger_state, wrapper, args
            )
            payload = torch.load(
                artifact, map_location="cpu", weights_only=True
            )

            self.assertEqual(
                set(payload["state_dict"]), set(merger_state)
            )
            self.assertEqual(
                payload["config"]["training_scope"],
                "visual_merger_only",
            )
            self.assertEqual(
                payload["config"]["module_path"],
                "model.visual.merger",
            )
            self.assertNotIn("language_model", payload["state_dict"])
            self.assertNotIn("visual.encoder", payload["state_dict"])


if __name__ == "__main__":
    unittest.main()
