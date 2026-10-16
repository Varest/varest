import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

import llmoe.compressor.duet.duet_qwen_tr as qwen_training
from llmoe.compressor.duet.duet import DUETCompressor
from llmoe.compressor.duet.duet_qwen import QwenDuetAdapter
from llmoe.compressor.duet.duet_qwen_tr import (
    DuetQwenTrainingDecoder,
    DuetTrainingQwenForConditionalGeneration,
    _visual_spans,
)
from llmoe.compressor.duet.stage1 import duet_stage1_plan
from llmoe.compressor.duet.duet_tr import (
    _build_training_args,
    _compressor_for_training,
    duet_metadata,
)




def test_borrowed_backbone_modules_do_not_alias_duet_state_dict(tmp_path) -> None:
    base_model = nn.Module()
    base_model.language_model = nn.Linear(4, 4, bias=False)
    base_model.vision_tower = nn.Sequential(nn.Linear(4, 4, bias=False))

    compressor = DUETCompressor(max_tokens=128, pretrained=False)
    compressor._set_borrowed_module("_language_model", base_model.language_model)
    compressor._set_borrowed_module("_target_layer", base_model.vision_tower[0])

    training_model = nn.Module()
    training_model.base_model = base_model
    training_model.compressor = compressor
    state = training_model.state_dict()

    assert "_language_model" not in compressor._modules
    assert "_target_layer" not in compressor._modules
    assert not any(key.startswith("compressor._language_model.") for key in state)
    assert not any(key.startswith("compressor._target_layer.") for key in state)
    assert "base_model.language_model.weight" in state
    assert "base_model.vision_tower.0.weight" in state
    save_file(state, tmp_path / "checkpoint.safetensors")

class _RotaryModel:
    def rotary_emb(
        self, _hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table = position_ids.to(torch.float32)
        return table, table.clone()


def _adapter(*, use_stage1: bool, use_stage2: bool) -> QwenDuetAdapter:
    return QwenDuetAdapter(
        dominant_ratio=0.25,
        contextual_ratio=0.25,
        cluster_width=2,
        pdrop_layers=(2,),
        pdrop_keep_ratios=(0.5,),
        text_rater="last",
        score_reduction="mean",
        use_stage1=use_stage1,
        use_stage2=use_stage2,
        track_source=True,
        attention_chunk=1,
    )


class DuetPolicyTest(unittest.TestCase):
    def test_stage1_plan_separates_dominant_anchor_and_merge_rows(self) -> None:
        plan = duet_stage1_plan(
            torch.tensor([[0.1, 0.9, 0.8, 0.7]]),
            torch.eye(4).unsqueeze(0),
            dominant=1,
            contextual=1,
            cluster_width=2,
        )

        torch.testing.assert_close(
            plan.dominant_index, torch.tensor([[1]])
        )
        torch.testing.assert_close(
            plan.anchor_index, torch.tensor([[2]])
        )
        torch.testing.assert_close(
            plan.merge_index, torch.tensor([[3]])
        )


class DuetQwenAdapterTest(unittest.TestCase):
    def test_stage1_uses_shared_prompt_cut_and_tracks_sources(self) -> None:
        adapter = _adapter(use_stage1=True, use_stage2=False)
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor([[7, 99, 99, 99, 99, 8]])
        adapter._image_grid_thw = torch.tensor([[1, 4, 4]])
        adapter._received = torch.ones(1)
        adapter._keys = torch.ones(1)
        adapter._merged_token_signals = lambda: (
            torch.tensor([0.1, 0.9, 0.8, 0.7]),
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
        expected = hidden[:, [0, 2, 3, 5]].clone()
        expected[:, 2] = torch.tensor([14.0, 16.0])
        torch.testing.assert_close(kwargs["inputs_embeds"], expected)
        torch.testing.assert_close(
            adapter.last_keep_mask,
            torch.tensor([True, False, True, True, False, True]),
        )
        torch.testing.assert_close(
            adapter.get_source_history()[-1],
            torch.tensor(
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
            ),
        )
        self.assertEqual(adapter._prefill_state["kv_len"], 4)

    def test_boundary_prune_reuses_shared_mrope_and_mask_geometry(self) -> None:
        adapter = _adapter(use_stage1=False, use_stage2=True)
        adapter._image_start = torch.tensor([1])
        adapter._visual_count = 4
        adapter._keep_counts = (2,)
        adapter._position_ids = torch.stack(
            [
                torch.arange(6),
                torch.arange(10, 16),
                torch.arange(20, 26),
                torch.arange(30, 36),
            ]
        ).unsqueeze(1)
        adapter._language_model = _RotaryModel()
        adapter._boundary_scores = lambda *args, **kwargs: torch.tensor(
            [[0.1, 0.9, 0.8, 0.2]]
        )
        identity = torch.eye(4)
        adapter._composed_source = identity.clone()
        adapter._source_history = [identity.clone()]

        hidden = torch.arange(12).view(1, 6, 2).float()
        kwargs = {
            "position_embeddings": None,
            "attention_mask": torch.tril(
                torch.ones((1, 1, 6, 6), dtype=torch.bool)
            ),
            "cache_position": torch.arange(6),
        }

        pruned = adapter._prune(object(), hidden, kwargs, stage=0)

        torch.testing.assert_close(pruned, hidden[:, [0, 2, 3, 5]])
        state = adapter._stage_state[0]
        self.assertEqual(state["kv_len"], 4)
        self.assertEqual(state["attention_mask"].shape, (1, 1, 4, 4))
        torch.testing.assert_close(
            state["position_ids"], torch.arange(4).unsqueeze(0)
        )
        torch.testing.assert_close(
            adapter.get_source_history()[-1], identity[[1, 2]]
        )


class _TrainingRotary(nn.Module):
    def forward(
        self, _hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table = position_ids.to(torch.float32)
        return table, table.clone()


class _TrainingTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            rope_parameters={"mrope_section": [1, 1, 1]},
            num_hidden_layers=0,
        )
        self.embed_tokens = nn.Embedding(128, 4)
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.norm = nn.Identity()
        self.rotary_emb = _TrainingRotary()
        self.gradient_checkpointing = False


class DuetQwenTrainingTest(unittest.TestCase):
    @staticmethod
    def _stage1_wrapper() -> DuetTrainingQwenForConditionalGeneration:
        wrapper = object.__new__(DuetTrainingQwenForConditionalGeneration)
        nn.Module.__init__(wrapper)
        wrapper.compressor = SimpleNamespace(
            _dominant=None,
            _contextual=None,
            _retain_ratio=0.5,
            _contextual_ratio=0.25,
            _cluster_width=2,
            _use_stage1=True,
        )
        # __init__ is bypassed here, so supply the Stage-1 split it would have
        # back-solved from the layer-averaged retain_ratio.
        wrapper._stage1_ratios = (0.25, 0.25)
        wrapper.image_token_id = 99
        wrapper.pad_token_id = 0
        return wrapper

    def test_qwen_training_resolves_ratio_schedule_and_metadata(self) -> None:
        config = SimpleNamespace(
            custom_args={
                "compressor_args": {
                    "retain_ratio": 0.7,
                    "contextual_ratio": 0.05,
                }
            }
        )

        compressor = _compressor_for_training(config, backbone="qwen")
        metadata = duet_metadata(compressor, backbone="qwen")

        self.assertEqual(compressor._pdrop_layers, (14, 21))
        self.assertEqual(compressor._pdrop_keep_ratios, (0.5, 0.25))
        self.assertEqual(metadata["budget"], 0.7)
        self.assertEqual(metadata["retain_ratio"], 0.7)
        self.assertIsNone(metadata["stage1_tokens"])
        self.assertFalse(metadata["released_training_recipe"])

    def test_qwen_training_args_allow_batched_microbatch(self) -> None:
        config = SimpleNamespace(
            custom_args={
                "training_args": {
                    "per_device_train_batch_size": 4,
                }
            },
            input_dir="/tmp/data",
            output_dir="/tmp/results",
            dataset_limit=None,
        )
        backbone = SimpleNamespace(
            _model_id="Qwen/Qwen2.5-VL-7B-Instruct"
        )
        compressor_config = SimpleNamespace(
            custom_args={
                "compressor_args": {
                    "retain_ratio": 0.7,
                    "contextual_ratio": 0.05,
                }
            }
        )
        compressor = _compressor_for_training(
            compressor_config, backbone="qwen"
        )

        with patch.dict("os.environ", {"WORLD_SIZE": "1"}), patch(
            "llmoe.compressor.duet.duet_tr.resolve_training_data",
            return_value=("/tmp/data.json", "/tmp/images"),
        ):
            args = _build_training_args(
                backbone,
                config,
                "pretrain",
                compressor,
                backbone_kind="qwen",
            )

        self.assertEqual(args.per_device_train_batch_size, 4)
        self.assertEqual(args.gradient_accumulation_steps, 64)
        self.assertEqual(args.stage1_module_path, "model.visual.merger")
        self.assertEqual(args.stage1_artifact_name, "mm_merger.bin")
        self.assertTrue(args.output_dir.endswith("qwen2.5-vl-7b-duet-r0.7-pretrain"))

    def test_qwen_mrope_filters_text_only_dummy_grids(self) -> None:
        wrapper = self._stage1_wrapper()

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
                    3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long
                )
                return positions, torch.zeros(input_ids.shape[0], 1)

        rope_model = RopeModel()
        wrapper.base_model = SimpleNamespace(model=rope_model)
        input_ids = torch.tensor(
            [[10, 99, 20], [11, 12, 0], [13, 99, 21]]
        )
        mask = torch.tensor(
            [[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool
        )
        grids = torch.tensor([[1, 4, 4], [1, 2, 2], [1, 6, 6]])

        wrapper._position_ids(input_ids, mask, grids, None)

        torch.testing.assert_close(
            rope_model.grids, torch.tensor([[1, 4, 4], [1, 6, 6]])
        )

        flat_positions = torch.arange(3).expand(3, -1)
        normalized = wrapper._position_ids(
            input_ids, mask, grids, flat_positions
        )
        torch.testing.assert_close(normalized[0], flat_positions)

    def test_stage1_batches_different_qwen_image_sizes_and_keeps_gradients(
        self,
    ) -> None:
        wrapper = self._stage1_wrapper()
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
        mrope_ids = torch.arange(11).view(1, 1, 11).expand(3, 2, -1).clone()
        feature_rows = [
            torch.randn(4, 4, requires_grad=True),
            torch.randn(8, 4, requires_grad=True),
        ]
        score_rows = [
            torch.tensor([0.1, 0.9, 0.8, 0.2]),
            torch.arange(8, dtype=torch.float32),
        ]
        metric_rows = [torch.eye(4), torch.eye(8)]

        hidden, positions, ids, mask, updated_labels, counts = wrapper._apply_stage1(
            inputs_embeds,
            input_ids,
            mrope_ids,
            attention_mask,
            labels,
            feature_rows,
            score_rows,
            metric_rows,
        )

        self.assertEqual(counts, [2, 4])
        self.assertEqual(tuple(hidden.shape), (2, 7, 4))
        self.assertEqual(mask.sum(dim=1).tolist(), [5, 7])
        self.assertEqual(
            [span.count for span in _visual_spans(ids, 99, mask)],
            [2, 4],
        )
        torch.testing.assert_close(updated_labels[0, 3:5], torch.tensor([30, 31]))
        torch.testing.assert_close(updated_labels[1, 5:7], torch.tensor([40, 41]))
        self.assertEqual(int(positions[0, 0, 3]), 5)
        self.assertEqual(int(positions[0, 1, 5]), 9)

        hidden.sum().backward()
        self.assertGreater(float(feature_rows[0].grad.abs().sum()), 0.0)
        self.assertGreater(float(feature_rows[1].grad.abs().sum()), 0.0)

    def test_stage2_batches_different_keep_counts_and_aligns_labels(self) -> None:
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
        decoder = DuetQwenTrainingDecoder(
            _TrainingTextModel(),
            image_token_id=99,
            pdrop_layers=(1,),
            pdrop_keep_ratios=(0.5,),
            text_rater="last",
            score_reduction="mean",
            tokenizer=tokenizer,
            use_stage2=True,
        )
        input_ids = torch.tensor(
            [
                [10, 99, 99, 99, 99, 20, 21, 0, 0],
                [11, 99, 99, 99, 99, 99, 99, 22, 23],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        labels = torch.full_like(input_ids, -100)
        labels[0, 5:7] = torch.tensor([30, 31])
        labels[1, 7:9] = torch.tensor([40, 41])
        hidden = torch.randn(2, 9, 4, requires_grad=True)
        mrope_ids = torch.arange(9).view(1, 1, 9).expand(3, 2, -1).clone()

        def fake_attention(
            _layer, states, _positions, _mask, raters, **_kwargs
        ):
            columns = torch.arange(
                states.shape[1], dtype=torch.float32, device=states.device
            )
            return columns.view(1, 1, -1).expand(
                states.shape[0], raters.shape[1], -1
            )

        with patch.object(
            qwen_training, "decoder_attention_rows", side_effect=fake_attention
        ):
            new_hidden, _, new_ids, new_mask, new_labels = decoder._prune(
                object(),
                hidden,
                mrope_ids,
                input_ids,
                attention_mask,
                labels,
                [4, 6],
                0.5,
                pad_token_id=0,
            )

        self.assertEqual(tuple(new_hidden.shape), (2, 6, 4))
        self.assertEqual(new_mask.sum(dim=1).tolist(), [5, 6])
        self.assertEqual(
            [span.count for span in _visual_spans(new_ids, 99, new_mask)],
            [2, 3],
        )
        torch.testing.assert_close(new_labels[0, 3:5], torch.tensor([30, 31]))
        torch.testing.assert_close(new_labels[1, 4:6], torch.tensor([40, 41]))
        new_hidden.sum().backward()
        self.assertGreater(float(hidden.grad.abs().sum()), 0.0)

    def test_stage2_text_only_batch_still_touches_selector_for_zero3(self) -> None:
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
        decoder = DuetQwenTrainingDecoder(
            _TrainingTextModel(),
            image_token_id=99,
            pdrop_layers=(1,),
            pdrop_keep_ratios=(0.5,),
            text_rater="last",
            score_reduction="mean",
            tokenizer=tokenizer,
            use_stage2=True,
        )
        input_ids = torch.tensor([[10, 20, 0], [11, 21, 22]])
        attention_mask = torch.tensor(
            [[1, 1, 0], [1, 1, 1]], dtype=torch.bool
        )
        labels = torch.full_like(input_ids, -100)
        hidden = torch.randn(2, 3, 4)
        mrope_ids = torch.arange(3).view(1, 1, 3).expand(3, 2, -1)

        with patch.object(
            qwen_training,
            "decoder_attention_rows",
            return_value=torch.ones(2, 1, 1),
        ) as selector:
            result = decoder._prune(
                object(),
                hidden,
                mrope_ids,
                input_ids,
                attention_mask,
                labels,
                [0, 0],
                0.5,
                pad_token_id=0,
            )

        self.assertEqual(tuple(selector.call_args.args[1].shape), (2, 1, 4))
        torch.testing.assert_close(result[0], hidden)
        torch.testing.assert_close(result[2], input_ids)

if __name__ == "__main__":
    unittest.main()
