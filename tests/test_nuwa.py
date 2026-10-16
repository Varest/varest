import unittest

import torch

from llmoe.compressor.nuwa import NuwaCompressor
from llmoe.compressor.nuwa.nuwa_internvl import InternVLNuwaAdapter
from llmoe.compressor.nuwa.nuwa_qwen import QwenNuwaAdapter
from llmoe.compressor.nuwa.policy import (
    nuwa_stage1,
    pool_grid_blocks,
    pool_merge_groups,
    region_candidate_selection,
    select_by_text_attention,
    tile_stitch_index,
)


class NuwaPolicyTest(unittest.TestCase):
    def test_regions_contribute_their_strongest_token(self) -> None:
        scores = torch.arange(16).view(1, 16).float()

        selected = region_candidate_selection(
            scores, grid_h=4, grid_w=4, top_n_per_region=1
        )

        torch.testing.assert_close(selected, torch.tensor([[5, 7, 13, 15]]))

    def test_stage1_uses_one_shared_selection_and_aggregation_pipeline(self) -> None:
        patches = torch.arange(48).view(1, 16, 3).float()
        similarity = patches.roll(shifts=1, dims=-1)
        scores = torch.arange(16).view(1, 16).float()

        merged, benchmark, source = nuwa_stage1(
            patches,
            similarity,
            scores,
            grid_h=4,
            grid_w=4,
            num_tokens=4,
            top_n_per_region=1,
            distance_threshold=4.0,
        )

        torch.testing.assert_close(benchmark, torch.tensor([[5, 7, 13, 15]]))
        self.assertEqual(merged.shape, (1, 4, 3))
        self.assertEqual(source.shape, (1, 4, 16))
        self.assertTrue(bool(torch.isfinite(merged).all()))

    def test_backbone_pooling_and_tile_mapping_keep_spatial_order(self) -> None:
        values = torch.arange(16).float()
        torch.testing.assert_close(
            pool_merge_groups(values),
            torch.tensor([1.5, 5.5, 9.5, 13.5]),
        )
        torch.testing.assert_close(
            pool_grid_blocks(values, grid_h=4, grid_w=4),
            torch.tensor([2.5, 4.5, 10.5, 12.5]),
        )
        torch.testing.assert_close(
            tile_stitch_index(rows=2, cols=2, tile_h=2, tile_w=2),
            torch.tensor(
                [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
            ),
        )

    def test_internvl_preserves_upstream_thumbnail_and_local_mosaic_groups(self) -> None:
        adapter = InternVLNuwaAdapter(
            stage1_count=15,
            top_n_per_region=1,
            distance=280.0,
            pillar_threshold_percentile=0.5,
            stage2_layer=2,
            vision_feature_layer=-1,
            sim_layers=(0, 1),
            track_source=False,
        )

        groups = adapter._tile_groups(
            tiles=5,
            rows=2,
            cols=2,
            tile_side=2,
            has_thumbnail=True,
            per_tile=3,
            device=torch.device("cpu"),
        )

        # Upstream Nuwa compresses the thumbnail on its own, then stitches all
        # local tiles and compresses that mosaic once. The adapter may process
        # those independent groups in either order because it restores original
        # tile-major positions before returning the features.
        self.assertEqual(len(groups), 2)
        local_order, local_h, local_w, local_budget = groups[0]
        thumbnail_order, thumb_h, thumb_w, thumb_budget = groups[1]
        torch.testing.assert_close(local_order, tile_stitch_index(2, 2, 2, 2))
        torch.testing.assert_close(thumbnail_order, torch.arange(16, 20))
        self.assertEqual((local_h, local_w, local_budget), (4, 4, 12))
        self.assertEqual((thumb_h, thumb_w, thumb_budget), (2, 2, 3))
        self.assertEqual(local_budget + thumb_budget, 15)

    def test_qwen_attention_rule_uses_three_rows_and_head_max(self) -> None:
        attention = torch.zeros((1, 2, 6, 6))
        vision_positions = torch.tensor([[1, 2, 3]])
        attention[:, 0, 3:, 1:4] = torch.tensor([0.6, 0.2, 0.1])
        attention[:, 1, 3:, 1:4] = torch.tensor([0.1, 0.9, 0.8])

        selected = select_by_text_attention(
            attention,
            vision_positions,
            k_keep=2,
            query_rows=3,
            head_reduction="max",
        )

        torch.testing.assert_close(selected, torch.tensor([[1, 2]]))

    def test_qwen_padding_preserves_raster_layout(self) -> None:
        values = torch.tensor([[1.0], [2.0], [3.0]])

        padded = QwenNuwaAdapter._pad_grid(
            values, height=3, width=1, pad_h=1, pad_w=1
        )

        torch.testing.assert_close(
            padded.squeeze(-1),
            torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
        )


class NuwaBackboneParityTest(unittest.TestCase):
    def test_llava_and_qwen_apply_the_same_stage2_keep_set(self) -> None:
        hidden_states = torch.arange(12).view(1, 6, 2).float()
        vision_positions = torch.tensor([[1, 2, 3]])
        attention = torch.zeros((1, 2, 6, 6))
        attention[..., 1] = 0.1
        attention[..., 2] = 0.9
        attention[..., 3] = 0.2
        mask = torch.tril(torch.ones((1, 1, 6, 6), dtype=torch.bool))
        rope = torch.arange(12).view(1, 6, 2).float()
        identity = torch.eye(3)

        llava = NuwaCompressor(
            max_tokens=1, grid_h=2, grid_w=2, track_source=True
        )
        llava._vision_positions = vision_positions.clone()
        llava._stage2_attention = attention.clone()
        llava._stage2_keep = 1
        llava._stage1_source = identity.clone()
        llava._source_history = [identity.clone(), identity.clone()]
        llava_kwargs = {
            "position_ids": torch.arange(6).unsqueeze(0),
            "position_embeddings": (rope.clone(), rope.clone()),
            "attention_mask": mask.clone(),
            "cache_position": torch.arange(6),
        }
        llava_hidden = llava._prune(hidden_states, llava_kwargs)

        qwen = QwenNuwaAdapter(
            retain_ratio=0.5,
            stage1_ratio=0.6,
            stage2_ratio=0.34,
            top_n_per_region=3,
            pillar_threshold_percentile=0.1,
            distance_ratio=0.8,
            stage2_layer=2,
            sim_layers=(0, 1),
            attention_chunk=1,
            track_source=True,
        )
        qwen._image_start = 1
        qwen._visual_count = 3
        qwen._vision_positions = vision_positions.clone()
        qwen._stage2_attention = attention.clone()
        qwen._composed_source = identity.clone()
        qwen._source_history = [identity.clone(), identity.clone()]
        qwen_position_ids = torch.stack(
            [
                torch.arange(6),
                torch.arange(10, 16),
                torch.arange(20, 26),
                torch.arange(30, 36),
            ]
        ).unsqueeze(1)
        qwen_kwargs = {
            "position_ids": qwen_position_ids,
            "position_embeddings": (rope.clone(), rope.clone()),
            "attention_mask": mask.clone(),
            "cache_position": torch.arange(6),
        }
        qwen_hidden = qwen._prune(hidden_states, qwen_kwargs)

        expected = hidden_states[:, [0, 2, 4, 5]]
        torch.testing.assert_close(llava_hidden, expected)
        torch.testing.assert_close(qwen_hidden, expected)
        torch.testing.assert_close(
            llava.get_source_history()[-1], qwen.get_source_history()[-1]
        )
        self.assertEqual(llava._prune_state["kv_len"], 4)
        self.assertEqual(qwen._prune_state["kv_len"], 4)
        self.assertEqual(llava._prune_state["attention_mask"].shape, (1, 1, 4, 4))
        self.assertEqual(qwen._prune_state["attention_mask"].shape, (1, 1, 4, 4))
        torch.testing.assert_close(
            qwen._prune_state["position_ids"], torch.arange(4).unsqueeze(0)
        )


if __name__ == "__main__":
    unittest.main()
