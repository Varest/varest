"""Lightweight contracts for CORE's ConvNeXt/OMG-Seg + Llama-3 port."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from llmoe.backbone import BACKBONES
from llmoe.compressor.core.core import (
    CORECompressor,
    CoreConfig,
    CoreLlama3VLM,
    CoreModel,
    CoreTokenBudget,
    CoreVLM,
    INFERENCE_ATTENTION_IMPLEMENTATION,
    ObjectCentricMerger,
    StubSegmentationVision,
    TRAINING_ATTENTION_IMPLEMENTATION,
    checkpoint_signature,
    format_inference_prompt,
    retention_target,
    validate_checkpoint_compatibility,
)
from llmoe.compressor.core.core_data import (
    CoreParquetConversation,
    CoreParquetDataset,
    RowGroupDistributedSampler,
)
from llmoe.compressor.core.core_llama3_lm import (
    CORE_LLAMA3_IMPLEMENTATION,
    DEFAULT_CORE_LLAMA3_MODEL,
    DEFAULT_CORE_LLAMA3_REVISION,
    LLAMA3_LORA_TARGETS,
    _load_prefixed_llama,
    resolve_llama3_source_path,
    source_config,
    text_config,
    validate_llama3_source,
    validate_llama3_tokenizer,
)
from llmoe.compressor.core.core_tr import (
    IGNORE_INDEX,
    REQUIRED_FLASH_ATTN_VERSION,
    Conversation,
    CoreDataCollator,
    TrainArgs,
    _resolve_training_data,
    _train_args,
    checkpoint_config_payload,
    require_flash_attention_2,
    training_provenance,
)


TARGET_SNAPSHOT = Path(
    "/workspace/output/hf_cache/models--llava-hf--llama3-llava-next-8b-hf/"
    "snapshots/b041c0d0ea0dd0196d147206c210c8d1752fc2da"
)
PRETRAIN_ROOT = Path("/workspace/input/LLaVA-Pretrain")
INSTRUCTION_ROOT = Path("/workspace/input/LLaVA-NeXT-Data")


def _source_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        llm_name_or_path=str(TARGET_SNAPSHOT),
        llm_revision=None,
    )


def _tokenizer():
    from transformers import AutoTokenizer

    if not TARGET_SNAPSHOT.is_dir():
        raise unittest.SkipTest("pinned Llama-3 checkpoint metadata is not cached")
    return AutoTokenizer.from_pretrained(
        TARGET_SNAPSHOT,
        use_fast=True,
        local_files_only=True,
    )


def _image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


class Llama3SourceTests(unittest.TestCase):
    def test_default_model_id_resolves_the_supplied_flat_offline_cache(self) -> None:
        if not TARGET_SNAPSHOT.is_dir():
            self.skipTest("pinned checkpoint metadata is unavailable")
        with patch.dict(
            "os.environ",
            {"HF_HOME": "/workspace/output/hf_cache", "HF_HUB_CACHE": ""},
        ):
            resolved = resolve_llama3_source_path(
                DEFAULT_CORE_LLAMA3_MODEL,
                DEFAULT_CORE_LLAMA3_REVISION,
            )
        self.assertEqual(Path(resolved).resolve(), TARGET_SNAPSHOT.resolve())

    def test_pinned_wrapper_contains_exact_llama3_8b_instruct_metadata(self) -> None:
        if not TARGET_SNAPSHOT.is_dir():
            self.skipTest("pinned checkpoint metadata is unavailable")
        source = source_config(_source_namespace(), local_files_only=True)
        nested = validate_llama3_source(source)
        self.assertEqual(nested.hidden_size, 4096)
        self.assertEqual(nested.num_hidden_layers, 32)
        self.assertEqual(nested.num_key_value_heads, 8)
        self.assertEqual(nested.max_position_embeddings, 8192)
        self.assertEqual(getattr(source, "model_type"), "llava_next")

    def test_pinned_tokenizer_ids_and_template(self) -> None:
        tokenizer = _tokenizer()
        validate_llama3_tokenizer(tokenizer)
        self.assertEqual(tokenizer.convert_tokens_to_ids("<image>"), 128256)
        self.assertEqual(tokenizer.convert_tokens_to_ids("<|eot_id|>"), 128009)
        self.assertIn("start_header_id", tokenizer.chat_template)

    def test_inference_prompt_uses_source_chat_template(self) -> None:
        tokenizer = _tokenizer()
        prompt = format_inference_prompt(
            "<image>", "What is shown?", "llama3", tokenizer=tokenizer
        )
        self.assertTrue(prompt.startswith("<|begin_of_text|>"))
        self.assertIn("<|start_header_id|>user<|end_header_id|>", prompt)
        self.assertTrue(
            prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
        )
        self.assertEqual(prompt.count("<image>"), 1)

    def test_wrapper_prefix_is_mapped_into_plain_llama(self) -> None:
        if not TARGET_SNAPSHOT.is_dir():
            self.skipTest("pinned checkpoint metadata is unavailable")
        source = source_config(_source_namespace(), local_files_only=True)
        fake_model = SimpleNamespace(config=text_config(source))
        loading = {
            "missing_keys": [],
            "unexpected_keys": ["vision_tower.vision_model.embeddings.patch_embedding.weight"],
            "mismatched_keys": [],
        }
        with patch(
            "transformers.LlamaForCausalLM.from_pretrained",
            return_value=(fake_model, loading),
        ) as loader:
            result = _load_prefixed_llama(
                str(TARGET_SNAPSHOT),
                source,
                dtype=torch.bfloat16,
                local_files_only=True,
                attention_implementation="flash_attention_2",
                revision=DEFAULT_CORE_LLAMA3_REVISION,
            )
        self.assertIs(result, fake_model)
        kwargs = loader.call_args.kwargs
        self.assertEqual(kwargs["key_mapping"], {r"^language_model\.": ""})
        self.assertEqual(kwargs["config"].model_type, "llama")
        self.assertEqual(kwargs["config"].hidden_size, 4096)

    def test_checkpoint_index_has_separate_language_and_unused_vision_keys(self) -> None:
        index_path = TARGET_SNAPSHOT / "model.safetensors.index.json"
        if not index_path.is_file():
            self.skipTest("checkpoint index is unavailable")
        weight_map = json.loads(index_path.read_text())["weight_map"]
        self.assertIn("language_model.model.layers.0.self_attn.q_proj.weight", weight_map)
        self.assertTrue(any(key.startswith("vision_tower.") for key in weight_map))
        self.assertIn("multi_modal_projector.linear_1.weight", weight_map)
        self.assertIn("image_newline", weight_map)


class AttentionBackendTests(unittest.TestCase):
    def test_inference_defaults_to_sdpa(self) -> None:
        config = CoreConfig()
        self.assertEqual(
            config.attention_implementation,
            INFERENCE_ATTENTION_IMPLEMENTATION,
        )
        self.assertEqual(INFERENCE_ATTENTION_IMPLEMENTATION, "sdpa")

    def test_training_rejects_sdpa_before_package_import(self) -> None:
        config = SimpleNamespace(attention_implementation="sdpa")
        with patch(
            "llmoe.compressor.core.core_tr.distribution_version"
        ) as package_version:
            with self.assertRaisesRegex(ValueError, "only attention_implementation"):
                require_flash_attention_2(config)
        package_version.assert_not_called()

    def test_training_requires_the_exact_compiled_fa2_build(self) -> None:
        config = SimpleNamespace(
            attention_implementation=TRAINING_ATTENTION_IMPLEMENTATION
        )
        fake_flash_attn = SimpleNamespace(flash_attn_func=lambda *args: None)
        with (
            patch(
                "llmoe.compressor.core.core_tr.distribution_version",
                return_value=REQUIRED_FLASH_ATTN_VERSION,
            ),
            patch(
                "llmoe.compressor.core.core_tr.importlib.import_module",
                side_effect=lambda name: (
                    fake_flash_attn if name == "flash_attn" else SimpleNamespace()
                ),
            ) as importer,
        ):
            require_flash_attention_2(config)
        self.assertEqual(
            [call.args[0] for call in importer.call_args_list],
            ["flash_attn", "flash_attn_2_cuda"],
        )

    def test_training_rejects_a_different_flash_attn_version(self) -> None:
        config = SimpleNamespace(
            attention_implementation=TRAINING_ATTENTION_IMPLEMENTATION
        )
        with patch(
            "llmoe.compressor.core.core_tr.distribution_version",
            return_value="2.8.2",
        ):
            with self.assertRaisesRegex(RuntimeError, REQUIRED_FLASH_ATTN_VERSION):
                require_flash_attention_2(config)


class ArchitectureAndBudgetTests(unittest.TestCase):
    def _local_vlm(self) -> CoreLlama3VLM:
        if not TARGET_SNAPSHOT.is_dir():
            self.skipTest("pinned checkpoint metadata is unavailable")
        return CoreLlama3VLM(
            config={
                "llm_name_or_path": str(TARGET_SNAPSHOT),
                "llm_revision": None,
            },
            pretrained=False,
            device="cpu",
        )

    def test_registration_is_decoder_specific_and_lazy(self) -> None:
        self.assertIs(BACKBONES["core"], CoreVLM)
        self.assertIs(BACKBONES["core-llama3-8b"], CoreLlama3VLM)
        self.assertNotEqual(BACKBONES["llava-next-8b"], CoreLlama3VLM)
        vlm = self._local_vlm()
        self.assertIsNone(vlm._model)
        self.assertEqual(vlm.weights_key, "core-llama3-8b")
        self.assertEqual(vlm.core_config.llm_family, "llama3")
        self.assertEqual(vlm.core_config.num_visual_tokens, 1024)
        self.assertEqual(vlm.core_config.vision_timm_model,
                         "convnext_large_mlp.clip_laion2b_ft_soup_320")
        self.assertEqual(vlm.core_config.num_queries, 300)

    def test_core_model_uses_stub_of_the_same_f3_and_mask_geometry(self) -> None:
        config = CoreConfig(
            image_size=64,
            feature_stride=32,
            fixed_tokens=4,
            llm_hidden_size=8,
        )
        model = CoreModel(config, segmentation=StubSegmentationVision(config))
        self.assertIsInstance(model.vision, StubSegmentationVision)
        self.assertEqual(config.grid_size, 2)
        self.assertEqual(config.num_visual_tokens, 4)
        self.assertEqual(model.projector.fc1.in_features, 1536)
        self.assertEqual(model.projector.fc2.out_features, 8)

    def test_model_captures_budget_after_f3_assembly_before_merge(self) -> None:
        config = CoreConfig(
            image_size=64,
            feature_stride=32,
            fixed_tokens=None,
            retain_ratio=0.5,
            llm_hidden_size=8,
        )
        model = CoreModel(config, segmentation=StubSegmentationVision(config))
        projected = model.encode_images(torch.zeros(2, 3, 64, 64))
        self.assertEqual([row.shape[0] for row in projected], [2, 2])
        self.assertEqual(len(model.last_budget), 2)
        for budget in model.last_budget:
            self.assertEqual(budget.original_content_tokens, 4)
            self.assertEqual(budget.target_content_tokens, 2)
            self.assertEqual(budget.realized_content_tokens, 2)
            self.assertEqual(budget.structural_tokens, 0)

    def test_inference_expands_one_marker_to_the_exact_core_budget(self) -> None:
        vlm = self._local_vlm()
        vlm._tokenizer = _tokenizer()
        prompt = format_inference_prompt(
            "<image>", "Name the object.", "llama3", tokenizer=vlm._tokenizer
        )
        ids, attention = vlm._tokenize_with_image_runs([prompt], [160])
        image_id = vlm._tokenizer.convert_tokens_to_ids("<image>")
        self.assertEqual(int((ids == image_id).sum()), 160)
        self.assertEqual(int(attention.sum()), ids.shape[1])
        positions = (ids[0] == image_id).nonzero(as_tuple=True)[0]
        self.assertEqual(int(positions[-1] - positions[0] + 1), 160)
        self.assertEqual(int((ids == vlm._tokenizer.bos_token_id).sum()), 1)

    def test_fixed_budget_is_exact_without_view_rounding(self) -> None:
        features = torch.arange(1024 * 3, dtype=torch.float32).reshape(1024, 3)
        masks = torch.rand(19, 32, 32, generator=torch.Generator().manual_seed(7))
        for budget in (1, 160, 1024):
            merger = ObjectCentricMerger(
                merge_mode="hard", rate="fixed", fixed_tokens=budget
            )
            result = merger.merge(features, masks)
            self.assertEqual(result.tokens.shape, (budget, 3))
            self.assertEqual(result.source.shape, (budget, 1024))
            torch.testing.assert_close(result.source.sum(dim=1), torch.ones(budget))
            self.assertTrue(torch.equal(result.centroids, result.centroids.sort().values))

    def test_ratio_target_uses_each_original_content_length(self) -> None:
        self.assertEqual(retention_target(1152, 0.25), 288)
        self.assertEqual(retention_target(1024, 0.25), 256)
        self.assertEqual(retention_target(3, 0.1), 1)

        merger = ObjectCentricMerger(
            merge_mode="hard", rate="fixed", retain_ratio=0.25
        )
        for original, height, width in ((12, 3, 4), (8, 2, 4)):
            features = torch.arange(original * 3, dtype=torch.float32).reshape(
                original, 3
            )
            masks = torch.rand(
                5,
                height,
                width,
                generator=torch.Generator().manual_seed(original),
            )
            result = merger.merge(features, masks)
            expected = max(1, round(0.25 * original))
            self.assertEqual(result.tokens.shape[0], expected)
            self.assertEqual(
                merger.last_budget,
                CoreTokenBudget(
                    original_content_tokens=original,
                    requested_retention_ratio=0.25,
                    target_content_tokens=expected,
                    realized_content_tokens=expected,
                    realized_retention_ratio=expected / original,
                    structural_tokens=0,
                ),
            )

    def test_budget_records_are_kept_per_evaluation_sample(self) -> None:
        records = (
            CoreTokenBudget(1152, 0.25, 288, 288, 0.25, 0),
            CoreTokenBudget(1024, 0.25, 256, 256, 0.25, 0),
        )
        compressor = CORECompressor(retain_ratio=0.25, pretrained=False)
        compressor._last_budget = records
        serialized = compressor.get_benchmark_record_metadata(2)
        self.assertEqual(serialized[0]["compression_budget"]["original_content_tokens"], 1152)
        self.assertEqual(serialized[1]["compression_budget"]["target_content_tokens"], 256)
        self.assertEqual(serialized[0]["compression_budget"]["structural_tokens"], 0)

    def test_llama3_target_rejects_legacy_absolute_budget(self) -> None:
        vlm = self._local_vlm()
        compressor = CORECompressor(max_tokens=160, pretrained=False)
        with self.assertRaisesRegex(ValueError, "requires retain_ratio"):
            compressor.install(vlm)

    def test_config_rejects_more_than_the_single_32x32_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds the visual grid"):
            CoreConfig(fixed_tokens=1025).validate()

    def test_compressor_rejects_native_llava_next(self) -> None:
        native = SimpleNamespace(compressor_family="llava_next")
        compressor = CORECompressor(max_tokens=160, pretrained=False)
        with self.assertRaisesRegex(TypeError, "ConvNeXt-L/Mask2Former"):
            compressor.install(native)

    def test_install_and_uninstall_handshake_uses_core_model(self) -> None:
        config = CoreConfig(image_size=64, feature_stride=32, fixed_tokens=4)
        built = SimpleNamespace(merger=None, get_source_history=lambda: [])

        class FakeCore:
            core_config = config
            _model = built
            _adapter_dir = None
            _track_source = False
            loaded = True

            @property
            def model(self):
                return self._model

        compressor = CORECompressor(max_tokens=2, pretrained=False)
        fake = FakeCore()
        compressor.install(fake)
        self.assertIs(built.merger, compressor.merger)
        self.assertEqual(config.fixed_tokens, 2)
        compressor.uninstall()
        self.assertIsNone(compressor._installed_vlm)


class SequenceAndCacheTests(unittest.TestCase):
    class _EmbeddingLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(300, 8)

        def get_input_embeddings(self):
            return self.embedding

    def test_training_expands_marker_mask_and_labels_together(self) -> None:
        config = CoreConfig(
            image_size=64,
            feature_stride=32,
            fixed_tokens=3,
            llm_hidden_size=8,
        )
        model = CoreModel(
            config,
            segmentation=StubSegmentationVision(config),
            llm=self._EmbeddingLM(),
        )
        model.set_image_token_id(256)
        input_ids = torch.tensor([[1, 12, 256, 13, 14]])
        attention = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, -100, 13, 14]])
        visual = [torch.randn(3, 8)]
        embeds, expanded_attention, expanded_labels = model._prepare_multimodal_inputs(
            input_ids, visual, attention, labels, max_length=16
        )
        self.assertEqual(embeds.shape, (1, 7, 8))
        self.assertEqual(expanded_attention.tolist(), [[1] * 7])
        self.assertEqual(expanded_labels.tolist(), [[-100] * 5 + [13, 14]])
        torch.testing.assert_close(embeds[0, 2:5], visual[0])

    def test_text_only_row_keeps_no_visual_positions(self) -> None:
        config = CoreConfig(
            image_size=64,
            feature_stride=32,
            fixed_tokens=4,
            llm_hidden_size=8,
        )
        model = CoreModel(
            config,
            segmentation=StubSegmentationVision(config),
            llm=self._EmbeddingLM(),
        )
        model.set_image_token_id(256)
        ids = torch.tensor([[1, 7, 8]])
        labels = torch.tensor([[-100, 7, 8]])
        embeds, mask, output_labels = model._prepare_multimodal_inputs(
            ids,
            [torch.randn(4, 8)],
            torch.ones_like(ids),
            labels,
            has_image=torch.tensor([False]),
        )
        self.assertEqual(embeds.shape[1], 3)
        self.assertEqual(mask.tolist(), [[1, 1, 1]])
        self.assertEqual(output_labels.tolist(), labels.tolist())

    def test_tiny_llama_prefill_and_decode_use_dynamic_cache(self) -> None:
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.cache_utils import DynamicCache

        tiny = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
            )
        ).eval()
        embeds = torch.randn(1, 5, 16)
        with torch.inference_mode():
            prefill = tiny(
                inputs_embeds=embeds,
                attention_mask=torch.ones(1, 5, dtype=torch.long),
                use_cache=True,
            )
            self.assertIsInstance(prefill.past_key_values, DynamicCache)
            self.assertEqual(prefill.past_key_values.get_seq_length(), 5)
            decode = tiny(
                input_ids=torch.tensor([[3]]),
                attention_mask=torch.ones(1, 6, dtype=torch.long),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
        self.assertEqual(decode.logits.shape, (1, 1, 64))
        self.assertEqual(decode.past_key_values.get_seq_length(), 6)


class CollatorAndDatasetTests(unittest.TestCase):
    def _collator(self, *, stage: int, visual_tokens: int = 4) -> CoreDataCollator:
        return CoreDataCollator(
            _tokenizer(),
            lambda images: torch.zeros(len(images), 3, 16, 16),
            str(INSTRUCTION_ROOT),
            image_token="<image>",
            max_length=512,
            visual_tokens=visual_tokens,
            stage=stage,
            llm_family="llama3",
        )

    def test_stage1_uses_plain_caption_recipe(self) -> None:
        collator = self._collator(stage=1)
        record = CoreParquetConversation(
            "sample",
            "test",
            _image_bytes(),
            None,
            (("human", "<image>\nDescribe."), ("gpt", "A blue square.")),
        )
        batch = collator([record])
        self.assertEqual(batch["pixel_values"].shape, (1, 3, 16, 16))
        self.assertTrue(batch["has_image"].item())
        ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        image_id = collator.tokenizer.convert_tokens_to_ids("<image>")
        self.assertTrue(torch.all(labels[ids == image_id] == IGNORE_INDEX))
        supervised = ids[labels != IGNORE_INDEX].tolist()
        self.assertIn("blue square", collator.tokenizer.decode(supervised).lower())

    def test_stage2_uses_llama3_template_and_assistant_only_labels(self) -> None:
        collator = self._collator(stage=2)
        ids, labels = collator._render_ids_labels(
            [
                ("human", "<image>\nWhat colour?"),
                ("gpt", "Blue."),
                ("human", "Be concise."),
                ("gpt", "Okay."),
            ],
            has_image=True,
        )
        decoded = collator.tokenizer.decode(ids)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", decoded)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", decoded)
        supervised = collator.tokenizer.decode(
            [token for token, label in zip(ids, labels, strict=True) if label != IGNORE_INDEX]
        )
        self.assertIn("Blue.", supervised)
        self.assertIn("Okay.", supervised)
        self.assertNotIn("What colour?", supervised)
        self.assertIn("<|eot_id|>", supervised)

    def test_stage2_text_only_row_has_no_image_marker(self) -> None:
        collator = self._collator(stage=2)
        record = Conversation(
            image_path=None,
            turns=[("human", "Count to two."), ("gpt", "One, two.")],
        )
        batch = collator([record])
        image_id = collator.tokenizer.convert_tokens_to_ids("<image>")
        self.assertFalse(bool((batch["input_ids"] == image_id).any()))
        self.assertFalse(batch["has_image"].item())
        self.assertTrue(bool((batch["labels"] != IGNORE_INDEX).any()))

    def test_stage1_resolves_local_llava_558k(self) -> None:
        if not PRETRAIN_ROOT.is_dir():
            self.skipTest("LLaVA-558K is unavailable")
        config = SimpleNamespace(input_dir="/workspace/input")
        annotations, images = _resolve_training_data(config, 1, {})
        self.assertEqual(annotations.name, "blip_laion_cc_sbu_558k.json")
        self.assertEqual(images.resolve(), (PRETRAIN_ROOT / "images").resolve())

    def test_stage2_reads_supplied_parquet_directly(self) -> None:
        if not INSTRUCTION_ROOT.is_dir():
            self.skipTest("LLaVA-NeXT-Data is unavailable")
        dataset = CoreParquetDataset(INSTRUCTION_ROOT)
        self.assertEqual(len(dataset.shards), 250)
        self.assertGreater(len(dataset), 0)
        record = dataset[0]
        self.assertTrue(record.turns)
        provenance = dataset.provenance()
        self.assertEqual(provenance["dataset"], "LLaVA-NeXT-Data")
        self.assertEqual(provenance["parquet_shards"], 250)
        self.assertGreater(len(dataset.row_groups), 0)

        class TinyDataset:
            row_groups = (
                SimpleNamespace(start=0, length=3),
                SimpleNamespace(start=3, length=3),
            )

            def __len__(self):
                return 6

        sampler = RowGroupDistributedSampler(TinyDataset(), num_replicas=2, rank=0, seed=9)
        sampled = list(iter(sampler))
        self.assertEqual(len(sampled), len(sampler))
        self.assertTrue(all(0 <= index < 6 for index in sampled))

    def test_training_arg_routes_keep_stage_datasets_distinct(self) -> None:
        stage1_config = SimpleNamespace(
            input_dir="/workspace/input",
            output_dir="/tmp/output",
            custom_args={
                "training_args": {
                    "data_json": str(PRETRAIN_ROOT / "blip_laion_cc_sbu_558k.json"),
                    "image_root": str(PRETRAIN_ROOT / "images"),
                }
            },
        )
        stage1 = _train_args(stage1_config, 1)
        self.assertIsNone(stage1.dataset_path)
        self.assertIn("blip_laion_cc_sbu_558k.json", stage1.data_json)

        stage2_config = SimpleNamespace(
            input_dir="/workspace/input",
            output_dir="/tmp/output",
            custom_args={
                "training_args": {
                    "dataset_path": str(INSTRUCTION_ROOT),
                    "projector_checkpoint": "/tmp/stage1/final/projector.pt",
                }
            },
        )
        stage2 = _train_args(stage2_config, 2)
        self.assertEqual(Path(stage2.dataset_path), INSTRUCTION_ROOT)
        self.assertEqual(stage2.data_json, "")


class ProvenanceAndConfigTests(unittest.TestCase):
    def _config(self) -> CoreConfig:
        if not TARGET_SNAPSHOT.is_dir():
            self.skipTest("pinned checkpoint metadata is unavailable")
        return CoreConfig(
            llm_name_or_path=str(TARGET_SNAPSHOT),
            llm_revision=DEFAULT_CORE_LLAMA3_REVISION,
            fixed_tokens=1024,
            attention_implementation=TRAINING_ATTENTION_IMPLEMENTATION,
        )

    def test_checkpoint_payload_identifies_decoder_and_core_vision(self) -> None:
        config = self._config()
        payload = checkpoint_config_payload(config)
        self.assertEqual(payload["_llm_family"], "llama3")
        self.assertEqual(payload["_implementation"], CORE_LLAMA3_IMPLEMENTATION)
        self.assertEqual(
            payload["_evaluation_budget_convention"],
            "per_sample_round_ratio_from_premerge_content_tokens",
        )
        self.assertEqual(payload["_evaluation_structural_tokens"], 0)
        signature = checkpoint_signature(config, include_lora=True)
        self.assertEqual(signature["vision"]["native_tokens"], 1024)
        self.assertIn("ConvNeXt-L", signature["vision"]["architecture"])
        self.assertEqual(signature["lora"]["target_modules"], list(LLAMA3_LORA_TARGETS))

    def test_checkpoint_rejects_a_vicuna_family_tag(self) -> None:
        config = self._config()
        with TemporaryDirectory() as directory:
            payload = checkpoint_config_payload(config)
            payload["_llm_family"] = "llama"
            payload["_implementation"] = "core-convnext-omgseg-v1"
            Path(directory, "core_config.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                validate_checkpoint_compatibility(
                    config,
                    directory,
                    include_lora=False,
                    purpose="test checkpoint",
                )

    def test_provenance_names_558k_and_parquet_stages(self) -> None:
        config = self._config()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "blip_laion_cc_sbu_558k.json"
            annotations.write_text("[]")
            images = root / "images"
            images.mkdir()
            stage1 = TrainArgs(
                data_json=str(annotations),
                image_root=str(images),
                output_dir=str(root / "out1"),
                stage=1,
            )
            manifest1 = training_provenance(config, stage1)
            self.assertEqual(manifest1["dataset"]["name"], "LLaVA-558K")
            self.assertEqual(manifest1["prompt_recipe"], "llava_plain")
            self.assertEqual(
                manifest1["attention"]["training_backend"],
                TRAINING_ATTENTION_IMPLEMENTATION,
            )
            self.assertEqual(
                manifest1["attention"]["required_flash_attn_version"],
                REQUIRED_FLASH_ATTN_VERSION,
            )
            self.assertEqual(
                manifest1["attention"]["inference_backend"],
                INFERENCE_ATTENTION_IMPLEMENTATION,
            )
            self.assertEqual(
                manifest1["merging"]["evaluation_budget_semantics"],
                "max(1, round(requested_retention_ratio * "
                "original_content_tokens)) per source image",
            )

            parquet = root / "next" / "data"
            parquet.mkdir(parents=True)
            (parquet / "part-000.parquet").write_bytes(b"metadata-only-test")
            stage2 = TrainArgs(
                data_json="",
                image_root=str(root / "next"),
                dataset_path=str(root / "next"),
                output_dir=str(root / "out2"),
                stage=2,
            )
            manifest2 = training_provenance(config, stage2)
            self.assertEqual(manifest2["dataset"]["name"], "LLaVA-NeXT-Data")
            self.assertEqual(manifest2["prompt_recipe"], "llama3_instruct")
            self.assertIn("shard_manifest_sha256", manifest2["dataset"])


if __name__ == "__main__":
    unittest.main()
