"""Dataset and answer-mask tests for the NeXT adapter training path."""

from io import BytesIO
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

import llmoe.compressor.quietprune.quietprune_tr as training_module
from llmoe.compressor.quietprune.quietprune_tr import (
    _ParquetConversationDataset,
    _answer_labels,
    _open_training_image,
    _question_and_answer,
    _require_effective_llava_next_flash_attention_2,
    _require_flash_attention_2_package,
    _require_llava_next_flash_attention_2,
)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 3), color).save(stream, format="JPEG")
    return stream.getvalue()


def test_staged_parquet_reader_streams_embedded_images_without_extraction(tmp_path) -> None:
    rows = [
        {
            "id": str(index),
            "conversations": [
                {"from": "human", "value": f"<image>\nquestion {index}"},
                {"from": "gpt", "value": f"answer {index}"},
            ],
            "data_source": "llava-next-instruct",
            "image": {"bytes": _jpeg_bytes((index, 2, 3)), "path": f"{index}.jpg"},
        }
        for index in range(5)
    ]
    path = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    dataset = _ParquetConversationDataset(
        [path], limit=3, seed=7, shuffle_buffer=2, parquet_batch_size=1
    )
    records = list(dataset)

    assert len(records) == 3
    question, answer = _question_and_answer(records[0])
    assert question.startswith("question ") and answer.startswith("answer ")
    image = _open_training_image(dataset, records[0])
    try:
        assert image.mode == "RGB" and image.size == (4, 3)
    finally:
        image.close()
    assert not dataset._usable(
        {
            "image": {"bytes": _jpeg_bytes((0, 0, 0)), "path": "empty.jpg"},
            "conversations": [
                {"from": "human", "value": "<image>\nquestion"},
                {"from": "gpt", "value": ""},
            ],
        }
    )
    assert not dataset._usable(
        {
            "image": {"bytes": None, "path": "not-extracted.jpg"},
            "conversations": rows[0]["conversations"],
        }
    )


def test_answer_labels_follow_right_aligned_active_tokens() -> None:
    ids = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
    attention = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    labels = _answer_labels(ids, attention, [2, 1])
    assert labels.tolist() == [
        [-100, -100, -100, 2, 3],
        [-100, -100, -100, -100, 7],
    ]


def test_next_training_requires_explicit_fa2_but_other_backbones_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdpa = SimpleNamespace(
        compressor_family="llava_next",
        _attn_implementation="sdpa",
        _dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="FlashAttention-2-only"):
        _require_llava_next_flash_attention_2(sdpa)

    monkeypatch.setattr(
        training_module, "_require_flash_attention_2_package", lambda: "2.8.3.post1"
    )
    fa2 = SimpleNamespace(
        compressor_family="llava_next",
        _attn_implementation="flash_attention_2",
        _dtype=torch.bfloat16,
    )
    assert _require_llava_next_flash_attention_2(fa2) == "2.8.3.post1"
    assert _require_llava_next_flash_attention_2(SimpleNamespace()) is None


def test_pinned_external_fa2_distribution_and_extension_are_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_module, "version", lambda _name: "2.8.2")
    with pytest.raises(RuntimeError, match="got 2.8.2"):
        _require_flash_attention_2_package()

    package = SimpleNamespace(flash_attn_func=lambda *_args, **_kwargs: None)
    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        return package if name == "flash_attn" else object()

    monkeypatch.setattr(training_module, "version", lambda _name: "2.8.3.post1")
    monkeypatch.setattr(training_module, "import_module", fake_import)
    assert _require_flash_attention_2_package() == "2.8.3.post1"
    assert imported == ["flash_attn", "flash_attn_2_cuda"]


def test_effective_fa2_is_required_for_both_llama_and_clip() -> None:
    def model(text: str, vision: str) -> SimpleNamespace:
        return SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=text)),
                vision_tower=SimpleNamespace(config=SimpleNamespace(_attn_implementation=vision)),
            )
        )

    assert _require_effective_llava_next_flash_attention_2(
        model("flash_attention_2", "flash_attention_2")
    ) == ("flash_attention_2", "flash_attention_2")
    with pytest.raises(RuntimeError, match="vision='sdpa'"):
        _require_effective_llava_next_flash_attention_2(model("flash_attention_2", "sdpa"))
