"""Two defects that stopped LLaVA-NeXT training before the first step.

DUET: `install()` on the NeXT compressor captures `self._vlm.generate`, but the
training path wraps the model in `_TrainingBackbone`, which supplied only
`model`, `processor` and a `prepare_inputs` stub -- so install raised
`AttributeError: '_TrainingBackbone' object has no attribute 'generate'`.

HiDrop: its NeXT trainer sets `data_path` to the Parquet shard directory
(`hidrop_next_tr.py`: `data_path=str(data_root / "data")`), but provenance
fingerprinted `data_path` strictly as a file, raising `FileNotFoundError`.
"""
import hashlib
import json

import pytest

from llmoe.compressor.duet.duet_tr import _TrainingBackbone
from llmoe.compressor.hidrop.provenance import _fingerprint_dataset, _fingerprint_file


# --- DUET ------------------------------------------------------------------

def training_backbone():
    return _TrainingBackbone(object(), object())


def test_install_can_capture_and_wrap_generate():
    """What install() actually does: read the attribute, then replace it."""
    backbone = training_backbone()
    original = backbone.generate           # the capture that used to raise
    assert callable(original)
    backbone.generate = lambda *a, **k: "wrapped"
    assert backbone.generate() == "wrapped"
    backbone.generate = original           # uninstall restores it
    assert backbone.generate is original


def test_generate_and_prepare_inputs_refuse_to_run():
    """Stubs exist to be wrapped, never called; both must say so, not return None."""
    backbone = training_backbone()
    for name in ("generate", "prepare_inputs"):
        with pytest.raises(RuntimeError):
            getattr(backbone, name)()


# --- HiDrop ----------------------------------------------------------------

def make_shards(root, contents):
    root.mkdir(parents=True, exist_ok=True)
    for name, blob in contents.items():
        (root / name).write_bytes(blob)
    return root


def test_single_file_corpus_is_unchanged(tmp_path):
    """LLaVA-1.5 passes one JSON; that path must behave exactly as before."""
    f = tmp_path / "mix665k.json"
    f.write_text(json.dumps([{"id": 1}]))
    assert _fingerprint_dataset(f) == _fingerprint_file(f)


def test_shard_directory_is_fingerprinted(tmp_path):
    root = make_shards(tmp_path / "data", {"a-00001.parquet": b"one", "a-00002.parquet": b"two"})
    fp = _fingerprint_dataset(root)
    assert fp["size_bytes"] == 6
    assert [s["path"] for s in fp["shards"]] == ["a-00001.parquet", "a-00002.parquet"]
    assert fp["sha256"] == _fingerprint_dataset(root)["sha256"]  # deterministic


def test_fingerprint_detects_edited_content_not_just_names(tmp_path):
    """The guarantee the single-file hash gave: a changed corpus is refused."""
    root = make_shards(tmp_path / "data", {"a.parquet": b"one", "b.parquet": b"two"})
    before = _fingerprint_dataset(root)["sha256"]
    (root / "b.parquet").write_bytes(b"TWO")          # same name, same size
    assert _fingerprint_dataset(root)["sha256"] != before


def test_fingerprint_detects_added_and_removed_shards(tmp_path):
    root = make_shards(tmp_path / "data", {"a.parquet": b"one"})
    before = _fingerprint_dataset(root)["sha256"]
    (root / "b.parquet").write_bytes(b"two")
    added = _fingerprint_dataset(root)["sha256"]
    assert added != before
    (root / "b.parquet").unlink()
    assert _fingerprint_dataset(root)["sha256"] == before


def test_ordering_is_stable_regardless_of_creation_order(tmp_path):
    a = make_shards(tmp_path / "a", {"x.parquet": b"1", "y.parquet": b"2"})
    b = make_shards(tmp_path / "b", {"y.parquet": b"2", "x.parquet": b"1"})
    assert _fingerprint_dataset(a)["sha256"] == _fingerprint_dataset(b)["sha256"]


def test_directory_without_shards_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No dataset shards"):
        _fingerprint_dataset(tmp_path / "empty")


def test_missing_path_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _fingerprint_dataset(tmp_path / "nope")


# --- DUET: text-only rows desync ZeRO-3 -------------------------------------

def write_next_shard(path, rows):
    """A LLaVA-NeXT-shaped shard: `image` is a struct whose `path` may be null."""
    import io
    import pyarrow as pa, pyarrow.parquet as pq
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), (127, 127, 127)).save(buf, format="PNG")
    blob = buf.getvalue()
    image = [None if r is None else {"path": r, "bytes": blob} for r in rows]
    table = pa.table({
        "image": pa.array(image, type=pa.struct([("path", pa.string()), ("bytes", pa.binary())])),
        "conversations": pa.array(
            [[{"from": "human", "value": "<image>\nhi"}, {"from": "gpt", "value": "ok"}]] * len(rows)
        ),
    })
    pq.write_table(table, path, row_group_size=max(1, len(rows)))
    return path


def build(tmp_path, rows):
    from llmoe.compressor.duet.duet_next_tr import LlavaNextParquetDataset
    root = tmp_path / "data"; root.mkdir(parents=True, exist_ok=True)
    write_next_shard(root / "shard-00000.parquet", rows)
    return LlavaNextParquetDataset(root)


def test_text_only_rows_are_excluded(tmp_path):
    """A text-only micro-batch skips the vision modules; under ZeRO-3 that rank
    never issues their parameter all-gather and the job hangs."""
    ds = build(tmp_path, ["a.jpg", None, "b.jpg", None, "c.jpg"])
    assert len(ds) == 3


def test_surviving_rows_are_the_image_bearing_ones_in_order(tmp_path):
    ds = build(tmp_path, ["a.jpg", None, "b.jpg"])
    seen = [ds[i]["image"] is not None for i in range(len(ds))]
    assert seen == [True, True]


def test_all_image_shard_is_unchanged(tmp_path):
    ds = build(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
    assert len(ds) == 3
    assert all(ds[i]["image"] is not None for i in range(len(ds)))


def test_row_groups_cover_exactly_the_kept_rows(tmp_path):
    """The sampler walks row_groups; it must not address dropped rows."""
    ds = build(tmp_path, ["a.jpg", None, "b.jpg", None])
    flat = [i for group in ds.row_groups for i in group]
    assert flat == list(range(len(ds)))
