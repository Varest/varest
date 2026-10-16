"""Direct reader for the supplied LLaVA-NeXT-Data CORE training corpus.

The corpus stores conversations and optional images in Parquet.  CORE still
uses its own 1024x1024 ConvNeXt preprocessing; this module only exposes the
records and does not invoke a LLaVA-NeXT processor or construct AnyRes crops.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class CoreParquetConversation:
    """One standard LLaVA conversation with an embedded or referenced image."""

    sample_id: str
    data_source: str
    image_bytes: bytes | None
    image_path: str | None
    turns: tuple[tuple[str, str], ...]

    @property
    def has_image(self) -> bool:
        return self.image_bytes is not None or self.image_path is not None


@dataclass(frozen=True)
class _RowGroup:
    path: str
    index: int
    start: int
    length: int
    row_offsets: tuple[int, ...] | None = None


class CoreParquetDataset(Dataset[CoreParquetConversation]):
    """Random-access Parquet reader with a small per-worker row-group cache.

    The Llama-3 stage-2 route retains text-only rows so they can train the Llama
    adapter, while :class:`CoreDataCollator` marks them with ``has_image=False``
    and therefore never inserts their dummy vision features.  The optional
    ``multimodal_only`` view remains useful for focused corpus inspection, but
    stage 1 does not use this reader: it uses the LLaVA-558K JSON/image tree.
    """

    columns = ("id", "conversations", "data_source", "image")

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        cached_row_groups: int = 2,
        multimodal_only: bool = False,
    ) -> None:
        import pyarrow.parquet as pq

        self.root = Path(dataset_path).expanduser().resolve()
        files = sorted((self.root / "data").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet shards found under {self.root / 'data'}")

        self.row_groups: list[_RowGroup] = []
        start = 0
        corpus_rows = 0
        multimodal_rows = 0
        first_text_only_index: int | None = None
        for path in files:
            parquet = pq.ParquetFile(path)
            metadata = parquet.metadata
            try:
                path_column = next(
                    index
                    for index in range(metadata.num_columns)
                    if parquet.schema.column(index).path == "image.path"
                )
                bytes_column = next(
                    index
                    for index in range(metadata.num_columns)
                    if parquet.schema.column(index).path == "image.bytes"
                )
            except StopIteration as error:
                raise RuntimeError(
                    f"{path} does not have the expected image.path/image.bytes schema"
                ) from error

            for group_index in range(metadata.num_row_groups):
                physical_length = int(metadata.row_group(group_index).num_rows)
                corpus_start = corpus_rows
                corpus_rows += physical_length
                statistics = metadata.row_group(group_index).column(path_column).statistics
                bytes_statistics = metadata.row_group(group_index).column(bytes_column).statistics
                if statistics is None or bytes_statistics is None:
                    raise RuntimeError(
                        f"Missing image validity statistics in {path}:{group_index}"
                    )
                nulls = int(statistics.null_count)
                if int(bytes_statistics.null_count) != nulls:
                    raise RuntimeError(
                        f"image bytes/path validity disagrees in {path}:{group_index}"
                    )
                multimodal_rows += physical_length - nulls

                valid: list[bool] | None = None
                if nulls and nulls != physical_length:
                    image = parquet.read_row_group(
                        group_index, columns=["image.path"]
                    ).column("image").combine_chunks()
                    valid = image.field("path").is_valid().to_pylist()
                if first_text_only_index is None and nulls:
                    first_offset = 0 if valid is None else valid.index(False)
                    first_text_only_index = corpus_start + first_offset

                offsets: tuple[int, ...] | None = None
                if multimodal_only:
                    if nulls == physical_length:
                        continue
                    if nulls:
                        assert valid is not None
                        offsets = tuple(index for index, keep in enumerate(valid) if keep)
                length = physical_length if offsets is None else len(offsets)
                self.row_groups.append(
                    _RowGroup(str(path), group_index, start, length, offsets)
                )
                start += length

        self._length = start
        self._starts = [group.start for group in self.row_groups]
        self.cached_row_groups = max(1, int(cached_row_groups))
        self.multimodal_only = bool(multimodal_only)
        self.corpus_rows = corpus_rows
        self.multimodal_rows = multimodal_rows
        self.text_only_rows = corpus_rows - multimodal_rows
        self.first_text_only_index = first_text_only_index
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self.shards = tuple(str(path) for path in files)

    def __len__(self) -> int:
        return self._length

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def row_group_for_index(self, index: int) -> int:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return bisect_right(self._starts, index) - 1

    def _read_group(self, group_index: int) -> Any:
        import pyarrow.parquet as pq

        if group_index in self._cache:
            table = self._cache.pop(group_index)
            self._cache[group_index] = table
            return table
        group = self.row_groups[group_index]
        table = pq.ParquetFile(group.path).read_row_group(
            group.index,
            columns=list(self.columns),
        )
        self._cache[group_index] = table
        while len(self._cache) > self.cached_row_groups:
            self._cache.popitem(last=False)
        return table

    def __getitem__(self, index: int) -> CoreParquetConversation:
        group_index = self.row_group_for_index(index)
        group = self.row_groups[group_index]
        if index < 0:
            index += self._length
        row = index - group.start
        if group.row_offsets is not None:
            row = group.row_offsets[row]
        record = self._read_group(group_index).slice(row, 1).to_pylist()[0]
        image = record.get("image") or {}
        payload = image.get("bytes")
        if payload is not None:
            payload = bytes(payload)
        return CoreParquetConversation(
            sample_id=str(record.get("id", "")),
            data_source=str(record.get("data_source", "")),
            image_bytes=payload,
            image_path=str(image["path"]) if image.get("path") else None,
            turns=tuple(
                (str(turn["from"]), str(turn["value"]))
                for turn in record.get("conversations", [])
            ),
        )

    def provenance(self) -> dict[str, Any]:
        shard_stats = [Path(path).stat() for path in self.shards]
        manifest = "\n".join(
            f"{Path(path).name}:{stat.st_size}:{stat.st_mtime_ns}"
            for path, stat in zip(self.shards, shard_stats, strict=True)
        )
        return {
            "dataset": "LLaVA-NeXT-Data",
            "root": str(self.root),
            "rows": len(self),
            "corpus_rows": self.corpus_rows,
            "multimodal_rows": self.multimodal_rows,
            "text_only_rows": self.text_only_rows,
            "parquet_shards": len(self.shards),
            "row_groups": len(self.row_groups),
            "total_parquet_bytes": sum(stat.st_size for stat in shard_stats),
            "shard_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
            "selection": "multimodal rows only" if self.multimodal_only else "all rows",
            "storage": "direct Parquet; embedded image bytes; no duplicated dataset",
        }


class RowGroupDistributedSampler(Sampler[int]):
    """Deterministically shuffle while keeping reads local to row groups."""

    def __init__(
        self,
        dataset: CoreParquetDataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} is invalid for {num_replicas} replicas")
        if not len(dataset):
            raise ValueError("Cannot sample an empty CORE Parquet dataset")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        group_count = len(self.dataset.row_groups)
        order = (
            torch.randperm(group_count, generator=generator).tolist()
            if self.shuffle
            else list(range(group_count))
        )
        indices: list[int] = []
        for group_index in order:
            group = self.dataset.row_groups[group_index]
            local = list(range(group.start, group.start + group.length))
            if self.shuffle:
                permutation = torch.randperm(group.length, generator=generator).tolist()
                local = [local[position] for position in permutation]
            indices.extend(local)
        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            repeats = math.ceil(padding / len(indices))
            indices.extend((indices * repeats)[:padding])
        return iter(indices[self.rank : self.total_size : self.num_replicas])


__all__ = [
    "CoreParquetConversation",
    "CoreParquetDataset",
    "RowGroupDistributedSampler",
]
