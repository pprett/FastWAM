"""FastWAM video samples streamed from a Rerun catalog."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rerun.catalog import CatalogClient
from rerun.experimental.dataloader import (
    BlockShuffle,
    ColumnDecoder,
    DataSource,
    DecodeRequest,
    Field,
    FieldBatch,
    NoShuffle,
    NumericDecoder,
    RerunIterableDataset,
    SampleIndex,
    VideoFrameDecoder,
)
from rerun.experimental.dataloader._shuffle import _contiguous_shard

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

DEFAULT_CATALOG_URL = "rerun+http://127.0.0.1:51234"
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
INDEX = "frame_index"
SEGMENT_SEP = "__episode_"


def segment_id(suite: str, episode_index: int) -> str:
    """Return the catalog segment id used for one LIBERO episode."""
    return f"{suite}{SEGMENT_SEP}{episode_index:05d}"


def _shape_meta(value: DictConfig | dict[str, Any]) -> dict[str, Any]:
    resolved: Any = OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value
    if not isinstance(resolved, dict):
        raise TypeError(f"shape_meta must resolve to a dict, got {type(resolved)}")
    return resolved


def _entity(kind: str, key: str) -> str:
    if kind == "images":
        return "observation.images" if key == "default" else f"observation.images.{key}"
    if kind == "state":
        return "observation.state" if key == "default" else f"observation.state.{key}"
    return "action" if key == "default" else f"action.{key}"


def _catalog_source(
    segment_prefixes: Sequence[str],
    *,
    catalog_url: str | None,
    dataset_name: str,
) -> tuple[DataSource, list[str]]:
    url = catalog_url or os.environ.get("RERUN_CATALOG_URL") or DEFAULT_CATALOG_URL
    entry = CatalogClient(url).get_dataset(dataset_name)
    available = entry.segment_ids()
    segments: list[str] = []
    for prefix in segment_prefixes:
        matches = sorted(segment for segment in available if segment.startswith(prefix))
        if not matches:
            raise ValueError(f"No Rerun segments found for prefix {prefix!r}")
        segments.extend(matches)
    return DataSource(entry, segments=segments), segments


def _full_sample_index(source: DataSource, state_entity: str) -> SampleIndex:
    return SampleIndex.build(
        source,
        INDEX,
        {"index": Field(f"/{state_entity}:Scalars:scalars", decode=NumericDecoder())},
    )


class TextDecoder(ColumnDecoder[torch.Tensor]):
    """Decode a TextDocument cell to UTF-8 bytes in a tensor."""

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[torch.Tensor | None]:
        """Decode all requested text cells."""
        values: list[torch.Tensor | None] = []
        for request in requests:
            cell = batch.take_output_rows(request)[0].as_py()
            text = cell[0] if isinstance(cell, list) and cell else cell
            values.append(None if text is None else torch.tensor(list(str(text).encode()), dtype=torch.uint8))
        return values


class SampleMetadataDecoder(ColumnDecoder[torch.Tensor]):
    """Expose global sample position and episode bounds to the FastWAM processor."""

    def __init__(
        self,
        segment_numbers: dict[str, int],
        segment_offsets: Sequence[int],
        first_frames: Sequence[int],
        lengths: Sequence[int],
    ) -> None:
        self._segment_numbers = segment_numbers
        self._segment_offsets = tuple(int(value) for value in segment_offsets)
        self._first_frames = tuple(int(value) for value in first_frames)
        self._lengths = tuple(int(value) for value in lengths)

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[torch.Tensor | None]:
        """Build metadata tensors from request coordinates."""
        del batch
        values: list[torch.Tensor | None] = []
        for request in requests:
            segment = self._segment_numbers[request.segment_id]
            position = int(request.index_value) - self._first_frames[segment]
            values.append(
                torch.tensor(
                    [self._segment_offsets[segment] + position, position, self._lengths[segment]],
                    dtype=torch.int64,
                )
            )
        return values


def _pad_order_for_equal_ranks(
    indices: NDArray[np.int64],
    block_bounds: NDArray[np.int64],
    world_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Pad by at most world_size-1 samples so every rank receives the same count."""
    if world_size <= 1 or len(indices) == 0:
        return indices, block_bounds
    samples_per_rank = (len(indices) + world_size - 1) // world_size
    padding = samples_per_rank * world_size - len(indices)
    if padding == 0:
        return indices, block_bounds

    original_length = len(indices)
    prefix_bounds = block_bounds[(block_bounds > 0) & (block_bounds < padding)]
    padded_prefix_bounds = np.append(prefix_bounds, padding).astype(np.int64, copy=False)
    return (
        np.concatenate([indices, indices[:padding]]),
        np.concatenate([block_bounds, original_length + padded_prefix_bounds]),
    )


class _RankShardedRerunIterableDataset(RerunIterableDataset):
    """Rerun iterable with rank sharding that survives spawn/forkserver workers."""

    def __init__(self, *args: Any, rank: int, world_size: int, **kwargs: Any) -> None:
        self._fastwam_rank = rank
        self._fastwam_world_size = world_size
        super().__init__(*args, **kwargs)

    def _worker_order(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        indices, block_bounds = self._shuffle_strategy.epoch_order(
            self._sample_index,
            fetch_block_size=self._fetch_block_size,
            seed=self._epoch,
        )
        if self._fastwam_world_size > 1:
            indices, block_bounds = _pad_order_for_equal_ranks(
                indices,
                block_bounds,
                self._fastwam_world_size,
            )
            indices, block_bounds = _contiguous_shard(
                indices,
                block_bounds,
                rank=self._fastwam_rank,
                world_size=self._fastwam_world_size,
            )

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            indices, block_bounds = _contiguous_shard(
                indices,
                block_bounds,
                rank=worker_info.id,
                world_size=worker_info.num_workers,
            )
        return indices, block_bounds


def compute_dataset_stats(data_config: DictConfig) -> dict[str, Any]:
    """Compute action/state normalization statistics from a numeric-only iterable stream."""
    shape_meta = _shape_meta(data_config.shape_meta)
    state_meta = shape_meta["state"]
    action_meta = shape_meta["action"]
    processor = instantiate(data_config.processor)
    action_size = int(data_config.num_frames) - 1
    sample_stride = int(data_config.get("global_sample_stride", 1))
    source, segments = _catalog_source(
        [str(value) for value in data_config.segment_prefixes],
        catalog_url=data_config.get("rerun_catalog_url"),
        dataset_name=str(data_config.rerun_dataset),
    )
    first_state_entity = _entity("state", state_meta[0]["key"])
    sample_index = _full_sample_index(source, first_state_entity)
    starts = sample_index.segment_offsets.astype(np.int64, copy=True)

    names: dict[tuple[str, str], str] = {}
    fields: dict[str, Field] = {}
    for meta in state_meta:
        key = meta["key"]
        name = f"state::{key}"
        names[("state", key)] = name
        fields[name] = Field(
            f"/{_entity('state', key)}:Scalars:scalars",
            window=(0,),
            decode=NumericDecoder(),
        )
    action_offsets = tuple(index * sample_stride for index in range(action_size))
    for meta in action_meta:
        key = meta["key"]
        name = f"action::{key}"
        names[("action", key)] = name
        fields[name] = Field(
            f"/{_entity('action', key)}:Scalars:scalars",
            window=action_offsets,
            decode=NumericDecoder(),
        )
    numeric = RerunIterableDataset(
        source,
        INDEX,
        fields,
        fetch_block_size=int(data_config.get("stats_fetch_block_size", 4096)),
        shuffle_strategy=NoShuffle(),
        decode_threads=int(data_config.get("decode_threads", 4)),
        max_consecutive_skipped_samples=0,
    )
    # Windowed fields normally trim the final action horizon from every episode. Restore the
    # state-only index so Rerun resolves those windows by repeating the episode's final row.
    numeric._sample_index = sample_index

    metrics = ("min", "max", "mean", "var", "q01", "q99")
    collected: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {
        "state": {meta["key"]: {metric: [] for metric in metrics} for meta in state_meta},
        "action": {meta["key"]: {metric: [] for metric in metrics} for meta in action_meta},
    }

    def summarize_episode(
        state_rows: dict[str, list[torch.Tensor]],
        action_rows: dict[str, list[torch.Tensor]],
        expected_length: int,
    ) -> None:
        actual_length = len(state_rows[state_meta[0]["key"]])
        if actual_length != expected_length:
            raise RuntimeError(f"Expected {expected_length} samples in episode, received {actual_length}")
        batch = cast(
            dict[str, dict[str, torch.Tensor]],
            processor.action_state_transform({
                "state": {key: torch.stack(values).float() for key, values in state_rows.items()},
                "action": {key: torch.stack(values).float() for key, values in action_rows.items()},
            }),
        )
        for kind, metadata in (("state", state_meta), ("action", action_meta)):
            for meta in metadata:
                values = batch[kind][meta["key"]]
                buckets = collected[kind][meta["key"]]
                buckets["min"].append(values.amin(0))
                buckets["max"].append(values.amax(0))
                buckets["mean"].append(values.mean(0))
                buckets["var"].append(values.var(0))
                buckets["q01"].append(torch.quantile(values, 0.01, dim=0))
                buckets["q99"].append(torch.quantile(values, 0.99, dim=0))

    state_rows: dict[str, list[torch.Tensor]] = {meta["key"]: [] for meta in state_meta}
    action_rows: dict[str, list[torch.Tensor]] = {meta["key"]: [] for meta in action_meta}
    episodes_processed = 0
    samples_processed = 0
    for samples_processed, sample in enumerate(numeric, start=1):
        for meta in state_meta:
            state_rows[meta["key"]].append(cast(torch.Tensor, sample[names[("state", meta["key"])]]))
        for meta in action_meta:
            action_rows[meta["key"]].append(cast(torch.Tensor, sample[names[("action", meta["key"])]]))

        if samples_processed == int(starts[episodes_processed + 1]):
            expected_length = int(starts[episodes_processed + 1] - starts[episodes_processed])
            summarize_episode(state_rows, action_rows, expected_length)
            episodes_processed += 1
            state_rows = {meta["key"]: [] for meta in state_meta}
            action_rows = {meta["key"]: [] for meta in action_meta}

    if samples_processed != int(starts[-1]) or episodes_processed != len(segments):
        raise RuntimeError(
            f"Expected {len(segments)} episodes and {int(starts[-1])} samples, "
            f"processed {episodes_processed} episodes and {samples_processed} samples"
        )

    def mean_std(means: list[torch.Tensor], variances: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        means_tensor, variances_tensor = torch.stack(means), torch.stack(variances)
        stepwise_mean = means_tensor.mean(0)
        stepwise_std = (variances_tensor + (means_tensor - stepwise_mean) ** 2).mean(0).sqrt()
        global_mean = means_tensor.mean((0, 1))
        global_std = (variances_tensor + (means_tensor - global_mean) ** 2).mean((0, 1)).sqrt()
        return stepwise_mean, stepwise_std, global_mean, global_std

    stats: dict[str, Any] = {
        "state": defaultdict(dict),
        "action": defaultdict(dict),
        "num_episodes": len(segments),
        "num_transition": int(starts[-1]),
    }
    for kind, metadata in (("state", state_meta), ("action", action_meta)):
        for meta in metadata:
            source_stats = collected[kind][meta["key"]]
            field_stats: dict[str, Any] = stats[kind][meta["key"]]
            field_stats["stepwise_min"] = torch.stack(source_stats["min"]).amin(0)
            field_stats["stepwise_max"] = torch.stack(source_stats["max"]).amax(0)
            field_stats["global_min"] = field_stats["stepwise_min"].amin(0)
            field_stats["global_max"] = field_stats["stepwise_max"].amax(0)
            field_stats["stepwise_q01"] = torch.stack(source_stats["q01"]).amin(0)
            field_stats["stepwise_q99"] = torch.stack(source_stats["q99"]).amax(0)
            field_stats["global_q01"] = field_stats["stepwise_q01"].amin(0)
            field_stats["global_q99"] = field_stats["stepwise_q99"].amax(0)
            (
                field_stats["stepwise_mean"],
                field_stats["stepwise_std"],
                field_stats["global_mean"],
                field_stats["global_std"],
            ) = mean_std(source_stats["mean"], source_stats["var"])
    return stats


class RobotVideoDataset(torch.utils.data.IterableDataset[dict[str, Any]]):
    """Iterable-only FastWAM dataset backed by a Rerun catalog."""

    def __init__(
        self,
        segment_prefixes: Sequence[str],
        shape_meta: DictConfig | dict[str, Any],
        processor: Any,
        pretrained_norm_stats: str,
        num_frames: int = 33,
        video_size: Sequence[int] = (224, 448),
        text_embedding_cache_dir: str | None = None,
        use_text_embed_cache: bool = True,
        context_len: int = 128,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 4,
        rerun_catalog_url: str | None = None,
        rerun_dataset: str = "libero",
        fetch_block_size: int = 32,
        shuffle_buffer_size: int | None = 4096,
        shuffle_buffer_min_fill: int | None = 2048,
        decode_threads: int = 4,
        video_decode_threads: int = 1,
        camera_codec: str = "av1",
        **_: Any,
    ) -> None:
        super().__init__()
        if multiprocessing.get_start_method(allow_none=True) != "forkserver":
            multiprocessing.set_start_method("forkserver", force=True)

        self.shape_meta = _shape_meta(shape_meta)
        self.image_meta = self.shape_meta["images"]
        self.action_meta = self.shape_meta["action"]
        self.state_meta = self.shape_meta["state"]
        self.action_size = num_frames - 1
        self.video_size = tuple(int(value) for value in video_size)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.use_text_embed_cache = bool(use_text_embed_cache)
        self.context_len = context_len

        if self.action_size % action_video_freq_ratio != 0:
            raise ValueError("num_frames-1 must be divisible by action_video_freq_ratio")
        if (self.action_size // action_video_freq_ratio) % 4 != 0:
            raise ValueError("The number of video transitions must be divisible by 4")
        video_offsets = list(
            range(0, num_frames * global_sample_stride, action_video_freq_ratio * global_sample_stride)
        )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(args={"img_w": self.video_size[1], "img_h": self.video_size[0]})
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        source, segments = _catalog_source(
            segment_prefixes,
            catalog_url=rerun_catalog_url,
            dataset_name=rerun_dataset,
        )
        first_state_entity = _entity("state", self.state_meta[0]["key"])
        full_sample_index = _full_sample_index(source, first_state_entity)
        first_frames = np.asarray([segment.index_start for segment in full_sample_index.segments], dtype=np.int64)
        lengths = np.asarray([segment.num_samples for segment in full_sample_index.segments], dtype=np.int64)
        starts = full_sample_index.segment_offsets.astype(np.int64, copy=True)

        self._field_names: dict[tuple[str, str], str] = {}
        self._offsets: dict[str, list[int]] = {}
        fields: dict[str, Field] = {}
        for meta in self.image_meta:
            name = f"images::{meta['key']}"
            self._field_names[("images", meta["key"])] = name
            self._offsets[name] = video_offsets
            fields[name] = Field(
                f"/{_entity('images', meta['key'])}:VideoStream:sample",
                window=tuple(video_offsets),
                decode=VideoFrameDecoder(
                    codec=camera_codec,
                    thread_count=video_decode_threads,
                    window_storage="view",
                ),
            )
        for kind, metadata, horizon in (
            ("state", self.state_meta, num_frames),
            ("action", self.action_meta, self.action_size),
        ):
            offsets = [index * global_sample_stride for index in range(horizon)]
            for meta in metadata:
                name = f"{kind}::{meta['key']}"
                self._field_names[(kind, meta["key"])] = name
                self._offsets[name] = offsets
                fields[name] = Field(
                    f"/{_entity(kind, meta['key'])}:Scalars:scalars",
                    window=tuple(offsets),
                    decode=NumericDecoder(),
                )
        fields["task"] = Field("/task:TextDocument:text", decode=TextDecoder())
        fields["metadata"] = Field(
            f"/{first_state_entity}:Scalars:scalars",
            decode=SampleMetadataDecoder(
                {segment.segment_id: index for index, segment in enumerate(full_sample_index.segments)},
                starts[:-1].tolist(),
                first_frames.tolist(),
                lengths.tolist(),
            ),
        )

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed topology: rank={rank}, world_size={world_size}")
        self._iterable = _RankShardedRerunIterableDataset(
            source,
            INDEX,
            fields,
            fetch_block_size=fetch_block_size,
            shuffle_strategy=BlockShuffle(buffer_size=shuffle_buffer_size, min_fill=shuffle_buffer_min_fill),
            decode_threads=decode_threads,
            rank=rank,
            world_size=world_size,
        )
        self._iterable._sample_index = full_sample_index
        self.num_samples = int(starts[-1])

        resolved_processor = instantiate(processor) if isinstance(processor, DictConfig) else processor
        resolved_processor.num_image_steps = len(video_offsets)
        resolved_processor.set_normalizer_from_stats(load_dataset_stats_from_json(pretrained_norm_stats))
        self.processor: Any = resolved_processor.train()

    def __len__(self) -> int:
        """Return the global number of training samples."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle order for an epoch."""
        self._iterable.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Stream this rank and worker's processed FastWAM samples."""
        for sample in self._iterable:
            yield self._format_sample(self._preprocess_sample(sample))

    @staticmethod
    def _decode_text(value: torch.Tensor) -> str:
        return bytes(value.tolist()).decode()

    def _preprocess_sample(self, raw: dict[str, Any]) -> dict[str, Any]:
        metadata = raw["metadata"]
        if not isinstance(metadata, torch.Tensor):
            raise TypeError(f"Expected tensor sample metadata, got {type(metadata)}")
        sample_index, position, length = (int(value) for value in metadata.tolist())
        sample: dict[str, Any] = {
            "idx": sample_index,
            "task": self._decode_text(cast(torch.Tensor, raw["task"])),
            "images": {},
            "state": {},
            "action": {},
        }
        for kind, metadata_fields in (
            ("images", self.image_meta),
            ("state", self.state_meta),
            ("action", self.action_meta),
        ):
            for meta in metadata_fields:
                name = self._field_names[(kind, meta["key"])]
                value = raw[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Expected tensor field {name!r}, got {type(value)}")
                if kind != "images":
                    value = value.float()
                    if value.ndim == 1:
                        value = value.unsqueeze(-1)
                sample[kind][meta["key"]] = value

        def pad_mask(name: str) -> torch.Tensor:
            return torch.tensor([not (0 <= position + offset < length) for offset in self._offsets[name]])

        sample["image_is_pad"] = pad_mask(self._field_names[("images", self.image_meta[0]["key"])])
        sample["state_is_pad"] = pad_mask(self._field_names[("state", self.state_meta[0]["key"])])
        sample["action_is_pad"] = pad_mask(self._field_names[("action", self.action_meta[0]["key"])])
        return cast(dict[str, Any], self.processor.preprocess(sample))

    def _format_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        video = sample["pixel_values"]
        if video.ndim != 5:
            raise ValueError(f"Expected [camera,time,channel,height,width], got {tuple(video.shape)}")
        video = torch.cat([camera for camera in video], dim=-1)
        video = self.normalize_transform(self.crop_transform(self.resize_transform(video))).permute(1, 0, 2, 3)
        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1 or action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(f"Action/video horizon mismatch: action={action.shape[0]}, video={video.shape[1]}")

        instruction = DEFAULT_PROMPT.format(task=sample["instruction"])
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": sample["image_is_pad"],
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.use_text_embed_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            context[~context_mask] = 0.0
            data["context"] = context
            data["context_mask"] = torch.ones_like(context_mask)
        return data

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set")
        path = os.path.join(
            self.text_embedding_cache_dir,
            f"{hashlib.sha256(prompt.encode()).hexdigest()}.t5_len{self.context_len}.wan22ti2v5b.pt",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing text embedding cache: {path}")
        payload = torch.load(path, map_location="cpu")
        context, mask = payload["context"], payload["mask"].bool()
        if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != self.context_len or len(mask) != self.context_len:
            raise ValueError(f"Invalid cached text context in {path}")
        return context, mask
