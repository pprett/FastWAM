"""FastWAM video dataset backed directly by Rerun's iterable dataloader.

The training path is a :class:`RerunIterableDataset`: it fetches contiguous shuffled
blocks so overlapping video windows share server queries and decoder work. A map-style
Rerun dataset is retained only for FastWAM's random validation previews, while a
numeric-only map is used to compute normalization statistics.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
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
    NumericDecoder,
    RerunIterableDataset,
    RerunMapDataset,
    SampleIndex,
    VideoFrameDecoder,
)
from rerun.experimental.dataloader._shuffle import _contiguous_shard

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from fastwam.utils import misc
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_CATALOG_URL = "rerun+http://127.0.0.1:51234"
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
INDEX = "frame_index"
SEGMENT_SEP = "__episode_"


def segment_id(suite: str, episode_index: int) -> str:
    return f"{suite}{SEGMENT_SEP}{episode_index:05d}"


def suite_segments(segment_ids: Sequence[str], suite: str) -> list[str]:
    return sorted(s for s in segment_ids if s.startswith(suite + SEGMENT_SEP))


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
        self._segment_offsets = tuple(int(v) for v in segment_offsets)
        self._first_frames = tuple(int(v) for v in first_frames)
        self._lengths = tuple(int(v) for v in lengths)

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[torch.Tensor | None]:
        """Build metadata tensors from request coordinates."""
        del batch
        values: list[torch.Tensor | None] = []
        for request in requests:
            segment = self._segment_numbers[request.segment_id]
            position = int(request.index_value) - self._first_frames[segment]
            values.append(
                torch.tensor(
                    [self._segment_offsets[segment] + position, segment, position, self._lengths[segment]],
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
    """Rerun iterable with rank sharding that also works in spawn/forkserver workers."""

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


def _sliding_window_with_replication(values: torch.Tensor, window_size: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"Expected [N,D], got {tuple(values.shape)}")
    positions = torch.arange(len(values)).unsqueeze(1) + torch.arange(window_size).unsqueeze(0)
    return values[positions.clamp(max=len(values) - 1)]


class RobotVideoDataset(torch.utils.data.IterableDataset[dict[str, Any]]):
    """Native FastWAM dataset that streams model samples from a Rerun catalog."""

    def __init__(
        self,
        dataset_dirs: list[str],
        shape_meta: DictConfig | dict[str, Any],
        num_frames: int = 33,
        video_size: Sequence[int] = (384, 640),
        camera_key: str | None = None,
        processor: Any | None = None,
        text_embedding_cache_dir: str | None = None,
        use_text_embed_cache: bool = True,
        context_len: int = 128,
        pretrained_norm_stats: str | None = None,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str | None = "horizontal",
        override_instruction: str | None = None,
        tolerance_s: float | None = None,
        video_backend: str | None = None,
        rerun_catalog_url: str | None = None,
        rerun_dataset: str = "libero",
        fps: int = 20,
        fetch_block_size: int = 32,
        shuffle_buffer_size: int | None = 4096,
        shuffle_buffer_min_fill: int | None = 2048,
        decode_threads: int = 4,
        video_decode_threads: int = 1,
        camera_codec: str = "av1",
        seed: int = 42,
    ) -> None:
        super().__init__()
        del camera_key, max_padding_retry, tolerance_s, video_backend, fps
        if skip_padding_as_possible:
            raise ValueError("skip_padding_as_possible is incompatible with distributed iterable training")
        if multiprocessing.get_start_method(allow_none=True) != "forkserver":
            multiprocessing.set_start_method("forkserver", force=True)

        resolved_shape_meta: Any = (
            OmegaConf.to_container(shape_meta, resolve=True) if isinstance(shape_meta, DictConfig) else shape_meta
        )
        if not isinstance(resolved_shape_meta, dict):
            raise TypeError(f"shape_meta must resolve to a dict, got {type(resolved_shape_meta)}")
        self.shape_meta: dict[str, Any] = resolved_shape_meta
        self.image_meta = resolved_shape_meta["images"]
        self.action_meta = resolved_shape_meta["action"]
        self.state_meta = resolved_shape_meta["state"]
        self.num_frames = num_frames
        self.action_size = num_frames - 1
        self.action_video_freq_ratio = action_video_freq_ratio
        self.video_sample_indices = list(range(0, num_frames, action_video_freq_ratio))
        if self.action_size % action_video_freq_ratio != 0:
            raise ValueError("num_frames-1 must be divisible by action_video_freq_ratio")
        if (self.action_size // action_video_freq_ratio) % 4 != 0:
            raise ValueError("The number of video transitions must be divisible by 4")

        self.video_size = tuple(int(v) for v in video_size)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.use_text_embed_cache = bool(use_text_embed_cache)
        self.context_len = context_len
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.is_training_set = is_training_set

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(args={"img_w": self.video_size[1], "img_h": self.video_size[0]})
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        catalog_url: str = rerun_catalog_url or os.environ.get("RERUN_CATALOG_URL") or DEFAULT_CATALOG_URL
        entry = CatalogClient(catalog_url).get_dataset(rerun_dataset)
        available = entry.segment_ids()
        segments: list[str] = []
        for dataset_dir in dataset_dirs:
            candidates = suite_segments(available, Path(dataset_dir).name)
            if val_set_proportion >= 1e-6:
                order = np.arange(len(candidates))
                np.random.default_rng(seed).shuffle(order)
                split = int(len(candidates) * (1 - val_set_proportion))
                selected = order[:split] if is_training_set else order[split:]
                candidates = [candidates[i] for i in sorted(int(value) for value in selected)]
            if not candidates:
                raise ValueError(f"No Rerun segments found for suite {Path(dataset_dir).name!r}")
            segments.extend(candidates)
        self.segments = segments
        source = DataSource(entry, segments=segments)

        self._field_names: dict[tuple[str, str], str] = {}
        self._offsets: dict[str, list[int]] = {}
        for meta in self.image_meta:
            key = meta["key"]
            entity = f"observation.images.{key}" if key != "default" else "observation.images"
            self._field_names[("images", key)] = f"images::{key}"
            self._offsets[f"images::{key}"] = list(
                range(0, num_frames * global_sample_stride, action_video_freq_ratio * global_sample_stride)
            )
            meta["rerun_entity"] = entity
        for kind, metadata, horizon in (
            ("state", self.state_meta, num_frames),
            ("action", self.action_meta, self.action_size),
        ):
            for meta in metadata:
                key = meta["key"]
                entity = kind if key == "default" else f"{kind}.{key}"
                if kind == "state":
                    entity = "observation.state" if key == "default" else f"observation.state.{key}"
                field_name = f"{kind}::{key}"
                self._field_names[(kind, key)] = field_name
                self._offsets[field_name] = [i * global_sample_stride for i in range(horizon)]
                meta["rerun_entity"] = entity

        first_state_entity = self.state_meta[0]["rerun_entity"]
        index_dataset = RerunMapDataset(
            source,
            INDEX,
            {"index": Field(f"/{first_state_entity}:Scalars:scalars", decode=NumericDecoder())},
            decode_threads=1,
        )
        full_sample_index = SampleIndex(index_dataset.sample_index.segments)
        segment_numbers = {segment.segment_id: i for i, segment in enumerate(full_sample_index.segments)}
        self._first_frames = np.asarray([s.index_start for s in full_sample_index.segments], dtype=np.int64)
        self._lengths = np.asarray([s.num_samples for s in full_sample_index.segments], dtype=np.int64)
        self._starts = full_sample_index.segment_offsets.astype(np.int64, copy=True)

        fields: dict[str, Field] = {}
        for meta in self.image_meta:
            name = self._field_names[("images", meta["key"])]
            fields[name] = Field(
                f"/{meta['rerun_entity']}:VideoStream:sample",
                window=tuple(self._offsets[name]),
                decode=VideoFrameDecoder(
                    codec=camera_codec,
                    thread_count=video_decode_threads,
                    window_storage="view",
                ),
            )
        for kind, metadata in (("state", self.state_meta), ("action", self.action_meta)):
            for meta in metadata:
                name = self._field_names[(kind, meta["key"])]
                fields[name] = Field(
                    f"/{meta['rerun_entity']}:Scalars:scalars",
                    window=tuple(self._offsets[name]),
                    decode=NumericDecoder(),
                )
        fields["task"] = Field("/task:TextDocument:text", decode=TextDecoder())
        fields["metadata"] = Field(
            f"/{first_state_entity}:Scalars:scalars",
            decode=SampleMetadataDecoder(
                segment_numbers,
                self._starts[:-1].tolist(),
                self._first_frames.tolist(),
                self._lengths.tolist(),
            ),
        )

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed topology: rank={rank}, world_size={world_size}")
        shuffle = BlockShuffle(buffer_size=shuffle_buffer_size, min_fill=shuffle_buffer_min_fill)
        self._iterable = _RankShardedRerunIterableDataset(
            source,
            INDEX,
            fields,
            fetch_block_size=fetch_block_size,
            shuffle_strategy=shuffle,
            decode_threads=decode_threads,
            rank=rank,
            world_size=world_size,
        )
        self._iterable._sample_index = full_sample_index
        self._map = RerunMapDataset(source, INDEX, fields, decode_threads=decode_threads)
        self._map._sample_index = full_sample_index

        numeric_fields = {}
        for kind, metadata in (("state", self.state_meta), ("action", self.action_meta)):
            for meta in metadata:
                name = self._field_names[(kind, meta["key"])]
                numeric_fields[name] = Field(
                    f"/{meta['rerun_entity']}:Scalars:scalars",
                    decode=NumericDecoder(),
                )
        self._numeric_map = RerunMapDataset(source, INDEX, numeric_fields, decode_threads=decode_threads)
        self._numeric_map._sample_index = full_sample_index

        self.num_samples = int(self._starts[-1])
        self.local_num_samples = (self.num_samples + world_size - 1) // world_size
        resolved_processor = instantiate(processor) if isinstance(processor, DictConfig) else processor
        if resolved_processor is None:
            raise ValueError("The Rerun FastWAM dataset requires a processor")
        self.processor: Any = resolved_processor
        self.processor.num_image_steps = len(self.video_sample_indices)

        if not pretrained_norm_stats:
            if not is_training_set:
                raise ValueError("pretrained_norm_stats is required for validation/test datasets")
            if PartialState().is_main_process:
                logger.info("Calculating Rerun dataset stats for normalization...")
                dataset_stats = self.get_dataset_stats(self.processor)
                save_dataset_stats_to_json(dataset_stats, os.path.join(misc.get_work_dir(), "dataset_stats.json"))
            else:
                dataset_stats = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                values = [dataset_stats]
                torch.distributed.broadcast_object_list(values, src=0)
                dataset_stats = values[0]
        else:
            dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
            logger.info("Using dataset stats: %s", pretrained_norm_stats)
            if PartialState().is_main_process:
                save_dataset_stats_to_json(dataset_stats, os.path.join(misc.get_work_dir(), "dataset_stats.json"))
        self.processor.set_normalizer_from_stats(dataset_stats)
        self.processor.train() if is_training_set else self.processor.eval()

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Fetch one processed sample for FastWAM's validation preview."""
        return self._format_sample(self._preprocess_sample(self._map[index]))

    @staticmethod
    def _decode_text(value: torch.Tensor) -> str:
        return bytes(value.tolist()).decode()

    def _preprocess_sample(self, raw: dict[str, Any]) -> dict[str, Any]:
        metadata = raw["metadata"]
        if not isinstance(metadata, torch.Tensor):
            raise TypeError(f"Expected tensor sample metadata, got {type(metadata)}")
        sample_index, _, position, length = (int(v) for v in metadata.tolist())
        sample: dict[str, Any] = {
            "idx": sample_index,
            "task": self._decode_text(raw["task"]),
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
        processed = self.processor.preprocess(sample)
        if not isinstance(processed, dict):
            raise TypeError(f"Processor must return a dict, got {type(processed)}")
        return processed

    def _format_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        image_is_pad = sample["image_is_pad"]
        video = sample["pixel_values"]
        num_cameras = 1
        if video.ndim == 5:
            num_cameras, video_frames, channels, height, width = video.shape
        elif video.ndim == 4:
            video_frames, channels, height, width = video.shape
        else:
            raise ValueError(f"Expected a 4D/5D video tensor, got {tuple(video.shape)}")
        video = video.view(num_cameras, video_frames, channels, height, width)

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError("concat_multi_camera='robotwin' requires exactly 3 cameras")
            top = transforms_F.resize(video[0], [256, 320], antialias=True)
            left = transforms_F.resize(video[1], [128, 160], antialias=True)
            right = transforms_F.resize(video[2], [128, 160], antialias=True)
            video = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(f"Invalid concat_multi_camera: {self.concat_multi_camera}")
        else:
            video = video.squeeze(0)

        video = self.normalize_transform(self.crop_transform(self.resize_transform(video)))
        video = video.permute(1, 0, 2, 3)
        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1 or action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(f"Action/video horizon mismatch: action={action.shape[0]}, video={video.shape[1]}")

        task = self.override_instruction if self.override_instruction is not None else sample["instruction"]
        instruction = DEFAULT_PROMPT.format(task=task)
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
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
        hashed = hashlib.sha256(prompt.encode()).hexdigest()
        path = os.path.join(
            self.text_embedding_cache_dir,
            f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing text embedding cache: {path}")
        payload = torch.load(path, map_location="cpu")
        context, mask = payload["context"], payload["mask"].bool()
        if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != self.context_len or len(mask) != self.context_len:
            raise ValueError(f"Invalid cached text context in {path}")
        return context, mask

    def _episode_data(self, episode: int) -> dict[str, dict[str, torch.Tensor]]:
        start, end = int(self._starts[episode]), int(self._starts[episode + 1])
        rows = self._numeric_map.__getitems__(list(range(start, end)))
        state: dict[str, torch.Tensor] = {}
        action: dict[str, torch.Tensor] = {}
        for meta in self.state_meta:
            name = self._field_names[("state", meta["key"])]
            values = torch.stack([cast(torch.Tensor, row[name]) for row in rows]).float()
            state[meta["key"]] = values.unsqueeze(1)
        for meta in self.action_meta:
            name = self._field_names[("action", meta["key"])]
            values = torch.stack([cast(torch.Tensor, row[name]) for row in rows]).float()
            action[meta["key"]] = _sliding_window_with_replication(values, self.action_size)
        return {"state": state, "action": action}

    def get_dataset_stats(self, preprocessor: Any) -> dict[str, Any]:
        """Compute FastWAM's action/state normalization statistics."""
        metrics = ("min", "max", "mean", "var", "q01", "q99")
        collected: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {
            "state": {meta["key"]: {metric: [] for metric in metrics} for meta in self.state_meta},
            "action": {meta["key"]: {metric: [] for metric in metrics} for meta in self.action_meta},
        }

        def process_episode(episode: int) -> dict[str, dict[str, torch.Tensor]]:
            return cast(
                dict[str, dict[str, torch.Tensor]],
                preprocessor.action_state_transform(self._episode_data(episode)),
            )

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_episode, episode) for episode in range(len(self.segments))]
            for future in as_completed(futures):
                batch = future.result()
                for kind, metadata_fields in (("state", self.state_meta), ("action", self.action_meta)):
                    for meta in metadata_fields:
                        key = meta["key"]
                        values = batch[kind][key]
                        buckets = collected[kind][key]
                        buckets["min"].append(values.amin(0))
                        buckets["max"].append(values.amax(0))
                        buckets["mean"].append(values.mean(0))
                        buckets["var"].append(values.var(0))
                        buckets["q01"].append(torch.quantile(values, 0.01, dim=0))
                        buckets["q99"].append(torch.quantile(values, 0.99, dim=0))

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
            "num_episodes": len(self.segments),
            "num_transition": self.num_samples,
        }
        for kind, metadata_fields in (("state", self.state_meta), ("action", self.action_meta)):
            for meta in metadata_fields:
                key = meta["key"]
                source = collected[kind][key]
                field_stats: dict[str, Any] = stats[kind][key]
                field_stats["stepwise_min"] = torch.stack(source["min"]).amin(0)
                field_stats["stepwise_max"] = torch.stack(source["max"]).amax(0)
                field_stats["global_min"] = field_stats["stepwise_min"].amin(0)
                field_stats["global_max"] = field_stats["stepwise_max"].amax(0)
                field_stats["stepwise_q01"] = torch.stack(source["q01"]).amin(0)
                field_stats["stepwise_q99"] = torch.stack(source["q99"]).amax(0)
                field_stats["global_q01"] = field_stats["stepwise_q01"].amin(0)
                field_stats["global_q99"] = field_stats["stepwise_q99"].amax(0)
                (
                    field_stats["stepwise_mean"],
                    field_stats["stepwise_std"],
                    field_stats["global_mean"],
                    field_stats["global_std"],
                ) = mean_std(source["mean"], source["var"])
        return stats
