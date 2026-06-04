#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                ep_length = end_index - start_index
                if drop_n_first_frames + drop_n_last_frames >= ep_length:
                    logger.warning(
                        "Episode %d has %d frames but drop_n_first_frames=%d and "
                        "drop_n_last_frames=%d removes all frames. Skipping.",
                        episode_idx,
                        ep_length,
                        drop_n_first_frames,
                        drop_n_last_frames,
                    )
                    continue
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class InstructionAlignedSampler:
    def __init__(
        self,
        episodes_metadata,
        *,
        dataset_root: str | Path | None = None,
        shuffle: bool = False,
    ):
        """Sampler that restricts observation starts for instruction-aligned episodes.

        Regular episodes keep the default frame-level sliding behavior. Episodes marked
        as ``instruction_aligned_episode`` are sampled only from local frame ranges stored
        in ``meta/aligned_sampling.json`` or, as a fallback, in extra ``meta/episodes``
        columns.
        """

        aligned_by_episode = (
            load_aligned_sampling_sidecar(Path(dataset_root)) if dataset_root is not None else {}
        )
        indices: list[int] = []
        aligned_episode_count = 0
        aligned_frame_count = 0
        default_frame_count = 0

        for row_idx in range(len(episodes_metadata)):
            row = episodes_metadata[row_idx]
            episode_idx = int(row["episode_index"])
            dataset_from_index = int(row["dataset_from_index"])
            dataset_to_index = int(row["dataset_to_index"])
            episode_length = dataset_to_index - dataset_from_index

            aligned_entry = aligned_by_episode.get(episode_idx)
            if aligned_entry is None:
                aligned_entry = aligned_entry_from_episode_row(row)

            if aligned_entry is None:
                indices.extend(range(dataset_from_index, dataset_to_index))
                default_frame_count += episode_length
                continue

            valid_indices = valid_instruction_aligned_indices(
                dataset_from_index=dataset_from_index,
                episode_length=episode_length,
                ranges=aligned_entry["allowed_obs_start_frame_ranges"],
            )
            if not valid_indices:
                logger.warning(
                    "Instruction-aligned episode %d has no valid obs starts after clamping. Skipping.",
                    episode_idx,
                )
                continue
            indices.extend(valid_indices)
            aligned_episode_count += 1
            aligned_frame_count += len(valid_indices)

        if not indices:
            raise ValueError("InstructionAlignedSampler did not find any valid frame indices.")

        self.indices = indices
        self.shuffle = shuffle
        self.aligned_episode_count = aligned_episode_count
        self.aligned_frame_count = aligned_frame_count
        self.default_frame_count = default_frame_count

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)

    def summary(self) -> dict[str, int]:
        return {
            "total_samples": len(self.indices),
            "aligned_episodes": self.aligned_episode_count,
            "aligned_samples": self.aligned_frame_count,
            "default_samples": self.default_frame_count,
        }


def load_aligned_sampling_sidecar(dataset_root: Path) -> dict[int, dict[str, Any]]:
    sidecar_path = dataset_root / "meta" / "aligned_sampling.json"
    if not sidecar_path.exists():
        return {}
    with sidecar_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError(f"{sidecar_path} must contain an episodes list")

    by_episode: dict[int, dict[str, Any]] = {}
    for entry in episodes:
        if not isinstance(entry, dict):
            continue
        episode_idx = int(entry["episode_index"])
        ranges = normalize_aligned_ranges(entry.get("allowed_obs_start_frame_ranges"))
        if not ranges:
            raise ValueError(f"{sidecar_path} episode {episode_idx} has no valid aligned ranges")
        by_episode[episode_idx] = {
            **entry,
            "allowed_obs_start_frame_ranges": ranges,
        }
    return by_episode


def aligned_entry_from_episode_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("episode_type") != "instruction_aligned_episode":
        return None
    ranges = normalize_aligned_ranges(row.get("alignment_allowed_obs_start_frame_ranges_json"))
    if not ranges:
        raise ValueError(f"episode {row.get('episode_index')} is aligned but has no valid ranges")
    return {"allowed_obs_start_frame_ranges": ranges}


def normalize_aligned_ranges(value: Any) -> list[list[int]]:
    if isinstance(value, str):
        if not value.strip():
            return []
        value = json.loads(value)
    if not isinstance(value, list):
        return []

    ranges: list[list[int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start = int(item[0])
        end = int(item[1])
        if start > end:
            start, end = end, start
        ranges.append([start, end])
    return ranges


def valid_instruction_aligned_indices(
    *,
    dataset_from_index: int,
    episode_length: int,
    ranges: list[list[int]],
) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for start, end in ranges:
        clamped_start = max(0, start)
        clamped_end = min(episode_length - 1, end)
        if clamped_start > clamped_end:
            continue
        for local_idx in range(clamped_start, clamped_end + 1):
            global_idx = dataset_from_index + local_idx
            if global_idx not in seen:
                indices.append(global_idx)
                seen.add(global_idx)
    return indices
