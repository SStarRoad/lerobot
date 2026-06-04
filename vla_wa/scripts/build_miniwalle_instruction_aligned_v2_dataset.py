#!/usr/bin/env python
"""Build the MiniWalle dance/atomic + instruction-aligned v2 training dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.dataset_tools import delete_episodes  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from vla_wa.scripts.convert_sim_to_lerobot import (  # noqa: E402
    patch_episode_metadata_columns,
    write_aligned_sampling_metadata,
)


BASELINE_ROOT = REPO_ROOT / "vla_wa/data/lerobot_dataset/miniwalle_basic_temporal_spatial_dance_text2action_v1"
BASELINE_REPO_ID = "local/miniwalle_basic_temporal_spatial_dance_text2action_v1"
V2_FRAMES_ROOT = Path(
    "/data/kirby/miniwalle-robotics/motion_dataset/frames/synthetic_instruction_actions_v2_aligned"
)

BASELINE_SUBSET_ROOT = REPO_ROOT / "vla_wa/data/lerobot_dataset/miniwalle_dance_atomic_text2action_v1"
BASELINE_SUBSET_REPO_ID = "local/miniwalle_dance_atomic_text2action_v1"
V2_STANDALONE_ROOT = REPO_ROOT / "vla_wa/data/lerobot_dataset/miniwalle_instruction_aligned_v2"
V2_STANDALONE_REPO_ID = "local/miniwalle_instruction_aligned_v2"
MIXED_ROOT = REPO_ROOT / "vla_wa/data/lerobot_dataset/miniwalle_dance_atomic_plus_instruction_aligned_v2"
MIXED_REPO_ID = "local/miniwalle_dance_atomic_plus_instruction_aligned_v2"

SELECTED_TASKS = {
    "跳舞",
    "眼睛上下转动",
    "眉毛跳动",
    "脖子上下动",
    "点头",
    "上下抬手",
    "摇头",
    "左手挥手",
    "右手挥手",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--baseline-repo-id", default=BASELINE_REPO_ID)
    parser.add_argument("--v2-frames-root", type=Path, default=V2_FRAMES_ROOT)
    parser.add_argument("--baseline-subset-root", type=Path, default=BASELINE_SUBSET_ROOT)
    parser.add_argument("--baseline-subset-repo-id", default=BASELINE_SUBSET_REPO_ID)
    parser.add_argument("--v2-standalone-root", type=Path, default=V2_STANDALONE_ROOT)
    parser.add_argument("--v2-standalone-repo-id", default=V2_STANDALONE_REPO_ID)
    parser.add_argument("--mixed-root", type=Path, default=MIXED_ROOT)
    parser.add_argument("--mixed-repo-id", default=MIXED_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache/huggingface"))

    args = parse_args()
    validate_inputs(args)
    prepare_outputs(
        [args.baseline_subset_root, args.v2_standalone_root, args.mixed_root],
        overwrite=args.overwrite,
    )

    build_baseline_subset(args)
    convert_v2_standalone(args)
    aggregate_mixed_dataset(args)
    write_mixed_aligned_sidecar(args)
    print_final_summary(args)


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.baseline_root.exists():
        raise FileNotFoundError(args.baseline_root)
    if not args.v2_frames_root.exists():
        raise FileNotFoundError(args.v2_frames_root)


def prepare_outputs(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        if not path.exists():
            continue
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)


def build_baseline_subset(args: argparse.Namespace) -> None:
    source = LeRobotDataset(args.baseline_repo_id, root=args.baseline_root, return_uint8=True)
    keep_episodes: list[int] = []
    for row in source.meta.episodes:
        tasks = row["tasks"]
        if any(task in SELECTED_TASKS for task in tasks):
            keep_episodes.append(int(row["episode_index"]))
    if not keep_episodes:
        raise ValueError("baseline task filter matched zero episodes")

    delete_indices = [idx for idx in range(source.meta.total_episodes) if idx not in set(keep_episodes)]
    print(
        f"Baseline subset: keep={len(keep_episodes)} delete={len(delete_indices)} "
        f"from={source.meta.total_episodes}"
    )
    delete_episodes(
        source,
        delete_indices,
        output_dir=args.baseline_subset_root,
        repo_id=args.baseline_subset_repo_id,
    )
    write_aligned_sampling_metadata(args.baseline_subset_root, [])


def convert_v2_standalone(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "vla_wa/scripts/convert_sim_to_lerobot.py"),
        "--input",
        str(args.v2_frames_root),
        "--output",
        str(args.v2_standalone_root),
        "--repo-id",
        args.v2_standalone_repo_id,
        "--dummy-image-key",
        "observation.images.context",
        "--overwrite",
    ]
    print("Converting v2 aligned standalone dataset")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def aggregate_mixed_dataset(args: argparse.Namespace) -> None:
    print("Aggregating baseline subset + v2 aligned")
    aggregate_datasets(
        repo_ids=[args.baseline_subset_repo_id, args.v2_standalone_repo_id],
        roots=[args.baseline_subset_root, args.v2_standalone_root],
        aggr_repo_id=args.mixed_repo_id,
        aggr_root=args.mixed_root,
    )


def write_mixed_aligned_sidecar(args: argparse.Namespace) -> None:
    baseline_meta = LeRobotDatasetMetadata(args.baseline_subset_repo_id, root=args.baseline_subset_root)
    sidecar_path = args.v2_standalone_root / "meta/aligned_sampling.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    offset = int(baseline_meta.total_episodes)
    entries = []
    for entry in payload.get("episodes", []):
        shifted = dict(entry)
        shifted["episode_index"] = int(entry["episode_index"]) + offset
        entries.append(shifted)
    write_aligned_sampling_metadata(args.mixed_root, entries)
    patch_episode_metadata_columns(args.mixed_root, entries)


def print_final_summary(args: argparse.Namespace) -> None:
    for repo_id, root in [
        (args.baseline_subset_repo_id, args.baseline_subset_root),
        (args.v2_standalone_repo_id, args.v2_standalone_root),
        (args.mixed_repo_id, args.mixed_root),
    ]:
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        print(
            f"{repo_id}: root={root} episodes={meta.total_episodes} "
            f"frames={meta.total_frames} tasks={meta.total_tasks} fps={meta.fps}"
        )


if __name__ == "__main__":
    main()
