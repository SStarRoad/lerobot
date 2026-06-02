#!/usr/bin/env python
"""Convert MiniWalle frame-motion JSON files into a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from vla_wa.robot_schema import MiniWalleSchema


ActionSource = Literal["current", "next"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MiniWalle motion_dataset/frames JSON files to LeRobot format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=Path("/data/kirby/miniwalle-robotics/motion_dataset/frames"),
        help="Input JSON file(s) or directories containing *.json motion episodes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vla_wa/data/lerobot_dataset/miniwalle_motion_v1"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/miniwalle_motion_v1",
        help="LeRobot dataset repo_id stored in metadata.",
    )
    parser.add_argument(
        "--robot-type",
        default="miniwalle",
        help="Robot type stored in LeRobot metadata.",
    )
    parser.add_argument(
        "--state-profile",
        choices=("upper_body_v1", "body_chassis_v1"),
        default="upper_body_v1",
        help="MiniWalle observation.state profile.",
    )
    parser.add_argument(
        "--action-profile",
        choices=("upper_body_v1", "body_chassis_v1"),
        default="upper_body_v1",
        help="MiniWalle action profile.",
    )
    parser.add_argument(
        "--action-source",
        choices=("current", "next"),
        default="current",
        help="Use current-frame or next-frame target_state as the per-frame action.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override dataset fps. By default, uses the first motion file's fps.",
    )
    parser.add_argument(
        "--task-template",
        default="{instruction}",
        help="Task text template. Available fields: motion_id, instruction, style, intensity, tempo.",
    )
    parser.add_argument(
        "--dummy-image-key",
        default=None,
        help=(
            "Optional image feature key to add as a black dummy camera, e.g. "
            "observation.images.context. SmolVLA model forward currently expects at least one image."
        ),
    )
    parser.add_argument(
        "--dummy-image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(64, 64),
        help="Dummy image size as HEIGHT WIDTH.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned conversion without writing a dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(args.input)
    motions = [load_motion(path) for path in input_files]
    if not motions:
        raise ValueError(f"no motion JSON files found under {args.input}")

    fps = args.fps or infer_dataset_fps(motions)
    schema = MiniWalleSchema(state_profile=args.state_profile, action_profile=args.action_profile)
    features = schema.lerobot_features()
    if args.dummy_image_key:
        add_dummy_image_feature(features, args.dummy_image_key, tuple(args.dummy_image_size))

    print(
        f"Converting {len(motions)} episode(s): fps={fps}, "
        f"state_dim={schema.state_dim}, action_dim={schema.action_dim}"
    )
    print(f"Output: {args.output}")
    if args.dry_run:
        for motion in motions:
            print_episode_summary(motion, args.task_template)
        return

    prepare_output_dir(args.output, overwrite=args.overwrite)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=args.output,
        robot_type=args.robot_type,
        use_videos=False,
    )

    try:
        for motion in motions:
            add_motion_episode(
                dataset,
                motion,
                schema=schema,
                task_template=args.task_template,
                action_source=args.action_source,
                dummy_image_key=args.dummy_image_key,
                dummy_image_size=tuple(args.dummy_image_size),
            )
            dataset.save_episode()
        dataset.finalize()
    except Exception:
        dataset.clear_episode_buffer(delete_images=True)
        raise

    print(f"Wrote {dataset.meta.total_episodes} episode(s), {dataset.meta.total_frames} frame(s).")


def discover_input_files(paths: Path | list[Path]) -> list[Path]:
    if isinstance(paths, Path):
        paths = [paths]

    files: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_file():
            candidates = [path]
        else:
            if not path.exists():
                raise FileNotFoundError(path)
            candidates = sorted(path.glob("*.json"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                files.append(candidate)
                seen.add(resolved)
    return files


def load_motion(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        motion = json.load(f)
    motion["_source_path"] = str(path)
    validate_motion(motion)
    return motion


def validate_motion(motion: dict[str, Any]) -> None:
    frames = motion.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{motion.get('_source_path', '<memory>')} must contain non-empty frames")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{index}] must be an object")
        if "joints" not in frame or not isinstance(frame["joints"], dict):
            raise ValueError(f"frames[{index}].joints must be an object")
        if "t" not in frame:
            raise ValueError(f"frames[{index}].t is required")


def infer_dataset_fps(motions: list[dict[str, Any]]) -> int:
    fps_values = {int(motion.get("fps")) for motion in motions if motion.get("fps") is not None}
    if not fps_values:
        raise ValueError("no fps found in input motions; pass --fps")
    if len(fps_values) > 1:
        raise ValueError(f"mixed fps values are not supported in one LeRobot dataset: {sorted(fps_values)}")
    return fps_values.pop()


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def add_motion_episode(
    dataset: LeRobotDataset,
    motion: dict[str, Any],
    *,
    schema: MiniWalleSchema,
    task_template: str,
    action_source: ActionSource,
    dummy_image_key: str | None = None,
    dummy_image_size: tuple[int, int] = (64, 64),
) -> None:
    frames = motion["frames"]
    task = format_task(motion, task_template)
    dummy_image = make_dummy_image(dummy_image_size) if dummy_image_key else None
    for index, frame in enumerate(frames):
        state = schema.vectorize_state(frame["joints"])
        action_frame = select_action_frame(frames, index, action_source)
        action = schema.vectorize_action(action_frame["joints"])
        out_frame = {
            OBS_STATE: np.asarray(state, dtype=np.float32),
            ACTION: np.asarray(action, dtype=np.float32),
            "task": task,
        }
        if dummy_image_key and dummy_image is not None:
            out_frame[dummy_image_key] = dummy_image
        dataset.add_frame(out_frame)


def add_dummy_image_feature(
    features: dict[str, dict[str, object]],
    key: str,
    size: tuple[int, int],
) -> None:
    if not key.startswith("observation.images."):
        raise ValueError("--dummy-image-key must start with observation.images.")
    height, width = size
    features[key] = {
        "dtype": "image",
        "shape": (height, width, 3),
        "names": ["height", "width", "channel"],
    }


def make_dummy_image(size: tuple[int, int]) -> np.ndarray:
    height, width = size
    return np.zeros((height, width, 3), dtype=np.uint8)


def select_action_frame(frames: list[dict[str, Any]], index: int, action_source: ActionSource) -> dict[str, Any]:
    if action_source == "current":
        return frames[index]
    if action_source == "next":
        return frames[min(index + 1, len(frames) - 1)]
    raise ValueError(f"unsupported action source: {action_source}")


def format_task(motion: dict[str, Any], template: str) -> str:
    values = {
        "motion_id": motion.get("motion_id") or Path(motion.get("_source_path", "motion")).stem,
        "instruction": motion.get("instruction") or motion.get("motion_id") or "MiniWalle motion",
        "style": motion.get("style") or "",
        "intensity": motion.get("intensity") if motion.get("intensity") is not None else "",
        "tempo": motion.get("tempo") if motion.get("tempo") is not None else "",
    }
    return template.format(**values).strip()


def print_episode_summary(motion: dict[str, Any], task_template: str) -> None:
    source = motion.get("_source_path", "<memory>")
    frames = motion["frames"]
    print(
        f"- {source}: motion_id={motion.get('motion_id')}, fps={motion.get('fps')}, "
        f"frames={len(frames)}, task={format_task(motion, task_template)!r}"
    )


if __name__ == "__main__":
    main()
