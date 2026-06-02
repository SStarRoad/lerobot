#!/usr/bin/env python
"""Export MiniWalle task action chunks, frames, and GIFs from SmolVLA checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache" / "hf_datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.processor import (  # noqa: E402
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION  # noqa: E402
from vla_wa.robot_schema.miniwalle_schema import JOINT_SPEC_BY_NAME  # noqa: E402


DEFAULT_CHECKPOINTS = "outputs/train/miniwalle_smolvla_basic_dance_text2action_v1_win/checkpoints"
DEFAULT_DATASET_ROOT = "vla_wa/data/lerobot_dataset/miniwalle_basic_dance_text2action_v1"
DEFAULT_REPO_ID = "local/miniwalle_basic_dance_text2action_v1"
DEFAULT_ACTION_CHUNKS_OUT = "outputs/eval/miniwalle_basic_dance_action_chunks"
DEFAULT_FRAMES_OUT = "../miniwalle-robotics/motion_dataset/frames/basic_dance_eval"
DEFAULT_GIFS_OUT = "../miniwalle-robotics/motion_dataset/renders/gifs/basic_dance_eval"
DEFAULT_PREFIX = "miniwalle_smolvla_basic_dance_text2action_v1_win"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--steps", default="", help="Comma-separated checkpoint steps. Empty exports all numeric dirs.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--tasks", default="", help="Comma-separated tasks. Empty reads meta/tasks.parquet.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--action-chunks-out-dir", default=DEFAULT_ACTION_CHUNKS_OUT)
    parser.add_argument("--frames-out-dir", default=DEFAULT_FRAMES_OUT)
    parser.add_argument("--gifs-out-dir", default=DEFAULT_GIFS_OUT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gif-width", type=int, default=640)
    parser.add_argument("--gif-height", type=int, default=480)
    parser.add_argument(
        "--gif-python",
        default=os.environ.get("MINIWALLE_PYTHON", default_gif_python()),
        help="Python executable with miniwalle/mujoco/pillow installed for GIF rendering.",
    )
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    checkpoints_dir = Path(args.checkpoints_dir)
    dataset_root = Path(args.dataset_root)
    action_chunks_out_dir = Path(args.action_chunks_out_dir)
    frames_out_dir = Path(args.frames_out_dir)
    gifs_out_dir = Path(args.gifs_out_dir)

    checkpoints = discover_checkpoints(checkpoints_dir, parse_steps(args.steps))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint pretrained_model dirs found under {checkpoints_dir}")
    tasks = parse_tasks(args.tasks) or load_dataset_tasks(dataset_root)
    if not tasks:
        raise ValueError("no tasks to export")

    meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root, revision=args.revision)
    action_names = action_feature_names(dataset_root)
    fps = int(meta.fps)
    dataset = build_dataset(args, meta)
    base_frame = dict(dataset[args.dataset_index])

    action_chunks_out_dir.mkdir(parents=True, exist_ok=True)
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    gifs_out_dir.mkdir(parents=True, exist_ok=True)

    generated_frames: list[Path] = []
    for step_name, checkpoint in checkpoints:
        print(f"loading step={step_name} checkpoint={checkpoint}")
        predictor = load_predictor(checkpoint=checkpoint, device=args.device)
        try:
            for task in tasks:
                generated_frames.extend(
                    export_one_task(
                        args=args,
                        step_name=step_name,
                        checkpoint=checkpoint,
                        task=task,
                        base_frame=base_frame,
                        predictor=predictor,
                        action_names=action_names,
                        fps=fps,
                        dataset_root=dataset_root,
                        action_chunks_out_dir=action_chunks_out_dir,
                        frames_out_dir=frames_out_dir,
                    )
                )
        finally:
            del predictor
            if args.device == "cuda":
                torch.cuda.empty_cache()

    if not args.skip_gifs:
        render_gifs(generated_frames, gifs_out_dir, width=args.gif_width, height=args.gif_height, python=args.gif_python)


def export_one_task(
    *,
    args: argparse.Namespace,
    step_name: str,
    checkpoint: Path,
    task: str,
    base_frame: dict[str, Any],
    predictor: dict[str, Any],
    action_names: list[str],
    fps: int,
    dataset_root: Path,
    action_chunks_out_dir: Path,
    frames_out_dir: Path,
) -> list[Path]:
    task_slug = slug_task(task)
    stem = f"{args.prefix}_checkpoints_{step_name}_{task_slug}"
    action_path = action_chunks_out_dir / f"{stem}_action_chunk.json"
    raw_path = frames_out_dir / f"{stem}_raw.json"
    filtered_path = frames_out_dir / f"{stem}_filtered.json"
    if not args.overwrite and any(path.exists() for path in (action_path, raw_path, filtered_path)):
        raise FileExistsError(f"{stem} outputs already exist; pass --overwrite")

    print(f"  task={task!r}")
    actions = predict_action_chunk(base_frame=base_frame, task=task, predictor=predictor)
    action_payload = {
        "motion_id": f"{stem}_action_chunk",
        "task": task,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_index": args.dataset_index,
        "action_names": action_names,
        "shape": [len(actions), len(action_names)],
        "actions": actions,
    }
    raw_motion = build_motion(
        motion_id=f"{stem}_raw",
        actions=actions,
        action_names=action_names,
        fps=fps,
        task=task,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        dataset_index=args.dataset_index,
        filtered=False,
    )
    filtered_motion = build_motion(
        motion_id=f"{stem}_filtered",
        actions=actions,
        action_names=action_names,
        fps=fps,
        task=task,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        dataset_index=args.dataset_index,
        filtered=True,
        alpha=args.alpha,
    )
    write_json(action_path, action_payload)
    write_json(raw_path, raw_motion)
    write_json(filtered_path, filtered_motion)
    print_step_metrics("    raw", raw_motion)
    print_step_metrics("    filtered", filtered_motion)
    return [raw_path, filtered_path]


def load_predictor(*, checkpoint: Path, device: str) -> dict[str, Any]:
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg, local_files_only=True)
    policy.to(torch.device(device))
    policy.eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return {"policy": policy, "preprocessor": preprocessor, "postprocessor": postprocessor}


def predict_action_chunk(*, base_frame: dict[str, Any], task: str, predictor: dict[str, Any]) -> list[list[float]]:
    frame = dict(base_frame)
    frame["task"] = task
    processed = predictor["preprocessor"](frame)
    with torch.inference_mode():
        normalized_chunk = predictor["policy"].predict_action_chunk(processed)
        action_chunk = predictor["postprocessor"](normalized_chunk)
    action_chunk = action_chunk.detach().cpu()
    if action_chunk.ndim == 3:
        if action_chunk.shape[0] != 1:
            raise ValueError(f"expected batch size 1 action chunk, got {tuple(action_chunk.shape)}")
        action_chunk = action_chunk[0]
    if not torch.isfinite(action_chunk).all():
        raise ValueError("generated non-finite action values")
    return [[float(value) for value in row] for row in action_chunk.tolist()]


def build_dataset(args: argparse.Namespace, meta: LeRobotDatasetMetadata) -> LeRobotDataset:
    cfg = PreTrainedConfig.from_pretrained(Path(args.checkpoints_dir) / discover_checkpoints(Path(args.checkpoints_dir), parse_steps(args.steps))[0][0] / "pretrained_model", local_files_only=True)
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(
        args.repo_id,
        root=Path(args.dataset_root),
        revision=args.revision,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    if args.dataset_index < 0 or args.dataset_index >= len(dataset):
        raise IndexError(f"dataset-index {args.dataset_index} out of range [0, {len(dataset)})")
    return dataset


def discover_checkpoints(checkpoints_dir: Path, requested_steps: set[int]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if requested_steps:
        width = max(6, max(len(str(step)) for step in requested_steps))
        names = [f"{step:0{width}d}" for step in sorted(requested_steps)]
    else:
        names = sorted(path.name for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit())
    for name in names:
        pretrained_model = checkpoints_dir / name / "pretrained_model"
        if (pretrained_model / "config.json").is_file():
            out.append((name, pretrained_model))
    return out


def parse_steps(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_tasks(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_dataset_tasks(dataset_root: Path) -> list[str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    df = pd.read_parquet(tasks_path)
    return [str(task) for task in df.index.tolist()]


def action_feature_names(dataset_root: Path) -> list[str]:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    names = info["features"][ACTION].get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{info_path} does not contain action feature names")
    return names


def build_motion(
    *,
    motion_id: str,
    actions: list[list[float]],
    action_names: list[str],
    fps: int,
    task: str,
    checkpoint: Path,
    dataset_root: Path,
    dataset_index: int,
    filtered: bool,
    alpha: float = 0.35,
) -> dict[str, Any]:
    dt = 1.0 / fps
    filter_state: dict[str, float] | None = None
    speed_limited_counts: dict[str, int] = {}
    frames: list[dict[str, Any]] = []
    for index, row in enumerate(actions):
        target = clip_joints({name: value for name, value in zip(action_names, row, strict=True)})
        if filtered:
            filter_result = apply_ema_speed_limit(target, previous=filter_state, alpha=alpha, dt=dt)
            joints = filter_result["filtered_joints"]
            filter_state = joints
            for name in filter_result["speed_limited_joints"]:
                speed_limited_counts[name] = speed_limited_counts.get(name, 0) + 1
            source = {
                "type": "vla_action_chunk_ema_speed_limit",
                "source_index": index,
                "checkpoint": str(checkpoint),
                "filter": {
                    "type": "ema_speed_limit",
                    "alpha": alpha,
                    "speed_limited_joints": filter_result["speed_limited_joints"],
                },
            }
        else:
            joints = {name: round(value, 4) for name, value in target.items()}
            source = {"type": "vla_action_chunk_raw", "source_index": index, "checkpoint": str(checkpoint)}
        frames.append({"t": round(index * dt, 4), "joints": joints, "source": source})

    motion = {
        "motion_id": motion_id,
        "instruction": task,
        "aliases": [],
        "style": "vla_smolvla_filtered" if filtered else "vla_smolvla_raw",
        "intensity": 1.0,
        "tempo": 1.0,
        "fps": fps,
        "duration": round(len(frames) * dt, 4),
        "source_duration": round(len(frames) * dt, 4),
        "frames": frames,
        "meta": {
            "source": "lerobot_checkpoint_action_chunk",
            "checkpoint": str(checkpoint),
            "dataset_root": str(dataset_root),
            "dataset_index": dataset_index,
            "task": task,
            "shape": [len(actions), len(action_names)],
        },
    }
    if filtered:
        motion["meta"]["filter"] = {
            "type": "ema_speed_limit",
            "alpha": alpha,
            "speed_limited_counts": dict(sorted(speed_limited_counts.items())),
        }
    return motion


def clip_joints(joints: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, spec in JOINT_SPEC_BY_NAME.items():
        out[name] = spec.clip(float(joints.get(name, spec.default)))
    return out


def apply_ema_speed_limit(
    target: dict[str, float],
    *,
    previous: dict[str, float] | None,
    alpha: float,
    dt: float,
) -> dict[str, Any]:
    active_alpha = max(0.01, min(1.0, float(alpha)))
    if previous is None:
        return {
            "filtered_joints": {name: round(value, 4) for name, value in target.items()},
            "speed_limited_joints": [],
        }

    filtered: dict[str, float] = {}
    speed_limited: list[str] = []
    for name, spec in JOINT_SPEC_BY_NAME.items():
        previous_value = float(previous.get(name, spec.default))
        target_value = float(target.get(name, previous_value))
        joint_alpha = alpha_for_joint(name, active_alpha)
        ema_value = previous_value + joint_alpha * (target_value - previous_value)
        max_delta = max(0.0, float(spec.max_speed)) * max(0.001, dt)
        limited_delta = max(-max_delta, min(max_delta, ema_value - previous_value))
        value = previous_value + limited_delta
        if abs(value - ema_value) > 1e-6:
            speed_limited.append(name)
        filtered[name] = round(spec.clip(value), 4)
    return {"filtered_joints": filtered, "speed_limited_joints": sorted(speed_limited)}


def alpha_for_joint(name: str, alpha: float) -> float:
    if name.endswith("_shoulder_yaw"):
        return min(alpha, 0.12)
    if name.endswith("_shoulder_pitch"):
        return min(alpha, 0.18)
    if name in {"head_pitch", "head_yaw"}:
        return min(alpha, 0.22)
    return alpha


def render_gifs(frame_paths: list[Path], out_dir: Path, *, width: int, height: int, python: str) -> None:
    if not frame_paths:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-m",
        "miniwalle.tools.render_motion_gif",
        *[str(path.resolve()) for path in frame_paths],
        "--out-dir",
        str(out_dir.resolve()),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    print(f"rendering {len(frame_paths)} gif(s) to {out_dir}")
    subprocess.run(cmd, cwd=REPO_ROOT.parent / "miniwalle-robotics", check=True)


def default_gif_python() -> str:
    candidate = Path("D:/apps/uv/mujoco/.venv/Scripts/python.exe")
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def slug_task(task: str) -> str:
    mapping = {
        "眼睛上下转动": "eye_vertical",
        "眉毛跳动": "eyebrow_bounce",
        "脖子上下动": "neck_bob",
        "点头": "nod_head",
        "上下抬手": "raise_arms",
        "摇头": "shake_head",
        "左手挥手": "wave_left",
        "右手挥手": "wave_right",
        "跳舞": "dance",
        "跳一段舞": "dance",
    }
    if task in mapping:
        return mapping[task]
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", task).strip("._")
    return slug or "task"


def print_step_metrics(label: str, motion: dict[str, Any]) -> None:
    frames = motion["frames"]
    if len(frames) < 2:
        return
    metrics = []
    for name in JOINT_SPEC_BY_NAME:
        values = [float(frame["joints"][name]) for frame in frames]
        max_delta = max(abs(values[i] - values[i - 1]) for i in range(1, len(values)))
        metrics.append((max_delta, name))
    top = ", ".join(f"{name}={delta:.2f}" for delta, name in sorted(metrics, reverse=True)[:5])
    print(f"{label} max_step_top5: {top}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
