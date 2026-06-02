#!/usr/bin/env python
"""Generate one MiniWalle SmolVLA action chunk from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
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


DEFAULT_CHECKPOINT = "outputs/train/miniwalle_smolvla_dance_text2action_v1/checkpoints/last/pretrained_model"
DEFAULT_DATASET_ROOT = "vla_wa/data/lerobot_dataset/miniwalle_dance_text2action_v1"
DEFAULT_REPO_ID = "local/miniwalle_dance_text2action_v1"
DEFAULT_OUTPUT = "outputs/eval/miniwalle_dance_action_chunk_10000.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default="跳一段舞")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="v3.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = resolve_pretrained_model_dir(Path(args.checkpoint))
    dataset_root = Path(args.dataset_root)
    output = Path(args.output)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root, revision=args.revision)
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.device = args.device

    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
        config=cfg,
        local_files_only=True,
    )
    policy.to(torch.device(args.device))
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": args.device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        revision=args.revision,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    if args.dataset_index < 0 or args.dataset_index >= len(dataset):
        raise IndexError(f"dataset-index {args.dataset_index} out of range [0, {len(dataset)})")

    frame = dict(dataset[args.dataset_index])
    frame["task"] = args.task
    processed = preprocessor(frame)

    with torch.inference_mode():
        normalized_chunk = policy.predict_action_chunk(processed)
        action_chunk = postprocessor(normalized_chunk)

    action_chunk = action_chunk.detach().cpu()
    if action_chunk.ndim == 3:
        if action_chunk.shape[0] != 1:
            raise ValueError(f"expected batch size 1 action chunk, got shape {tuple(action_chunk.shape)}")
        action_chunk = action_chunk[0]

    expected_shape = (cfg.chunk_size, cfg.action_feature.shape[0])
    if tuple(action_chunk.shape) != expected_shape:
        raise ValueError(f"expected action chunk shape {expected_shape}, got {tuple(action_chunk.shape)}")
    if not torch.isfinite(action_chunk).all():
        raise ValueError("generated action chunk contains non-finite values")

    actions = action_chunk.tolist()
    action_names = action_feature_names(dataset_root)
    fps = int(meta.fps)
    payload = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_index": args.dataset_index,
        "task": args.task,
        "fps": fps,
        "dt": 1.0 / fps,
        "shape": list(action_chunk.shape),
        "action_names": action_names,
        "actions": actions,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"wrote={output}")
    print(f"shape={payload['shape']} fps={fps} dt={payload['dt']:.6f}")
    print(f"action_names={action_names}")
    print(f"first_action={format_row(actions[0])}")
    print(f"last_action={format_row(actions[-1])}")


def action_feature_names(dataset_root: Path) -> list[str]:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    names = info["features"][ACTION].get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{info_path} does not contain action feature names")
    return names


def resolve_pretrained_model_dir(path: Path) -> Path:
    """Accept either a pretrained_model directory or a checkpoint directory."""
    if (path / "config.json").is_file():
        return path
    pretrained_model = path / "pretrained_model"
    if (pretrained_model / "config.json").is_file():
        return pretrained_model
    raise FileNotFoundError(f"could not find config.json in {path} or {pretrained_model}")


def format_row(row: list[Any]) -> str:
    return json.dumps([round_float(value) for value in row], ensure_ascii=False)


def round_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite action value: {value!r}")
    return round(out, 4)


if __name__ == "__main__":
    main()
