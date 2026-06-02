#!/usr/bin/env python
"""Inspect a MiniWalle LeRobot dataset sample against the SmolVLA data path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import ProcessorStep
from lerobot.utils.feature_utils import dataset_to_policy_features


class MockTokenizerProcessorStep(ProcessorStep):
    """Pass-through tokenizer replacement for offline data-path checks."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, transition):
        return transition

    def transform_features(self, features):
        return features

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_config(self) -> dict:
        return {"mock_tokenizer": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a SmolVLA-ready sample from a LeRobot dataset.")
    parser.add_argument("--root", default="vla_wa/data/lerobot_dataset/miniwalle_motion_v1")
    parser.add_argument("--repo-id", default="local/miniwalle_motion_v1")
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--use-real-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root, revision=args.revision)
    config = build_config(meta, args)
    delta_timestamps = resolve_delta_timestamps(config, meta)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        revision=args.revision,
        delta_timestamps=delta_timestamps,
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size)))

    print_header("Dataset")
    print(f"root={dataset.root}")
    print(f"episodes={dataset.num_episodes} frames={dataset.num_frames} fps={dataset.fps}")
    print(f"features={meta.features}")
    print(f"stats_keys={list(meta.stats)}")

    print_header("SmolVLA Config")
    print(f"chunk_size={config.chunk_size} n_action_steps={config.n_action_steps}")
    print(f"input_features={config.input_features}")
    print(f"output_features={config.output_features}")
    print(f"image_features={config.image_features}")
    if not config.image_features:
        print("warning=SmolVLA model forward requires at least one image feature; processor-only check will pass.")

    print_header("Delta Timestamps")
    print(delta_timestamps)

    print_header("Raw DataLoader Batch")
    print_shapes(batch)

    processor_factory = make_smolvla_pre_post_processors
    if args.use_real_tokenizer:
        preprocessor, _ = processor_factory(config, meta.stats)
    else:
        with patch(
            "lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep",
            MockTokenizerProcessorStep,
        ):
            preprocessor, _ = processor_factory(config, meta.stats)

    processed = preprocessor(batch)
    print_header("Processed SmolVLA Batch")
    print_shapes(processed)
    print_sample_values(processed)


def build_config(meta: LeRobotDatasetMetadata, args: argparse.Namespace) -> SmolVLAConfig:
    kwargs = {"device": "cpu"}
    if args.chunk_size is not None:
        kwargs["chunk_size"] = args.chunk_size
    if args.n_action_steps is not None:
        kwargs["n_action_steps"] = args.n_action_steps
    config = SmolVLAConfig(**kwargs)
    features = dataset_to_policy_features(meta.features)
    config.output_features = {key: ft for key, ft in features.items() if ft.type.value == "ACTION"}
    config.input_features = {key: ft for key, ft in features.items() if key not in config.output_features}
    return config


def print_header(title: str) -> None:
    print(f"\n== {title} ==")


def print_shapes(batch: dict) -> None:
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype} device={getattr(value, 'device', '-')}")
        else:
            print(f"{key}: {value!r}")


def print_sample_values(batch: dict) -> None:
    state = batch.get("observation.state")
    action = batch.get("action")
    action_is_pad = batch.get("action_is_pad")
    task = batch.get("task")
    if state is not None:
        print(f"sample.observation.state[0,-1,:6]={state[0, -1, :6].tolist()}")
    if action is not None:
        print(f"sample.action[0,0,:6]={action[0, 0, :6].tolist()}")
        print(f"sample.action[0,-1,:6]={action[0, -1, :6].tolist()}")
    if action_is_pad is not None:
        print(f"sample.action_is_pad.sum={int(action_is_pad[0].sum().item())}")
    if task is not None:
        print(f"sample.task={task!r}")


if __name__ == "__main__":
    main()
