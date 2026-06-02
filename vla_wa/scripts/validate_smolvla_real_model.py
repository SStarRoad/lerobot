#!/usr/bin/env python
"""Validate a local SmolVLA checkpoint on a MiniWalle LeRobot sample."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real SmolVLA forward pass on one dataset batch.")
    parser.add_argument("--policy-path", default="vla_wa/models/smolvla_base")
    parser.add_argument("--vlm-path", default="vla_wa/models/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--dataset-root", default="vla_wa/data/lerobot_dataset/miniwalle_motion_v1_dummy_image")
    parser.add_argument("--repo-id", default="local/miniwalle_motion_v1_dummy_image")
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root, revision=args.revision)
    cfg = build_config(args, meta)

    print("== Config ==")
    print(f"policy_path={Path(args.policy_path).resolve()}")
    print(f"vlm_path={Path(args.vlm_path).resolve()}")
    print(f"input_features={cfg.input_features}")
    print(f"output_features={cfg.output_features}")
    print(f"chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps}")

    policy = SmolVLAPolicy.from_pretrained(
        args.policy_path,
        config=cfg,
        local_files_only=True,
        strict=args.strict,
    )

    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        revision=args.revision,
        delta_timestamps=delta_timestamps,
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size)))
    preprocessor, _ = make_smolvla_pre_post_processors(cfg, meta.stats)
    processed = preprocessor(batch)

    print("== Batch ==")
    print(f"observation.state={tuple(processed['observation.state'].shape)}")
    print(f"action={tuple(processed['action'].shape)}")
    for key in sorted(k for k in processed if k.startswith("observation.images.")):
        print(f"{key}={tuple(processed[key].shape)}")
    print(f"task={processed.get('task')!r}")
    if "action_is_pad" in processed:
        print(f"action_is_pad.sum={int(processed['action_is_pad'][0].sum().item())}")

    loss, info = policy(processed)
    print("== Forward ==")
    print(f"loss={float(loss.detach().cpu())}")
    print(f"loss_info={info}")


def build_config(args: argparse.Namespace, meta: LeRobotDatasetMetadata):
    cfg = PreTrainedConfig.from_pretrained(args.policy_path, local_files_only=True)
    cfg.device = args.device
    cfg.vlm_model_name = str(Path(args.vlm_path).resolve())

    features = dataset_to_policy_features(meta.features)
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type.value == "ACTION"}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    return cfg


if __name__ == "__main__":
    main()
