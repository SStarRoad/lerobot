#!/usr/bin/env python
"""Train SmolVLA's action expert on MiniWalle dance motion data.

This script intentionally bypasses ``lerobot-train`` for the first MiniWalle
prototype so that the local VLM path, dataset-derived feature space, and fresh
processors are all explicit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.common.train_utils import (  # noqa: E402
    get_step_checkpoint_dir,
    save_training_state,
)
from lerobot.configs import FeatureType, PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.sampler import InstructionAlignedSampler  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR  # noqa: E402
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402


DEFAULT_DATASET_ROOT = "vla_wa/data/lerobot_dataset/miniwalle_dance_text2action_v1"
DEFAULT_REPO_ID = "local/miniwalle_dance_text2action_v1"
DEFAULT_POLICY_PATH = "vla_wa/models/smolvla_base"
DEFAULT_VLM_PATH = "vla_wa/models/SmolVLM2-500M-Video-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/train/miniwalle_smolvla_dance_text2action_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--vlm-path", default=DEFAULT_VLM_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of future action steps predicted per model invocation. Defaults to policy config.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Number of predicted action steps used for execution. Defaults to --chunk-size or policy config.",
    )
    parser.add_argument(
        "--save-steps",
        default="",
        help="Comma-separated extra checkpoint steps to save, e.g. 10,100,1000.",
    )
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--strict", action="store_true", help="Strictly load pretrained policy weights.")
    parser.add_argument(
        "--instruction-aligned-sampling",
        action="store_true",
        help=(
            "For episode_type=instruction_aligned_episode, sample obs starts only from "
            "allowed_obs_start_frame_ranges; regular episodes keep default frame-level sampling."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Build everything and run one no-update batch.")
    parser.add_argument(
        "--print-trainable",
        action="store_true",
        help="Print every trainable parameter name instead of only a summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    policy_path = Path(args.policy_path)
    vlm_path = Path(args.vlm_path)
    output_dir = Path(args.output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    if output_dir.exists() and any(output_dir.iterdir()) and not (args.overwrite or args.dry_run):
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to write into it")

    meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root, revision=args.revision)
    cfg = build_config(args, meta, policy_path, vlm_path, device)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(cfg, meta.stats)
    policy = SmolVLAPolicy.from_pretrained(
        policy_path,
        config=cfg,
        local_files_only=True,
        strict=args.strict,
    )
    policy.to(device)
    policy.train()

    print_dataset_summary(meta, cfg)
    print_trainable_summary(policy, print_names=args.print_trainable)

    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        revision=args.revision,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    sampler = None
    shuffle = True
    if args.instruction_aligned_sampling:
        sampler = InstructionAlignedSampler(dataset.meta.episodes, dataset_root=dataset_root, shuffle=True)
        shuffle = False
        print(f"instruction_aligned_sampler={json.dumps(sampler.summary(), ensure_ascii=False)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=args.lr if args.lr is not None else cfg.optimizer_lr,
        betas=tuple(cfg.optimizer_betas),
        eps=cfg.optimizer_eps,
        weight_decay=args.weight_decay if args.weight_decay is not None else cfg.optimizer_weight_decay,
    )
    scheduler = make_scheduler(optimizer, cfg, args.steps)
    grad_clip_norm = args.grad_clip_norm if args.grad_clip_norm is not None else cfg.optimizer_grad_clip_norm

    if args.dry_run:
        batch = next(iter(dataloader))
        batch = preprocessor(batch)
        with torch.no_grad():
            loss, info = policy(batch)
        print(f"dry_run_loss={float(loss.detach().cpu()):.6f}")
        print(f"dry_run_info={json.dumps(info, ensure_ascii=False)}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(output_dir, args, cfg, meta)
    train(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        steps=args.steps,
        save_freq=args.save_freq,
        save_steps=parse_save_steps(args.save_steps),
        log_freq=args.log_freq,
        grad_clip_norm=grad_clip_norm,
    )


def build_config(
    args: argparse.Namespace,
    meta: LeRobotDatasetMetadata,
    policy_path: Path,
    vlm_path: Path,
    device: torch.device,
):
    cfg = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    cfg.pretrained_path = policy_path
    cfg.device = str(device)
    cfg.vlm_model_name = str(vlm_path.resolve())
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.train_state_proj = True
    cfg.load_vlm_weights = True
    cfg.push_to_hub = False
    cfg.use_amp = False
    if args.chunk_size is not None:
        if args.chunk_size <= 0:
            raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}")
        cfg.chunk_size = args.chunk_size
    if args.n_action_steps is not None:
        if args.n_action_steps <= 0:
            raise ValueError(f"--n-action-steps must be positive, got {args.n_action_steps}")
        cfg.n_action_steps = args.n_action_steps
    elif args.chunk_size is not None:
        cfg.n_action_steps = args.chunk_size
    if cfg.n_action_steps > cfg.chunk_size:
        raise ValueError(
            f"n_action_steps ({cfg.n_action_steps}) cannot exceed chunk_size ({cfg.chunk_size})"
        )

    features = dataset_to_policy_features(meta.features)
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    return cfg


def train(
    *,
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    steps: int,
    save_freq: int,
    save_steps: set[int],
    log_freq: int,
    grad_clip_norm: float,
) -> None:
    step = 0
    started_at = time.perf_counter()
    while step < steps:
        for batch in dataloader:
            step += 1
            batch = preprocessor(batch)
            loss, info = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {float(loss.detach().cpu())}")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step == 1 or step % log_freq == 0:
                elapsed_s = time.perf_counter() - started_at
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"step={step} loss={float(loss.detach().cpu()):.6f} "
                    f"grad_norm={float(grad_norm):.4f} lr={lr:.3e} elapsed_s={elapsed_s:.1f}"
                )
                if info:
                    print(f"  info={json.dumps(sanitize_info(info), ensure_ascii=False)}")

            is_save_freq_step = save_freq > 0 and step % save_freq == 0
            is_explicit_save_step = step in save_steps
            if is_save_freq_step or is_explicit_save_step:
                save_artifacts(
                    output_dir=output_dir,
                    step=step,
                    total_steps=steps,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    checkpoint=True,
                )

            if step >= steps:
                break

    save_artifacts(
        output_dir=output_dir,
        step=step,
        total_steps=steps,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        checkpoint=False,
    )
    print(f"finished steps={step} output_dir={output_dir}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    peak_lr = cfg.optimizer_lr
    decay_lr = cfg.scheduler_decay_lr
    warmup_steps = min(max(0, cfg.scheduler_warmup_steps), max(1, steps))
    decay_steps = max(1, min(cfg.scheduler_decay_steps, steps))
    min_ratio = decay_lr / peak_lr if peak_lr > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(1e-8, (current_step + 1) / warmup_steps)
        progress = (current_step - warmup_steps) / max(1, decay_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_save_steps(value: str) -> set[int]:
    if not value.strip():
        return set()
    save_steps: set[int] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        step = int(item)
        if step <= 0:
            raise ValueError(f"--save-steps values must be positive, got {step}")
        save_steps.add(step)
    return save_steps


def update_last_checkpoint_compatible(checkpoint_dir: Path) -> None:
    """Update checkpoints/last, using a directory copy when symlinks are blocked.

    Windows often requires elevated privileges for directory symlinks. LeRobot's
    checkpoint loader can read a real directory at checkpoints/last too, so local
    training falls back to copying the just-saved checkpoint.
    """

    last_checkpoint_dir = checkpoint_dir.parent / "last"
    if last_checkpoint_dir.exists() or last_checkpoint_dir.is_symlink():
        if last_checkpoint_dir.is_symlink() or last_checkpoint_dir.is_file():
            last_checkpoint_dir.unlink()
        else:
            shutil.rmtree(last_checkpoint_dir)

    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    try:
        last_checkpoint_dir.symlink_to(relative_target)
    except OSError as exc:
        shutil.copytree(checkpoint_dir, last_checkpoint_dir)
        marker = {
            "target": str(relative_target),
            "fallback": "copytree",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        with (last_checkpoint_dir / "last_checkpoint_fallback.json").open("w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)


def save_artifacts(
    *,
    output_dir: Path,
    step: int,
    total_steps: int,
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    preprocessor,
    postprocessor,
    checkpoint: bool,
) -> None:
    if checkpoint:
        checkpoint_dir = get_step_checkpoint_dir(output_dir, total_steps, step)
        pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(pretrained_dir)
        preprocessor.save_pretrained(pretrained_dir)
        postprocessor.save_pretrained(pretrained_dir)
        save_training_state(checkpoint_dir, step, optimizer, scheduler)
        update_last_checkpoint_compatible(checkpoint_dir)
        print(f"saved_checkpoint={checkpoint_dir}")
    else:
        pretrained_dir = output_dir / PRETRAINED_MODEL_DIR
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(pretrained_dir)
        preprocessor.save_pretrained(pretrained_dir)
        postprocessor.save_pretrained(pretrained_dir)
        save_training_state(output_dir, step, optimizer, scheduler)


def print_dataset_summary(meta: LeRobotDatasetMetadata, cfg: Any) -> None:
    print("== Dataset ==")
    print(f"episodes={meta.total_episodes} frames={meta.total_frames} fps={meta.fps}")
    print(f"input_features={cfg.input_features}")
    print(f"output_features={cfg.output_features}")


def print_trainable_summary(policy: SmolVLAPolicy, *, print_names: bool) -> None:
    trainable = [(name, p.numel()) for name, p in policy.named_parameters() if p.requires_grad]
    frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    trainable_count = sum(count for _, count in trainable)
    total = frozen + trainable_count
    print("== Trainable Parameters ==")
    print(f"trainable={trainable_count} total={total} ratio={trainable_count / max(1, total):.6f}")
    prefixes = ("model.state_proj", "model.action_in_proj", "model.action_out_proj", "model.action_time")
    expected = {prefix: any(name.startswith(prefix) for name, _ in trainable) for prefix in prefixes}
    expected["model.vlm_with_expert.lm_expert"] = any(
        name.startswith("model.vlm_with_expert.lm_expert") for name, _ in trainable
    )
    expected["model.vlm_with_expert.vlm"] = any(
        name.startswith("model.vlm_with_expert.vlm") for name, _ in trainable
    )
    print(f"trainable_groups={json.dumps(expected, ensure_ascii=False)}")
    if print_names:
        for name, count in trainable:
            print(f"{name} {count}")


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    cfg: Any,
    meta: LeRobotDatasetMetadata,
) -> None:
    payload = {
        "args": vars(args),
        "dataset": {
            "repo_id": args.repo_id,
            "root": args.dataset_root,
            "episodes": meta.total_episodes,
            "frames": meta.total_frames,
            "fps": meta.fps,
        },
        "policy": {
            "type": cfg.type,
            "vlm_model_name": cfg.vlm_model_name,
            "chunk_size": cfg.chunk_size,
            "n_action_steps": cfg.n_action_steps,
            "freeze_vision_encoder": cfg.freeze_vision_encoder,
            "train_expert_only": cfg.train_expert_only,
            "train_state_proj": cfg.train_state_proj,
            "load_vlm_weights": cfg.load_vlm_weights,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "miniwalle_train_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sanitize_info(info: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                out[key] = float(value.detach().cpu())
            else:
                out[key] = list(value.shape)
        elif isinstance(value, float):
            out[key] = value if math.isfinite(value) else str(value)
        else:
            out[key] = value
    return out


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
