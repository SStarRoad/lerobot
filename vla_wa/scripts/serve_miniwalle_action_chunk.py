#!/usr/bin/env python
"""Serve raw MiniWalle SmolVLA action chunks over HTTP."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache" / "hf_datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_CHECKPOINTS_DIR = (
    "outputs/train/miniwalle_smolvla_basic_temporal_spatial_dance_text2action_v1_win_chunk100/checkpoints"
)
DEFAULT_ROBOT_SCHEMA = "vla_wa/configs/robot_schema.yaml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8782
DEFAULT_FPS = 10
DEFAULT_PROFILE = "upper_body_v1"
DEFAULT_RETURN_MQTT = False
DEFAULT_FILTER_ALPHA = 0.35
DEFAULT_USE_FILTER = True
DEFAULT_MQTT_SOURCE = "lerobot_smolvla_service"
OBS_STATE = "observation.state"
OBS_CONTEXT_IMAGE = "observation.images.context"
ACTION = "action"
PROTOCOL_VERSION = "miniwalle.atomic_motion.v1"
EPSILON = 1e-6
DEFAULT_MIN_JOINT_DELTA = 10.0
DEFAULT_VELOCITY_DELTA = 80.0
DEFAULT_MAX_INTERVAL_SEC = 1.5

JOINT_SPECS = {
    "left_eyebrow": {"default": 10.0, "min": 0.0, "max": 20.0, "max_speed": 120.0},
    "right_eyebrow": {"default": 10.0, "min": 0.0, "max": 20.0, "max_speed": 120.0},
    "left_eye": {"default": 7.0, "min": 0.0, "max": 15.0, "max_speed": 120.0},
    "right_eye": {"default": 7.0, "min": 0.0, "max": 15.0, "max_speed": 120.0},
    "head_pitch": {"default": 0.0, "min": -25.0, "max": 28.0, "max_speed": 90.0},
    "head_yaw": {"default": 0.0, "min": -80.0, "max": 80.0, "max_speed": 100.0},
    "neck": {"default": 60.0, "min": 0.0, "max": 120.0, "max_speed": 100.0},
    "left_shoulder_pitch": {"default": 0.0, "min": 0.0, "max": 90.0, "max_speed": 100.0},
    "right_shoulder_pitch": {"default": 0.0, "min": 0.0, "max": 90.0, "max_speed": 100.0},
    "left_shoulder_yaw": {"default": -90.0, "min": -90.0, "max": 0.0, "max_speed": 100.0},
    "right_shoulder_yaw": {"default": -90.0, "min": -90.0, "max": 0.0, "max_speed": 100.0},
    "left_arm": {"default": 0.0, "min": -28.0, "max": 28.0, "max_speed": 100.0},
    "right_arm": {"default": 0.0, "min": -28.0, "max": 28.0, "max_speed": 100.0},
}


@dataclass(frozen=True)
class CheckpointCandidate:
    step: int
    source: Path


class CheckpointResolutionError(RuntimeError):
    """Raised when a checkpoint path cannot be resolved to a pretrained_model directory."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--checkpoint", default="", help="Specific checkpoint dir or pretrained_model dir")
    parser.add_argument("--robot-schema", default=DEFAULT_ROBOT_SCHEMA)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = MiniWalleActionChunkService(
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        checkpoints_dir=Path(args.checkpoints_dir),
        robot_schema=Path(args.robot_schema),
        fps=args.fps,
        device=args.device,
    )
    service.load()
    server = build_server(args.host, args.port, service)
    print(f"MiniWalle action chunk service listening on http://{args.host}:{args.port}")
    print(f"checkpoint={service.checkpoint}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        server.server_close()


class MiniWalleActionChunkService:
    def __init__(
        self,
        *,
        checkpoint: Path | None,
        checkpoints_dir: Path,
        robot_schema: Path,
        fps: int,
        device: str,
        predictor: Any | None = None,
    ) -> None:
        self.requested_checkpoint = checkpoint
        self.checkpoints_dir = checkpoints_dir
        self.robot_schema = robot_schema
        self.fps = int(fps)
        self.device = device
        self.predictor = predictor
        self.checkpoint: Path | None = None
        self.action_names: list[str] = []
        self.state_names: list[str] = []
        self.chunk_size = 0
        self.action_dim = 0
        self.context_image_shape: tuple[int, int, int] = (3, 64, 64)
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.predictor is not None and self.checkpoint is not None

    def load(self) -> None:
        if self.requested_checkpoint is not None:
            checkpoint = resolve_checkpoint_path(self.requested_checkpoint)
        else:
            checkpoint = resolve_latest_checkpoint(self.checkpoints_dir)

        self.checkpoint = checkpoint
        self._load_schema()
        if self.predictor is None:
            self._load_real_predictor(checkpoint)
        elif isinstance(self.predictor, dict) and self.predictor.get("cfg") is not None:
            self._load_checkpoint_metadata(self.predictor["cfg"])

    def _load_real_predictor(self, checkpoint: Path) -> None:
        import torch

        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            policy_action_to_transition,
            transition_to_policy_action,
        )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

        cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        cfg.device = self.device
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg, local_files_only=True)
        policy.to(torch.device(self.device))
        policy.eval()

        preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            overrides={"device_processor": {"device": self.device}},
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.predictor = {
            "cfg": cfg,
            "policy": policy,
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "torch": torch,
        }
        self._load_checkpoint_metadata(cfg)

    def _load_schema(self) -> None:
        schema = load_robot_schema_fields(self.robot_schema)
        self.state_names = schema["state_fields"]
        self.action_names = schema["action_fields"]

    def _load_checkpoint_metadata(self, cfg: Any) -> None:
        self.chunk_size = int(getattr(cfg, "chunk_size", 0) or 0)
        input_features = getattr(cfg, "input_features", {}) or {}
        output_features = getattr(cfg, "output_features", {}) or {}
        state_feature = input_features.get(OBS_STATE)
        action_feature = getattr(cfg, "action_feature", None)
        if action_feature is None:
            action_feature = output_features.get(ACTION)
        state_shape = getattr(state_feature, "shape", None)
        action_shape = getattr(action_feature, "shape", None)
        if state_shape and int(state_shape[0]) != len(self.state_names):
            raise ValueError(f"state schema has {len(self.state_names)} fields but checkpoint expects {state_shape}")
        if action_shape and int(action_shape[0]) != len(self.action_names):
            raise ValueError(f"action schema has {len(self.action_names)} fields but checkpoint expects {action_shape}")
        self.action_dim = int(action_shape[0]) if action_shape else len(self.action_names)
        context_shape = getattr(input_features.get(OBS_CONTEXT_IMAGE), "shape", None)
        if context_shape:
            self.context_image_shape = tuple(int(value) for value in context_shape)

    def metadata(self) -> dict[str, Any]:
        return {
            "ok": True,
            "checkpoint": str(self.checkpoint) if self.checkpoint else "",
            "device": self.device,
            "loaded": self.loaded,
            "fps": self.fps,
            "dt": (1.0 / self.fps) if self.fps else None,
            "chunk_size": self.chunk_size,
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
        }

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "checkpoint": str(self.checkpoint) if self.checkpoint else "",
            "device": self.device,
            "loaded": self.loaded,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_started_at = time.perf_counter()
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("instruction is required")
        current_joints = payload.get("current_joints")
        if not isinstance(current_joints, dict):
            raise ValueError("current_joints must be a JSON object")

        seed = payload.get("seed")
        if seed is not None:
            set_seed(int(seed), self.predictor)

        with self._lock:
            inference_started_at = time.perf_counter()
            frame = build_model_input(
                instruction=instruction,
                current_joints=current_joints,
                state_names=self.state_names,
                context_image_shape=self.context_image_shape,
            )
            action_chunk = self._predict_action_chunk(frame)
            inference_ms = (time.perf_counter() - inference_started_at) * 1000.0

        response = build_action_chunk_response(
            checkpoint=self.checkpoint or Path(""),
            instruction=instruction,
            fps=self.fps,
            action_names=self.action_names,
            action_chunk=action_chunk,
        )
        if parse_bool(payload.get("return_mqtt", DEFAULT_RETURN_MQTT)):
            motion = build_motion_from_action_response(
                response,
                current_joints=current_joints,
                filter_alpha=float(payload.get("filter_alpha", DEFAULT_FILTER_ALPHA)),
                use_filter=parse_bool(payload.get("use_filter", DEFAULT_USE_FILTER)),
                motion_id=str(payload.get("motion_id") or build_motion_id()),
            )
            response["frames"] = motion
            response["mqtt_payload"] = build_motion_payload(
                motion,
                source=str(payload.get("mqtt_source") or DEFAULT_MQTT_SOURCE),
                dense=parse_bool(payload.get("dense_mqtt", False)),
            )
        total_ms = (time.perf_counter() - request_started_at) * 1000.0
        print(
            "predict "
            f"instruction={instruction!r} "
            f"shape={response['shape']} "
            f"inference_ms={inference_ms:.1f} "
            f"total_ms={total_ms:.1f}",
            flush=True,
        )
        return response

    def _predict_action_chunk(self, frame: dict[str, Any]) -> Any:
        if not isinstance(self.predictor, dict):
            return self.predictor.predict_action_chunk(frame)

        torch = self.predictor["torch"]
        processed = self.predictor["preprocessor"](frame)
        processed = move_tensors_to_device(processed, torch.device(self.device), torch)
        with torch.inference_mode():
            normalized_chunk = self.predictor["policy"].predict_action_chunk(processed)
            action_chunk = self.predictor["postprocessor"](normalized_chunk)
        return action_chunk


def resolve_latest_checkpoint(checkpoints_dir: Path) -> Path:
    candidates = discover_checkpoint_candidates(checkpoints_dir)
    if not candidates:
        raise CheckpointResolutionError(f"no checkpoint directories found under {checkpoints_dir}")
    candidate = max(candidates, key=lambda item: item.step)
    return resolve_checkpoint_path(candidate.source)


def discover_checkpoint_candidates(checkpoints_dir: Path) -> list[CheckpointCandidate]:
    if not checkpoints_dir.is_dir():
        return []

    candidates: list[CheckpointCandidate] = []
    for path in checkpoints_dir.iterdir():
        if path.is_dir() and path.name.isdigit() and (path / "pretrained_model" / "config.json").is_file():
            candidates.append(CheckpointCandidate(step=int(path.name), source=path))
    return candidates


def resolve_checkpoint_path(path: Path) -> Path:
    if (path / "config.json").is_file():
        return path
    pretrained_model = path / "pretrained_model"
    if (pretrained_model / "config.json").is_file():
        return pretrained_model
    raise CheckpointResolutionError(f"could not find pretrained_model config in {path}")


def build_model_input(
    *,
    instruction: str,
    current_joints: dict[str, Any],
    state_names: list[str],
    context_image_shape: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        "task": instruction,
        OBS_STATE: current_joints_to_state_tensor(current_joints, state_names),
        f"{OBS_STATE}_is_pad": observation_padding_mask(),
        OBS_CONTEXT_IMAGE: black_context_image(context_image_shape),
        f"{OBS_CONTEXT_IMAGE}_is_pad": observation_padding_mask(),
    }


def current_joints_to_state(current_joints: dict[str, Any], state_names: list[str]) -> list[float]:
    missing = [name for name in state_names if name not in current_joints]
    if missing:
        raise ValueError(f"current_joints is missing required joints: {', '.join(missing)}")
    state: list[float] = []
    for name in state_names:
        try:
            state.append(finite_float(current_joints[name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"current_joints[{name!r}] must be a finite number") from exc
    return state


def current_joints_to_state_tensor(current_joints: dict[str, Any], state_names: list[str]) -> Any:
    state = current_joints_to_state(current_joints, state_names)
    try:
        import torch
    except ImportError:
        return [state]
    return torch.tensor([state], dtype=torch.float32)


def black_context_image(shape: tuple[int, int, int]) -> Any:
    try:
        import torch
    except ImportError:
        channels, height, width = shape
        return [[[[0 for _ in range(width)] for _ in range(height)] for _ in range(channels)]]
    # Keep the placeholder image in floating point so downstream resize/upsample
    # processors can use bilinear interpolation on CPU/CUDA.
    return torch.zeros((1, *shape), dtype=torch.float32)


def observation_padding_mask() -> Any:
    try:
        import torch
    except ImportError:
        return [False]
    return torch.zeros((1,), dtype=torch.bool)


def load_robot_schema_fields(path: Path, profile: str = DEFAULT_PROFILE) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    profile_indent: int | None = None
    active_list: str | None = None
    fields: dict[str, list[str]] = {"state_fields": [], "action_fields": []}

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))

        if stripped == f"{profile}:":
            profile_indent = indent
            active_list = None
            continue
        if profile_indent is None:
            continue
        if indent <= profile_indent:
            break
        if stripped in {"state_fields:", "action_fields:"}:
            active_list = stripped[:-1]
            continue
        if active_list and stripped.startswith("- "):
            fields[active_list].append(stripped[2:].strip())

    if not fields["state_fields"] or not fields["action_fields"]:
        raise ValueError(f"{path} does not contain {profile} state_fields/action_fields")
    return fields


def set_seed(seed: int, predictor: Any) -> None:
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_tensors_to_device(value: Any, device: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_device(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_device(item, device, torch) for item in value)
    return value


def build_action_chunk_response(
    *,
    checkpoint: Path,
    instruction: str,
    fps: int,
    action_names: list[str],
    action_chunk: Any,
) -> dict[str, Any]:
    actions = action_chunk_to_list(action_chunk)
    shape = [len(actions), len(actions[0]) if actions else 0]
    if not actions:
        raise ValueError("generated action chunk is empty")
    if shape[1] != len(action_names):
        raise ValueError(f"action dim {shape[1]} does not match {len(action_names)} action names")

    return {
        "ok": True,
        "checkpoint": str(checkpoint),
        "instruction": instruction,
        "fps": int(fps),
        "dt": 1.0 / int(fps),
        "shape": shape,
        "action_names": list(action_names),
        "actions": actions,
    }


def action_chunk_to_list(action_chunk: Any) -> list[list[float]]:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(action_chunk, torch.Tensor):
        action_chunk = action_chunk.detach().cpu()
        if action_chunk.ndim == 3:
            if action_chunk.shape[0] != 1:
                raise ValueError(f"expected batch size 1 action chunk, got {tuple(action_chunk.shape)}")
            action_chunk = action_chunk[0]
        values = action_chunk.tolist()
    else:
        values = action_chunk
        if values and isinstance(values[0], list) and values and values[0] and isinstance(values[0][0], list):
            if len(values) != 1:
                raise ValueError(f"expected batch size 1 action chunk, got {len(values)}")
            values = values[0]

    if not isinstance(values, list) or not values:
        raise ValueError("action chunk must be a non-empty 2D list or tensor")
    out: list[list[float]] = []
    for row in values:
        if not isinstance(row, list):
            raise ValueError("action chunk must be a 2D list or tensor")
        out_row = [finite_float(value) for value in row]
        out.append(out_row)
    return out


def finite_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite action value: {value!r}")
    return out


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def build_motion_id() -> str:
    return f"smolvla_{int(time.time() * 1000)}"


def build_motion_from_action_response(
    response: dict[str, Any],
    *,
    current_joints: dict[str, Any],
    filter_alpha: float,
    use_filter: bool,
    motion_id: str,
) -> dict[str, Any]:
    action_names = response.get("action_names")
    actions = response.get("actions")
    fps = int(response.get("fps") or DEFAULT_FPS)
    if not isinstance(action_names, list) or not action_names:
        raise ValueError("action response must contain non-empty action_names")
    if not isinstance(actions, list) or not actions:
        raise ValueError("action response must contain non-empty actions")

    dt = 1.0 / fps
    previous = complete_joints(clip_joints(current_joints))
    frames = []
    speed_limited_counts: dict[str, int] = {}
    for index, row in enumerate(actions):
        if not isinstance(row, list) or len(row) != len(action_names):
            raise ValueError(f"actions[{index}] has invalid shape")
        target = complete_joints(clip_joints({name: value for name, value in zip(action_names, row, strict=True)}))
        if use_filter:
            filter_result = apply_ema_speed_limit(target, previous=previous, alpha=filter_alpha, dt=dt)
            joints = filter_result["filtered_joints"]
            for name in filter_result["speed_limited_joints"]:
                speed_limited_counts[name] = speed_limited_counts.get(name, 0) + 1
            source = {
                "type": "vla_action_chunk_ema_speed_limit",
                "source_index": index,
                "checkpoint": response.get("checkpoint"),
                "filter": {
                    "type": "ema_speed_limit",
                    "alpha": max(0.01, min(1.0, float(filter_alpha))),
                    "speed_limited_joints": filter_result["speed_limited_joints"],
                },
            }
        else:
            joints = {name: round(value, 4) for name, value in target.items()}
            source = {
                "type": "vla_action_chunk_raw",
                "source_index": index,
                "checkpoint": response.get("checkpoint"),
            }
        frames.append({"t": round(index * dt, 4), "joints": joints, "source": source})
        previous = joints

    duration = round(len(frames) / fps, 6)
    return {
        "motion_id": motion_id,
        "instruction": response.get("instruction"),
        "aliases": [],
        "style": "vla_smolvla",
        "intensity": 1.0,
        "tempo": 1.0,
        "fps": fps,
        "duration": duration,
        "source_duration": duration,
        "frames": frames,
        "meta": {
            "source": "lerobot_smolvla_action_chunk",
            "shape": response.get("shape"),
            "filter": {
                "enabled": bool(use_filter),
                "type": "ema_speed_limit" if use_filter else "none",
                "alpha": max(0.01, min(1.0, float(filter_alpha))),
                "speed_limited_counts": speed_limited_counts,
            },
        },
    }


def apply_ema_speed_limit(
    target: dict[str, float],
    *,
    previous: dict[str, float],
    alpha: float,
    dt: float,
) -> dict[str, Any]:
    active_alpha = max(0.01, min(1.0, float(alpha)))
    filtered: dict[str, float] = {}
    speed_limited = []
    for name, spec in JOINT_SPECS.items():
        previous_value = float(previous.get(name, spec["default"]))
        target_value = float(target.get(name, previous_value))
        joint_alpha = alpha_for_joint(name, active_alpha)
        ema_value = previous_value + joint_alpha * (target_value - previous_value)
        max_delta = max(0.0, float(spec["max_speed"])) * max(0.001, dt)
        limited_delta = max(-max_delta, min(max_delta, ema_value - previous_value))
        value = previous_value + limited_delta
        if abs(value - ema_value) > EPSILON:
            speed_limited.append(name)
        filtered[name] = round(clamp_joint(name, value), 4)
    return {"filtered_joints": filtered, "speed_limited_joints": sorted(speed_limited)}


def alpha_for_joint(name: str, alpha: float) -> float:
    if name.endswith("_shoulder_yaw"):
        return min(alpha, 0.12)
    if name.endswith("_shoulder_pitch"):
        return min(alpha, 0.18)
    if name in {"head_pitch", "head_yaw"}:
        return min(alpha, 0.22)
    return alpha


def clip_joints(joints: dict[str, Any]) -> dict[str, float]:
    clipped: dict[str, float] = {}
    for name, value in joints.items():
        if str(name) in JOINT_SPECS:
            clipped[str(name)] = clamp_joint(str(name), finite_float(value))
    return clipped


def complete_joints(joints: dict[str, float]) -> dict[str, float]:
    return {name: float(joints.get(name, spec["default"])) for name, spec in JOINT_SPECS.items()}


def clamp_joint(name: str, value: float) -> float:
    spec = JOINT_SPECS[name]
    return max(float(spec["min"]), min(float(spec["max"]), float(value)))


def build_motion_payload(motion: dict[str, Any], *, source: str, dense: bool = False) -> dict[str, Any]:
    frames = motion.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("motion must contain non-empty frames")
    normalized = normalize_motion_frames(frames)
    groups = frames_to_dense_groups(normalized) if dense else frames_to_groups(frames, normalized)
    return {
        "trace": str(uuid4()),
        "type": "multimodal_action",
        "command": "motor_control",
        "payload": groups,
        "meta": {
            "schema_version": PROTOCOL_VERSION,
            "source": source,
            "mode": "frame_motion_dense" if dense else "frame_motion",
            "motion_id": motion.get("motion_id"),
            "fps": motion.get("fps"),
            "duration": motion.get("duration"),
            "source_duration": motion.get("source_duration"),
            "source_frame_count": len(frames),
            "group_count": len(groups),
            "compaction": {
                "type": "none_dense_frame_stream" if dense else "velocity_redundancy_compaction",
                "source": "lerobot.vla_wa.scripts.serve_miniwalle_action_chunk",
            },
        },
    }


def frames_to_groups(
    frames: list[dict[str, Any]],
    normalized: list[tuple[float, dict[str, float]]],
) -> list[dict[str, Any]]:
    _, first_joints = normalized[0]
    changed_names = changed_joint_names(normalized)
    first_actions = joint_actions(
        {name: value for name, value in first_joints.items() if name in changed_names},
        duration_ms=1,
    )
    groups = [{"group_id": 1, "duration_ms": 1, "actions": first_actions}] if first_actions else []
    if len(normalized) == 1:
        return groups

    group_id = 2 if groups else 1
    compacted = compact_motion_frames(frames)
    for start, end in zip(compacted, compacted[1:]):
        group = segment_group(group_id, start["t"], start["joints"], end["t"], end["joints"])
        if group is not None:
            groups.append(group)
            group_id += 1
    return groups


def frames_to_dense_groups(frames: list[tuple[float, dict[str, float]]]) -> list[dict[str, Any]]:
    default_interval_ms = default_interval_ms(frames)
    _, first_joints = frames[0]
    groups = [{"group_id": 1, "duration_ms": default_interval_ms, "actions": joint_actions(first_joints, duration_ms=default_interval_ms)}]
    group_id = 2
    for (start_t, start_joints), (end_t, end_joints) in zip(frames, frames[1:]):
        duration_ms = max(1, int(round((end_t - start_t) * 1000)))
        changed = {
            name: value
            for name, value in end_joints.items()
            if abs(float(value) - float(start_joints.get(name, value))) > EPSILON
        }
        actions = joint_actions(changed, duration_ms=duration_ms)
        if actions:
            groups.append({"group_id": group_id, "duration_ms": duration_ms, "actions": actions})
            group_id += 1
    return groups


def segment_group(
    group_id: int,
    start_t: float,
    start_joints: dict[str, float],
    end_t: float,
    end_joints: dict[str, float],
) -> dict[str, Any] | None:
    duration_ms = max(1, int(round((end_t - start_t) * 1000)))
    changed = {
        name: value
        for name, value in end_joints.items()
        if abs(float(value) - float(start_joints.get(name, value))) > EPSILON
    }
    actions = joint_actions(changed, duration_ms=duration_ms)
    if not actions:
        return None
    return {"group_id": group_id, "duration_ms": duration_ms, "actions": actions}


def joint_actions(joints: dict[str, Any], *, duration_ms: int) -> list[dict[str, Any]]:
    actions = []
    for name, value in sorted(joints.items()):
        action = joint_action(str(name), float(value), duration_ms=duration_ms)
        if action is not None:
            actions.append(action)
    return actions


def joint_action(name: str, value: float, *, duration_ms: int) -> dict[str, Any] | None:
    action: dict[str, Any] = {"duration_ms": max(1, int(duration_ms)), "angle": clean_number(value)}
    if name in {"left_eyebrow", "right_eyebrow"}:
        action.update({"type": "eyebrow", "side": name.removesuffix("_eyebrow")})
        return action
    if name in {"left_eye", "right_eye"}:
        action.update({"type": "eye", "side": name.removesuffix("_eye")})
        return action
    if name in {"head_pitch", "head_yaw"}:
        action.update({"type": "head", "direction": name.removeprefix("head_")})
        return action
    if name == "neck":
        action.update({"type": "neck"})
        return action
    if name in {"left_shoulder_pitch", "right_shoulder_pitch", "left_shoulder_yaw", "right_shoulder_yaw"}:
        side, _, direction = name.partition("_shoulder_")
        action.update({"type": "shoulder", "side": side, "direction": direction})
        return action
    if name in {"left_arm", "right_arm"}:
        action.update({"type": "arm", "side": name.removesuffix("_arm")})
        return action
    return None


def compact_motion_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_motion_frames(frames)
    indices = compact_motion_indices(normalized)
    return [{"index": index, "t": normalized[index][0], "joints": normalized[index][1]} for index in indices]


def compact_motion_indices(frames: list[tuple[float, dict[str, float]]]) -> list[int]:
    if not frames:
        return []
    if len(frames) == 1:
        return [0]
    selected = {0, len(frames) - 1}
    selected.update(velocity_compaction_indices(frames))
    selected.update(local_extrema_indices(frames, min_joint_delta=DEFAULT_MIN_JOINT_DELTA * 2.0))
    selected = enforce_max_interval(frames, sorted(selected))
    return sorted(selected)


def velocity_compaction_indices(frames: list[tuple[float, dict[str, float]]]) -> set[int]:
    selected: set[int] = set()
    if len(frames) < 3:
        return selected

    anchor_index = 0
    previous_velocity = joint_velocity(frames[0], frames[1])
    for index in range(2, len(frames)):
        current_velocity = joint_velocity(frames[index - 1], frames[index])
        candidate_index = index - 1
        candidate_delta = max_joint_delta(frames[anchor_index][1], frames[candidate_index][1])
        elapsed = frames[candidate_index][0] - frames[anchor_index][0]
        velocity_changed = max_joint_delta(previous_velocity, current_velocity) >= DEFAULT_VELOCITY_DELTA
        moved_enough = candidate_delta >= DEFAULT_MIN_JOINT_DELTA
        interval_expired = elapsed >= DEFAULT_MAX_INTERVAL_SEC
        if (velocity_changed and moved_enough) or interval_expired:
            selected.add(candidate_index)
            anchor_index = candidate_index
            previous_velocity = current_velocity
        else:
            previous_velocity = blend_velocity(previous_velocity, current_velocity)
    return selected


def local_extrema_indices(frames: list[tuple[float, dict[str, float]]], *, min_joint_delta: float) -> set[int]:
    selected = set()
    for index in range(1, len(frames) - 1):
        previous = frames[index - 1][1]
        current = frames[index][1]
        following = frames[index + 1][1]
        for name, value in current.items():
            left = previous.get(name, value)
            right = following.get(name, value)
            if abs(value - left) < min_joint_delta and abs(value - right) < min_joint_delta:
                continue
            if (value >= left and value >= right) or (value <= left and value <= right):
                if abs(value - left) >= min_joint_delta or abs(value - right) >= min_joint_delta:
                    selected.add(index)
                    break
    return selected


def enforce_max_interval(frames: list[tuple[float, dict[str, float]]], indices: list[int]) -> set[int]:
    selected = set(indices)
    changed = True
    while changed:
        changed = False
        ordered = sorted(selected)
        for left, right in zip(ordered, ordered[1:]):
            left_t = frames[left][0]
            right_t = frames[right][0]
            if right_t - left_t > DEFAULT_MAX_INTERVAL_SEC:
                midpoint_t = (left_t + right_t) / 2.0
                midpoint_index = min(range(left + 1, right), key=lambda item: abs(frames[item][0] - midpoint_t))
                if midpoint_index not in selected:
                    selected.add(midpoint_index)
                    changed = True
    return selected


def normalize_motion_frames(frames: list[dict[str, Any]]) -> list[tuple[float, dict[str, float]]]:
    normalized = []
    for frame in frames:
        joints = frame.get("joints")
        if not isinstance(joints, dict):
            raise ValueError("frame.joints must be an object")
        normalized.append((float(frame.get("t", 0.0) or 0.0), {str(name): float(value) for name, value in joints.items()}))
    return normalized


def changed_joint_names(frames: list[tuple[float, dict[str, float]]]) -> set[str]:
    first = frames[0][1]
    changed = set()
    for _, joints in frames[1:]:
        for name, value in joints.items():
            if abs(float(value) - float(first.get(name, value))) > EPSILON:
                changed.add(name)
    return changed


def joint_velocity(left: tuple[float, dict[str, float]], right: tuple[float, dict[str, float]]) -> dict[str, float]:
    left_t, left_joints = left
    right_t, right_joints = right
    dt = max(EPSILON, right_t - left_t)
    names = set(left_joints) | set(right_joints)
    return {
        name: round((right_joints.get(name, left_joints.get(name, 0.0)) - left_joints.get(name, 0.0)) / dt, 6)
        for name in names
    }


def max_joint_delta(left: dict[str, float], right: dict[str, float]) -> float:
    names = set(left) | set(right)
    if not names:
        return 0.0
    return max(abs(right.get(name, left.get(name, 0.0)) - left.get(name, 0.0)) for name in names)


def blend_velocity(left: dict[str, float], right: dict[str, float], *, alpha: float = 0.35) -> dict[str, float]:
    names = set(left) | set(right)
    return {name: round(float(left.get(name, 0.0)) * (1.0 - alpha) + float(right.get(name, 0.0)) * alpha, 6) for name in names}


def default_interval_ms(frames: list[tuple[float, dict[str, float]]]) -> int:
    for (left_t, _), (right_t, _) in zip(frames, frames[1:]):
        delta_ms = int(round((right_t - left_t) * 1000))
        if delta_ms > 0:
            return delta_ms
    return 100


def clean_number(value: float) -> int | float:
    rounded = round(float(value), 4)
    if abs(rounded - int(rounded)) <= EPSILON:
        return int(rounded)
    return rounded


def build_server(host: str, port: int, service: MiniWalleActionChunkService) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/health":
                    self._json(service.health())
                    return
                if path == "/metadata":
                    self._json(service.metadata())
                    return
                self._json({"ok": False, "error": "not_found", "message": f"unknown path {path}"}, status=404)
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self._error(exc)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            request_started_at = time.perf_counter()
            try:
                if path == "/predict":
                    payload = self._read_json()
                    instruction = str(payload.get("instruction") or "").strip()
                    print(f"request path=/predict instruction={instruction!r}", flush=True)
                    self._json(service.predict(payload))
                    elapsed_ms = (time.perf_counter() - request_started_at) * 1000.0
                    print(f"response path=/predict status=200 total_http_ms={elapsed_ms:.1f}", flush=True)
                    return
                self._json({"ok": False, "error": "not_found", "message": f"unknown path {path}"}, status=404)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - request_started_at) * 1000.0
                print(
                    f"response path={path} status=400 error={type(exc).__name__} total_http_ms={elapsed_ms:.1f}",
                    flush=True,
                )
                self._error(exc)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, exc: Exception) -> None:
            self._json(
                {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                status=400,
            )

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    main()
