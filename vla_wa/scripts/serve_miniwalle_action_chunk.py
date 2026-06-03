#!/usr/bin/env python
"""Serve raw MiniWalle SmolVLA action chunks over HTTP."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
OBS_STATE = "observation.state"
OBS_CONTEXT_IMAGE = "observation.images.context"
ACTION = "action"


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
            frame = build_model_input(
                instruction=instruction,
                current_joints=current_joints,
                state_names=self.state_names,
                context_image_shape=self.context_image_shape,
            )
            action_chunk = self._predict_action_chunk(frame)

        return build_action_chunk_response(
            checkpoint=self.checkpoint or Path(""),
            instruction=instruction,
            fps=self.fps,
            action_names=self.action_names,
            action_chunk=action_chunk,
        )

    def _predict_action_chunk(self, frame: dict[str, Any]) -> Any:
        if not isinstance(self.predictor, dict):
            return self.predictor.predict_action_chunk(frame)

        torch = self.predictor["torch"]
        processed = self.predictor["preprocessor"](frame)
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
        OBS_STATE: current_joints_to_state(current_joints, state_names),
        OBS_CONTEXT_IMAGE: black_context_image(context_image_shape),
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


def black_context_image(shape: tuple[int, int, int]) -> Any:
    try:
        import torch
    except ImportError:
        channels, height, width = shape
        return [[[0 for _ in range(width)] for _ in range(height)] for _ in range(channels)]
    # Keep the placeholder image in floating point so downstream resize/upsample
    # processors can use bilinear interpolation on CPU/CUDA.
    return torch.zeros(shape, dtype=torch.float32)


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
            try:
                if path == "/predict":
                    payload = self._read_json()
                    self._json(service.predict(payload))
                    return
                self._json({"ok": False, "error": "not_found", "message": f"unknown path {path}"}, status=404)
            except Exception as exc:
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
