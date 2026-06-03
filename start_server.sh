#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${LEROBOT_ROOT:-/data/kirby/lerobot}"
PYTHON="${PYTHON:-.venv/bin/python}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-outputs/train/miniwalle_smolvla_basic_temporal_spatial_dance_text2action_v1_win_chunk100/checkpoints}"
CHECKPOINT="${CHECKPOINT:-}"
ROBOT_SCHEMA="${ROBOT_SCHEMA:-vla_wa/configs/robot_schema.yaml}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8782}"

cd "$REPO_ROOT"

args=(
  vla_wa/scripts/serve_miniwalle_action_chunk.py
  --robot-schema "$ROBOT_SCHEMA"
  --fps "$FPS"
  --device "$DEVICE"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$CHECKPOINT" ]]; then
  args+=(--checkpoint "$CHECKPOINT")
else
  args+=(--checkpoints-dir "$CHECKPOINTS_DIR")
fi

exec "$PYTHON" "${args[@]}"
