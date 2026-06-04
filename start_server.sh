#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-./.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-outputs/train/miniwalle_smolvla_dance_atomic_instruction_aligned_v2_chunk100/checkpoints/last/pretrained_model}"
ROBOT_SCHEMA="${ROBOT_SCHEMA:-vla_wa/configs/robot_schema.yaml}"
DEVICE="${DEVICE:-cuda}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8782}"
FPS="${FPS:-10}"

exec "$PYTHON" \
  vla_wa/scripts/serve_miniwalle_action_chunk.py \
  --robot-schema "$ROBOT_SCHEMA" \
  --device "$DEVICE" \
  --host "$HOST" \
  --port "$PORT" \
  --fps "$FPS" \
  --checkpoint "$CHECKPOINT"
