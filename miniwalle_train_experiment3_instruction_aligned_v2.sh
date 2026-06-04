#!/usr/bin/env bash
set -euo pipefail

# Experiment 3: baseline dance + 8 atomic actions + instruction_aligned v2 temporal/spatial data.
# Dataset target:
#   vla_wa/data/lerobot_dataset/miniwalle_dance_atomic_plus_instruction_aligned_v2
# This experiment intentionally does not mix Experiment 2 v1_2 data.
# Aligned episodes use meta/aligned_sampling.json to restrict observation start frames.
# Generated on 2026-06-03:
#   baseline dance/8 atomic: episodes=878 frames=69533 tasks=9 fps=10
#   instruction aligned v2:  episodes=6507 frames=312055 tasks=400 fps=10
#   mixed dataset:           episodes=7385 frames=381588 tasks=409 fps=10
# Rebuild command:
#   .venv/bin/python vla_wa/scripts/build_miniwalle_instruction_aligned_v2_dataset.py --overwrite

cd /data/kirby/lerobot

PYTHON="${PYTHON:-.venv/bin/python}" \
DATASET_ROOT="${DATASET_ROOT:-vla_wa/data/lerobot_dataset/miniwalle_dance_atomic_plus_instruction_aligned_v2}" \
REPO_ID="${REPO_ID:-local/miniwalle_dance_atomic_plus_instruction_aligned_v2}" \
EXP_NAME="${EXP_NAME:-miniwalle_smolvla_dance_atomic_instruction_aligned_v2_chunk100}" \
GPU="${GPU:-3}" \
CHUNK_SIZE="${CHUNK_SIZE:-100}" \
N_ACTION_STEPS="${N_ACTION_STEPS:-100}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
STEPS="${STEPS:-100000}" \
SAVE_FREQ="${SAVE_FREQ:-20000}" \
SAVE_STEPS="${SAVE_STEPS:-5000,10000}" \
LOG_FREQ="${LOG_FREQ:-50}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
SEED="${SEED:-1000}" \
OVERWRITE="${OVERWRITE:-1}" \
INSTRUCTION_ALIGNED_SAMPLING="${INSTRUCTION_ALIGNED_SAMPLING:-1}" \
bash ./minimalle_train_dance_local.sh "$@"
