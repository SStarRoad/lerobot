#!/usr/bin/env bash
set -euo pipefail

# Experiment 2: baseline MiniWalle text2action data + v1_2 temporal/spatial mixture.
# Dataset was generated on 2026-06-03:
#   vla_wa/data/lerobot_dataset/miniwalle_basic_temporal_spatial_dance_text2action_v1_plus_v1_2
#   episodes=2478 frames=160214 tasks=746 fps=10

cd /data/kirby/lerobot

PYTHON="${PYTHON:-.venv/bin/python}" \
DATASET_ROOT="${DATASET_ROOT:-vla_wa/data/lerobot_dataset/miniwalle_basic_temporal_spatial_dance_text2action_v1_plus_v1_2}" \
REPO_ID="${REPO_ID:-local/miniwalle_basic_temporal_spatial_dance_text2action_v1_plus_v1_2}" \
EXP_NAME="${EXP_NAME:-miniwalle_smolvla_basic_temporal_spatial_dance_text2action_v1_plus_v1_2_chunk100}" \
GPU="${GPU:-2}" \
CHUNK_SIZE="${CHUNK_SIZE:-100}" \
N_ACTION_STEPS="${N_ACTION_STEPS:-100}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
STEPS="${STEPS:-200000}" \
SAVE_FREQ="${SAVE_FREQ:-50000}" \
SAVE_STEPS="${SAVE_STEPS:-5000,10000,20000}" \
LOG_FREQ="${LOG_FREQ:-50}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
SEED="${SEED:-1000}" \
OVERWRITE="${OVERWRITE:-1}" \
bash ./minimalle_train_dance_local.sh "$@"
