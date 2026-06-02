#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-miniwalle_smolvla_dance_text2action_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${EXP_NAME}}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"

env HF_HOME="${HF_HOME:-/data/kirby/lerobot/.cache/huggingface}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
/data/kirby/lerobot/.venv/bin/python -B \
vla_wa/scripts/train_smolvla_miniwalle.py \
--device "${DEVICE}" \
--batch-size "${BATCH_SIZE}" \
--steps "${STEPS}" \
--save-freq "${SAVE_FREQ}" \
--log-freq "${LOG_FREQ}" \
--num-workers "${NUM_WORKERS}" \
--output-dir "${OUTPUT_DIR}" \
--overwrite \
"$@"
