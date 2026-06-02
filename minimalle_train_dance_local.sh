#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EXP_NAME="${EXP_NAME:-miniwalle_smolvla_dance_text2action_local_v1_win}"
DATASET_ROOT="${DATASET_ROOT:-vla_wa/data/lerobot_dataset/miniwalle_dance_text2action_v1}"
REPO_ID="${REPO_ID:-local/miniwalle_dance_text2action_v1}"
POLICY_PATH="${POLICY_PATH:-vla_wa/models/smolvla_base}"
VLM_PATH="${VLM_PATH:-vla_wa/models/SmolVLM2-500M-Video-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${EXP_NAME}}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
SAVE_STEPS="${SAVE_STEPS:-10,100,1000,5000,10000,15000,20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"
OVERWRITE="${OVERWRITE:-1}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python venv not found or not executable: ${PYTHON}" >&2
  exit 1
fi

for required_path in "${DATASET_ROOT}" "${POLICY_PATH}" "${VLM_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path not found: ${required_path}" >&2
    exit 1
  fi
done

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/.cache/hf_datasets}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

args=(
  -B
  vla_wa/scripts/train_smolvla_miniwalle.py
  --dataset-root "${DATASET_ROOT}"
  --repo-id "${REPO_ID}"
  --policy-path "${POLICY_PATH}"
  --vlm-path "${VLM_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --save-freq "${SAVE_FREQ}"
  --save-steps "${SAVE_STEPS}"
  --log-freq "${LOG_FREQ}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

echo "MiniWalle SmolVLA training"
echo "  dataset:    ${DATASET_ROOT}"
echo "  policy:     ${POLICY_PATH}"
echo "  vlm:        ${VLM_PATH}"
echo "  output:     ${OUTPUT_DIR}"
echo "  device:     ${DEVICE}"
echo "  steps:      ${STEPS}"
echo "  save steps: ${SAVE_STEPS}"
echo "  batch size: ${BATCH_SIZE}"
echo

"${PYTHON}" "${args[@]}" "$@"
