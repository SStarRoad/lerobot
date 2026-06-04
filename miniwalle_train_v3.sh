cd /data/kirby/lerobot

PYTHON=.venv/bin/python \
DATASET_ROOT=vla_wa/data/lerobot_dataset/miniwalle_basic_temporal_spatial_dance_text2action_v1 \
REPO_ID=local/miniwalle_basic_temporal_spatial_dance_text2action_v1 \
EXP_NAME=miniwalle_smolvla_basic_temporal_spatial_dance_text2action_v1_win_chunk100 \
GPU=2 \
CHUNK_SIZE=100 \
N_ACTION_STEPS=100 \
BATCH_SIZE=8 \
STEPS=100000 \
SAVE_FREQ=10000 \
SAVE_STEPS=100,1000,5000 \
bash ./minimalle_train_dance_local.sh
