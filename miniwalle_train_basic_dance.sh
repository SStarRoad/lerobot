DATASET_ROOT=vla_wa/data/lerobot_dataset/miniwalle_basic_dance_text2action_v1 \
REPO_ID=local/miniwalle_basic_dance_text2action_v1 \
EXP_NAME=miniwalle_smolvla_basic_dance_text2action_v1_win \
BATCH_SIZE=4 STEPS=20000 SAVE_FREQ=5000 SAVE_STEPS=10,100,500,1000 \
./minimalle_train_dance_local.sh
