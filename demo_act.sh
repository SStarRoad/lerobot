export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/kirby/.cache/huggingface
export HF_LEROBOT_HOME=/data/kirby/.cache/huggingface/lerobot



CUDA_VISIBLE_DEVICES=6,7  lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=act \
  --output_dir=outputs/train/act_pusht \
  --job_name=act_pusht \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
