# LeRobot MiniWalle-SmolVLA Experiment Registry

本文档记录 MiniWalle-SmolVLA 训练实验的复现入口、核心配置和主要验证问题。当前只补录最近两次实验；更早 baseline 暂不追溯。

## Experiment 2: v1_2 Mixture Scaling

- 实验名称: `miniwalle_smolvla_basic_temporal_spatial_dance_text2action_v1_plus_v1_2_chunk100`
- 训练入口: `/data/kirby/lerobot/miniwalle_train_experiment2_v1_2_mixture.sh`
- dataset: `/data/kirby/lerobot/vla_wa/data/lerobot_dataset/miniwalle_basic_temporal_spatial_dance_text2action_v1_plus_v1_2`
- checkpoint/output: `/data/kirby/lerobot/outputs/train/miniwalle_smolvla_basic_temporal_spatial_dance_text2action_v1_plus_v1_2_chunk100`
- dataset 概况: `episodes=2478`, `frames=160214`, `tasks=746`, `fps=10`
- 主要配置: `chunk_size=100`, `n_action_steps=100`, `batch_size=8`, `steps=200000`
- 保存配置: `save_freq=50000`, `save_steps=5000,10000,20000`

复现命令:

```bash
cd /data/kirby/lerobot
bash ./miniwalle_train_experiment2_v1_2_mixture.sh
```

主要想验证的问题:

- 在原有训练范式下，扩大 temporal/spatial 组合动作数据量是否能改善组合动作表现。
- `chunk_size=100` 是否足够覆盖长组合动作。
- scaling 数据后，集内/集外组合动作是否能自然泛化。

当前观察结论:

- 数据量更大，但仍存在 suffix frame 使用完整 instruction 训练的问题。
- 对部分组合动作，尤其后半段动作和重复计数，表现仍不稳定。
- raw action 仍有抽风，需要滤波兜底。

后续关联实验:

- Experiment 3 用 instruction-aligned v2 sampling 修复完整 instruction 与中后段 suffix frame 的错配问题。
- 后续 velocity/smoothness loss 实验用于缓解 raw action 抽风。

## Experiment 3: Instruction-Aligned v2 Sampling

- 实验名称: `miniwalle_smolvla_dance_atomic_instruction_aligned_v2_chunk100`
- 训练入口: `/data/kirby/lerobot/miniwalle_train_experiment3_instruction_aligned_v2.sh`
- dataset: `/data/kirby/lerobot/vla_wa/data/lerobot_dataset/miniwalle_dance_atomic_plus_instruction_aligned_v2`
- checkpoint/output: `/data/kirby/lerobot/outputs/train/miniwalle_smolvla_dance_atomic_instruction_aligned_v2_chunk100`
- dataset 概况: `episodes=7385`, `frames=381588`, `tasks=409`, `fps=10`
- 数据组成: baseline 跳舞/8 原子动作 `episodes=878`, `frames=69533`, `tasks=9`; instruction aligned v2 `episodes=6507`, `frames=312055`, `tasks=400`
- 主要配置: `chunk_size=100`, `n_action_steps=100`, `batch_size=8`, `steps=100000`
- 保存配置: `save_freq=20000`, `save_steps=5000,10000`
- dataset rebuild: `.venv/bin/python vla_wa/scripts/build_miniwalle_instruction_aligned_v2_dataset.py --overwrite`

复现命令:

```bash
cd /data/kirby/lerobot
bash ./miniwalle_train_experiment3_instruction_aligned_v2.sh
```

主要想验证的问题:

- 修复完整 instruction 从中后段 suffix frame 采样训练的问题后，集内组合动作是否改善。
- 对“同时动作”和“两步动作”这类 instruction-aligned 数据，aligned sampling 是否比普通 sliding sampling 更有效。
- 不混入 v1_2 的情况下，只用跳舞 + 8 原子动作 + v2 aligned temporal/spatial 是否能提升组合动作质量。

当前观察结论:

- 集内交集 prompt 表现改善明显，例如 `一边抬起双手，一边点头一下`、`第一步挥右手一下，第二步点头一下`。
- 集外长程组合仍一般，例如 `点头三次后挥手两次` 中后半段挥手不够明显。
- aligned sampling 对 instruction/frame 对齐问题有效，但还不能完全解决长程计数、后半段动作幅度、raw action 抽风和平滑问题。

后续关联实验:

- velocity/smoothness loss: 解决 raw action 抽风，减少对推理后处理滤波的依赖。
- short-action termination: 解决固定 100 frames 输出导致短动作后半段冗余的问题。
- action history / previous action chunk: 解决长程任务进度不可见的问题。

