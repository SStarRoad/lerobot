# README: MiniWalle VLA / WA 接入路线

## 0. 当前背景

本项目目标不是简单做“LLM 调用预定义动作库”，而是逐步构建一个 **VLA / WA motion model**，让模型根据任务、机器人状态、仿真状态和后续视觉反馈，生成机器人可执行的连续动作。

当前机器人系统已具备：

```text
1. 真实机器人状态 ↔ 仿真状态 实时映射
2. 底层舵机 / 表情 / 头手眼等控制能力
3. 可通过仿真环境回放和采集动作轨迹
4. 后续可以让模型或规则系统在仿真里自动生成大量数据
```

当前重点不是马上让 VLA 控制真机，而是：

```text
先跑通一个 VLA / WA 风格模型流程
→ 理解输入输出格式
→ 反推本项目的数据采集 schema
→ 再做自己的 motion model
```

---

## 1. 总体架构目标

目标架构如下：

```text
VLM / Robot Brain
  负责：感知物理世界、理解用户意图、下达高层任务
  示例：跳舞、敬礼、激动地动、去拿杯子、回充电桩

        ↓

WA / VLA Motion Model
  负责：根据 instruction + robot_state + optional image/history
       生成未来一段 action chunk

        ↓

Simulator / State Mapping
  负责：仿真回放、动作评估、状态同步、数据采集

        ↓

Executor
  负责：把 action chunk 转成底层舵机 / MQTT / 控制协议
```

---

## 2. 当前阶段划分

### Stage 1: 非环境交互动作

先做不依赖实时环境反馈的 expressive motion：

```text
跳舞
敬礼
开心
激动
沮丧
害羞
点头
摇头
自然摆动
```

特点：

```text
1. 不需要知道杯子、充电桩、障碍物的位置
2. 主要依赖 instruction + 当前 robot_state
3. 目标是生成自然、连续、有节奏的动作轨迹
4. 本质更像 motion generation，而不是 manipulation policy
```

推荐建模：

```text
instruction + robot_state/history
→ future action chunk
```

### Stage 2: 引入环境实时反馈

后续再做依赖视觉闭环的任务：

```text
去拿杯子
回充电桩
跟随人
绕开障碍物
室内巡航
```

特点：

```text
1. 需要 image / depth / object position / task progress
2. 需要 observe → act → observe → replan 的闭环
3. 更接近传统 VLA manipulation / navigation policy
```

推荐建模：

```text
image + instruction + robot_state + history
→ short-horizon action chunk
→ execute partial chunk
→ re-observe
```

---

## 3. 当前模型路线判断

我们讨论过 OpenVLA-OFT、SmolVLA / LeRobot、π0 / openpi、π0-FAST / FAST。结论是：

```text
当前主线：SmolVLA / LeRobot
辅助参考：OpenVLA-OFT
中长期参考：π0 / openpi
长期探索：π0-FAST / FAST action tokenizer
```

### 路线定位

| 路线 | 代表模型 | 核心思路 | 当前价值 | 风险 |
|---|---|---|---|---|
| 连续 chunk 回归 | OpenVLA-OFT | 直接回归连续 action chunk | 适合看 VLA 输入输出和工程结构 | 对 expressive motion 可能平均化 / 背轨迹 |
| 轻量 Action Expert | SmolVLA / LeRobot | VLM context + action expert 生成 chunk | 最适合作为当前 WA 原型参考 | 需要适配本项目 action space |
| Flow / Diffusion 动作生成 | π0 / openpi | 从噪声生成合理动作轨迹 | 适合多解动作，如跳舞、情绪动作 | 工程复杂，数据要求高 |
| Action Token 自回归 | π0-FAST / FAST | 把动作轨迹 token 化后自回归生成 | 长期适合动作语言和跨机器人动作空间 | tokenizer 有重建误差，调试复杂 |
| 规则动作库 | 当前已有系统 | 人工预定义动作 / 参数 | 只作为冷启动、fallback、baseline | 不能作为主路线，否则退回旧方案 |

---

## 4. 为什么当前主线是 SmolVLA / LeRobot

当前项目最缺的不是“大模型能力”，而是：

```text
1. episode 数据怎么组织
2. robot_state / action 怎么定义
3. action chunk 怎么切
4. 采集数据怎么喂给模型
5. 模型输出怎么接回仿真 / 真机
```

SmolVLA / LeRobot 更适合作为这个阶段的主参考，因为它强调：

```text
1. 轻量 VLA
2. 自定义机器人数据微调
3. action expert
4. action chunk
5. LeRobot dataset / replay / training pipeline
```

当前主线应理解为：

```text
先用 SmolVLA / LeRobot 跑通“数据 → action chunk → 仿真/真机执行”的闭环；
OpenVLA-OFT 只用来学习 VLA chunk/proprio 结构；
π0 / FAST 作为后续动作生成和动作 token 化的长期技术路线。
```

---

## 5. 当前不应该做什么

当前不要把重点放在：

```text
1. 直接让 OpenVLA-OFT 控制本机器人
2. 一开始就训练完整端到端 VLA
3. 一开始就做 action token
4. 一开始就强依赖图像闭环
5. 回到“LLM 调动作库”作为主系统
```

规则动作库可以保留，但定位只能是：

```text
1. 冷启动数据生成器
2. baseline 对照
3. 安全 fallback
4. 人工动作质量参考
```

不能让规则库成为主能力上限。

---

## 6. Action 表示建议

当前第一版 action 不建议用高层动作名，也不建议直接 action token。

建议先用：

```text
future N-step continuous action chunk
```

例如：

```json
{
  "dt": 0.02,
  "chunk_size": 50,
  "action_dim": 8,
  "actions": [
    [0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 1.0, 0.0],
    [0.1, 0.2, 0.25, 0.35, 0.0, 0.0, 1.0, 0.0]
  ]
}
```

其中 action_dim 需要根据机器人状态定义，例如：

```text
head_yaw
head_pitch
neck
left_arm
right_arm
left_brow
right_brow
eye_expr
```

注意：

```text
1. action 可以先表示 target_state
2. 后续可以比较 target_state vs delta_action
3. 原始数据一定要存连续轨迹
4. action token 只能作为后处理派生结果
```

---

## 7. Action Chunk 执行方式

推荐使用 receding horizon 方式：

```text
每次模型预测未来 H step
只执行前 K step
然后重新观测当前状态
再生成下一段 H step
```

例如：

```text
控制频率：50Hz
chunk_size: 50 step = 1s
execute_prefix: 10 step = 0.2s
replan_interval: 0.2s
```

流程：

```text
t = 0.0
obs_t → predict 50-step chunk
execute step 0~9

t = 0.2
new_obs_t → predict new 50-step chunk
execute step 0~9

t = 0.4
continue
```

这个方式同时适用于：

```text
1. continuous action chunk
2. flow-generated action chunk
3. token-decoded action chunk
```

---

## 8. 关于 Expressive Motion 的重要判断

“开心跳舞”这类动作和“叠衣服 / 拿杯子”不同。

叠衣服有明确阶段：

```text
衣服乱
→ 摆正
→ 整理袖子
→ 对折
→ 完成
```

但开心跳舞没有唯一阶段：

```text
可以先动头
也可以先动手
也可以先眨眼
也可以双手一起动
```

所以直接用 L1 / MSE regression 学单一轨迹，可能会导致：

```text
1. 背轨迹
2. 动作平均化
3. 动作幅度变小
4. 节奏变钝
5. 多样性不足
```

因此：

```text
OpenVLA-OFT 的 continuous chunk regression 可以作为 baseline
但 expressive motion 的长期主线更适合 flow / diffusion action expert
```

SmolVLA / π0 这类 action expert / flow-based 思路更适合建模“一条指令对应多种合理动作”的动作分布。

---

## 9. Action Token 的定位

Action token 不是当前第一版主线。

它的作用是：

```text
把高频连续动作轨迹
压缩成离散 token 序列
再用自回归模型生成
```

当前建议：

```text
1. 数采时永远保存连续高频轨迹
2. 不要只保存 token
3. 后续可以离线训练 / 测试 action tokenizer
4. token 版本作为长期路线，不影响当前数据采集
```

---

## 10. 推荐近期 Roadmap

### Step 1: 跑通 SmolVLA / LeRobot 最小 demo

目标：

```text
1. 安装环境
2. 跑通官方推理或训练 demo
3. 看清 dataset 格式
4. 看清 action chunk 格式
5. 看清 robot processor / policy 输入输出
```

需要重点阅读：

```text
LeRobot dataset format
SmolVLA policy config
action_dim
state_dim
chunk_size
robot processor
training script
inference / rollout script
```

### Step 2: 旁路跑 OpenVLA-OFT inference

目标不是用它控制机器人，而是看：

```text
1. image 怎么进模型
2. instruction 怎么进模型
3. proprio/state 怎么接入
4. action_head 怎么输出 chunk
5. rollout loop 怎么写
```

重点看：

```text
get_processor
get_proprio_projector
get_action_head
get_vla_action
NUM_ACTIONS_CHUNK
PROPRIO_DIM
```

### Step 3: 定义本项目 robot_state/action schema

需要先明确：

```text
robot_state_dim = ?
action_dim = ?
dt = ?
chunk_size = ?
action 是 target_state 还是 delta_action？
是否包含 eye_expr / brow_expr 这类离散状态？
是否需要归一化？
```

建议第一版：

```text
dt = 0.02
control_fps = 50
chunk_size = 50
execute_prefix = 10
action = target_state
```

后续再比较：

```text
target_state vs delta_action
50Hz vs 20Hz
chunk_size 25 / 50 / 100
```

### Step 4: 用仿真采集 100 条 expressive motion episode

第一批任务：

```text
happy_dance
sad_motion
excited_motion
salute
wave
nod
shake_head
cute_motion
thinking_motion
```

每条 episode 至少包含：

```json
{
  "episode_id": "happy_dance_000001",
  "task_type": "expressive_motion",
  "instruction": "开心地跳个舞",
  "duration": 4.0,
  "fps": 50,
  "observations": [
    {
      "t": 0.0,
      "robot_state": [],
      "image": "optional/path/to/frame.png"
    }
  ],
  "actions": [
    {
      "t": 0.0,
      "target_state": []
    }
  ],
  "metadata": {
    "emotion": "happy",
    "energy": 0.8,
    "style": "cute",
    "quality_score": 0.9
  }
}
```

### Step 5: 转成 LeRobot-style dataset

目标：

```text
把本项目仿真采集的 episode
转换成 LeRobot / SmolVLA 可训练格式
```

需要实现：

```text
scripts/convert_sim_to_lerobot.py
```

输入：

```text
raw_sim_episodes/
```

输出：

```text
lerobot_dataset/
```

需要保留：

```text
1. instruction
2. robot_state
3. action
4. timestamp
5. optional image
6. episode_index
7. frame_index
```

### Step 6: 训练第一版 WA prototype

第一版不要直接追求完整 VLA。

推荐任务：

```text
instruction + robot_state/history
→ future action chunk
```

可选输入：

```text
image 暂时 optional
```

第一版目标：

```text
1. 能在仿真里生成连续动作
2. 能跟当前状态自然接上
3. 能稳定回放
4. 能区分 happy / sad / excited / salute 等动作类型
```

### Step 7: 加入图像和环境反馈

当 Stage 1 跑通后，再进入 Stage 2：

```text
image + instruction + robot_state/history
→ action chunk
```

任务升级：

```text
看向人
朝人挥手
靠近目标
回充电桩
拿杯子
```

---

## 11. 建议的工程目录

```text
vla_wa/
  README_vla_wa_roadmap.md

  configs/
    robot_schema.yaml
    smolvla_train.yaml
    dataset_convert.yaml

  robot_schema/
    state_schema.py
    action_schema.py
    normalization.py

  data/
    raw_sim_episodes/
    lerobot_dataset/
    samples/

  scripts/
    run_smolvla_demo.py
    inspect_lerobot_dataset.py
    convert_sim_to_lerobot.py
    collect_sim_episode.py
    replay_episode.py

  policies/
    smolvla_adapter.py
    wa_chunk_policy.py
    openvla_oft_probe.py

  sim/
    sim_state_bridge.py
    sim_executor.py
    trajectory_replay.py

  executor/
    chunk_executor.py
    mqtt_executor.py
    safety_filter.py

  notebooks/
    inspect_action_chunk.ipynb
    plot_robot_state_action.ipynb
```

---

## 12. Codex 当前任务建议

Codex 可以优先做这些事情：

```text
1. 检查当前项目里 robot_state 的字段定义
2. 整理出 state_dim / action_dim 候选表
3. 新建 vla_wa/ 目录
4. 写 robot_schema.py
5. 写一个标准 ActionChunk dataclass / pydantic schema
6. 写 replay_episode.py，用于回放一条 episode
7. 写 convert_sim_to_lerobot.py 的初版骨架
8. 跑通 SmolVLA / LeRobot 官方 demo
9. 打印一个 sample 的 observation/action 格式
10. 对齐本项目 schema 和 LeRobot schema
```

---

## 13. 当前最终结论

当前路线不是回到“动作原子库”，而是：

```text
用 SmolVLA / LeRobot 作为主参考
建立自己的 robot_state/action_chunk 数据闭环
先做 expressive motion 的 WA prototype
再逐步加入视觉反馈和环境交互
```

模型选型结论：

```text
现在主线：SmolVLA / LeRobot
辅助学习：OpenVLA-OFT
后续升级：π0 / openpi flow-based action model
长期探索：FAST / action token
```

一句话总结：

```text
先用 SmolVLA / LeRobot 跑通“数据 → action chunk → 仿真/真机执行”的闭环；
OpenVLA-OFT 只用来学习 VLA chunk/proprio 结构；
π0 / FAST 作为后续动作生成和动作 token 化的长期技术路线。
```
