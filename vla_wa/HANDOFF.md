# MiniWalle VLA / WA Handoff

This document summarizes the current MiniWalle WA/VLA integration state so the next session can continue without re-discovering the setup.

## Environment

Use this Python environment:

```bash
/data/kirby/lerobot/.venv/bin/python
```

For commands that load local Hugging Face assets, use:

```bash
export HF_HOME=/data/kirby/lerobot/.cache/huggingface
export HF_DATASETS_CACHE=/tmp/hf_datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Notes:

- The `.venv` initially did not have `transformers`.
- Installed during this work:
  - `transformers==5.5.4`
  - `tokenizers==0.22.2`
  - `regex==2026.5.9`
  - `num2words==0.5.14`
  - `docopt==0.6.2`
- `uv` cache was redirected to `/data/kirby/lerobot/.cache/uv` when installing.

## Local Models

Downloaded local model paths:

```text
/data/kirby/lerobot/vla_wa/models/smolvla_base
/data/kirby/lerobot/vla_wa/models/SmolVLM2-500M-Video-Instruct
```

Approximate sizes:

```text
smolvla_base: 873M
SmolVLM2-500M-Video-Instruct: 1.9G
```

Important:

- `smolvla_base/config.json` still points at remote `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
- When loading locally, override:

```python
cfg.vlm_model_name = "/data/kirby/lerobot/vla_wa/models/SmolVLM2-500M-Video-Instruct"
```

Download note:

- `hf-mirror.com` failed in this environment with a Hugging Face Hub metadata error.
- Direct Hugging Face download worked.
- The SmolVLM2 repo includes a large `onnx/` directory that is not needed for PyTorch/LeRobot loading. It was excluded/removed after partial download.

## Implemented Files

Schema:

```text
vla_wa/configs/robot_schema.yaml
vla_wa/robot_schema/__init__.py
vla_wa/robot_schema/miniwalle_schema.py
```

Scripts:

```text
vla_wa/scripts/convert_sim_to_lerobot.py
vla_wa/scripts/inspect_smolvla_sample.py
vla_wa/scripts/validate_smolvla_real_model.py
```

Datasets generated:

```text
vla_wa/data/lerobot_dataset/miniwalle_motion_v1
vla_wa/data/lerobot_dataset/miniwalle_motion_v1_dummy_image
```

## MiniWalle Schema

Default profile:

```text
upper_body_v1
```

State/action dimension:

```text
state_dim = 13
action_dim = 13
```

Field order:

```text
left_eyebrow
right_eyebrow
left_eye
right_eye
head_pitch
head_yaw
neck
left_shoulder_pitch
right_shoulder_pitch
left_shoulder_yaw
right_shoulder_yaw
left_arm
right_arm
```

Design choices:

- `chunk_size` and `n_action_steps` are not schema constants.
- First action representation is continuous `target_state`.
- `body_chassis_v1` exists as an optional extension with:
  - `chassis_linear_velocity`
  - `chassis_angular_velocity`
- Two pose presets are defined:
  - `hardware_default`
  - `omni_neutral`

Source references used:

```text
/data/kirby/miniwalle-robotics/motion_dataset/configs/joints.yaml
/data/kirby/miniwalle-robotics/miniwalle/real_robot/payload.py
/data/kirby/walle-omni-demo/skills/motion/data/neutral_pose.json
```

## Dataset Conversion

Default source data:

```text
/data/kirby/miniwalle-robotics/motion_dataset/frames/wave_001.json
```

Default conversion:

```bash
/data/kirby/lerobot/.venv/bin/python -B \
vla_wa/scripts/convert_sim_to_lerobot.py
```

Black-image mock dataset conversion:

```bash
/data/kirby/lerobot/.venv/bin/python -B \
vla_wa/scripts/convert_sim_to_lerobot.py \
--output vla_wa/data/lerobot_dataset/miniwalle_motion_v1_dummy_image \
--repo-id local/miniwalle_motion_v1_dummy_image \
--dummy-image-key observation.images.context \
--overwrite
```

Why black image:

- Current SmolVLA model forward requires at least one visual feature.
- Stage 1 expressive motion does not depend on real vision yet.
- The black image is a temporary interface shim so the pipeline can use `instruction + robot_state/action chunk`.

Current dummy visual feature:

```text
observation.images.context
shape = (3, 64, 64)
```

## Validation Results

### LeRobot Dataset + SmolVLA Processor

Command:

```bash
env HF_DATASETS_CACHE=/tmp/hf_datasets \
/data/kirby/lerobot/.venv/bin/python -B \
vla_wa/scripts/inspect_smolvla_sample.py \
--root vla_wa/data/lerobot_dataset/miniwalle_motion_v1_dummy_image \
--repo-id local/miniwalle_motion_v1_dummy_image
```

Observed:

```text
episodes=1
frames=47
fps=20
observation.state=(1, 1, 13)
action=(1, 50, 13)
observation.images.context=(1, 1, 3, 64, 64)
task=['挥手\n']
action_is_pad.sum=3
```

The 3 padded steps are expected because the episode has 47 frames and default SmolVLA chunk size is 50.

### Real SmolVLA Forward Pass

Command:

```bash
env HF_HOME=/data/kirby/lerobot/.cache/huggingface \
HF_DATASETS_CACHE=/tmp/hf_datasets \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
/data/kirby/lerobot/.venv/bin/python -B \
vla_wa/scripts/validate_smolvla_real_model.py
```

Observed:

```text
input_features:
  observation.state: shape=(13,)
  observation.images.context: shape=(3, 64, 64)

output_features:
  action: shape=(13,)

batch:
  observation.state=(1, 1, 13)
  action=(1, 50, 13)
  observation.images.context=(1, 1, 3, 64, 64)
  action_is_pad.sum=3

forward:
  loss ~= 1.28
```

Conclusion:

```text
MiniWalle frames JSON
-> LeRobot dataset
-> SmolVLA delta/action chunk sample
-> SmolVLA real processor
-> SmolVLA real model forward loss
```

This path is working.

## Important Compatibility Notes

`smolvla_base` pretrained config is originally:

```text
state_dim = 6
action_dim = 6
camera1/camera2/camera3
```

For MiniWalle validation, the script overrides config features from dataset metadata:

```text
state_dim = 13
action_dim = 13
observation.images.context
```

The checkpoint is loaded with `strict=False`, so newly shaped state/action projection weights are allowed to be missing or mismatched. This is acceptable for pipeline validation and future fine-tuning setup, but the model is not yet a trained MiniWalle policy.

## Current Limitations

- Only one motion episode is available in the current converted dataset: `wave_001`.
- The visual input is a black dummy image.
- No training run has been started yet.
- No rollout/inference action chunk has been decoded back to MiniWalle executor payload yet.
- Real tokenizer/model forward now works, but CPU is slow. There was no CUDA backend available in this environment.

## Recommended Next Tasks

1. Add more MiniWalle motion JSON episodes under `/data/kirby/miniwalle-robotics/motion_dataset/frames`.
2. Re-run conversion for a larger dataset with `--dummy-image-key observation.images.context`.
3. Create a minimal SmolVLA fine-tuning config for MiniWalle:
   - `policy.path=/data/kirby/lerobot/vla_wa/models/smolvla_base`
   - override `policy.vlm_model_name=/data/kirby/lerobot/vla_wa/models/SmolVLM2-500M-Video-Instruct`
   - dataset root/repo id pointing to the local MiniWalle dataset.
4. Run a tiny training smoke test, e.g. 1-5 optimizer steps, just to verify training loop.
5. Add a script to convert predicted action chunks back into MiniWalle joint frames.
6. Add replay/plot utility for `ActionChunk`:
   - inspect state/action curves
   - verify limits
   - compare predicted vs target trajectory.

## Useful Commands

Inspect sample:

```bash
env HF_DATASETS_CACHE=/tmp/hf_datasets \
/data/kirby/lerobot/.venv/bin/python -B \
vla_wa/scripts/inspect_smolvla_sample.py \
--root vla_wa/data/lerobot_dataset/miniwalle_motion_v1_dummy_image \
--repo-id local/miniwalle_motion_v1_dummy_image
```

Validate real model:

```bash
env HF_HOME=/data/kirby/lerobot/.cache/huggingface \
HF_DATASETS_CACHE=/tmp/hf_datasets \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
/data/kirby/lerobot/.venv/bin/python -B \
vla_wa/scripts/validate_smolvla_real_model.py
```

Regenerate dummy-image dataset:

```bash
/data/kirby/lerobot/.venv/bin/python -B \
vla_wa/scripts/convert_sim_to_lerobot.py \
--output vla_wa/data/lerobot_dataset/miniwalle_motion_v1_dummy_image \
--repo-id local/miniwalle_motion_v1_dummy_image \
--dummy-image-key observation.images.context \
--overwrite
```
