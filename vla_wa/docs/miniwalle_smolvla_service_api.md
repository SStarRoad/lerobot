# MiniWalle SmolVLA Service API

服务默认由 `lerobot/start_server.sh` 启动，监听本机所有网卡：

```bash
cd /data/kirby/lerobot
./start_server.sh
```

默认地址：

```text
http://<这台机器的 IP>:8782
```

如果在同一台机器上调用，也可以用：

```text
http://127.0.0.1:8782
```

## GET /health

检查服务是否已加载模型。

```bash
curl -s http://127.0.0.1:8782/health
```

返回示例：

```json
{
  "ok": true,
  "checkpoint": "outputs/train/.../pretrained_model",
  "device": "cuda",
  "loaded": true
}
```

## GET /metadata

查看模型 fps、action 维度、关节字段顺序。

```bash
curl -s http://127.0.0.1:8782/metadata
```

## POST /predict

默认行为保持原来的 action chunk 输出，不带 MQTT：

```bash
curl -s http://127.0.0.1:8782/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "instruction": "挥手打招呼",
    "current_joints": {
      "left_eyebrow": 10,
      "right_eyebrow": 10,
      "left_eye": 7,
      "right_eye": 7,
      "head_pitch": 0,
      "head_yaw": 0,
      "neck": 60,
      "left_shoulder_pitch": 0,
      "right_shoulder_pitch": 0,
      "left_shoulder_yaw": -90,
      "right_shoulder_yaw": -90,
      "left_arm": 0,
      "right_arm": 0
    }
  }'
```

返回字段：

```text
ok              是否成功
checkpoint      当前 checkpoint
instruction     输入指令
fps / dt         动作频率
shape           [chunk_len, action_dim]
action_names    actions 每一列对应的关节名
actions         SmolVLA 原始 action chunk
```

## POST /predict with return_mqtt

需要直接给机器人侧 MQTT 下发 payload 时，传 `return_mqtt: true`。服务会把 SmolVLA action chunk 转成 MiniWalle frame motion，并应用原先 MiniWalle 逻辑里的关节限位、EMA 和每关节速度限制，然后生成 `multimodal_action / motor_control` payload。

```bash
curl -s http://127.0.0.1:8782/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "instruction": "挥手打招呼",
    "return_mqtt": true,
    "motion_id": "hello_wave_001",
    "current_joints": {
      "left_eyebrow": 10,
      "right_eyebrow": 10,
      "left_eye": 7,
      "right_eye": 7,
      "head_pitch": 0,
      "head_yaw": 0,
      "neck": 60,
      "left_shoulder_pitch": 0,
      "right_shoulder_pitch": 0,
      "left_shoulder_yaw": -90,
      "right_shoulder_yaw": -90,
      "left_arm": 0,
      "right_arm": 0
    }
  }'
```

新增返回字段：

```text
frames          MiniWalle frame motion，已限位和滤波
mqtt_payload    可以直接给后续 MQTT 发布逻辑使用的 motor_control payload
```

常用可选参数：

```text
return_mqtt     默认 false；true 时返回 frames 和 mqtt_payload
motion_id       可选；不传则服务自动生成
filter_alpha    默认 0.35；EMA 滤波强度
use_filter      默认 true；false 时只做关节限位，不做 EMA/速度限制
dense_mqtt      默认 false；true 时保留逐帧 timing，不做冗余压缩
mqtt_source     默认 lerobot_smolvla_service；写入 mqtt_payload.meta.source
seed            可选；设置推理随机种子
```

`mqtt_payload` 顶层结构：

```json
{
  "trace": "...",
  "type": "multimodal_action",
  "command": "motor_control",
  "payload": [
    {
      "group_id": 1,
      "duration_ms": 1,
      "actions": [
        {"duration_ms": 1, "angle": 12.3, "type": "head", "direction": "yaw"}
      ]
    }
  ],
  "meta": {
    "schema_version": "miniwalle.atomic_motion.v1",
    "source": "lerobot_smolvla_service",
    "mode": "frame_motion",
    "motion_id": "hello_wave_001"
  }
}
```

注意：

- `current_joints` 必须包含 `/metadata` 里的全部 `state_names`，单位是度。
- HTTP 服务只返回 JSON，不直接发布 MQTT；调用方把 `mqtt_payload` 交给现有 MQTT 发布逻辑即可。
- 如果同事从另一台机器访问，把 `127.0.0.1` 换成这台服务器的内网 IP，端口仍是 `8782`。
