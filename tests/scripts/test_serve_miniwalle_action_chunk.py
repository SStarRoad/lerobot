from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
import sys
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vla_wa.scripts.serve_miniwalle_action_chunk import (
    OBS_CONTEXT_IMAGE,
    OBS_STATE,
    MiniWalleActionChunkService,
    build_action_chunk_response,
    build_server,
    current_joints_to_state,
    discover_checkpoint_candidates,
    load_robot_schema_fields,
    resolve_checkpoint_path,
    resolve_latest_checkpoint,
)


ACTION_NAMES = [
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "head_pitch",
    "head_yaw",
    "neck",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_arm",
    "right_arm",
]


class MiniWalleActionChunkServiceTest(unittest.TestCase):
    def test_resolve_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pretrained = tmp_path / "050000" / "pretrained_model"
            pretrained.mkdir(parents=True)
            (pretrained / "config.json").write_text("{}", encoding="utf-8")

            self.assertEqual(resolve_checkpoint_path(tmp_path / "050000"), pretrained)
            self.assertEqual(resolve_checkpoint_path(pretrained), pretrained)

    def test_latest_checkpoint_prefers_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            for step in ("050000", "060000"):
                pretrained = checkpoints / step / "pretrained_model"
                pretrained.mkdir(parents=True)
                (pretrained / "config.json").write_text("{}", encoding="utf-8")

            self.assertEqual(resolve_latest_checkpoint(checkpoints), checkpoints / "060000" / "pretrained_model")

    def test_load_robot_schema_fields(self) -> None:
        schema = load_robot_schema_fields(REPO_ROOT / "vla_wa" / "configs" / "robot_schema.yaml")

        self.assertEqual(schema["state_fields"], ACTION_NAMES)
        self.assertEqual(schema["action_fields"], ACTION_NAMES)

    def test_current_joints_to_state_uses_schema_order(self) -> None:
        current_joints = {name: index for index, name in enumerate(ACTION_NAMES)}

        state = current_joints_to_state(current_joints, list(reversed(ACTION_NAMES)))

        self.assertEqual(state, list(reversed([float(index) for index in range(len(ACTION_NAMES))])))

    def test_current_joints_to_state_requires_all_joints(self) -> None:
        current_joints = {name: 0 for name in ACTION_NAMES}
        del current_joints["head_yaw"]

        with self.assertRaisesRegex(ValueError, "head_yaw"):
            current_joints_to_state(current_joints, ACTION_NAMES)

    def test_current_joints_to_state_requires_numeric_joints(self) -> None:
        current_joints = {name: 0 for name in ACTION_NAMES}
        current_joints["head_yaw"] = "not-a-number"

        with self.assertRaisesRegex(ValueError, "head_yaw"):
            current_joints_to_state(current_joints, ACTION_NAMES)

    def test_build_action_chunk_response_schema(self) -> None:
        response = build_action_chunk_response(
            checkpoint=Path("outputs/train/checkpoints/050000/pretrained_model"),
            instruction="jump",
            fps=10,
            action_names=["a", "b"],
            action_chunk=[[[1, 2], [3, 4]]],
        )

        self.assertEqual(
            response,
            {
                "ok": True,
                "checkpoint": str(Path("outputs/train/checkpoints/050000/pretrained_model")),
                "instruction": "jump",
                "fps": 10,
                "dt": 0.1,
                "shape": [2, 2],
                "action_names": ["a", "b"],
                "actions": [[1.0, 2.0], [3.0, 4.0]],
            },
        )

    def test_http_smoke_with_fake_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            robot_schema = tmp_path / "robot_schema.yaml"
            robot_schema.write_text(robot_schema_text(), encoding="utf-8")
            checkpoint = tmp_path / "checkpoints" / "050000" / "pretrained_model"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")

            service = MiniWalleActionChunkService(
                checkpoint=checkpoint,
                checkpoints_dir=checkpoint.parent.parent,
                robot_schema=robot_schema,
                fps=10,
                device="cpu",
                predictor=FakePredictor(),
            )
            service.load()
            service.chunk_size = 2
            server = build_server("127.0.0.1", 0, service)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                self.assertTrue(get_json(f"{base_url}/health")["loaded"])
                metadata = get_json(f"{base_url}/metadata")
                self.assertEqual(metadata["state_names"], ACTION_NAMES)
                self.assertEqual(metadata["action_names"], ACTION_NAMES)
                current_joints = {name: 0 for name in ACTION_NAMES}
                current_joints["head_yaw"] = 4.0
                response = post_json(
                    f"{base_url}/predict",
                    {"instruction": "jump", "current_joints": current_joints},
                )
                self.assertTrue(response["ok"])
                self.assertEqual(response["instruction"], "jump")
                self.assertEqual(response["shape"], [2, len(ACTION_NAMES)])
                self.assertEqual(response["actions"][0][ACTION_NAMES.index("head_yaw")], 4.0)
            finally:
                server.shutdown()
                server.server_close()


class FakePredictor:
    def predict_action_chunk(self, frame: dict[str, object]) -> list[list[float]]:
        assert frame["task"] == "jump"
        assert OBS_CONTEXT_IMAGE in frame
        state = list(frame[OBS_STATE])
        return [state, [value + 1.0 for value in state]]


def robot_schema_text() -> str:
    state_lines = "\n".join(f"      - {name}" for name in ACTION_NAMES)
    return f"""profiles:
  upper_body_v1:
    state_fields:
{state_lines}
    action_fields:
{state_lines}
"""


def get_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
