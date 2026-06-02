"""MiniWalle state/action schema for action-chunk motion models.

The first WA prototype uses continuous target-state actions. The canonical
upper-body order comes from miniwalle-robotics/motion_dataset/configs/joints.yaml
and matches the motor_control payload mapping in miniwalle/real_robot/payload.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Number = int | float
MotionProfile = Literal["upper_body_v1", "body_chassis_v1"]


@dataclass(frozen=True)
class JointSpec:
    name: str
    motor_type: str
    default: float
    min_value: float
    max_value: float
    max_speed: float
    unit: str = "degree"
    side: str | None = None
    direction: str | None = None

    def clip(self, value: Number) -> float:
        return min(self.max_value, max(self.min_value, float(value)))


JOINT_SPECS: tuple[JointSpec, ...] = (
    JointSpec("left_eyebrow", "eyebrow", 10, 0, 20, 120, side="left"),
    JointSpec("right_eyebrow", "eyebrow", 10, 0, 20, 120, side="right"),
    JointSpec("left_eye", "eye", 7, 0, 15, 120, side="left"),
    JointSpec("right_eye", "eye", 7, 0, 15, 120, side="right"),
    JointSpec("head_pitch", "head", 0, -25, 28, 90, direction="pitch"),
    JointSpec("head_yaw", "head", 0, -80, 80, 100, direction="yaw"),
    JointSpec("neck", "neck", 60, 0, 120, 100),
    JointSpec("left_shoulder_pitch", "shoulder", 0, 0, 90, 100, side="left", direction="pitch"),
    JointSpec("right_shoulder_pitch", "shoulder", 0, 0, 90, 100, side="right", direction="pitch"),
    JointSpec("left_shoulder_yaw", "shoulder", -90, -90, 0, 100, side="left", direction="yaw"),
    JointSpec("right_shoulder_yaw", "shoulder", -90, -90, 0, 100, side="right", direction="yaw"),
    JointSpec("left_arm", "arm", 0, -28, 28, 100, side="left"),
    JointSpec("right_arm", "arm", 0, -28, 28, 100, side="right"),
)

JOINT_ORDER: tuple[str, ...] = tuple(spec.name for spec in JOINT_SPECS)
JOINT_SPEC_BY_NAME: dict[str, JointSpec] = {spec.name: spec for spec in JOINT_SPECS}

CHASSIS_ORDER: tuple[str, ...] = ("chassis_linear_velocity", "chassis_angular_velocity")
CHASSIS_LIMITS: dict[str, tuple[float, float, str]] = {
    "chassis_linear_velocity": (-1000.0, 1000.0, "mm_per_s"),
    "chassis_angular_velocity": (-60.0, 60.0, "deg_per_s"),
}

POSE_PRESETS: dict[str, dict[str, float]] = {
    "hardware_default": {spec.name: float(spec.default) for spec in JOINT_SPECS},
    "omni_neutral": {
        "left_eyebrow": 10.0,
        "right_eyebrow": 10.0,
        "left_eye": 7.0,
        "right_eye": 7.0,
        "head_pitch": 5.0,
        "head_yaw": 0.0,
        "neck": 50.0,
        "left_shoulder_pitch": 10.0,
        "right_shoulder_pitch": 10.0,
        "left_shoulder_yaw": -60.0,
        "right_shoulder_yaw": -60.0,
        "left_arm": 0.0,
        "right_arm": 0.0,
    },
}
for preset in POSE_PRESETS.values():
    preset.update({"chassis_linear_velocity": 0.0, "chassis_angular_velocity": 0.0})

STATE_PROFILES: dict[str, tuple[str, ...]] = {
    "upper_body_v1": JOINT_ORDER,
    "body_chassis_v1": JOINT_ORDER + CHASSIS_ORDER,
}

ACTION_PROFILES: dict[str, tuple[str, ...]] = {
    # Main Stage-1 expressive motion profile: model outputs target joint state.
    "upper_body_v1": JOINT_ORDER,
    # Optional profile for future chassis-aware motions; keep out of default training.
    "body_chassis_v1": JOINT_ORDER + CHASSIS_ORDER,
}

DEFAULT_STATE_PROFILE: MotionProfile = "upper_body_v1"
DEFAULT_ACTION_PROFILE: MotionProfile = "upper_body_v1"


@dataclass(frozen=True)
class ActionChunk:
    """A future action chunk with variable horizon and control dt.

    `actions` is shaped as [horizon, action_dim]. The horizon is intentionally
    not tied to SmolVLA's chunk_size or n_action_steps.
    """

    dt: float
    actions: tuple[tuple[float, ...], ...]
    action_profile: MotionProfile = DEFAULT_ACTION_PROFILE
    action_type: Literal["target_state", "delta"] = "target_state"

    @property
    def horizon(self) -> int:
        return len(self.actions)

    @property
    def action_dim(self) -> int:
        return len(ACTION_PROFILES[self.action_profile])

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        expected_dim = self.action_dim
        for index, row in enumerate(self.actions):
            if len(row) != expected_dim:
                raise ValueError(f"actions[{index}] has dim {len(row)}, expected {expected_dim}")


class MiniWalleSchema:
    """Canonical vector order and conversion helpers for MiniWalle motion data."""

    state_profile: MotionProfile
    action_profile: MotionProfile

    def __init__(
        self,
        *,
        state_profile: MotionProfile = DEFAULT_STATE_PROFILE,
        action_profile: MotionProfile = DEFAULT_ACTION_PROFILE,
    ) -> None:
        self.state_profile = state_profile
        self.action_profile = action_profile

    @property
    def state_names(self) -> tuple[str, ...]:
        return STATE_PROFILES[self.state_profile]

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_PROFILES[self.action_profile]

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    def neutral_state(
        self,
        *,
        preset: Literal["hardware_default", "omni_neutral"] = "hardware_default",
    ) -> dict[str, float]:
        state = POSE_PRESETS[preset]
        return {name: state[name] for name in self.state_names}

    def vectorize_state(self, state: dict[str, Number], *, fill_neutral: bool = True) -> tuple[float, ...]:
        base = self.neutral_state() if fill_neutral else {}
        base.update({key: float(value) for key, value in state.items()})
        return tuple(self._clip_name(name, base[name]) for name in self.state_names)

    def devectorize_state(self, vector: list[Number] | tuple[Number, ...]) -> dict[str, float]:
        self._check_dim("state", vector, self.state_dim)
        return {
            name: self._clip_name(name, value)
            for name, value in zip(self.state_names, vector, strict=True)
        }

    def vectorize_action(self, target_state: dict[str, Number], *, fill_neutral: bool = True) -> tuple[float, ...]:
        base = self.neutral_state() if fill_neutral else {}
        base.update({key: float(value) for key, value in target_state.items()})
        return tuple(self._clip_name(name, base[name]) for name in self.action_names)

    def devectorize_action(self, vector: list[Number] | tuple[Number, ...]) -> dict[str, float]:
        self._check_dim("action", vector, self.action_dim)
        return {
            name: self._clip_name(name, value)
            for name, value in zip(self.action_names, vector, strict=True)
        }

    def lerobot_features(self) -> dict[str, dict[str, object]]:
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": list(self.state_names),
            },
            "action": {
                "dtype": "float32",
                "shape": (self.action_dim,),
                "names": list(self.action_names),
            },
        }

    @staticmethod
    def _check_dim(kind: str, vector: list[Number] | tuple[Number, ...], expected_dim: int) -> None:
        if len(vector) != expected_dim:
            raise ValueError(f"{kind} vector has dim {len(vector)}, expected {expected_dim}")

    @staticmethod
    def _clip_name(name: str, value: Number) -> float:
        if name in JOINT_SPEC_BY_NAME:
            return JOINT_SPEC_BY_NAME[name].clip(value)
        if name in CHASSIS_LIMITS:
            min_value, max_value, _ = CHASSIS_LIMITS[name]
            return min(max_value, max(min_value, float(value)))
        raise KeyError(f"unknown MiniWalle state/action field: {name}")
