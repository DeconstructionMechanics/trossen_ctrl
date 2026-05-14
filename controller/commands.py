"""Shared 6 + 1 degree command model.

The first six values are Cartesian pose deltas in the order expected by the
Trossen driver:
    x, y, z, roll, pitch, yaw

The seventh value is gripper delta. Keyboard and Xbox controllers both map
their inputs into this same model so the arm action loop can stay input-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


Pose = Tuple[float, float, float, float, float, float]

X_INDEX = 0
Y_INDEX = 1
Z_INDEX = 2
ROLL_INDEX = 3
PITCH_INDEX = 4
YAW_INDEX = 5
POSE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")

ZERO_POSE: Pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

ACTION_DIRECTIONS: dict[str, Pose] = {
    "forward": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "leftward": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "rightward": (0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    "upward": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "downward": (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
    "pitchup": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "pitchdown": (0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    "yawleft": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "yawright": (0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    "rollleft": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "rollright": (0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
}

GRIPPER_DIRECTIONS = {
    "gripperopen": 1.0,
    "gripperclose": -1.0,
}

SENSITIVITY_ACTIONS = {"sensitivityup", "sensitivitydown"}


@dataclass(frozen=True)
class MotionCommand:
    """One input event mapped to robot target-state changes."""

    source: str
    action: str
    pose_delta: Pose = ZERO_POSE
    gripper_delta: float = 0.0
    reset: bool = False
    quit: bool = False
    sensitivity: float = 1.0


@dataclass(frozen=True)
class TargetState:
    """Robot target state: Cartesian pose plus gripper position."""

    pose: Pose
    gripper: float

    def as_vector(self) -> Tuple[float, float, float, float, float, float, float]:
        return (*self.pose, self.gripper)

    @classmethod
    def from_iterables(cls, pose: Iterable[float], gripper: float) -> "TargetState":
        values = tuple(float(value) for value in pose)
        if len(values) != 6:
            raise ValueError(f"Pose must contain 6 values, got {len(values)}")
        return cls(values, float(gripper))


def scaled_pose_delta(action: str, linear_step: float, angular_step: float) -> Pose:
    direction = ACTION_DIRECTIONS[action]
    return tuple(
        linear_step * value if index < 3 else angular_step * value
        for index, value in enumerate(direction)
    )
