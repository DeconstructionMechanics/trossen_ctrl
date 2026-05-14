"""Maintain and print the arm target state."""

from __future__ import annotations

import math
from typing import Iterable

from .commands import (
    MotionCommand,
    POSE_LABELS,
    Pose,
    TargetState,
    X_INDEX,
    YAW_INDEX,
    Y_INDEX,
    ZERO_POSE,
)


class TargetStateController:
    """Applies controller commands to a persistent 6 + 1 target state."""

    def __init__(
        self,
        initial_pose: Iterable[float],
        initial_gripper: float,
        return_pose: Iterable[float] = ZERO_POSE,
        return_gripper: float = 0.0,
        gripper_min: float = 0.0,
        gripper_max: float = 0.044,
        yaw_follow_xy: bool = True,
    ):
        self.return_state = TargetState.from_iterables(return_pose, return_gripper)
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self.yaw_follow_xy = yaw_follow_xy
        self._state = TargetState.from_iterables(initial_pose, initial_gripper)
        self._yaw_offset = self._state.pose[YAW_INDEX] - self._yaw_base(self._state.pose)
        if self.yaw_follow_xy:
            self._state = self._state_with_followed_yaw(self._state)

    @property
    def state(self) -> TargetState:
        return self._state

    def reset(self) -> TargetState:
        return self.set_state(self.return_state.pose, self.return_state.gripper)

    def set_state(self, pose: Iterable[float], gripper: float) -> TargetState:
        self._state = TargetState.from_iterables(pose, gripper)
        self._yaw_offset = self._state.pose[YAW_INDEX] - self._yaw_base(self._state.pose)
        if self.yaw_follow_xy:
            self._state = self._state_with_followed_yaw(self._state)
        return self._state

    def iter_reset_steps(self, steps: int):
        """Yield intermediate states from the current target to the return target."""
        start_pose = self._state.pose
        start_gripper = self._state.gripper
        end_pose = self.return_state.pose
        end_gripper = self.return_state.gripper

        for index in range(1, max(1, steps) + 1):
            amount = index / max(1, steps)
            pose = tuple(
                start + (end - start) * amount
                for start, end in zip(start_pose, end_pose)
            )
            gripper = start_gripper + (end_gripper - start_gripper) * amount
            yield self.set_state(pose, gripper)

    def apply(self, command: MotionCommand) -> TargetState:
        if command.reset:
            return self.reset()
        if command.quit:
            return self._state

        pose_delta = list(command.pose_delta)
        if self.yaw_follow_xy:
            self._yaw_offset += pose_delta[YAW_INDEX]
            pose_delta[YAW_INDEX] = 0.0

        pose = tuple(value + change for value, change in zip(self._state.pose, pose_delta))
        gripper = self._clamp_gripper(self._state.gripper + command.gripper_delta)
        self._state = TargetState(pose=pose, gripper=gripper)
        if self.yaw_follow_xy:
            self._state = self._state_with_followed_yaw(self._state)
        return self._state

    def _clamp_gripper(self, value: float) -> float:
        return min(max(value, self.gripper_min), self.gripper_max)

    def pose_and_gripper(self) -> tuple[Pose, float]:
        return self._state.pose, self._state.gripper

    def format_state(self) -> str:
        values = [round(value, 4) for value in self._state.as_vector()]
        labels = ", ".join((*POSE_LABELS, "gripper"))
        return f"target=[{labels}]={values}"

    def print_state(self, prefix: str = ""):
        print(f"\r{prefix}{self.format_state()}   ", end="", flush=True)

    def _state_with_followed_yaw(self, state: TargetState) -> TargetState:
        pose = list(state.pose)
        pose[YAW_INDEX] = self._normalize_angle(self._yaw_base(state.pose) + self._yaw_offset)
        return TargetState(pose=tuple(pose), gripper=state.gripper)

    def _yaw_base(self, pose: Pose) -> float:
        x = pose[X_INDEX]
        y = pose[Y_INDEX]
        if math.hypot(x, y) < 1e-9:
            return 0.0
        return math.atan2(y, x)

    def _normalize_angle(self, value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))
