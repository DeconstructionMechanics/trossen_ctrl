"""Keyboard input mapped to 6 + 1 degree arm commands.

Edit controller/config.yaml to change the terminal controls.

Movement:
    w/s: x forward/backward
    a/d: y left/right
    e/q: z up/down

Orientation:
    i/k: pitch up/down
    j/l: yaw left/right
    u/o: roll left/right

Gripper and session:
    x/z: open/close gripper
    c: reset to the startup target
    Esc or Ctrl-C: quit
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Optional

from .commands import (
    ACTION_DIRECTIONS,
    GRIPPER_DIRECTIONS,
    MotionCommand,
    SENSITIVITY_ACTIONS,
)
from .config import (
    ACTION_DESCRIPTIONS,
    load_keybind_config,
    sensitivity_config,
    validate_sensitivity_bounds,
)

QUIT_KEYS = {"\x03", "\x1b"}


class RawTerminal:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self._settings = None

    def __enter__(self):
        if self.stream.isatty():
            self._settings = termios.tcgetattr(self.stream)
            tty.setcbreak(self.stream.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._settings is not None:
            termios.tcsetattr(self.stream, termios.TCSADRAIN, self._settings)


class CartesianKeyboardController:
    """Poll terminal keys and emit elapsed-time-scaled MotionCommand values."""

    source_name = "keyboard"

    def __init__(
        self,
        linear_step: float = 0.01,
        angular_step: float = 0.05,
        gripper_step: float = 0.005,
        key_bindings: Optional[dict[str, str]] = None,
        config_path: Optional[str] = None,
        sensitivity: Optional[float] = None,
        sensitivity_factor: float = 1.25,
        min_sensitivity: Optional[float] = None,
        max_sensitivity: Optional[float] = None,
        key_release_timeout: float = 0.06,
    ):
        config = load_keybind_config(config_path)
        sensitivity_settings = sensitivity_config(config)
        self.key_bindings = key_bindings or config["keyboard"]
        self.key_to_action = {
            key.lower(): action for action, key in self.key_bindings.items()
        }
        self.linear_rate = linear_step
        self.angular_rate = angular_step
        self.gripper_rate = gripper_step
        self.sensitivity = (
            sensitivity
            if sensitivity is not None
            else sensitivity_settings["default_sensitivity"]
        )
        self.sensitivity_factor = sensitivity_factor
        self.min_sensitivity = (
            min_sensitivity
            if min_sensitivity is not None
            else sensitivity_settings["min_sensitivity"]
        )
        self.max_sensitivity = (
            max_sensitivity
            if max_sensitivity is not None
            else sensitivity_settings["max_sensitivity"]
        )
        validate_sensitivity_bounds(self.min_sensitivity, self.max_sensitivity)
        self.sensitivity = self._clamp_sensitivity(self.sensitivity)
        self.key_release_timeout = key_release_timeout
        self.active_actions: dict[str, float] = {}
        self.last_tick = time.monotonic()

    def __enter__(self):
        self.last_tick = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read_command(self, timeout: Optional[float] = None) -> Optional[MotionCommand]:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        now = time.monotonic()
        had_active_actions = bool(self.active_actions)

        if readable:
            for key in self._read_available_keys():
                command = self._process_key_event(key, now)
                if command is not None:
                    self.last_tick = now
                    return command

        self._expire_inactive_keys(now)
        if not had_active_actions and self.active_actions:
            self.last_tick = now
            return None
        dt = now - self.last_tick
        self.last_tick = now
        return self._continuous_command(dt)

    def read_key(self, timeout: Optional[float] = None) -> Optional[str]:
        command = self.read_command(timeout)
        if command is None:
            return None
        return command.action

    def command_for_key(self, key: Optional[str]) -> Optional[MotionCommand]:
        if key is None:
            return None
        return self._process_key_event(key.lower(), time.monotonic())

    def _read_available_keys(self) -> list[str]:
        keys = [sys.stdin.read(1)]
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                return keys
            keys.append(sys.stdin.read(1))

    def _process_key_event(self, key: str, now: float) -> Optional[MotionCommand]:
        key = key.lower()
        if key in QUIT_KEYS:
            return MotionCommand(source=self.source_name, action="quit", quit=True)

        action = self.key_to_action.get(key)
        if action is None:
            return None
        if action == "reset":
            self.active_actions.clear()
            return MotionCommand(source=self.source_name, action=action, reset=True)
        if action in SENSITIVITY_ACTIONS:
            self._change_sensitivity(action)
            return MotionCommand(
                source=self.source_name,
                action=action,
                sensitivity=self.sensitivity,
            )
        if action in ACTION_DIRECTIONS or action in GRIPPER_DIRECTIONS:
            self.active_actions[action] = now
        return None

    def _expire_inactive_keys(self, now: float):
        expired = [
            action
            for action, last_seen in self.active_actions.items()
            if now - last_seen > self.key_release_timeout
        ]
        for action in expired:
            del self.active_actions[action]

    def _continuous_command(self, dt: float) -> Optional[MotionCommand]:
        if dt <= 0.0 or not self.active_actions:
            return None

        pose_delta = [0.0] * 6
        gripper_delta = 0.0
        action_names = []
        scale = self.sensitivity * dt

        for action in self.active_actions:
            action_names.append(action)
            if action in GRIPPER_DIRECTIONS:
                gripper_delta += self.gripper_rate * GRIPPER_DIRECTIONS[action] * scale
                continue
            direction = ACTION_DIRECTIONS[action]
            for index, value in enumerate(direction):
                rate = self.linear_rate if index < 3 else self.angular_rate
                pose_delta[index] += rate * value * scale

        return MotionCommand(
            source=self.source_name,
            action="+".join(action_names),
            pose_delta=tuple(pose_delta),
            gripper_delta=gripper_delta,
            sensitivity=self.sensitivity,
        )

    def _change_sensitivity(self, action: str):
        if action == "sensitivityup":
            self.sensitivity *= self.sensitivity_factor
        elif action == "sensitivitydown":
            self.sensitivity /= self.sensitivity_factor
        self.sensitivity = self._clamp_sensitivity(self.sensitivity)

    def _clamp_sensitivity(self, value: float) -> float:
        return min(max(value, self.min_sensitivity), self.max_sensitivity)

    def print_help(self):
        print("Keyboard Cartesian control")
        for action, key in self.key_bindings.items():
            description = ACTION_DESCRIPTIONS.get(action, action)
            print(f"  {key}: {action} - {description}")
        print("  Esc or Ctrl-C: quit")
        print(
            "  Note: terminal keyboard input has no key-release event; movement "
            f"stops after {self.key_release_timeout:.2f}s without key repeats."
        )
