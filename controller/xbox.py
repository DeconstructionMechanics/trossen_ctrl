"""Xbox controller input mapped to 6 + 1 degree arm commands.

Edit controller/config.yaml to change the controls.

Buttons:
    A/B: close/open gripper
    Y: reset to the startup target
    Menu/Start: quit
    LB/RB: roll left/right

Axes:
    Left stick X/Y: y and x motion
    Right stick X/Y: yaw and pitch
    LT/RT: z down/up
    D-pad: coarse x/y motion
"""

from __future__ import annotations

from pathlib import Path
import select
from typing import Optional

from .commands import (
    ACTION_DIRECTIONS,
    GRIPPER_DIRECTIONS,
    MotionCommand,
)


try:
    from evdev import InputDevice, categorize, ecodes, list_devices
except ImportError:
    InputDevice = None
    categorize = None
    ecodes = None
    list_devices = None


def find_xbox_controller(device_path: Optional[str] = None):
    if InputDevice is None:
        raise SystemExit("Missing Python package 'evdev'.")
    if device_path:
        return InputDevice(device_path)

    paths = list(list_devices())
    paths.extend(str(path) for path in Path("/dev/input/by-id").glob("*event-joystick"))
    paths.extend(str(path) for path in Path("/dev/input").glob("event*"))

    seen: set[str] = set()
    devices = []
    for path in paths:
        real_path = str(Path(path).resolve())
        if real_path in seen:
            continue
        seen.add(real_path)
        try:
            devices.append(InputDevice(path))
        except PermissionError:
            raise SystemExit(
                f"Permission denied reading {path}. Try adding your user to the "
                "input group, then restart WSL:\n  sudo usermod -aG input $USER"
            )

    if not devices:
        raise SystemExit("No /dev/input event devices found. Is the controller attached?")

    for device in devices:
        name = device.name.lower()
        if "xbox" in name or "controller" in name or "x-input" in name or "xinput" in name:
            return device

    print("Input devices found:")
    for device in devices:
        print(f"  {device.path}: {device.name}")
    raise SystemExit("Could not find an Xbox controller input device.")


import time

from .commands import SENSITIVITY_ACTIONS
from .config import (
    ACTION_DESCRIPTIONS,
    XBOX_CONTROL_DESCRIPTIONS,
    load_keybind_config,
    sensitivity_config,
    validate_sensitivity_bounds,
)


class XboxController:
    """Poll Linux evdev input and emit elapsed-time-scaled MotionCommand values."""

    source_name = "xbox"

    def __init__(
        self,
        device_path: Optional[str] = None,
        linear_step: float = 0.01,
        angular_step: float = 0.05,
        gripper_step: float = 0.005,
        deadzone_fraction: float = 0.125,
        grab_device: bool = True,
        config_path: Optional[str] = None,
        sensitivity: Optional[float] = None,
        sensitivity_factor: float = 1.25,
        min_sensitivity: Optional[float] = None,
        max_sensitivity: Optional[float] = None,
    ):
        if InputDevice is None:
            raise SystemExit(
                "Missing Python package 'evdev'. Install it in the trossen conda "
                "environment with:\n  python -m pip install evdev"
            )

        config = load_keybind_config(config_path)
        sensitivity_settings = sensitivity_config(config)
        self.button_bindings = config["xbox_buttons"]
        self.axis_binding_config = config["xbox_axes"]
        self.button_to_action = {
            button: action
            for action, button in self.button_bindings.items()
            if ":" not in button
        }
        self.axis_button_bindings = _parse_axis_bindings(
            {
                action: binding
                for action, binding in self.button_bindings.items()
                if ":" in binding
            }
        )
        self.axis_bindings = _parse_axis_bindings(self.axis_binding_config)
        self.linear_rate = linear_step
        self.angular_rate = angular_step
        self.gripper_rate = gripper_step
        self.deadzone_fraction = deadzone_fraction
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
        self.active_buttons: set[str] = set()
        self.active_axis_buttons: dict[str, float] = {}
        self.axis_amounts: dict[str, float] = {}
        self.last_tick = time.monotonic()
        self.device = find_xbox_controller(device_path)
        self._grabbed = False
        if grab_device:
            try:
                self.device.grab()
                self._grabbed = True
            except OSError:
                pass

    def close(self):
        if self._grabbed:
            try:
                self.device.ungrab()
            except OSError:
                pass
            self._grabbed = False

    def __enter__(self):
        self.last_tick = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def read_command(self, timeout: Optional[float] = None) -> Optional[MotionCommand]:
        readable, _, _ = select.select([self.device.fd], [], [], timeout)
        now = time.monotonic()
        had_active_controls = bool(self.active_buttons or self.axis_amounts)

        if readable:
            for event in self.device.read():
                command = self._process_event(event)
                if command is not None:
                    self.last_tick = now
                    return command

        if not had_active_controls and (self.active_buttons or self.axis_amounts):
            self.last_tick = now
            return None
        dt = now - self.last_tick
        self.last_tick = now
        return self._continuous_command(dt)

    def _process_event(self, event) -> Optional[MotionCommand]:
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)
            action = self.button_to_action.get(key_event.keycode)
            if action is None:
                return None

            if action == "quit" and key_event.keystate:
                self.active_buttons.clear()
                self.active_axis_buttons.clear()
                self.axis_amounts.clear()
                return MotionCommand(source=self.source_name, action=action, quit=True)
            if action == "reset" and key_event.keystate:
                self.active_buttons.clear()
                self.active_axis_buttons.clear()
                self.axis_amounts.clear()
                return MotionCommand(source=self.source_name, action=action, reset=True)
            if action in SENSITIVITY_ACTIONS and key_event.keystate:
                self._change_sensitivity(action)
                return MotionCommand(
                    source=self.source_name,
                    action=action,
                    sensitivity=self.sensitivity,
                )

            if key_event.keystate:
                self.active_buttons.add(action)
            else:
                self.active_buttons.discard(action)
            return None

        if event.type == ecodes.EV_ABS:
            command = self._update_axis_buttons(event.code, event.value)
            if command is not None:
                return command
            self._update_axis_amounts(event.code, event.value)
        return None

    def _update_axis_buttons(self, code: int, value: int) -> Optional[MotionCommand]:
        code_name = ecodes.ABS.get(code, str(code))
        command = None

        for action, axis_name, sign in self.axis_button_bindings:
            if axis_name != code_name:
                continue
            amount = self._axis_amount(code_name, code, value, sign)
            was_active = action in self.active_axis_buttons

            if amount <= 0.0:
                self.active_axis_buttons.pop(action, None)
                continue

            self.active_axis_buttons[action] = amount
            if was_active:
                continue

            if action == "quit":
                self.active_buttons.clear()
                self.active_axis_buttons.clear()
                self.axis_amounts.clear()
                return MotionCommand(source=self.source_name, action=action, quit=True)
            if action == "reset":
                self.active_buttons.clear()
                self.active_axis_buttons.clear()
                self.axis_amounts.clear()
                return MotionCommand(source=self.source_name, action=action, reset=True)
            if action in SENSITIVITY_ACTIONS:
                self._change_sensitivity(action)
                command = MotionCommand(
                    source=self.source_name,
                    action=action,
                    sensitivity=self.sensitivity,
                )

        return command

    def _update_axis_amounts(self, code: int, value: int):
        code_name = ecodes.ABS.get(code, str(code))
        for action, axis_name, sign in self.axis_bindings:
            if axis_name != code_name:
                continue
            amount = self._axis_amount(code_name, code, value, sign)
            if amount > 0.0:
                self.axis_amounts[action] = amount
            else:
                self.axis_amounts.pop(action, None)

    def _axis_amount(self, code_name: str, code: int, value: int, sign: int) -> float:
        if code_name.startswith("ABS_HAT"):
            if sign < 0 and value < 0:
                return 1.0
            if sign > 0 and value > 0:
                return 1.0
            return 0.0

        absinfo = self.device.absinfo(code)
        span = absinfo.max - absinfo.min
        deadzone = max(1, int(span * self.deadzone_fraction))

        if code_name in {"ABS_Z", "ABS_RZ"}:
            raw = max(0, value - absinfo.min)
            if raw <= deadzone:
                return 0.0
            return min(1.0, (raw - deadzone) / max(1, span - deadzone))

        center = (absinfo.max + absinfo.min) / 2.0
        distance = (value - center) * sign
        if distance <= deadzone:
            return 0.0
        max_distance = max(abs(absinfo.max - center), abs(absinfo.min - center))
        return min(1.0, (distance - deadzone) / max(1.0, max_distance - deadzone))

    def _continuous_command(self, dt: float) -> Optional[MotionCommand]:
        action_amounts = dict(self.axis_amounts)
        for action in self.active_buttons:
            action_amounts[action] = max(action_amounts.get(action, 0.0), 1.0)
        for action, amount in self.active_axis_buttons.items():
            if action not in SENSITIVITY_ACTIONS:
                action_amounts[action] = max(action_amounts.get(action, 0.0), amount)

        if dt <= 0.0 or not action_amounts:
            return None

        pose_delta = [0.0] * 6
        gripper_delta = 0.0
        scale = self.sensitivity * dt

        for action, amount in action_amounts.items():
            if action in GRIPPER_DIRECTIONS:
                gripper_delta += (
                    self.gripper_rate * GRIPPER_DIRECTIONS[action] * amount * scale
                )
                continue
            direction = ACTION_DIRECTIONS.get(action)
            if direction is None:
                continue
            for index, value in enumerate(direction):
                rate = self.linear_rate if index < 3 else self.angular_rate
                pose_delta[index] += rate * value * amount * scale

        return MotionCommand(
            source=self.source_name,
            action="+".join(sorted(action_amounts)),
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
        print(f"Xbox Cartesian control on {self.device.name} at {self.device.path}")
        print("  Buttons:")
        for action, button in self.button_bindings.items():
            control = XBOX_CONTROL_DESCRIPTIONS.get(button, button)
            description = ACTION_DESCRIPTIONS.get(action, action)
            print(f"    {control} ({button}): {action} - {description}")
        print("  Axes:")
        for action_name, spec in self.axis_binding_config.items():
            action = action_name
            control = XBOX_CONTROL_DESCRIPTIONS.get(spec, spec)
            description = ACTION_DESCRIPTIONS.get(action, action)
            print(f"    {control} ({spec}): {action} - {description}")


def _parse_axis_bindings(bindings: dict[str, str]) -> list[tuple[str, str, int]]:
    parsed = []
    for action_name, spec in bindings.items():
        action = action_name
        axis_name, separator, direction = spec.partition(":")
        if not separator or direction not in {"+", "-"}:
            raise ValueError(f"Invalid Xbox axis binding: {action_name}: {spec}")
        parsed.append((action, axis_name, 1 if direction == "+" else -1))
    return parsed
