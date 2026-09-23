"""Input snapshots without time integration or early returns inside event batches."""

import math
import select
import sys
import termios
import time
import tty

from .commands import ACTION_DIRECTIONS, GRIPPER_DIRECTIONS
from .config import load_keybind_config
from .teleop import Intent
from .xbox import ecodes, find_xbox_controller


def response(value):
    return 0.5 * value + 0.5 * value ** 3


def radial_pair(x, y, deadzone):
    length = math.hypot(x, y)
    if length <= deadzone:
        return 0.0, 0.0
    scale = response(min(1.0, (length - deadzone) / (1 - deadzone))) / length
    return x * scale, y * scale


class GamepadInput:
    def __init__(self, settings, path=None, config_path=None, device=None, clock=time.monotonic):
        self.s, self.path, self.clock = settings, path, clock
        config = load_keybind_config(config_path)
        self.bindings = {**config["xbox_axes"], **config["xbox_buttons"]}
        self.device = device
        self.axes, self.keys, self.info = {}, set(), {}
        self.dropped = False
        self.retry_at = 0
        self.reset_since = None
        self.reset_fired = False
        self.previous = set()
        self.connected = False
        self.ever_connected = False
        if device is not None:
            self._attach(device)

    def _attach(self, device):
        self.device = device
        caps = device.capabilities(absinfo=True)
        self.info = dict(caps.get(ecodes.EV_ABS, []))
        required = {binding.split(":")[0] for binding in self.bindings.values() if ":" in binding}
        if any(getattr(ecodes, name) not in self.info for name in required):
            raise ValueError("Device lacks configured controller axes")
        self._snapshot()
        self.connected = True
        self.previous = self._actions()
        try:
            device.grab()
        except OSError:
            pass

    def _snapshot(self):
        self.axes = {code: self.device.absinfo(code).value for code in self.info}
        self.keys = set(self.device.active_keys())
        self.reset_since = None
        # A control already held when this snapshot is taken has no press this session,
        # so treat its hold as spent: otherwise attaching to a pad with the return
        # button down commands a return one hold-time later, with no operator action.
        self.reset_fired = "reset" in self._actions()

    def resync(self):
        """Absorb the backlog left by a blocking caller.

        A blocking SDK call stops this process reading the pad for long enough that
        the kernel drops events, which would otherwise be reported as input loss and
        send control straight back to pause. The current axis and key state is read
        back from the device, so nothing is assumed about what happened meanwhile.
        """
        if not self.connected:
            return
        try:
            for _ in self.device.read():
                pass
        except (BlockingIOError, OSError):
            pass
        self._snapshot()
        self.previous = self._actions()
        self.dropped = False

    def close(self):
        if self.device is not None:
            try:
                self.device.ungrab()
            except OSError:
                pass
            self.device.close()
            self.device = None
        self.connected = False

    def _axis(self, name):
        code = getattr(ecodes, name)
        info = self.info[code]
        value = self.axes.get(code, info.value)
        if name.startswith("ABS_HAT"):
            return float(value)
        span = info.max - info.min
        if span <= 0:
            raise ValueError(f"Invalid axis range: {name}")
        if name in {"ABS_Z", "ABS_RZ"}:
            amount = max(0.0, min(1.0, (value - info.min) / span))
            return response(max(0.0, (amount - self.s.trigger_deadzone) / (1 - self.s.trigger_deadzone)))
        return max(-1.0, min(1.0, (value - (info.min + info.max) / 2) / (span / 2)))

    def _amounts(self):
        axes = {binding.split(":")[0]: self._axis(binding.split(":")[0])
                for binding in self.bindings.values() if ":" in binding}
        for x, y in (("ABS_X", "ABS_Y"), ("ABS_RX", "ABS_RY")):
            if x in axes and y in axes:
                axes[x], axes[y] = radial_pair(axes[x], axes[y], self.s.stick_deadzone)
        amounts = {}
        for action, binding in self.bindings.items():
            if ":" in binding:
                axis, sign = binding.split(":")
                amount = max(0.0, axes[axis] * (1 if sign == "+" else -1))
            else:
                amount = float(getattr(ecodes, binding) in self.keys)
            if amount:
                amounts[action] = amount
        return amounts

    def _actions(self):
        return set(self._amounts())

    def poll(self, now):
        events = set()
        if not self.connected:
            if now < self.retry_at:
                return Intent(connected=False)
            self.retry_at = now + 1
            try:
                reconnect = self.ever_connected
                self._attach(find_xbox_controller(self.path))
                self.ever_connected = True
                if reconnect:
                    events.add("input_lost")
            except (OSError, SystemExit, ValueError):
                self.close()
                return Intent(connected=False)
        try:
            # Preserve rising edges even when press and release share one OS batch.
            for event in self.device.read():
                if event.type == ecodes.EV_SYN and event.code == ecodes.SYN_DROPPED:
                    self.dropped = True
                    events.add("input_lost")
                    continue
                if self.dropped:
                    if event.type == ecodes.EV_SYN and event.code == ecodes.SYN_REPORT:
                        self._snapshot()
                        self.previous = self._actions()
                        self.dropped = False
                    continue
                if event.type == ecodes.EV_KEY:
                    if event.value:
                        self.keys.add(event.code)
                    else:
                        self.keys.discard(event.code)
                elif event.type == ecodes.EV_ABS:
                    self.axes[event.code] = event.value
                actions = self._actions()
                for action in actions - self.previous:
                    mapped = {"sensitivityup": "speed_up", "sensitivitydown": "speed_down"}.get(action, action)
                    if mapped in {
                        "pause",
                        "resume",
                        "quit",
                        "speed_up",
                        "speed_down",
                        "record_toggle",
                        "accept",
                    }:
                        events.add(mapped)
                if "reset" not in actions:
                    self.reset_since, self.reset_fired = None, False
                self.previous = actions
        except BlockingIOError:
            pass
        except OSError:
            self.close()
            return Intent(events={"input_lost"}, connected=False)
        if self.dropped:
            return Intent(events={"input_lost"}, connected=False)
        amounts = self._amounts()
        if "reset" in amounts:
            if self.reset_since is None:
                self.reset_since = now
            if not self.reset_fired and now - self.reset_since >= self.s.reset_hold:
                events.add("reset")
                self.reset_fired = True
        linear, angular = [0.0] * 3, [0.0] * 3
        gripper = 0.0
        for action, amount in amounts.items():
            if action in ACTION_DIRECTIONS:
                direction = ACTION_DIRECTIONS[action]
                for index in range(3):
                    linear[index] += direction[index] * amount
                    angular[index] += direction[index + 3] * amount
            elif action in GRIPPER_DIRECTIONS:
                gripper += GRIPPER_DIRECTIONS[action] * amount
        return Intent(tuple(linear), tuple(angular), gripper, events)


class RawTerminal:
    """Put the terminal in cbreak mode so single keys arrive without Enter."""

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


class KeyboardInput:
    def __init__(self, config_path=None):
        self.bindings = {value: key for key, value in load_keybind_config(config_path)["keyboard"].items()}
        self.active = {}

    def resync(self):
        """Drop keys typed while a blocking caller held up the control loop."""
        while select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.read(1) == "":
                break
        self.active.clear()

    def close(self):
        pass

    def poll(self, now):
        events = set()
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == "":
                return Intent(events={"quit"})
            if key in {"\x03", "\x1b"}:
                events.add("quit")
            action = self.bindings.get(key.lower())
            event = {"sensitivityup": "speed_up", "sensitivitydown": "speed_down"}.get(action, action)
            if event in {
                "reset",
                "pause",
                "resume",
                "speed_up",
                "speed_down",
                "record_toggle",
                "accept",
            }:
                events.add(event)
            elif action in ACTION_DIRECTIONS or action in GRIPPER_DIRECTIONS:
                self.active[action] = now
        self.active = {key: value for key, value in self.active.items() if now - value < 0.06}
        vector, gripper = [0.0] * 6, 0.0
        for action in self.active:
            for index, value in enumerate(ACTION_DIRECTIONS.get(action, (0,) * 6)):
                vector[index] += value
            gripper += GRIPPER_DIRECTIONS.get(action, 0)
        return Intent(tuple(vector[:3]), tuple(vector[3:]), gripper, events)
