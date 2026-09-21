"""Locate the gamepad event device; the bindings live in controller/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError:
    InputDevice = None
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
    denied = []
    for path in paths:
        real_path = str(Path(path).resolve())
        if real_path in seen:
            continue
        seen.add(real_path)
        try:
            devices.append(InputDevice(path))
        except PermissionError:
            denied.append(path)
        except OSError:
            continue

    if not devices:
        if denied:
            raise SystemExit(f"Permission denied reading input devices: {denied}")
        raise SystemExit("No /dev/input event devices found. Is the controller attached?")

    for device in devices:
        name = device.name.lower()
        if "xbox" in name or "controller" in name or "x-input" in name or "xinput" in name:
            for other in devices:
                if other is not device:
                    other.close()
            return device

    print("Input devices found:")
    for device in devices:
        print(f"  {device.path}: {device.name}")
        device.close()
    raise SystemExit("Could not find an Xbox controller input device.")
