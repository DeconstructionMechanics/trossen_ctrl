"""Load controller settings from the project YAML config.

The config file intentionally uses a very small YAML subset so this project
does not need PyYAML just to read controller config: top-level sections plus
string key/value pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
LEGACY_CONFIG_PATH = Path(__file__).with_name("keybinds.yaml")

ACTION_DESCRIPTIONS = {
    "forward": "move +x",
    "backward": "move -x",
    "leftward": "move +y",
    "rightward": "move -y",
    "upward": "move +z",
    "downward": "move -z",
    "pitchup": "rotate +pitch",
    "pitchdown": "rotate -pitch",
    "yawleft": "rotate +yaw",
    "yawright": "rotate -yaw",
    "rollleft": "rotate +roll",
    "rollright": "rotate -roll",
    "gripperopen": "open gripper",
    "gripperclose": "close gripper",
    "reset": "smoothly return to startup target",
    "sensitivityup": "increase sensitivity",
    "sensitivitydown": "decrease sensitivity",
    "quit": "smoothly return to startup target, then quit",
}

XBOX_CONTROL_DESCRIPTIONS = {
    "BTN_SOUTH": "A",
    "BTN_EAST": "B",
    "BTN_NORTH": "Y",
    "BTN_WEST": "X",
    "BTN_TL": "LB",
    "BTN_TR": "RB",
    "BTN_SELECT": "View/Back",
    "BTN_START": "Menu/Start",
    "ABS_X:-": "left stick left",
    "ABS_X:+": "left stick right",
    "ABS_Y:-": "left stick up",
    "ABS_Y:+": "left stick down",
    "ABS_RX:-": "right stick left",
    "ABS_RX:+": "right stick right",
    "ABS_RY:-": "right stick up",
    "ABS_RY:+": "right stick down",
    "ABS_Z:+": "left trigger",
    "ABS_RZ:+": "right trigger",
    "ABS_HAT0X:-": "D-pad left",
    "ABS_HAT0X:+": "D-pad right",
    "ABS_HAT0Y:-": "D-pad up",
    "ABS_HAT0Y:+": "D-pad down",
}


def load_keybind_config(path: Optional[str | Path] = None) -> dict[str, dict[str, str]]:
    config_path = Path(path) if path else _default_config_path()
    config: dict[str, dict[str, str]] = {}
    current_section: Optional[str] = None

    for line_number, raw_line in enumerate(config_path.read_text().splitlines(), 1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue

        if not line.startswith((" ", "\t")):
            if not line.endswith(":"):
                raise ValueError(
                    f"{config_path}:{line_number}: expected a section ending with ':'"
                )
            current_section = line[:-1].strip()
            config[current_section] = {}
            continue

        if current_section is None:
            raise ValueError(f"{config_path}:{line_number}: key outside a section")

        key, separator, value = line.strip().partition(":")
        if not separator:
            raise ValueError(f"{config_path}:{line_number}: expected 'key: value'")
        config[current_section][key.strip()] = _unquote(value.strip())

    return config


def sensitivity_config(config: dict[str, dict[str, str]]) -> dict[str, float]:
    section = config.get("sensitivity", {})
    defaults = {
        "default_sensitivity": 1.0,
        "min_sensitivity": 0.1,
        "max_sensitivity": 10.0,
    }
    values = {
        key: _float_setting(section, key, default)
        for key, default in defaults.items()
    }
    validate_sensitivity_bounds(
        values["min_sensitivity"],
        values["max_sensitivity"],
    )
    return values


def validate_sensitivity_bounds(min_sensitivity: float, max_sensitivity: float):
    if min_sensitivity <= 0.0:
        raise ValueError("min_sensitivity must be greater than 0")
    if max_sensitivity < min_sensitivity:
        raise ValueError(
            "max_sensitivity must be greater than or equal to min_sensitivity"
        )


def _default_config_path() -> Path:
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return LEGACY_CONFIG_PATH


def _float_setting(section: dict[str, str], key: str, default: float) -> float:
    raw_value = section.get(key)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"sensitivity.{key} must be a number, got {raw_value!r}"
        ) from exc


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
