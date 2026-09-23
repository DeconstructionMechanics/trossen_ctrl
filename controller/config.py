"""Load controller settings and key bindings from the project YAML config."""

from pathlib import Path
from dataclasses import fields

import yaml

from .teleop import Settings


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
BINDING_SECTIONS = ("keyboard", "xbox_buttons", "xbox_axes")
SECTIONS = BINDING_SECTIONS + ("teleop", "home")


def load_keybind_config(path=None):
    config = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping of sections")
    unknown = sorted(set(config) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown configuration sections: {unknown}")
    for section in BINDING_SECTIONS:
        bindings = config.setdefault(section, {}) or {}
        if not isinstance(bindings, dict):
            raise ValueError(f"Section {section} must be a mapping of action to binding")
        for action, binding in bindings.items():
            # YAML reads bare yes, no, on and off as booleans, which would silently
            # rebind a key; ask for quotes rather than guessing what was meant.
            if isinstance(binding, bool):
                raise ValueError(f"{section}.{action}: quote the binding to keep it a key name")
            bindings[action] = str(binding)
        config[section] = bindings
    teleop = config.setdefault("teleop", {}) or {}
    if not isinstance(teleop, dict):
        raise ValueError("Section teleop must be a mapping of setting to value")
    config["teleop"] = teleop
    home = config.get("home") or {}
    if not isinstance(home, dict):
        raise ValueError("Section home must be a mapping with a joints list")
    joints = home.get("joints")
    if joints is not None:
        if not isinstance(joints, (list, tuple)) or not all(isinstance(x, (int, float)) for x in joints):
            raise ValueError("home.joints must be a list of joint angles in radians")
        home["joints"] = [float(x) for x in joints]
    config["home"] = home
    return config


def settings_from_config(path=None, **overrides):
    """Build validated controller settings from the shared YAML file."""
    settings = Settings()
    names = {item.name for item in fields(settings)}
    for key, value in load_keybind_config(path)["teleop"].items():
        if key not in names:
            raise ValueError(f"Unknown teleop setting: {key}")
        if key in {"linear_speeds", "angular_speeds"}:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} must be a list of three speeds")
            value = tuple(float(x) for x in value)
        elif key == "trajectory_samples":
            value = int(value)
        else:
            value = float(value)
        setattr(settings, key, value)
    for key, value in overrides.items():
        if key not in names:
            raise ValueError(f"Unknown teleop override: {key}")
        if value is not None:
            setattr(settings, key, value)
    settings.validate()
    return settings


def home_joints_from_config(path, limits):
    joints = load_keybind_config(path)["home"].get("joints")
    if joints is None:
        return None
    if limits and len(joints) != len(limits) - 1:
        raise ValueError(f"home.joints has {len(joints)} angles but the arm has {len(limits) - 1}")
    for index, (angle, limit) in enumerate(zip(joints, limits)):
        if not limit["position_min"] <= angle <= limit["position_max"]:
            raise ValueError(
                f"home.joints[{index}] = {angle} is outside the SDK limits "
                f"[{limit['position_min']}, {limit['position_max']}]"
            )
    return joints
