"""Run SDK teleoperation, hardware-free checks, or read-only calibration."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
import math
import time
import traceback

from controller.config import load_keybind_config
from controller.input_state import GamepadInput, KeyboardInput, RawTerminal
from controller.runtime import DiagnosticLog, SDKDriver, SimDriver, calibrate, load_workspace
from controller.teleop import Intent, Settings, Teleop, Workspace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("keyboard", "xbox"), default="keyboard")
    parser.add_argument("--device")
    parser.add_argument("--arm-ip", default="192.168.1.3")
    parser.add_argument("--config", default=None)
    parser.add_argument("--workspace", default="workspace.json")
    parser.add_argument("--no-workspace-limits", action="store_true",
                        help="Skip software XYZ workspace limits and calibration file")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Scripted fake input and driver; no hardware opened")
    modes.add_argument("--calibrate", action="store_true", help="Read-only TCP workspace calibration")
    parser.add_argument("--duration", type=float, default=5, help="Dry-run duration in seconds")
    parser.add_argument("--session-seconds", type=float, default=None,
                        help="Stop real control automatically after this many seconds")
    parser.add_argument("--log", default=None)
    for name in ("goal-time", "control-period", "reset-goal-time", "gripper-min", "gripper-max"):
        parser.add_argument("--" + name, type=float, default=None)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if args.session_seconds is not None and (not math.isfinite(args.session_seconds) or args.session_seconds <= 0):
        parser.error("--session-seconds must be finite and positive")
    return args


def settings_for(args):
    settings = Settings()
    names = {item.name for item in fields(settings)}
    for key, value in load_keybind_config(args.config)["teleop"].items():
        if key not in names:
            raise ValueError(f"Unknown teleop setting: {key}")
        if key in {"linear_speeds", "angular_speeds"}:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} must be a list of three speeds")
            setattr(settings, key, tuple(float(x) for x in value))
        else:
            setattr(settings, key, int(value) if key == "trajectory_samples" else float(value))
    for flag, name in (("goal_time", "goal_time"), ("control_period", "period"),
                       ("reset_goal_time", "reset_time"), ("gripper_min", "gripper_min"),
                       ("gripper_max", "gripper_max")):
        if getattr(args, flag) is not None:
            setattr(settings, name, getattr(args, flag))
    settings.validate()
    return settings


def home_joints_for(args, limits):
    """Return the configured home joint angles, checked against the SDK limits."""
    joints = load_keybind_config(args.config)["home"].get("joints")
    if joints is None:
        return None
    if limits and len(joints) != len(limits) - 1:
        raise ValueError(f"home.joints has {len(joints)} angles but the arm has {len(limits) - 1}")
    for index, (angle, limit) in enumerate(zip(joints, limits)):
        if not limit["position_min"] <= angle <= limit["position_max"]:
            raise ValueError(f"home.joints[{index}] = {angle} is outside the SDK limits "
                             f"[{limit['position_min']}, {limit['position_max']}]")
    return joints


def run(args):
    settings = settings_for(args)
    workspace = (Workspace([.1, -.4, .05], [.6, .4, .6], 0, settings.clearance)
                 if args.dry_run else None)
    if not args.dry_run and not args.calibrate and not args.no_workspace_limits:
        # Reject missing/unmatched calibration before opening the robot connection.
        workspace = load_workspace(args.workspace, args.arm_ip, settings.clearance)
    driver = SimDriver(settings) if args.dry_run else SDKDriver(args.arm_ip, settings)
    inputs = logger = engine = None
    sample = None
    enabled = False
    try:
        if args.calibrate:
            calibrate(driver, args.workspace, args.arm_ip, settings.clearance)
            return
        limits = driver.limits()
        sample = driver.sample()
        sample.validate()
        log_path = args.log or f"logs/teleop-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.jsonl"
        logger = DiagnosticLog(log_path)
        try:
            sdk_version = version("trossen-arm")
        except PackageNotFoundError:
            sdk_version = "unavailable"
        logger.emit(dict(event="startup", sdk_version=sdk_version, dry_run=args.dry_run,
                         settings=asdict(settings), joint_limits=limits, arguments=vars(args),
                         workspace=(dict(lower=workspace.lower.tolist(), upper=workspace.upper.tolist(),
                                         table_z=workspace.table_z, clearance=workspace.clearance,
                                         notice=workspace.notice) if workspace is not None else None)))
        if workspace is not None and workspace.notice:
            print(workspace.notice)
        if not args.dry_run:
            if workspace is None:
                print("Software XYZ workspace limits disabled for this session.")
            print("B pause / A resume; View long press returns to startup; Start or Ctrl-C stops and exits.")
            print("Keyboard: p pause, r resume, c return; existing motion keys retained.")
            input("Check the workspace is clear. Press Enter to enable position control: ")
            inputs = GamepadInput(settings, args.device, args.config) if args.controller == "xbox" else KeyboardInput(args.config)
        enabled = True
        driver.resume()
        if inputs is not None:
            # Position mode blocks for as long as it takes to settle; that backlog is
            # ours, not a controller failure.
            inputs.resync()
        sample = driver.sample()
        sample.validate()
        if workspace is not None and not workspace.contains(sample.pose[:3]):
            # Starting slightly outside the calibrated box is recoverable: the engine
            # blocks outward motion only, so the operator can drive back inside.
            warning = ("TCP starts outside the calibrated workspace: "
                       f"actual={sample.pose[:3].tolist()}, "
                       f"lower={workspace.lower.tolist()}, upper={workspace.upper.tolist()}. "
                       "Only motion back toward the box is allowed until the TCP is inside.")
            print(warning)
            logger.emit(dict(event="workspace_warning", time=time.monotonic(), message=warning))
        home = home_joints_for(args, limits)
        if home is not None:
            print(f"Return target: configured home joints {home}")
        engine = Teleop(driver, settings, workspace, sample, home)
        if workspace is None:
            engine.level = 0
        print(f"Diagnostic log: {log_path}")
        context = RawTerminal() if args.controller == "keyboard" and not args.dry_run else nullcontext()
        with context:
            start = last = time.monotonic()
            deadline, last_print = start + settings.period, 0.0
            last_stamp, stamp_changed = sample.timestamp, start
            while engine.state != "exiting":
                time.sleep(max(0.0, deadline - time.monotonic()))
                now = time.monotonic()
                dt, last = now - last, now
                deadline = now + settings.period
                intent = (Intent(linear=(0.2, 0, 0)) if .3 < now - start < args.duration * .6 else Intent()) if args.dry_run else inputs.poll(now)
                if (args.dry_run and now - start >= args.duration) or (
                    not args.dry_run and args.session_seconds is not None
                    and now - start >= args.session_seconds
                ):
                    intent.events.add("quit")
                try:
                    sample = driver.sample()
                    sample.validate()
                    if sample.timestamp != last_stamp:
                        last_stamp, stamp_changed = sample.timestamp, now
                    if now - stamp_changed > settings.max_dt:
                        raise RuntimeError("SDK feedback timestamp stale")
                except Exception:
                    logger.emit(dict(event="feedback_error", time=now, error=traceback.format_exc()))
                    driver.idle()
                    raise
                if time.monotonic() - now > settings.max_dt:
                    dt = settings.max_dt + settings.period
                previous_state = engine.state
                record = engine.tick(intent, sample, dt, now)
                try:
                    logger.emit(record)
                except Exception:
                    engine.stop(sample, "logging failure", "fault")
                    raise
                if now - last_print >= .2 or "error" in record:
                    outside = " OUTSIDE WORKSPACE (only inward motion allowed)" if record["outside_workspace"] else ""
                    print(f"{engine.state}: {engine.reason}; speed={engine.level + 1}; "
                          f"limits={record['limits']}{outside}")
                    last_print = now
                if "quit" in intent.events:
                    break
                # Mode recovery may block; never integrate its duration into a command,
                # and absorb the input backlog it caused instead of reading it as loss.
                if engine.state == "waiting_neutral":
                    if previous_state != "waiting_neutral" and inputs is not None:
                        inputs.resync()
                    last = time.monotonic()
                    deadline = last + settings.period
                    stamp_changed = last
        if engine.reason and "failed" in engine.reason:
            raise RuntimeError(engine.reason)
    except KeyboardInterrupt:
        if enabled:
            if engine is not None and sample is not None:
                try:
                    sample = driver.sample()
                    sample.validate()
                    engine.stop(sample, "KeyboardInterrupt", "exiting")
                except Exception:
                    driver.idle()
                print(engine.reason)
            else:
                driver.idle()
    except Exception:
        if logger is not None:
            try:
                logger.emit(dict(event="fatal_error", time=time.monotonic(), error=traceback.format_exc()))
            except Exception:
                pass
        raise
    finally:
        if enabled:
            try:
                parked = driver.sample()
            except Exception:
                parked = None
            if parked is not None and workspace is not None:
                above = float(parked.pose[2] - workspace.table_z)
                if above < settings.sag_allowance:
                    print(f"Parked {above * 1000:.0f} mm above the calibrated table: the arm settles "
                          "downward when the brakes engage. Lift the TCP before stopping.")
            try:
                # Idle brakes motion before cleanup; it does not command a return or close.
                driver.idle()
            except Exception as exc:
                print(f"Stop NOT confirmed: idle failed: {exc!r}")
            else:
                # Measure the braking sag instead of leaving it to folklore.
                if parked is not None:
                    try:
                        time.sleep(settings.sag_settle)
                        settled = driver.sample()
                        sag = float(parked.pose[2] - settled.pose[2])
                        print(f"Braking sag: {sag * 1000:.1f} mm "
                              f"(z {parked.pose[2]:.4f} -> {settled.pose[2]:.4f})")
                        if logger is not None:
                            logger.emit(dict(event="idle_sag", time=time.monotonic(), sag=sag,
                                             before=float(parked.pose[2]), after=float(settled.pose[2])))
                    except Exception as exc:
                        print(f"Braking sag not measured: {exc!r}")
        if logger is not None:
            try:
                logger.emit(dict(event="shutdown", time=time.monotonic(),
                                 state=engine.state if engine else "initialization",
                                 reason=engine.reason if engine else "initialization ended"))
            except Exception as exc:
                print(f"Shutdown log unavailable: {exc!r}")
        try:
            if inputs is not None:
                inputs.close()
        finally:
            try:
                driver.close()
            finally:
                if logger is not None:
                    logger.close()


if __name__ == "__main__":
    run(parse_args())
