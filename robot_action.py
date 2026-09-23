"""Run SDK teleoperation, hardware-free checks, or read-only calibration."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
import math
import time
import traceback

from controller.config import home_joints_from_config, settings_from_config
from controller.input_state import GamepadInput, KeyboardInput, RawTerminal
from controller.runtime import DiagnosticLog, SDKDriver, SimDriver, calibrate, load_workspace
from controller.session import ControlSession
from controller.teleop import Intent, Workspace


class ScriptedInput:
    def __init__(self, duration):
        self.duration = duration
        self.start = None

    def poll(self, now):
        if self.start is None:
            self.start = now
        elapsed = now - self.start
        intent = Intent(linear=(0.2, 0, 0)) if .3 < elapsed < self.duration * .6 else Intent()
        if elapsed >= self.duration:
            intent.events.add("quit")
        return intent

    def resync(self):
        pass

    def close(self):
        pass


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
    return settings_from_config(
        args.config,
        goal_time=args.goal_time,
        period=args.control_period,
        reset_time=args.reset_goal_time,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
    )


def home_joints_for(args, limits):
    """Return the configured home joint angles, checked against the SDK limits."""
    return home_joints_from_config(args.config, limits)


def run(args):
    settings = settings_for(args)
    workspace = (Workspace([.1, -.4, .05], [.6, .4, .6], 0, settings.clearance)
                 if args.dry_run else None)
    if not args.dry_run and not args.calibrate and not args.no_workspace_limits:
        # Reject missing/unmatched calibration before opening the robot connection.
        workspace = load_workspace(args.workspace, args.arm_ip, settings.clearance)
    driver = SimDriver(settings) if args.dry_run else SDKDriver(args.arm_ip, settings)
    inputs = logger = engine = session = None
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
        else:
            inputs = ScriptedInput(args.duration)
        home = home_joints_for(args, limits)
        session = ControlSession(
            driver=driver,
            inputs=inputs,
            settings=settings,
            workspace=workspace,
            home_joints=home,
            logger=logger,
        )
        sample = session.connect()
        engine = session.engine
        enabled = True
        if workspace is not None and not workspace.contains(sample.pose[:3]):
            # Starting slightly outside the calibrated box is recoverable: the engine
            # blocks outward motion only, so the operator can drive back inside.
            warning = ("TCP starts outside the calibrated workspace: "
                       f"actual={sample.pose[:3].tolist()}, "
                       f"lower={workspace.lower.tolist()}, upper={workspace.upper.tolist()}. "
                       "Only motion back toward the box is allowed until the TCP is inside.")
            print(warning)
            logger.emit(dict(event="workspace_warning", time=time.monotonic(), message=warning))
        if home is not None:
            print(f"Return target: configured home joints {home}")
        print(f"Diagnostic log: {log_path}")
        context = RawTerminal() if args.controller == "keyboard" and not args.dry_run else nullcontext()
        with context:
            start = time.monotonic()
            last_print = 0.0
            timeout_requested = False
            session.start()
            while session.is_running:
                time.sleep(.02)
                now = time.monotonic()
                if (
                    not timeout_requested
                    and not args.dry_run
                    and args.session_seconds is not None
                    and now - start >= args.session_seconds
                ):
                    session.request_event("quit")
                    timeout_requested = True
                transition = session.latest_transition()
                if transition is not None and now - last_print >= .2:
                    outside = " OUTSIDE WORKSPACE (only inward motion allowed)" if transition.record["outside_workspace"] else ""
                    print(f"{transition.state}: {transition.reason}; speed={engine.level + 1}; "
                          f"limits={list(transition.limit_flags)}{outside}")
                    last_print = now
            if session.error is not None:
                raise RuntimeError("Control session failed") from session.error
            sample = session.last_sample
        if engine.reason and "failed" in engine.reason:
            raise RuntimeError(engine.reason)
    except KeyboardInterrupt:
        if enabled:
            if session is not None and session.is_running:
                session.stop()
            driver.idle()
            if engine is not None:
                print(engine.reason)
    except Exception:
        if logger is not None:
            try:
                logger.emit(dict(event="fatal_error", time=time.monotonic(), error=traceback.format_exc()))
            except Exception:
                pass
        raise
    finally:
        if session is not None and session.is_running:
            try:
                session.stop()
            except Exception as exc:
                print(f"Control thread stop failed: {exc!r}")
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


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
