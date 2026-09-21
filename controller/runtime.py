"""SDK adapter, read-only calibration, simulator and asynchronous diagnostics."""

import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np

from .teleop import Sample, Workspace


class SDKDriver:
    def __init__(self, ip, settings):
        import trossen_arm as sdk
        self.sdk, self.s = sdk, settings
        self.raw = sdk.TrossenArmDriver()
        try:
            self.raw.configure(sdk.Model.wxai_v0, sdk.StandardEndEffector.wxai_v0_follower, ip, False)
        except Exception as exc:
            # The SDK reports this as "Resource temporarily unavailable", which says nothing
            # about the cause. A second read-only client connects fine; the observed failure
            # was a second connection while another process was actively controlling the arm.
            raise RuntimeError(
                f"Could not connect to the arm at {ip}: {exc} "
                "Check that no other process is currently controlling this arm, then check "
                "power, cabling and the IP address."
            ) from exc

    def limits(self):
        limits = self.raw.get_joint_limits()
        self.velocity_limits = np.array([x.velocity_max for x in limits[:-1]])
        if not np.all(np.isfinite(self.velocity_limits)) or np.any(self.velocity_limits <= 0):
            raise ValueError("Invalid SDK joint velocity limits")
        self.s.gripper_min = max(self.s.gripper_min, limits[-1].position_min)
        self.s.gripper_max = min(self.s.gripper_max, limits[-1].position_max)
        self.s.validate()
        return [dict(position_min=x.position_min, position_max=x.position_max, velocity_max=x.velocity_max)
                for x in limits]

    def sample(self):
        output = self.raw.get_robot_output()
        return Sample(np.array(output.joint.arm.positions, dtype=float),
                      np.array(output.cartesian.positions, dtype=float),
                      float(output.joint.gripper.position), int(output.header.timestamp))

    def resume(self, fault=False):
        if fault:
            self.raw.clear_error()
        elif all(mode == self.sdk.Mode.position for mode in self.raw.get_modes()):
            # Position mode was never left, so there is nothing to switch or settle;
            # skipping it keeps a plain resume instant instead of blocking for a second.
            return
        self.raw.set_arm_modes(self.sdk.Mode.position)
        self.raw.set_gripper_mode(self.sdk.Mode.position)
        time.sleep(0.5)
        self.hold(self.sample())
        previous = self.sample()
        steady = 0
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            time.sleep(0.1)
            current = self.sample()
            current.validate()
            if np.linalg.norm(current.pose[:3] - previous.pose[:3]) < 0.0004:
                steady += 1
                if steady >= 3:
                    return
            else:
                steady = 0
            previous = current
        raise RuntimeError("Arm did not settle after switching to position mode")

    def hold(self, sample, arm=True, gripper=True, gripper_position=None):
        if arm:
            self.raw.set_arm_positions(sample.joints.tolist(), goal_time=self.s.period, blocking=False)
        if gripper:
            position = sample.gripper if gripper_position is None else float(gripper_position)
            self.raw.set_gripper_position(position, goal_time=self.s.period, blocking=False)

    def idle(self):
        self.raw.set_arm_modes(self.sdk.Mode.idle)
        self.raw.set_gripper_mode(self.sdk.Mode.idle)

    def move(self, pose, duration, samples, feedforward=None):
        extra = ({} if feedforward is None
                 else dict(goal_feedforward_velocities=np.asarray(feedforward, dtype=float).tolist()))
        self.raw.set_cartesian_positions(pose.tolist(), self.sdk.InterpolationSpace.cartesian,
                                        goal_time=duration, blocking=False,
                                        num_trajectory_check_samples=samples, **extra)

    def move_gripper(self, value, duration):
        self.raw.set_gripper_position(value, goal_time=duration, blocking=False)

    def return_joints(self, joints, duration):
        self.raw.set_arm_positions(joints.tolist(), goal_time=duration, blocking=False)

    def close(self):
        self.raw.cleanup()


class SimDriver:
    """Deterministic command sink; not an IK or physics simulator."""
    def __init__(self, settings):
        self.s = settings
        self.pose = np.array([0.3, 0, 0.25, 0, 0, 0], dtype=float)
        self.joints = np.zeros(6)
        self.gripper = 0.02
        self.velocity_limits = np.ones(6)
        self.feedforward = None
        self.gripper_hold = None
        self.calls = []

    def limits(self):
        return []

    def sample(self):
        return Sample(self.joints.copy(), self.pose.copy(), self.gripper, time.monotonic_ns() // 1000)

    def resume(self, fault=False):
        self.calls.append(("resume", fault))

    def hold(self, sample, arm=True, gripper=True, gripper_position=None):
        self.calls.append(("hold", arm, gripper))
        if gripper and gripper_position is not None:
            self.gripper_hold = float(gripper_position)

    def idle(self):
        self.calls.append(("idle",))

    def move(self, pose, duration, samples, feedforward=None):
        self.calls.append(("move", pose.copy()))
        self.feedforward = None if feedforward is None else np.asarray(feedforward, dtype=float).copy()
        self.pose = pose.copy()

    def move_gripper(self, value, duration):
        self.calls.append(("gripper", value))
        self.gripper = value

    def return_joints(self, joints, duration):
        self.calls.append(("return", duration))
        self.joints = joints.copy()

    def close(self):
        pass


class DiagnosticLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")
        self.queue = queue.Queue(maxsize=1000)
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def emit(self, record):
        if self.error:
            raise RuntimeError(f"Diagnostic log failed: {self.error}")
        try:
            self.queue.put_nowait(record)
        except queue.Full as exc:
            self.error = "diagnostic queue full"
            raise RuntimeError(self.error) from exc

    def _write(self):
        try:
            while not self.done.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.stream.write(json.dumps(record, allow_nan=False) + "\n")
                self.stream.flush()
        except Exception as exc:
            self.error = repr(exc)
        finally:
            self.stream.close()

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Diagnostic writer did not finish within 2 seconds")
        if self.error:
            raise RuntimeError(f"Diagnostic log failed: {self.error}")


def load_workspace(path, ip, clearance):
    data = json.loads(Path(path).read_text())
    if data.get("arm_ip") != ip or data.get("model") != "wxai_v0_follower":
        raise ValueError("Calibration does not match selected arm/model")
    workspace = Workspace(data["lower"], data["upper"], data["table_z"], clearance)
    recorded = data.get("clearance")
    if recorded is not None and not math.isclose(float(recorded), clearance, rel_tol=0, abs_tol=1e-9):
        # The runtime setting decides; saying so beats silently ignoring the stored value.
        workspace.notice = (
            f"Calibration recorded clearance {float(recorded)} m but the configured clearance "
            f"{clearance} m is in effect; floor z = {workspace.lower[2]:.4f} m."
        )
    return workspace


def calibrate(driver, path, ip, clearance):
    print("Read-only calibration. This program will NOT enable motors or move the arm.")
    print("Use your existing positioning method. Values refer to the SDK TCP in base coordinates.")
    print("Position the TCP at table height only when physically appropriate; otherwise enter measured table Z.")
    table = input("Measured table Z in meters, or Enter to sample current TCP Z: ").strip()
    table_z = float(table) if table else float(driver.sample().pose[2])
    lower, upper = [], []
    for axis, name in enumerate("XYZ"):
        for label, values in (("minimum", lower), ("maximum", upper)):
            input(f"Position TCP at intended {name} {label}; press Enter to read: ")
            sample = driver.sample()
            sample.validate()
            values.append(float(sample.pose[axis]))
            print(f"Read {name} {label}: {values[-1]:.6f} m")
    workspace = Workspace(lower, upper, table_z, clearance)
    data = dict(arm_ip=ip, model="wxai_v0_follower", lower=lower, upper=upper,
                table_z=table_z, recorded_at=time.time(), clearance=clearance)
    print(json.dumps(data, indent=2))
    print(f"Effective lower boundary: {workspace.lower.tolist()}")
    if input("Type SAVE to write this calibration: ").strip() != "SAVE":
        print("Calibration not saved.")
        return
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2)
    print(f"Calibration written to {path}")
