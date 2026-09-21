"""Cartesian teleoperation, independent of input devices and SDK imports."""

from dataclasses import asdict, dataclass, field
import math
import traceback

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Settings:
    period: float = 0.02
    goal_time: float = 0.06
    max_dt: float = 0.1
    linear_speeds: tuple = (0.02, 0.06, 0.12)
    angular_speeds: tuple = tuple(math.radians(x) for x in (10, 30, 60))
    gripper_speed: float = 0.02
    linear_acceleration: float = 0.5
    angular_acceleration: float = 3.0
    position_lead: float = 0.005
    rotation_lead: float = math.radians(3)
    lead_time: float = 0.08
    feedforward: float = 1.0
    stick_deadzone: float = 0.1
    trigger_deadzone: float = 0.05
    reset_hold: float = 1.0
    reset_time: float = 3.0
    trajectory_samples: int = 3
    gripper_min: float = 0.0
    gripper_max: float = 0.04
    gripper_squeeze: float = 0.004
    clearance: float = 0.01
    sag_allowance: float = 0.03
    sag_settle: float = 0.5

    def validate(self):
        for name, value in asdict(self).items():
            values = value if isinstance(value, tuple) else (value,)
            if not all(math.isfinite(x) and x >= 0 for x in values):
                raise ValueError(f"Invalid setting {name}: {value}")
        if not 0 < self.period <= self.max_dt or self.goal_time <= 0:
            raise ValueError("Require 0 < period <= max_dt and goal_time > 0")
        if not 0 <= self.stick_deadzone < 1 or not 0 <= self.trigger_deadzone < 1:
            raise ValueError("Deadzones must be in [0, 1)")
        if self.gripper_min >= self.gripper_max:
            raise ValueError("Empty gripper range")
        if len(self.linear_speeds) != 3 or len(self.angular_speeds) != 3:
            raise ValueError("Exactly three speed levels are required")
        if self.trajectory_samples < 1:
            raise ValueError("Trajectory sampling must be enabled")
        if not 0 <= self.feedforward <= 1:
            raise ValueError("Feedforward scale must be in [0, 1]")


@dataclass
class Intent:
    linear: tuple = (0.0, 0.0, 0.0)
    angular: tuple = (0.0, 0.0, 0.0)
    gripper: float = 0.0
    events: set = field(default_factory=set)
    connected: bool = True

    @property
    def neutral(self):
        return not any(abs(x) > 1e-8 for x in (*self.linear, *self.angular, self.gripper))


@dataclass
class Sample:
    joints: np.ndarray
    pose: np.ndarray
    gripper: float
    timestamp: int = 0

    def validate(self):
        if self.pose.shape != (6,) or not np.all(np.isfinite(self.pose)):
            raise ValueError("Invalid Cartesian feedback")
        if not np.all(np.isfinite(self.joints)) or not math.isfinite(self.gripper):
            raise ValueError("Invalid joint feedback")

    def record(self):
        return dict(joints=self.joints.tolist(), pose=self.pose.tolist(),
                    gripper=self.gripper, timestamp=self.timestamp)


@dataclass
class Workspace:
    lower: np.ndarray
    upper: np.ndarray
    table_z: float
    clearance: float = 0.01
    notice: object = None

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float).copy()
        self.upper = np.asarray(self.upper, dtype=float).copy()
        if self.lower.shape != (3,) or self.upper.shape != (3,):
            raise ValueError("Workspace requires three lower and upper values")
        if not np.all(np.isfinite([*self.lower, *self.upper, self.table_z, self.clearance])):
            raise ValueError("Workspace values must be finite")
        if self.clearance < 0:
            raise ValueError("Clearance must be nonnegative")
        self.lower[2] = max(self.lower[2], self.table_z + self.clearance)
        if np.any(self.lower >= self.upper):
            raise ValueError("Workspace lower bounds must be below upper bounds")

    def contains(self, xyz):
        return bool(np.all(xyz >= self.lower - 1e-6) and np.all(xyz <= self.upper + 1e-6))

    def allow(self, xyz, velocity):
        """Drop the velocity components that would push an outside TCP further out."""
        velocity = np.asarray(velocity, dtype=float).copy()
        velocity[((xyz > self.upper + 1e-6) & (velocity > 0))
                 | ((xyz < self.lower - 1e-6) & (velocity < 0))] = 0.0
        return velocity

    def bounds_including(self, xyz):
        """Bounds widened to the measured TCP so recovery motion is not clipped away."""
        return np.minimum(self.lower, xyz), np.maximum(self.upper, xyz)


def limit_vector(vector, maximum):
    vector = np.asarray(vector, dtype=float)
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-15))


class Teleop:
    def __init__(self, driver, settings, workspace, initial, home_joints=None):
        settings.validate()
        initial.validate()
        self.driver, self.s, self.workspace = driver, settings, workspace
        if home_joints is None:
            self.start_joints = initial.joints.copy()
        else:
            self.start_joints = np.asarray(home_joints, dtype=float).copy()
            if self.start_joints.shape != initial.joints.shape or not np.all(np.isfinite(self.start_joints)):
                raise ValueError("Configured home joints do not match the arm's joint count")
        self.state = "waiting_neutral"
        self.level = 1
        self.outside = workspace is not None and not workspace.contains(initial.pose[:3])
        self.reason = "startup outside workspace; only inward motion allowed" if self.outside else "startup"
        self.reset_deadline = 0.0
        self.arm_active = self.gripper_active = False
        self.sync(initial)

    def sync(self, sample):
        self.target = sample.pose.copy()
        self.gripper = sample.gripper
        self.velocity = np.zeros(3)
        self.omega = np.zeros(3)

    def stop(self, sample, reason, state="paused"):
        self.state, self.reason = state, reason
        try:
            # Hold the commanded opening, not the measured one, so pausing while
            # holding an object does not release the squeeze that carries it.
            self.driver.hold(sample, gripper_position=self.gripper)
        except Exception as exc:
            self.state = "fault"
            self.reason += f"; hold failed: {exc!r}"
            try:
                self.driver.idle()
            except Exception as idle_exc:
                self.reason += f"; idle failed, stop NOT confirmed: {idle_exc!r}"
        self.sync(sample)
        self.arm_active = self.gripper_active = False

    def tick(self, intent, sample, dt, now):
        sample.validate()
        self.outside = self.workspace is not None and not self.workspace.contains(sample.pose[:3])
        result = dict(time=now, dt=dt, input={**asdict(intent), "events": sorted(intent.events)},
                      actual=sample.record(), limits=[])
        try:
            self._tick(intent, sample, dt, now, result)
        except Exception as exc:
            result["error"] = traceback.format_exc()
            result["error_category"] = "SDK IK error" if "kinematic" in str(exc).lower() else "control error"
            self.stop(sample, repr(exc), "fault")
        result.update(state=self.state, reason=self.reason, speed_level=self.level,
                      outside_workspace=self.outside,
                      target=self.target.tolist(), gripper_target=self.gripper)
        return result

    def _tick(self, intent, sample, dt, now, result):
        events = intent.events
        if "quit" in events:
            self.stop(sample, "quit", "exiting")
            return
        if not intent.connected or "input_lost" in events or "pause" in events:
            if self.state not in {"paused", "fault"}:
                self.stop(sample, "input unavailable" if not intent.connected else "pause/input resync")
            return
        if "speed_up" in events:
            self.level = min(2, self.level + 1)
        if "speed_down" in events:
            self.level = max(0, self.level - 1)
        # Recovery is offered before any check that could re-enter the same stop.
        if self.state in {"paused", "fault"}:
            if "resume" in events and intent.neutral:
                self.driver.resume(self.state == "fault")
                self.sync(self.driver.sample())
                self.state, self.reason = "waiting_neutral", "resume"
            return
        if dt > self.s.max_dt or dt <= 0 or not math.isfinite(dt):
            self.stop(sample, "control deadline missed")
            return
        if self.state == "waiting_neutral":
            self.sync(sample)
            if intent.neutral:
                self.state, self.reason = "running", "ready"
            return
        if self.state == "returning":
            if not intent.neutral:
                # Cancelling is the operator taking control back, so releasing the
                # controls is enough to continue; the cancelling input is not executed.
                self.stop(sample, "return cancelled by input", "waiting_neutral")
            elif now > self.reset_deadline:
                self.stop(sample, "return timeout", "fault")
            elif np.max(np.abs(sample.joints - self.start_joints)) < 0.01:
                self.stop(sample, "return complete")
            return
        if "reset" in events:
            if not intent.neutral:
                # Same situation as cancelling a return: the operator had controls
                # pushed, so releasing them is enough to carry on.
                self.stop(sample, "release motion controls before return", "waiting_neutral")
                return
            self.stop(sample, "return requested")
            if self.state == "fault":
                return
            duration = max(self.s.reset_time, float(np.max(
                np.abs(self.start_joints - sample.joints) / (0.25 * self.driver.velocity_limits))))
            self.driver.return_joints(self.start_joints, duration)
            self.state = "returning"
            self.reset_deadline = now + duration + 2.0
            return

        linear = limit_vector(intent.linear, 1) * self.s.linear_speeds[self.level]
        angular = limit_vector(intent.angular, 1) * self.s.angular_speeds[self.level]
        if self.workspace is not None:
            allowed = self.workspace.allow(sample.pose[:3], linear)
            if not np.array_equal(allowed, linear):
                result["limits"].append("workspace_recovery")
            # Masking the command as well as the state keeps the ramp from winding up.
            linear = allowed
            self.velocity = self.workspace.allow(sample.pose[:3], self.velocity)
        moving = bool(np.any(linear) or np.any(angular))
        gripping = abs(intent.gripper) > 1e-8
        if self.arm_active and not moving:
            self.driver.hold(sample, arm=True, gripper=False)
            self.target = sample.pose.copy()
            self.velocity[:] = self.omega[:] = 0
        if self.gripper_active and not gripping:
            # Releasing the trigger keeps the commanded opening; resynchronising it to
            # feedback would give away the squeeze that holds an object.
            self.driver.move_gripper(self.gripper, self.s.goal_time)
        if moving:
            if not np.any(linear) and np.any(self.velocity):
                self.target[:3] = sample.pose[:3]
                self.velocity[:] = 0
            if not np.any(angular) and np.any(self.omega):
                self.target[3:] = sample.pose[3:]
                self.omega[:] = 0
            self.velocity += limit_vector(linear - self.velocity, self.s.linear_acceleration * dt)
            self.omega += limit_vector(angular - self.omega, self.s.angular_acceleration * dt)
            measured = Rotation.from_rotvec(sample.pose[3:])
            rotation = Rotation.from_rotvec(self.target[3:]) * Rotation.from_rotvec(self.omega * dt)
            xyz = self.target[:3] + self.velocity * dt
            result["candidate"] = [*xyz.tolist(), *rotation.as_rotvec().tolist()]
            # The target leads feedback by the distance covered in lead_time, so the
            # bound scales with the selected speed instead of capping it.
            position_lead = max(self.s.position_lead,
                                float(np.linalg.norm(self.velocity)) * self.s.lead_time)
            rotation_lead = max(self.s.rotation_lead,
                                float(np.linalg.norm(self.omega)) * self.s.lead_time)
            lead = xyz - sample.pose[:3]
            if np.linalg.norm(lead) > position_lead:
                result["limits"].append("position_lead")
            xyz = sample.pose[:3] + limit_vector(lead, position_lead)
            if self.workspace is not None:
                lower, upper = self.workspace.bounds_including(sample.pose[:3])
                bounded = np.clip(xyz, lower, upper)
                if not np.allclose(xyz, bounded, atol=1e-12, rtol=0):
                    result["limits"].append("workspace")
            else:
                bounded = xyz
            relative = (measured.inv() * rotation).as_rotvec()
            if np.linalg.norm(relative) > rotation_lead:
                result["limits"].append("rotation_lead")
            rotation = measured * Rotation.from_rotvec(limit_vector(relative, rotation_lead))
            self.target = np.r_[bounded, rotation.as_rotvec()]
            # Feedforward keeps the servo from planning a stop at every waypoint;
            # omega is a tool-axis rate, the SDK expects it in the base frame.
            feedforward = self.s.feedforward * np.r_[self.velocity, measured.apply(self.omega)]
            self.driver.move(self.target, self.s.goal_time, self.s.trajectory_samples, feedforward)
            result["sent_pose"] = self.target.tolist()
            result["sent_velocity"] = feedforward.tolist()
        else:
            self.target = sample.pose.copy()
            self.velocity[:] = self.omega[:] = 0
        if gripping:
            # Closing may command past feedback by gripper_squeeze: in position mode that
            # position error is what produces grip force. Opening keeps the motion lead,
            # and both stay inside the configured and SDK limits.
            motion_lead = self.s.gripper_speed * self.s.goal_time
            squeeze = max(motion_lead, self.s.gripper_squeeze)
            self.gripper = float(np.clip(self.gripper + np.clip(intent.gripper, -1, 1) * self.s.gripper_speed * dt,
                                         sample.gripper - squeeze, sample.gripper + motion_lead))
            self.gripper = float(np.clip(self.gripper, self.s.gripper_min, self.s.gripper_max))
            self.driver.move_gripper(self.gripper, self.s.goal_time)
            result["sent_gripper"] = self.gripper
            result["gripper_error"] = self.gripper - sample.gripper
        self.arm_active, self.gripper_active = moving, gripping
