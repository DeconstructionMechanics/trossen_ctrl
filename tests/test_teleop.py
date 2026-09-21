import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from controller.config import load_keybind_config
from controller.input_state import GamepadInput, radial_pair
from controller.runtime import DiagnosticLog, SDKDriver, SimDriver, calibrate, load_workspace
from controller.teleop import Intent, Sample, Settings, Teleop, Workspace
from controller.xbox import ecodes
from robot_action import parse_args, run, settings_for


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.driver = SimDriver(self.settings)
        self.workspace = Workspace([.1, -.4, .05], [.6, .4, .6], 0)
        self.engine = Teleop(self.driver, self.settings, self.workspace, self.driver.sample())
        self.now = 0
        self.tick()

    def tick(self, intent=None, dt=.02, sample=None):
        self.now += dt
        return self.engine.tick(intent or Intent(), sample or self.driver.sample(), dt, self.now)

    def test_translation_preserves_orientation(self):
        original = Rotation.from_euler("xyz", [.3, -.4, 1.2])
        self.driver.pose[3:] = original.as_rotvec()
        self.engine.sync(self.driver.sample())
        for _ in range(30):
            self.tick(Intent(linear=(1, 1, 0)))
        self.assertLess((original.inv() * Rotation.from_rotvec(self.driver.pose[3:])).magnitude(), 1e-10)

    def test_rotation_is_about_tool_axis_across_pi(self):
        original = Rotation.from_euler("z", math.pi - .001)
        self.driver.pose[3:] = original.as_rotvec()
        self.engine.sync(self.driver.sample())
        self.tick(Intent(angular=(1, 0, 0)))
        delta = (original.inv() * Rotation.from_rotvec(self.driver.pose[3:])).as_rotvec()
        self.assertGreater(delta[0], 0)
        np.testing.assert_allclose(delta[1:], 0, atol=1e-10)

    def test_feedback_lead_bounded_and_no_windup(self):
        stationary = self.driver.sample()
        for _ in range(150):
            record = self.tick(Intent(linear=(1, 1, 1), angular=(1, 1, 1)), sample=stationary)
        self.assertLessEqual(np.linalg.norm(self.engine.target[:3] - stationary.pose[:3]), .00500001)
        self.assertLessEqual(Rotation.from_rotvec(self.engine.target[3:]).magnitude(), math.radians(3) + 1e-9)
        self.assertIn("position_lead", record["limits"])
        self.assertIn("rotation_lead", record["limits"])

    def test_release_holds_arm_and_keeps_gripper_command(self):
        self.tick(Intent(linear=(1, 0, 0), gripper=1))
        commanded = self.engine.gripper
        self.driver.calls.clear()
        self.tick()
        self.assertEqual(self.driver.calls, [("hold", True, False), ("gripper", commanded)])
        np.testing.assert_allclose(self.engine.target, self.driver.pose)
        self.assertEqual(self.engine.gripper, commanded)
        self.tick()
        self.assertEqual(len(self.driver.calls), 2)

    def test_closing_on_a_stalled_gripper_builds_squeeze(self):
        stalled = self.driver.sample()
        for _ in range(60):
            record = self.tick(Intent(gripper=-1), sample=stalled)
        error = stalled.gripper - record["sent_gripper"]
        self.assertAlmostEqual(error, self.settings.gripper_squeeze, places=9)
        self.assertAlmostEqual(record["gripper_error"], -self.settings.gripper_squeeze, places=9)
        # Releasing keeps the squeeze instead of resynchronising to feedback.
        self.tick(sample=stalled)
        self.assertAlmostEqual(self.engine.gripper, record["sent_gripper"], places=9)
        self.assertAlmostEqual(self.driver.gripper, record["sent_gripper"], places=9)

    def test_partial_gripper_input_gives_partial_rate(self):
        stalled = self.driver.sample()
        slow = fast = None
        for gripper, label in ((-.25, "slow"), (-1.0, "fast")):
            self.engine.sync(stalled)
            start = self.engine.gripper
            self.tick(Intent(gripper=gripper), sample=stalled)
            moved = start - self.engine.gripper
            slow, fast = (moved, fast) if label == "slow" else (slow, moved)
        self.assertGreater(fast, slow * 3.9)
        self.assertLess(fast, slow * 4.1)

    def test_pause_holds_the_commanded_gripper_opening(self):
        stalled = self.driver.sample()
        for _ in range(60):
            self.tick(Intent(gripper=-1), sample=stalled)
        commanded = self.engine.gripper
        self.tick(Intent(events={"pause"}), sample=stalled)
        self.assertEqual(self.engine.state, "paused")
        self.assertAlmostEqual(self.driver.gripper_hold, commanded, places=9)

    def test_boundary_reverse_has_no_accumulated_target(self):
        self.driver.pose[0] = .6
        self.engine.sync(self.driver.sample())
        for _ in range(100):
            self.tick(Intent(linear=(1, 0, 0)))
        self.assertAlmostEqual(self.engine.target[0], .6)
        for _ in range(12):
            self.tick(Intent(linear=(-1, 0, 0)))
        self.assertLess(self.engine.target[0], .6)

    def test_unbounded_mode_skips_software_workspace_only(self):
        engine = Teleop(self.driver, self.settings, None, self.driver.sample())
        engine.tick(Intent(), self.driver.sample(), .02, 0)
        engine.level = 0
        self.driver.pose[:3] = [1.0, 1.0, 1.0]
        engine.sync(self.driver.sample())
        record = engine.tick(Intent(linear=(1, 0, 0)), self.driver.sample(), .02, .02)
        self.assertEqual(engine.state, "running")
        self.assertGreater(record["sent_pose"][0], 1.0)
        self.assertLessEqual(record["sent_pose"][0] - 1.0, self.settings.position_lead)
        self.assertNotIn("workspace", record["limits"])

    def test_outside_workspace_blocks_only_outward_motion(self):
        self.driver.pose[0] = .62
        self.engine.sync(self.driver.sample())
        record = self.tick(Intent(linear=(1, 0, 0)))
        self.assertEqual(self.engine.state, "running")
        self.assertTrue(record["outside_workspace"])
        self.assertIn("workspace_recovery", record["limits"])
        self.assertNotIn("sent_pose", record)
        self.assertAlmostEqual(self.driver.pose[0], .62)
        for _ in range(10):
            record = self.tick(Intent(linear=(-1, 0, 0)))
        self.assertEqual(self.engine.state, "running")
        self.assertLess(self.driver.pose[0], .62)
        self.assertNotIn("workspace_recovery", record["limits"])

    def test_outside_workspace_keeps_orthogonal_and_inward_axes(self):
        self.driver.pose[:3] = [.62, 0, .3]
        self.engine.sync(self.driver.sample())
        for _ in range(5):
            self.tick(Intent(linear=(1, 1, 0)))
        self.assertAlmostEqual(self.driver.pose[0], .62)
        self.assertGreater(self.driver.pose[1], 0)

    def test_startup_outside_workspace_is_recoverable_not_fatal(self):
        self.driver.pose[0] = .65
        engine = Teleop(self.driver, self.settings, self.workspace, self.driver.sample())
        self.assertTrue(engine.outside)
        engine.tick(Intent(), self.driver.sample(), .02, 0)
        record = engine.tick(Intent(linear=(-1, 0, 0)), self.driver.sample(), .02, .04)
        self.assertEqual(engine.state, "running")
        self.assertLess(record["sent_pose"][0], .65)

    def test_fault_outside_workspace_can_still_be_resumed(self):
        self.driver.pose[0] = .62
        self.engine.sync(self.driver.sample())
        self.engine.stop(self.driver.sample(), "test fault", "fault")
        self.tick(Intent(events={"resume"}))
        self.assertIn(("resume", True), self.driver.calls)
        self.assertEqual(self.engine.state, "waiting_neutral")

    def test_lead_and_feedforward_scale_with_speed_level(self):
        stationary = self.driver.sample()
        self.engine.level = 2
        for _ in range(150):
            record = self.tick(Intent(linear=(1, 0, 0)), sample=stationary)
        lead = self.engine.target[0] - stationary.pose[0]
        self.assertGreater(lead, self.settings.position_lead)
        self.assertLessEqual(lead, self.settings.linear_speeds[2] * self.settings.lead_time + 1e-9)
        np.testing.assert_allclose(record["sent_velocity"][:3], [self.settings.linear_speeds[2], 0, 0], atol=1e-9)

    def test_feedforward_angular_rate_is_reported_in_the_base_frame(self):
        rotation = Rotation.from_euler("z", math.pi / 2)
        self.driver.pose[3:] = rotation.as_rotvec()
        self.engine.sync(self.driver.sample())
        for _ in range(60):
            record = self.tick(Intent(angular=(1, 0, 0)))
        measured = Rotation.from_rotvec(np.array(record["actual"]["pose"][3:]))
        np.testing.assert_allclose(record["sent_velocity"][3:],
                                   measured.apply([self.settings.angular_speeds[1], 0, 0]), atol=1e-9)

    def test_long_dt_stops_instead_of_integrating(self):
        self.tick(Intent(linear=(1, 0, 0)))
        pose = self.driver.pose.copy()
        self.tick(Intent(linear=(1, 0, 0)), dt=1)
        self.assertEqual(self.engine.state, "paused")
        np.testing.assert_array_equal(pose, self.driver.pose)

    def test_fault_latched_until_neutral_resume(self):
        def fail(*args):
            raise RuntimeError("inverse kinematics failure")
        self.driver.move = fail
        result = self.tick(Intent(linear=(1, 0, 0)))
        self.assertEqual(result["error_category"], "SDK IK error")
        self.assertIn("Traceback", result["error"])
        self.assertEqual(self.engine.state, "fault")
        self.tick(Intent(linear=(1, 0, 0), events={"resume"}))
        self.assertNotIn(("resume", True), self.driver.calls)
        self.tick(Intent(events={"resume"}))
        self.assertIn(("resume", True), self.driver.calls)
        self.assertEqual(self.engine.state, "waiting_neutral")
        self.tick(Intent(linear=(1, 0, 0)))
        self.assertEqual(self.engine.state, "waiting_neutral")
        self.tick()
        self.assertEqual(self.engine.state, "running")

    def test_return_cancelled_without_executing_input(self):
        self.driver.joints[:] = .2
        self.tick(Intent(events={"reset"}))
        self.assertEqual(self.engine.state, "returning")
        self.driver.calls.clear()
        pose = self.driver.pose.copy()
        self.tick(Intent(linear=(1, 0, 0)))
        # Cancelling stops in place and waits for neutral; it never executes that input.
        self.assertEqual(self.engine.state, "waiting_neutral")
        self.assertEqual(self.driver.calls, [("hold", True, True)])
        np.testing.assert_array_equal(pose, self.driver.pose)
        self.tick(Intent(linear=(1, 0, 0)))
        self.assertEqual(self.engine.state, "waiting_neutral")
        np.testing.assert_array_equal(pose, self.driver.pose)
        self.tick()
        self.assertEqual(self.engine.state, "running")
        self.assertNotIn(("resume", False), self.driver.calls)

    def test_return_requested_while_pushing_waits_for_neutral(self):
        self.driver.joints[:] = .2
        pose = self.driver.pose.copy()
        self.tick(Intent(linear=(1, 0, 0), events={"reset"}))
        self.assertEqual(self.engine.state, "waiting_neutral")
        self.assertEqual(self.engine.reason, "release motion controls before return")
        self.assertFalse(any(call[0] == "return" for call in self.driver.calls))
        np.testing.assert_array_equal(pose, self.driver.pose)
        self.tick()
        self.assertEqual(self.engine.state, "running")

    def test_completed_return_stays_paused_until_resumed(self):
        self.driver.joints[:] = .2
        self.tick(Intent(events={"reset"}))
        self.tick()
        self.assertEqual(self.engine.state, "paused")
        self.assertEqual(self.engine.reason, "return complete")
        self.tick()
        self.assertEqual(self.engine.state, "paused")
        self.tick(Intent(events={"resume"}))
        self.assertEqual(self.engine.state, "waiting_neutral")

    def test_configured_home_is_the_return_target(self):
        home = [.1, -.2, .3, -.4, .5, -.6]
        engine = Teleop(self.driver, self.settings, self.workspace, self.driver.sample(), home)
        np.testing.assert_allclose(engine.start_joints, home)
        engine.tick(Intent(), self.driver.sample(), .02, 0)
        engine.tick(Intent(events={"reset"}), self.driver.sample(), .02, .02)
        np.testing.assert_allclose(self.driver.joints, home)
        with self.assertRaises(ValueError):
            Teleop(self.driver, self.settings, self.workspace, self.driver.sample(), [0, 0, 0])

    def test_quit_priority_no_return_or_close(self):
        opening = self.driver.gripper
        self.tick(Intent(gripper=-1, events={"reset", "quit", "resume"}))
        self.assertEqual(self.engine.state, "exiting")
        self.assertEqual(self.driver.gripper, opening)
        self.assertFalse(any(call[0] == "return" for call in self.driver.calls))

    def test_disconnect_requires_explicit_resume(self):
        self.tick(Intent(connected=False))
        self.driver.calls.clear()
        self.tick(Intent(connected=False))
        self.assertEqual(self.driver.calls, [])
        self.tick()
        self.assertEqual(self.engine.state, "paused")

    def test_stop_failure_attempts_idle(self):
        def fail(*args, **kwargs):
            raise OSError("connection failed")
        self.driver.hold = fail
        self.tick(Intent(events={"pause"}))
        self.assertEqual(self.engine.state, "fault")
        self.assertIn(("idle",), self.driver.calls)

    def test_rotation_release_while_translating_stops_rotation(self):
        for _ in range(20):
            self.tick(Intent(linear=(1, 0, 0), angular=(0, 0, 1)))
        rotation = self.driver.pose[3:].copy()
        self.tick(Intent(linear=(1, 0, 0)))
        np.testing.assert_allclose(self.driver.pose[3:], rotation, atol=1e-10)

    def test_multi_axis_speed_is_norm_limited(self):
        for _ in range(30):
            self.tick(Intent(linear=(1, 1, 1), angular=(1, 1, 1)))
        self.assertLessEqual(np.linalg.norm(self.engine.velocity), .06000001)
        self.assertLessEqual(np.linalg.norm(self.engine.omega), math.radians(30) + 1e-8)

    def test_return_uses_startup_joints_and_preserves_gripper(self):
        self.driver.joints[:] = .5
        self.driver.gripper = .03
        self.tick(Intent(events={"reset"}))
        np.testing.assert_allclose(self.driver.joints, self.engine.start_joints)
        self.assertEqual(self.driver.gripper, .03)
        self.assertGreaterEqual(self.driver.calls[-1][1], 3)

    def test_failed_recovery_stays_faulted(self):
        self.engine.state = "fault"
        def fail(*args):
            raise OSError("recovery failed")
        self.driver.resume = fail
        self.tick(Intent(events={"resume"}))
        self.assertEqual(self.engine.state, "fault")

    def test_pause_wins_over_resume(self):
        self.tick(Intent(events={"pause", "resume"}))
        self.assertEqual(self.engine.state, "paused")
        self.assertFalse(any(x[0] == "resume" for x in self.driver.calls))


class FakeDevice:
    def __init__(self):
        names = ("ABS_X", "ABS_Y", "ABS_RX", "ABS_RY", "ABS_Z", "ABS_RZ", "ABS_HAT0X", "ABS_HAT0Y")
        self.info = {getattr(ecodes, n): SimpleNamespace(min=0 if n in {"ABS_Z", "ABS_RZ"} else -100,
                                                      max=100, value=0) for n in names}
        self.keys = set()
        self.events = []

    def capabilities(self, absinfo=True):
        return {ecodes.EV_ABS: list(self.info.items())}

    def absinfo(self, code):
        return self.info[code]

    def active_keys(self):
        return self.keys

    def grab(self):
        pass

    def ungrab(self):
        pass

    def close(self):
        pass

    def read(self):
        batch, self.events = self.events, []
        return iter(batch)

    def event(self, kind, code, value):
        self.events.append(SimpleNamespace(type=kind, code=code, value=value))


class InputTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.input = GamepadInput(Settings(), device=self.device)

    def test_speed_event_does_not_lose_center_event(self):
        self.device.event(ecodes.EV_ABS, ecodes.ABS_X, 100)
        self.assertFalse(self.input.poll(0).neutral)
        self.device.event(ecodes.EV_ABS, ecodes.ABS_HAT0Y, -1)
        self.device.event(ecodes.EV_ABS, ecodes.ABS_X, 0)
        intent = self.input.poll(.02)
        self.assertIn("speed_up", intent.events)
        self.assertTrue(intent.neutral)

    def test_held_stick_without_new_events(self):
        self.device.event(ecodes.EV_ABS, ecodes.ABS_X, 100)
        first = self.input.poll(0)
        self.assertEqual(first.linear, self.input.poll(2).linear)

    def test_held_reset_at_attach_needs_a_release_first(self):
        device = FakeDevice()
        device.keys.add(ecodes.BTN_SELECT)
        inputs = GamepadInput(Settings(), device=device)
        self.assertNotIn("reset", inputs.poll(0).events)
        self.assertNotIn("reset", inputs.poll(2).events)
        device.event(ecodes.EV_KEY, ecodes.BTN_SELECT, 0)
        self.assertNotIn("reset", inputs.poll(2.1).events)
        device.event(ecodes.EV_KEY, ecodes.BTN_SELECT, 1)
        self.assertNotIn("reset", inputs.poll(2.2).events)
        self.assertIn("reset", inputs.poll(3.3).events)

    def test_dropped_frame_resync_disarms_a_held_reset(self):
        self.device.event(ecodes.EV_KEY, ecodes.BTN_SELECT, 1)
        self.input.poll(0)
        self.device.event(ecodes.EV_SYN, ecodes.SYN_DROPPED, 0)
        self.device.event(ecodes.EV_SYN, ecodes.SYN_REPORT, 0)
        self.input.poll(.02)
        self.assertNotIn("reset", self.input.poll(1.5).events)

    def test_reset_requires_long_press(self):
        self.device.event(ecodes.EV_KEY, ecodes.BTN_SELECT, 1)
        self.assertNotIn("reset", self.input.poll(0).events)
        self.assertIn("reset", self.input.poll(1.1).events)
        self.assertNotIn("reset", self.input.poll(2).events)

    def test_press_release_same_batch_preserves_pause(self):
        self.device.event(ecodes.EV_KEY, ecodes.BTN_EAST, 1)
        self.device.event(ecodes.EV_KEY, ecodes.BTN_EAST, 0)
        self.assertIn("pause", self.input.poll(0).events)

    def test_syn_dropped_resynchronizes_and_pauses(self):
        self.device.event(ecodes.EV_ABS, ecodes.ABS_X, 100)
        self.input.poll(0)
        self.device.event(ecodes.EV_SYN, ecodes.SYN_DROPPED, 0)
        self.device.event(ecodes.EV_ABS, ecodes.ABS_X, 100)
        self.device.event(ecodes.EV_SYN, ecodes.SYN_REPORT, 0)
        intent = self.input.poll(.02)
        self.assertTrue(intent.neutral)
        self.assertIn("input_lost", intent.events)

    def test_triggers_drive_z_proportionally(self):
        self.device.event(ecodes.EV_ABS, ecodes.ABS_Z, 100)
        intent = self.input.poll(0)
        self.assertEqual(intent.linear, (0, 0, -1))
        self.assertEqual(intent.gripper, 0)
        self.device.event(ecodes.EV_ABS, ecodes.ABS_Z, 50)
        partial = self.input.poll(.02).linear[2]
        self.assertLess(-1, partial)
        self.assertLess(partial, -.1)
        self.device.event(ecodes.EV_ABS, ecodes.ABS_Z, 3)
        self.assertTrue(self.input.poll(.04).neutral)
        self.device.event(ecodes.EV_ABS, ecodes.ABS_RZ, 100)
        self.assertEqual(self.input.poll(.06).linear, (0, 0, 1))

    def test_bumpers_drive_roll_and_dpad_drives_gripper(self):
        self.device.event(ecodes.EV_KEY, ecodes.BTN_TR, 1)
        self.assertEqual(self.input.poll(0).angular, (1, 0, 0))
        self.device.event(ecodes.EV_KEY, ecodes.BTN_TR, 0)
        self.device.event(ecodes.EV_KEY, ecodes.BTN_TL, 1)
        self.assertEqual(self.input.poll(.02).angular, (-1, 0, 0))
        self.device.event(ecodes.EV_KEY, ecodes.BTN_TL, 0)
        self.device.event(ecodes.EV_ABS, ecodes.ABS_HAT0X, -1)
        intent = self.input.poll(.04)
        self.assertEqual(intent.gripper, -1)
        self.assertEqual(intent.linear, (0, 0, 0))
        self.device.event(ecodes.EV_ABS, ecodes.ABS_HAT0X, 1)
        self.assertEqual(self.input.poll(.06).gripper, 1)

    def test_resync_absorbs_backlog_without_reporting_loss(self):
        self.device.event(ecodes.EV_ABS, ecodes.ABS_X, 100)
        self.device.event(ecodes.EV_SYN, ecodes.SYN_DROPPED, 0)
        self.device.event(ecodes.EV_KEY, ecodes.BTN_EAST, 1)
        self.input.resync()
        intent = self.input.poll(.02)
        self.assertNotIn("input_lost", intent.events)
        self.assertEqual(intent.events, set())
        self.assertTrue(intent.connected)

    def test_resync_reads_held_controls_back_from_the_device(self):
        self.device.info[ecodes.ABS_X].value = 100
        self.device.keys.add(ecodes.BTN_TL)
        self.input.resync()
        intent = self.input.poll(.02)
        self.assertEqual(intent.linear[1], -1)
        self.assertEqual(intent.angular[0], -1)
        self.assertFalse(intent.neutral)

    def test_radial_deadzone_and_diagonal_bound(self):
        self.assertEqual(radial_pair(.05, .05, .1), (0, 0))
        self.assertGreater(radial_pair(.2, 0, .1)[0], 0)
        self.assertAlmostEqual(math.hypot(*radial_pair(1, 1, .1)), 1)

    def test_dropped_frame_across_polls_remains_disabled(self):
        self.device.event(ecodes.EV_SYN, ecodes.SYN_DROPPED, 0)
        self.assertFalse(self.input.poll(0).connected)
        self.assertFalse(self.input.poll(.02).connected)
        self.device.event(ecodes.EV_SYN, ecodes.SYN_REPORT, 0)
        self.assertTrue(self.input.poll(.04).connected)

    def test_disconnect_and_reconnect_snapshot(self):
        def fail():
            raise OSError("removed")
        self.device.read = fail
        self.assertFalse(self.input.poll(0).connected)
        new_device = FakeDevice()
        new_device.info[ecodes.ABS_X].value = 100
        self.input.ever_connected = True
        with patch("controller.input_state.find_xbox_controller", return_value=new_device):
            intent = self.input.poll(1.1)
        self.assertIn("input_lost", intent.events)
        self.assertFalse(intent.neutral)


class RuntimeTests(unittest.TestCase):
    def test_missing_calibration_never_connects(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("robot_action.SDKDriver") as driver:
                with self.assertRaises(FileNotFoundError):
                    run(parse_args(["--workspace", str(Path(directory) / "missing.json")]))
                driver.assert_not_called()

    def test_calibration_only_reads_and_does_not_overwrite(self):
        driver = SimDriver(Settings())
        samples = []
        for axis, bounds in enumerate(((.1, .6), (-.4, .4), (.05, .6))):
            for bound in bounds:
                sample = driver.sample()
                sample.pose[axis] = bound
                samples.append(sample)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace.json"
            with patch.object(driver, "sample", side_effect=samples), patch("builtins.input", side_effect=["0", "", "", "", "", "", "", "SAVE"]):
                calibrate(driver, path, "192.168.1.3", .01)
            self.assertEqual(driver.calls, [])
            workspace = load_workspace(path, "192.168.1.3", .01)
            self.assertTrue(workspace.contains(np.array([.3, 0, .25])))

    def test_sdk_adapter_uses_snapshot_and_async_commands(self):
        calls = []
        raw = SimpleNamespace(
            get_robot_output=lambda: SimpleNamespace(
                joint=SimpleNamespace(arm=SimpleNamespace(positions=[0.] * 6), gripper=SimpleNamespace(position=.02)),
                cartesian=SimpleNamespace(positions=[.3, 0, .2, 0, 0, 0]), header=SimpleNamespace(timestamp=123)),
            set_cartesian_positions=lambda *a, **kw: calls.append((a, kw)),
        )
        driver = SDKDriver.__new__(SDKDriver)
        driver.raw, driver.s = raw, Settings()
        driver.sdk = SimpleNamespace(InterpolationSpace=SimpleNamespace(cartesian="cartesian"))
        sample = driver.sample()
        self.assertEqual(sample.timestamp, 123)
        driver.move(sample.pose, .06, 3)
        self.assertFalse(calls[0][1]["blocking"])
        self.assertEqual(calls[0][1]["num_trajectory_check_samples"], 3)
        self.assertNotIn("goal_feedforward_velocities", calls[0][1])
        driver.move(sample.pose, .06, 3, np.array([.1, 0, 0, 0, 0, .2]))
        self.assertEqual(calls[1][1]["goal_feedforward_velocities"], [.1, 0, 0, 0, 0, .2])

    def test_sdk_resume_waits_for_position_feedback_to_settle(self):
        heights = iter((.155, .159, .161, .161, .161, .161))
        calls = []
        driver = SDKDriver.__new__(SDKDriver)
        driver.raw = SimpleNamespace(
            set_arm_modes=lambda mode: calls.append("arm_mode"),
            set_gripper_mode=lambda mode: calls.append("gripper_mode"),
        )
        driver.sdk = SimpleNamespace(Mode=SimpleNamespace(position="position"))
        driver.raw.get_modes = lambda: ["idle"] * 7
        driver.sample = lambda: Sample(np.zeros(6), np.array([.25, 0, next(heights), 0, 0, 0]), .02)
        driver.hold = lambda sample: calls.append("hold")
        with patch("controller.runtime.time.sleep", return_value=None):
            driver.resume()
        self.assertEqual(calls, ["arm_mode", "gripper_mode", "hold"])

    def test_sdk_resume_from_a_plain_pause_skips_the_mode_switch(self):
        calls = []
        driver = SDKDriver.__new__(SDKDriver)
        driver.sdk = SimpleNamespace(Mode=SimpleNamespace(position="position"))
        driver.raw = SimpleNamespace(
            get_modes=lambda: ["position"] * 7,
            set_arm_modes=lambda mode: calls.append("arm_mode"),
            set_gripper_mode=lambda mode: calls.append("gripper_mode"),
        )
        driver.resume()
        self.assertEqual(calls, [])

    def test_config_defaults_and_overrides(self):
        settings = settings_for(parse_args([]))
        self.assertEqual(settings.linear_speeds, (.02, .06, .12))
        self.assertEqual(settings.gripper_speed, .02)
        settings = settings_for(parse_args(["--goal-time", ".08", "--gripper-max", ".03"]))
        self.assertEqual(settings.goal_time, .08)
        self.assertEqual(settings.gripper_max, .03)

    def test_config_rejects_unknown_sections_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("teleop:\n  period: 0.02\nnonsense:\n  a: 1\n")
            with self.assertRaises(ValueError) as caught:
                load_keybind_config(path)
            self.assertIn("nonsense", str(caught.exception))
            path = Path(directory) / "other.yaml"
            path.write_text("teleop:\n  not_a_setting: 1\n")
            with self.assertRaises(ValueError) as caught:
                settings_for(parse_args(["--config", str(path)]))
            self.assertIn("not_a_setting", str(caught.exception))

    def test_config_keeps_key_bindings_as_text(self):
        config = load_keybind_config(None)
        self.assertEqual(config["keyboard"]["sensitivityup"], "=")
        self.assertEqual(config["xbox_axes"]["downward"], "ABS_Z:+")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("keyboard:\n  forward: on\n")
            with self.assertRaises(ValueError) as caught:
                load_keybind_config(path)
            self.assertIn("quote the binding", str(caught.exception))

    def test_invalid_workspace_rejected(self):
        with self.assertRaises(ValueError):
            Workspace([0, 0, 0], [1, 1, .01], .1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace.json"
            path.write_text(json.dumps(dict(arm_ip="wrong")))
            with self.assertRaises(ValueError):
                load_workspace(path, "192.168.1.3", .01)

    def test_recorded_clearance_mismatch_is_reported_not_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace.json"
            data = dict(arm_ip="192.168.1.3", model="wxai_v0_follower", lower=[.1, -.4, .008],
                        upper=[.6, .4, .6], table_z=.005, clearance=.005)
            path.write_text(json.dumps(data))
            workspace = load_workspace(path, "192.168.1.3", .005)
            self.assertIsNone(workspace.notice)
            self.assertAlmostEqual(workspace.lower[2], .01)
            workspace = load_workspace(path, "192.168.1.3", .02)
            self.assertIn("0.005", workspace.notice)
            self.assertIn("0.02", workspace.notice)
            self.assertAlmostEqual(workspace.lower[2], .025)

    def test_connection_failure_names_the_single_connection_limit(self):
        class Raw:
            def configure(self, *args):
                raise OSError("Failed to read TCP message from 192.168.1.3:50001 due to Resource temporarily unavailable")
        fake = SimpleNamespace(TrossenArmDriver=Raw, Model=SimpleNamespace(wxai_v0=0),
                               StandardEndEffector=SimpleNamespace(wxai_v0_follower=0))
        with patch.dict(sys.modules, {"trossen_arm": fake}):
            with self.assertRaises(RuntimeError) as caught:
                SDKDriver("192.168.1.3", Settings())
        message = str(caught.exception)
        self.assertIn("no other process is currently controlling this arm", message)
        self.assertIn("Resource temporarily unavailable", message)
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_shutdown_measures_braking_sag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            run(parse_args(["--dry-run", "--duration", ".2", "--log", str(path)]))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            sag = [r for r in records if r.get("event") == "idle_sag"]
            self.assertEqual(len(sag), 1)
            self.assertEqual(sag[0]["sag"], 0.0)
            self.assertEqual(sag[0]["before"], sag[0]["after"])

    def test_parked_low_warns_before_braking(self):
        settings = Settings()
        driver = SimDriver(settings)
        driver.pose[2] = .012
        with tempfile.TemporaryDirectory() as directory:
            with patch("robot_action.SimDriver", return_value=driver), patch("builtins.print") as printed:
                run(parse_args(["--dry-run", "--duration", ".2", "--log", str(Path(directory) / "log.jsonl")]))
        messages = " ".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertIn("Lift the TCP before stopping", messages)

    def test_log_written_and_failure_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            log = DiagnosticLog(path)
            log.emit({"event": "test", "time": 1})
            log.close()
            self.assertEqual(json.loads(path.read_text())["event"], "test")
            log = DiagnosticLog(Path(directory) / "bad.jsonl")
            log.emit({"bad": float("nan")})
            with self.assertRaises(RuntimeError):
                log.close()


if __name__ == "__main__":
    unittest.main()
