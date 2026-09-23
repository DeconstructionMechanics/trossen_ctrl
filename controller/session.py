"""Threaded control session shared by teleoperation and dataset recording."""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .teleop import Intent, Sample, Settings, Teleop, Workspace


@dataclass(frozen=True)
class ControlTransition:
    sequence: int
    sample_time_s: float
    send_time_s: float
    robot_timestamp_us: int
    control_dt_s: float
    intent: Intent
    actual_joint_positions: tuple[float, ...]
    actual_pose: tuple[float, ...]
    actual_gripper: float
    target_pose: tuple[float, ...]
    target_gripper: float
    command_valid: bool
    state: str
    reason: str
    limit_flags: tuple[str, ...]
    record: dict


class ControlSession:
    """Own a driver/input pair and publish ordered 50 Hz control transitions.

    The control thread never performs camera or dataset I/O. Consumers may wait
    for the first transition at or after an image timestamp without interfering
    with the control deadline.
    """

    def __init__(
        self,
        *,
        driver,
        inputs,
        settings: Settings,
        workspace: Workspace | None,
        home_joints=None,
        logger=None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        transition_buffer_size: int = 2048,
    ):
        settings.validate()
        if transition_buffer_size < 2:
            raise ValueError("transition_buffer_size must be at least 2")
        self.driver = driver
        self.inputs = inputs
        self.settings = settings
        self.workspace = workspace
        self.home_joints = home_joints
        self.logger = logger
        self.clock = clock
        self.sleep = sleep
        self.engine: Teleop | None = None
        self.joint_limits = None
        self._condition = threading.Condition()
        self._transitions: deque[ControlTransition] = deque(maxlen=transition_buffer_size)
        self._pending_events: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._error: BaseException | None = None
        self._sequence = 0
        self._last_sample: Sample | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def last_sample(self) -> Sample | None:
        return self._last_sample

    def connect(self) -> Sample:
        if self.engine is not None:
            raise RuntimeError("ControlSession is already connected")
        self.joint_limits = self.driver.limits()
        sample = self.driver.sample()
        sample.validate()
        self.driver.resume()
        if self.inputs is not None:
            self.inputs.resync()
        sample = self.driver.sample()
        sample.validate()
        self.engine = Teleop(
            self.driver,
            self.settings,
            self.workspace,
            sample,
            self.home_joints,
        )
        if self.workspace is None:
            self.engine.level = 0
        self._last_sample = sample
        return sample

    def start(self) -> None:
        if self.engine is None:
            raise RuntimeError("Call connect() before start()")
        if self.is_running:
            raise RuntimeError("ControlSession is already running")
        self._stop_requested.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="trossen-control", daemon=True)
        self._thread.start()

    def request_hold(self, reason: str = "external hold") -> None:
        del reason  # The Teleop state machine owns the canonical pause reason.
        with self._condition:
            self._pending_events.add("pause")

    def request_event(self, event: str) -> None:
        if event not in {"pause", "resume", "reset", "quit"}:
            raise ValueError(f"Unsupported control event: {event}")
        with self._condition:
            self._pending_events.add(event)

    def latest_transition(self) -> ControlTransition | None:
        with self._condition:
            return self._transitions[-1] if self._transitions else None

    def transitions_after(self, sequence: int) -> list[ControlTransition]:
        """Return buffered transitions newer than ``sequence`` in order."""
        with self._condition:
            if self._transitions and sequence < self._transitions[0].sequence - 1:
                raise RuntimeError("Requested control transitions have fallen out of the ring buffer")
            return [item for item in self._transitions if item.sequence > sequence]

    def wait_for_transition_at_or_after(
        self,
        timestamp_s: float,
        timeout_s: float = 0.25,
    ) -> ControlTransition:
        deadline = self.clock() + timeout_s
        with self._condition:
            while True:
                for transition in self._transitions:
                    if transition.sample_time_s >= timestamp_s:
                        return transition
                if self._error is not None:
                    raise RuntimeError("Control thread failed") from self._error
                if not self.is_running:
                    raise RuntimeError("Control thread stopped before a matching transition arrived")
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No control transition at or after {timestamp_s:.6f} within {timeout_s:.3f}s"
                    )
                self._condition.wait(remaining)

    def stop(self, timeout_s: float = 2.0) -> None:
        with self._condition:
            self._pending_events.add("quit")
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)
            if thread.is_alive():
                self._stop_requested.set()
                thread.join(0.25)
                if thread.is_alive():
                    raise RuntimeError("Control thread did not stop within the timeout")
        self._stop_requested.set()
        self._thread = None

    def close(self) -> None:
        try:
            if self._thread is not None:
                self.stop()
            try:
                self.driver.idle()
            except Exception:
                pass
        finally:
            try:
                if self.inputs is not None:
                    self.inputs.close()
            finally:
                self.driver.close()

    def __enter__(self):
        self.connect()
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _take_pending_events(self) -> set[str]:
        with self._condition:
            events = set(self._pending_events)
            self._pending_events.clear()
            return events

    def _run(self) -> None:
        assert self.engine is not None
        last = self.clock()
        deadline = last + self.settings.period
        last_stamp = self._last_sample.timestamp if self._last_sample is not None else None
        stamp_changed = last
        previous_state = self.engine.state
        try:
            while not self._stop_requested.is_set() and self.engine.state != "exiting":
                self.sleep(max(0.0, deadline - self.clock()))
                now = self.clock()
                dt, last = now - last, now
                deadline = now + self.settings.period
                intent = self.inputs.poll(now) if self.inputs is not None else Intent()
                intent.events.update(self._take_pending_events())

                sample_time = self.clock()
                sample = self.driver.sample()
                sample.validate()
                self._last_sample = sample
                if sample.timestamp != last_stamp:
                    last_stamp, stamp_changed = sample.timestamp, sample_time
                if sample_time - stamp_changed > self.settings.max_dt:
                    raise RuntimeError("SDK feedback timestamp stale")
                if self.clock() - sample_time > self.settings.max_dt:
                    dt = self.settings.max_dt + self.settings.period

                record = self.engine.tick(intent, sample, dt, now)
                send_time = self.clock()
                if self.logger is not None:
                    self.logger.emit(record)
                transition = self._transition(intent, sample, sample_time, send_time, dt, record)
                with self._condition:
                    self._transitions.append(transition)
                    self._condition.notify_all()

                if self.engine.state == "waiting_neutral":
                    if previous_state != "waiting_neutral" and self.inputs is not None:
                        self.inputs.resync()
                    last = self.clock()
                    deadline = last + self.settings.period
                    stamp_changed = last
                previous_state = self.engine.state
        except BaseException as exc:
            self._error = exc
            if self.logger is not None:
                try:
                    self.logger.emit(
                        {"event": "control_thread_error", "time": self.clock(), "error": traceback.format_exc()}
                    )
                except Exception:
                    pass
            try:
                if self._last_sample is not None:
                    self.engine.stop(self._last_sample, repr(exc), "fault")
                else:
                    self.driver.idle()
            except Exception:
                pass
        finally:
            with self._condition:
                self._condition.notify_all()

    def _transition(self, intent, sample, sample_time, send_time, dt, record) -> ControlTransition:
        self._sequence += 1
        state = str(record["state"])
        return ControlTransition(
            sequence=self._sequence,
            sample_time_s=float(sample_time),
            send_time_s=float(send_time),
            robot_timestamp_us=int(sample.timestamp),
            control_dt_s=float(dt),
            intent=Intent(
                linear=tuple(float(x) for x in intent.linear),
                angular=tuple(float(x) for x in intent.angular),
                gripper=float(intent.gripper),
                events=set(intent.events),
                connected=bool(intent.connected),
            ),
            actual_joint_positions=tuple(float(x) for x in sample.joints),
            actual_pose=tuple(float(x) for x in sample.pose),
            actual_gripper=float(sample.gripper),
            target_pose=tuple(float(x) for x in record["target"]),
            target_gripper=float(record["gripper_target"]),
            command_valid=state == "running" and "error" not in record,
            state=state,
            reason=str(record["reason"]),
            limit_flags=tuple(str(x) for x in record.get("limits", ())),
            record=record,
        )
