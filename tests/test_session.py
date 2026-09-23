import time
import unittest

from controller.runtime import SimDriver
from controller.session import ControlSession
from controller.teleop import Intent, Settings, Workspace


class NeutralInput:
    def __init__(self):
        self.closed = False

    def poll(self, now):
        return Intent()

    def resync(self):
        pass

    def close(self):
        self.closed = True


class SessionTests(unittest.TestCase):
    def make_session(self):
        settings = Settings(period=.005, max_dt=.1)
        driver = SimDriver(settings)
        inputs = NeutralInput()
        session = ControlSession(
            driver=driver,
            inputs=inputs,
            settings=settings,
            workspace=Workspace([.1, -.4, .05], [.6, .4, .6], 0),
        )
        return session, driver, inputs

    def test_publishes_ordered_transitions_and_holds_on_request(self):
        session, driver, inputs = self.make_session()
        try:
            session.connect()
            session.start()
            first = session.wait_for_transition_at_or_after(time.monotonic(), timeout_s=.2)
            second = session.wait_for_transition_at_or_after(first.sample_time_s + 1e-9, timeout_s=.2)
            self.assertGreater(second.sequence, first.sequence)
            self.assertGreaterEqual(second.sample_time_s, first.sample_time_s)
            session.request_hold("test")
            deadline = time.monotonic() + .2
            while time.monotonic() < deadline:
                transition = session.latest_transition()
                if transition is not None and transition.state == "paused":
                    break
                time.sleep(.005)
            self.assertEqual(session.latest_transition().state, "paused")
        finally:
            session.close()
        self.assertTrue(inputs.closed)
        self.assertIn(("idle",), driver.calls)
        self.assertEqual(session.engine.state, "exiting")

    def test_wait_rejects_when_session_is_not_running(self):
        session, _, _ = self.make_session()
        session.connect()
        with self.assertRaises(RuntimeError):
            session.wait_for_transition_at_or_after(time.monotonic(), timeout_s=.01)
        session.close()


if __name__ == "__main__":
    unittest.main()
