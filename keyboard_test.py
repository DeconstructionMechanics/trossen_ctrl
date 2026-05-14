"""Keyboard controller demo.

The implementation lives in controller/keyboard.py. This file is kept as a
short runnable entry point for the old filename.
"""

import time

from controller.control_logic import TargetStateController
from controller.keyboard import CartesianKeyboardController, RawTerminal

CONTROL_PERIOD = 0.02
RESET_STEPS = 60


def smooth_reset(state_controller):
    for _ in state_controller.iter_reset_steps(RESET_STEPS):
        state_controller.print_state(prefix="returning: ")
        time.sleep(CONTROL_PERIOD)


def main():
    input_controller = CartesianKeyboardController()
    state_controller = TargetStateController(
        initial_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        initial_gripper=0.0,
    )

    input_controller.print_help()
    state_controller.print_state(prefix="Demo ")

    with RawTerminal():
        while True:
            command = input_controller.read_command(timeout=CONTROL_PERIOD)
            if command is None:
                continue
            if command.quit:
                smooth_reset(state_controller)
                print("\nReturned to zero. Quit.")
                break
            if command.reset:
                smooth_reset(state_controller)
                continue
            state_controller.apply(command)
            state_controller.print_state(
                prefix=f"{command.action} sens={command.sensitivity:.2f}: "
            )


if __name__ == "__main__":
    main()
