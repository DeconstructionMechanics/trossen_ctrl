"""Run controller input and apply target Cartesian positions to the Trossen arm."""

from __future__ import annotations

import argparse
from contextlib import nullcontext

from controller import (
    CartesianKeyboardController,
    RawTerminal,
    TargetStateController,
    XboxController,
)
from controller.commands import POSE_LABELS


ARM_IP = "192.168.1.3"
LINEAR_STEP = 0.01
ANGULAR_STEP = 0.05
GRIPPER_STEP = 0.005
GOAL_TIME = 0.2
RESET_GOAL_TIME = 3.0
CONTROL_PERIOD = 0.05


def get_attr(obj, names: list[str]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"Cannot find any of {names} in {obj}")


def build_input_controller(args):
    kwargs = {
        "linear_step": args.linear_step,
        "angular_step": args.angular_step,
        "gripper_step": args.gripper_step,
        "sensitivity": args.sensitivity,
        "sensitivity_factor": args.sensitivity_factor,
    }
    if args.controller == "keyboard":
        return CartesianKeyboardController(**kwargs), RawTerminal()
    return XboxController(device_path=args.device, **kwargs), nullcontext()


def move_to_state(
    driver,
    state_controller,
    cartesian_interp,
    goal_time,
    blocking,
    set_gripper=True,
):
    pose, gripper = state_controller.pose_and_gripper()
    driver.set_cartesian_positions(
        goal_positions=list(pose),
        interpolation_space=cartesian_interp,
        goal_time=goal_time,
        blocking=blocking,
    )
    if set_gripper:
        driver.set_gripper_position(
            goal_position=gripper,
            goal_time=goal_time,
            blocking=blocking,
        )


def return_to_global_initial(driver, state_controller, args):
    arm_joint_count = len(driver.get_arm_positions())
    arm_positions = [0.0] * arm_joint_count
    if hasattr(driver, "set_all_positions"):
        driver.set_all_positions(
            goal_positions=[*arm_positions, args.return_gripper],
            goal_time=args.reset_goal_time,
            blocking=True,
        )
    else:
        driver.set_arm_positions(
            goal_positions=arm_positions,
            goal_time=args.reset_goal_time,
            blocking=True,
        )
        driver.set_gripper_position(
            goal_position=args.return_gripper,
            goal_time=args.reset_goal_time,
            blocking=True,
        )
    state_controller.set_state(
        driver.get_cartesian_positions(),
        driver.get_gripper_position(),
    )
    state_controller.print_state(prefix="returned: ")


def run(args):
    import numpy as np
    import trossen_arm

    model = get_attr(trossen_arm.Model, ["wxai_v0", "WXAI_V0"])
    ee = get_attr(
        trossen_arm.StandardEndEffector,
        ["wxai_v0_follower", "WXAI_V0_FOLLOWER"],
    )
    position_mode = get_attr(trossen_arm.Mode, ["position", "POSITION"])
    cartesian_interp = get_attr(
        trossen_arm.InterpolationSpace,
        ["cartesian", "CARTESIAN"],
    )

    driver = trossen_arm.TrossenArmDriver()
    pose0 = None
    state_controller = None

    try:
        print("Connecting...")
        driver.configure(model, ee, args.arm_ip, False)

        print("Setting arm to position mode...")
        driver.set_arm_modes(position_mode)
        driver.set_gripper_mode(position_mode)

        q0 = np.array(driver.get_arm_positions(), dtype=float)
        pose0 = np.array(driver.get_cartesian_positions(), dtype=float)
        gripper0 = float(driver.get_gripper_position())

        print("\nCurrent joint positions:")
        print(q0)
        print("\nCurrent Cartesian pose:")
        print(pose0)
        print("\nCurrent gripper position:")
        print(gripper0)
        print(f"\nPose format: [{', '.join(POSE_LABELS)}]")

        input(
            "\nSafety check: gripper should be far from table and obstacles. "
            f"Press Enter to start {args.controller} Cartesian control..."
        )

        input_controller, input_context = build_input_controller(args)
        state_controller = TargetStateController(
            initial_pose=pose0.tolist(),
            initial_gripper=gripper0,
            return_pose=pose0.tolist(),
            return_gripper=gripper0,
            gripper_min=args.gripper_min,
            gripper_max=args.gripper_max,
            yaw_follow_xy=args.yaw_follow_xy,
        )

        print()
        input_controller.print_help()
        print(
            f"linear_step={args.linear_step} m, angular_step={args.angular_step} rad, "
            f"gripper_step={args.gripper_step} m"
        )
        state_controller.print_state()

        with input_controller, input_context:
            while True:
                command = input_controller.read_command(timeout=args.control_period)
                if command is not None:
                    if command.quit:
                        print("\nQuit command received.")
                        return_to_global_initial(
                            driver,
                            state_controller,
                            args,
                        )
                        break

                    if command.reset:
                        print("\nMoving gently to global initial joint pose...")
                        return_to_global_initial(
                            driver,
                            state_controller,
                            args,
                        )
                        state_controller.print_state(prefix="reset done: ")
                        continue

                    state_controller.apply(command)
                    state_controller.print_state(
                        prefix=f"{command.action} sens={command.sensitivity:.2f}: "
                    )

                move_to_state(
                    driver,
                    state_controller,
                    cartesian_interp,
                    args.goal_time,
                    blocking=False,
                )

        print(f"\n{args.controller.capitalize()} Cartesian control finished.")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected.")
        if pose0 is not None and state_controller is not None:
            print("Trying to return to global initial joint pose...")
            try:
                return_to_global_initial(
                    driver,
                    state_controller,
                    args,
                )
            except Exception as exc:
                print("Failed to return automatically:", exc)

    finally:
        try:
            driver.cleanup()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=("keyboard", "xbox"),
        default="keyboard",
        help="Input source to use.",
    )
    parser.add_argument("--device", help="Xbox event device, for example /dev/input/event0.")
    parser.add_argument("--arm-ip", default=ARM_IP)
    parser.add_argument("--linear-step", type=float, default=LINEAR_STEP)
    parser.add_argument("--angular-step", type=float, default=ANGULAR_STEP)
    parser.add_argument("--gripper-step", type=float, default=GRIPPER_STEP)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--sensitivity-factor", type=float, default=1.25)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=0.044)
    parser.add_argument(
        "--return-gripper",
        type=float,
        default=0.0,
        help="Gripper return target in meters. Defaults to the global initial value.",
    )
    parser.add_argument(
        "--no-yaw-follow-xy",
        dest="yaw_follow_xy",
        action="store_false",
        help="Disable automatic yaw alignment to atan2(y, x).",
    )
    parser.set_defaults(yaw_follow_xy=True)
    parser.add_argument("--goal-time", type=float, default=GOAL_TIME)
    parser.add_argument("--reset-goal-time", type=float, default=RESET_GOAL_TIME)
    parser.add_argument("--control-period", type=float, default=CONTROL_PERIOD)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
