"""Run controller input and apply target Cartesian positions to the Trossen arm."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math

from controller import (
    CartesianKeyboardController,
    RawTerminal,
    TargetStateController,
    XboxController,
)
from controller.commands import (
    PITCH_INDEX,
    POSE_LABELS,
    ROLL_INDEX,
    X_INDEX,
    YAW_INDEX,
    Y_INDEX,
    Z_INDEX,
)


ARM_IP = "192.168.1.3"
LINEAR_STEP = 0.01
ANGULAR_STEP = 0.05
GRIPPER_STEP = 0.005
GOAL_TIME = 0.2
RESET_GOAL_TIME = 3.0
CONTROL_PERIOD = 0.05
MAX_REACH_RADIUS = 0.9
BASE_ORIENTATION_ORIGIN = (0.0, 0.0)


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
        "min_sensitivity": args.min_sensitivity,
        "max_sensitivity": args.max_sensitivity,
    }
    if args.controller == "keyboard":
        return CartesianKeyboardController(**kwargs), RawTerminal()
    return XboxController(device_path=args.device, **kwargs), nullcontext()


class RobotMover:
    """Apply target states to the driver without mutating the target on failure."""

    def __init__(
        self,
        driver,
        cartesian_interp,
        max_reach_radius: float,
        orientation_origin_xy,
    ):
        self.driver = driver
        self.cartesian_interp = cartesian_interp
        self.max_reach_radius = max_reach_radius
        self.orientation_origin_xy = tuple(orientation_origin_xy)
        self._motion_blocked = False
        self._radius_limited = False

    def move_to_state(
        self,
        state_controller,
        goal_time,
        blocking,
        set_gripper=True,
    ) -> bool:
        pose, gripper = state_controller.pose_and_gripper()
        driver_pose = target_pose_to_driver_pose(
            pose,
            self.max_reach_radius,
            self.orientation_origin_xy,
        )
        self._print_radius_limit_notice(pose, driver_pose)
        try:
            self.driver.set_cartesian_positions(
                goal_positions=list(driver_pose),
                interpolation_space=self.cartesian_interp,
                goal_time=goal_time,
                blocking=blocking,
            )
        except Exception as exc:
            if not self._motion_blocked:
                print(
                    "\nTarget is outside the reachable Cartesian range; "
                    f"holding the last valid arm position. Driver error: {exc}"
                )
            self._motion_blocked = True
            _try_clear_driver_error(self.driver)
            return False

        self._motion_blocked = False
        if set_gripper:
            try:
                self.driver.set_gripper_position(
                    goal_position=gripper,
                    goal_time=goal_time,
                    blocking=blocking,
                )
            except Exception as exc:
                print(
                    "\nCould not apply gripper target; "
                    f"holding gripper. Driver error: {exc}"
                )
                return False
        return True

    def _print_radius_limit_notice(self, target_pose, driver_pose):
        target_radius = _xyz_radius(target_pose)
        driver_radius = _xyz_radius(driver_pose)
        radius_limited = driver_radius < target_radius - 1e-9
        if radius_limited and not self._radius_limited:
            print(
                "\nTarget radius exceeds configured reach; "
                f"limiting API xyz radius to {driver_radius:.3f} m."
        )
        self._radius_limited = radius_limited


def target_pose_to_driver_pose(
    pose,
    max_reach_radius: float = MAX_REACH_RADIUS,
    orientation_origin_xy=BASE_ORIENTATION_ORIGIN,
):
    """Convert human target pose to driver Cartesian xyz plus angle-axis."""
    x, y, z, roll, pitch, yaw = pose
    limited_xyz = _limit_xyz_radius((x, y, z), max_reach_radius)
    base_yaw = _orientation_base_yaw(limited_xyz, orientation_origin_xy)
    rotation = _matmul(
        _rotation_z(base_yaw + yaw),
        _matmul(_rotation_y(pitch), _rotation_x(roll)),
    )
    return (*limited_xyz, *_matrix_to_axis_angle(rotation))


def driver_pose_to_target_pose(pose, orientation_origin_xy=BASE_ORIENTATION_ORIGIN):
    """Convert driver Cartesian angle-axis pose to human target roll/pitch/yaw."""
    x, y, z = pose[:3]
    rotation = _axis_angle_to_matrix(pose[3:])
    base_rotation = _rotation_z(
        -_orientation_base_yaw((x, y, z), orientation_origin_xy)
    )
    relative_rotation = _matmul(base_rotation, rotation)
    roll, pitch, yaw = _matrix_to_roll_pitch_yaw(relative_rotation)
    return (x, y, z, roll, pitch, yaw)


def _orientation_base_yaw(xyz, orientation_origin_xy) -> float:
    origin_x, origin_y = orientation_origin_xy
    return _xy_yaw(xyz[X_INDEX] - origin_x, xyz[Y_INDEX] - origin_y)


def _limit_xyz_radius(xyz, max_radius: float):
    if max_radius <= 0.0:
        return tuple(xyz)
    radius = _xyz_radius(xyz)
    if radius <= max_radius or radius < 1e-12:
        return tuple(xyz)
    scale = max_radius / radius
    return tuple(value * scale for value in xyz)


def _xyz_radius(pose_or_xyz) -> float:
    return math.sqrt(
        pose_or_xyz[X_INDEX] ** 2
        + pose_or_xyz[Y_INDEX] ** 2
        + pose_or_xyz[Z_INDEX] ** 2
    )


def _xy_yaw(x: float, y: float) -> float:
    if math.hypot(x, y) < 1e-12:
        return 0.0
    return math.atan2(y, x)


def _rotation_x(angle: float):
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return (
        (1.0, 0.0, 0.0),
        (0.0, cos_value, -sin_value),
        (0.0, sin_value, cos_value),
    )


def _rotation_y(angle: float):
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return (
        (cos_value, 0.0, sin_value),
        (0.0, 1.0, 0.0),
        (-sin_value, 0.0, cos_value),
    )


def _rotation_z(angle: float):
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return (
        (cos_value, -sin_value, 0.0),
        (sin_value, cos_value, 0.0),
        (0.0, 0.0, 1.0),
    )


def _matmul(left, right):
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][col] for inner in range(3))
            for col in range(3)
        )
        for row in range(3)
    )


def _axis_angle_to_matrix(vector):
    angle = math.sqrt(sum(value * value for value in vector))
    if angle < 1e-12:
        return _identity_matrix()
    axis = tuple(value / angle for value in vector)
    x, y, z = axis
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    one_minus_cos = 1.0 - cos_value
    return (
        (
            cos_value + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin_value,
            x * z * one_minus_cos + y * sin_value,
        ),
        (
            y * x * one_minus_cos + z * sin_value,
            cos_value + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin_value,
        ),
        (
            z * x * one_minus_cos - y * sin_value,
            z * y * one_minus_cos + x * sin_value,
            cos_value + z * z * one_minus_cos,
        ),
    )


def _matrix_to_axis_angle(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    angle = math.acos(_clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-12:
        return (0.0, 0.0, 0.0)

    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1e-8:
        return _matrix_to_axis_angle_near_pi(matrix, angle)

    scale = angle / (2.0 * sin_angle)
    return (
        (matrix[2][1] - matrix[1][2]) * scale,
        (matrix[0][2] - matrix[2][0]) * scale,
        (matrix[1][0] - matrix[0][1]) * scale,
    )


def _matrix_to_axis_angle_near_pi(matrix, angle):
    axis = [
        math.sqrt(max(0.0, (matrix[index][index] + 1.0) / 2.0))
        for index in range(3)
    ]
    if matrix[2][1] - matrix[1][2] < 0.0:
        axis[0] = -axis[0]
    if matrix[0][2] - matrix[2][0] < 0.0:
        axis[1] = -axis[1]
    if matrix[1][0] - matrix[0][1] < 0.0:
        axis[2] = -axis[2]
    length = math.sqrt(sum(value * value for value in axis))
    if length < 1e-12:
        return (angle, 0.0, 0.0)
    return tuple(angle * value / length for value in axis)


def _matrix_to_roll_pitch_yaw(matrix):
    pitch = math.asin(_clamp(-matrix[2][0], -1.0, 1.0))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-8:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    return roll, pitch, yaw


def _identity_matrix():
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def return_to_global_initial(
    driver,
    state_controller,
    args,
    orientation_origin_xy=BASE_ORIENTATION_ORIGIN,
):
    arm_positions_current = _try_driver_call(
        driver,
        driver.get_arm_positions,
        "read current arm joint positions",
    )
    if arm_positions_current is None:
        return False

    arm_joint_count = len(arm_positions_current)
    arm_positions = [0.0] * arm_joint_count
    if not _send_global_initial_joint_pose(driver, arm_positions, args):
        return False

    cartesian_pose = _try_driver_call(
        driver,
        driver.get_cartesian_positions,
        "read returned Cartesian pose",
    )
    gripper_position = _try_driver_call(
        driver,
        driver.get_gripper_position,
        "read returned gripper position",
    )
    if cartesian_pose is None or gripper_position is None:
        return False

    state_controller.set_state(
        driver_pose_to_target_pose(cartesian_pose, orientation_origin_xy),
        gripper_position,
    )
    state_controller.print_state(prefix="returned: ")
    return True


def _send_global_initial_joint_pose(driver, arm_positions, args) -> bool:
    try:
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
    except Exception as exc:
        print(f"\nFailed to return to global initial joint pose: {exc}")
        _try_clear_driver_error(driver)
        return False
    return True


def _try_driver_call(driver, callback, action_name: str):
    for attempt in range(2):
        try:
            return callback()
        except Exception as exc:
            if attempt == 0:
                print(f"\nCould not {action_name}; clearing driver error: {exc}")
                _try_clear_driver_error(driver)
                continue
            print(f"\nCould not {action_name}: {exc}")
            return None


def _try_clear_driver_error(driver):
    if not hasattr(driver, "clear_error"):
        return
    try:
        driver.clear_error()
    except Exception as exc:
        print(f"\nCould not clear driver error: {exc}")


def resolve_orientation_origin_xy(args, initial_pose):
    if args.orientation_origin == "base":
        return BASE_ORIENTATION_ORIGIN
    if args.orientation_origin == "initial":
        return (initial_pose[X_INDEX], initial_pose[Y_INDEX])
    return (args.orientation_origin_x, args.orientation_origin_y)


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
    orientation_origin_xy = BASE_ORIENTATION_ORIGIN

    try:
        print("Connecting...")
        driver.configure(model, ee, args.arm_ip, False)

        print("Setting arm to position mode...")
        driver.set_arm_modes(position_mode)
        driver.set_gripper_mode(position_mode)

        q0 = np.array(driver.get_arm_positions(), dtype=float)
        pose0 = np.array(driver.get_cartesian_positions(), dtype=float)
        orientation_origin_xy = resolve_orientation_origin_xy(args, pose0.tolist())
        target_pose0 = driver_pose_to_target_pose(
            pose0.tolist(),
            orientation_origin_xy,
        )
        gripper0 = float(driver.get_gripper_position())
        mover = RobotMover(
            driver,
            cartesian_interp,
            args.max_reach_radius,
            orientation_origin_xy,
        )

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
            initial_pose=target_pose0,
            initial_gripper=gripper0,
            return_pose=target_pose0,
            return_gripper=gripper0,
            gripper_min=args.gripper_min,
            gripper_max=args.gripper_max,
            yaw_follow_xy=args.controller_yaw_follow_xy,
        )

        print()
        input_controller.print_help()
        print(
            f"linear_step={args.linear_step} m, angular_step={args.angular_step} rad, "
            f"gripper_step={args.gripper_step} m"
        )
        print(f"max_reach_radius={args.max_reach_radius} m")
        origin_x, origin_y = orientation_origin_xy
        print(
            "orientation_origin="
            f"({origin_x:.3f}, {origin_y:.3f})"
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
                            orientation_origin_xy,
                        )
                        break

                    if command.reset:
                        print("\nMoving gently to global initial joint pose...")
                        returned = return_to_global_initial(
                            driver,
                            state_controller,
                            args,
                            orientation_origin_xy,
                        )
                        if returned:
                            state_controller.print_state(prefix="reset done: ")
                        continue

                    state_controller.apply(command)
                    state_controller.print_state(
                        prefix=f"{command.action} sens={command.sensitivity:.2f}: "
                    )

                mover.move_to_state(
                    state_controller,
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
                    orientation_origin_xy,
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
    parser.add_argument(
        "--device",
        help="Xbox event device, for example /dev/input/event0.",
    )
    parser.add_argument("--arm-ip", default=ARM_IP)
    parser.add_argument("--linear-step", type=float, default=LINEAR_STEP)
    parser.add_argument("--angular-step", type=float, default=ANGULAR_STEP)
    parser.add_argument("--gripper-step", type=float, default=GRIPPER_STEP)
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=None,
        help="Controller sensitivity override. Defaults to controller/config.yaml.",
    )
    parser.add_argument("--sensitivity-factor", type=float, default=1.25)
    parser.add_argument(
        "--min-sensitivity",
        type=float,
        default=None,
        help="Minimum sensitivity override. Defaults to controller/config.yaml.",
    )
    parser.add_argument(
        "--max-sensitivity",
        type=float,
        default=None,
        help="Maximum sensitivity override. Defaults to controller/config.yaml.",
    )
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=0.044)
    parser.add_argument(
        "--return-gripper",
        type=float,
        default=0.0,
        help="Gripper return target in meters. Defaults to the global initial value.",
    )
    parser.add_argument(
        "--controller-yaw-follow-xy",
        dest="controller_yaw_follow_xy",
        action="store_true",
        help=(
            "Let the controller package add atan2(y, x) to target yaw before "
            "robot_action applies its orientation-origin transform."
        ),
    )
    parser.add_argument(
        "--no-controller-yaw-follow-xy",
        "--no-yaw-follow-xy",
        dest="controller_yaw_follow_xy",
        action="store_false",
        help="Keep target yaw as direct human input in robot_action.",
    )
    parser.set_defaults(controller_yaw_follow_xy=False)
    parser.add_argument("--goal-time", type=float, default=GOAL_TIME)
    parser.add_argument("--reset-goal-time", type=float, default=RESET_GOAL_TIME)
    parser.add_argument("--control-period", type=float, default=CONTROL_PERIOD)
    parser.add_argument(
        "--max-reach-radius",
        type=float,
        default=MAX_REACH_RADIUS,
        help=(
            "Maximum API xyz radius in meters. Target xyz keeps its direction "
            "but is scaled down before sending to the robot."
        ),
    )
    parser.add_argument(
        "--orientation-origin",
        choices=("base", "initial", "point"),
        default="base",
        help=(
            "Origin used to decide the zero yaw direction. 'base' uses the "
            "robot base axis, 'initial' uses the startup gripper xy, and "
            "'point' uses --orientation-origin-x/y."
        ),
    )
    parser.add_argument(
        "--orientation-origin-x",
        type=float,
        default=0.0,
        help="X coordinate for --orientation-origin point.",
    )
    parser.add_argument(
        "--orientation-origin-y",
        type=float,
        default=0.0,
        help="Y coordinate for --orientation-origin point.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
