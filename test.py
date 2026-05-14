"""Backward-compatible entry point for running the Trossen arm controller."""

from robot_action import parse_args, run


if __name__ == "__main__":
    run(parse_args())
