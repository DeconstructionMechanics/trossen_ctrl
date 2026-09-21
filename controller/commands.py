"""Action directions shared by the keyboard and gamepad input layers.

Each direction is x, y, z, roll, pitch, yaw: the first three are base-frame
translation and the last three are rotation about the current tool axes.
"""

ACTION_DIRECTIONS = {
    "forward": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "leftward": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "rightward": (0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    "upward": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "downward": (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
    "pitchup": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "pitchdown": (0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    "yawleft": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "yawright": (0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    "rollleft": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "rollright": (0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
}

GRIPPER_DIRECTIONS = {
    "gripperopen": 1.0,
    "gripperclose": -1.0,
}
