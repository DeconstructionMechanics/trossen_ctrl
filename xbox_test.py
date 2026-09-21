"""Display normalized hand-controller input without connecting to a robot."""

import time

from controller.input_state import GamepadInput
from robot_action import parse_args, settings_for


def main():
    args = parse_args()
    settings = settings_for(args)
    device = GamepadInput(settings, args.device, args.config)
    print("Input-only monitor: no robot connection. Start or Ctrl-C exits.")
    try:
        while True:
            intent = device.poll(time.monotonic())
            print(intent)
            if "quit" in intent.events:
                break
            time.sleep(settings.period)
    except KeyboardInterrupt:
        pass
    finally:
        device.close()


if __name__ == "__main__":
    main()
