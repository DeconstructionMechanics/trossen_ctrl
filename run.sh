#!/usr/bin/env bash
# Start Xbox teleoperation with the calibrated workspace in this directory.
# Extra arguments are passed straight through, for example:
#   ./run.sh --session-seconds 300
#   ./run.sh --dry-run --duration 5
# PYTHON and DEVICE can be overridden from the environment.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/jiahao/miniconda3/envs/trossen/bin/python}"
DEVICE="${DEVICE:-/dev/input/by-id/usb-PowerA_Xbox_Series_X_Wired_Controller_Black_00000101A4059B03-event-joystick}"

if [ ! -x "$PYTHON" ]; then
    echo "No Python at $PYTHON. Activate the trossen environment or set PYTHON=..." >&2
    exit 1
fi

if [ ! -e "$DEVICE" ]; then
    echo "No gamepad at $DEVICE; looking for another one." >&2
    DEVICE="$(ls /dev/input/by-id/*event-joystick 2>/dev/null | head -1 || true)"
fi

if [ -z "$DEVICE" ] || [ ! -e "$DEVICE" ]; then
    echo "No gamepad found. Attach the controller, or run with DEVICE=/dev/input/eventN." >&2
    exit 1
fi

# The program reports a missing calibration itself, but only after the arguments
# that would replace it have been ruled out.
case " $* " in
    *" --workspace "*|*" --no-workspace-limits "*|*" --calibrate "*|*" --dry-run "*) ;;
    *)
        if [ ! -f workspace.json ]; then
            echo "workspace.json is missing. Calibrate first:" >&2
            echo "  ./run.sh --calibrate --workspace workspace.json" >&2
            exit 1
        fi
        ;;
esac

exec "$PYTHON" test.py --controller xbox --device "$DEVICE" "$@"
