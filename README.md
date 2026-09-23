# Trossen Arm Teleoperation

SDK position control for the WidowX AI follower, with Xbox or keyboard input.
Translation uses base axes; rotation uses the current tool axes. There is no
automatic yaw-follow transform or automatic IK recovery.

## Setup

```bash
conda activate trossen
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python test.py --controller xbox --dry-run --duration 5
```

The dry run uses scripted input and a fake command sink, without opening a
controller or robot connection. It verifies the runtime and logging, not IK,
physics, network timing, or stopping distance.

For WSL USB forwarding, use an administrator PowerShell:

```powershell
usbipd list
usbipd bind --busid x-x
usbipd attach --wsl --busid x-x
```

Use a stable `/dev/input/by-id/*event-joystick` path when available. Input-only
inspection (no robot connection):

```bash
python xbox_test.py --device /dev/input/event9
```

## Workspace Calibration

Real motion requires a workspace file matching the arm IP and model:

```bash
python test.py --calibrate --arm-ip 192.168.1.3 --workspace workspace.json
```

This command connects for state reads only: it never changes modes, clears
errors, or commands motion. Position the arm using your existing positioning
method, then press Enter at each requested boundary. Do not run two motion
controllers concurrently. Table Z can be entered as a measured value or sampled
at the TCP; account for the actual TCP/tool definition before sampling it.

Record the table height and each X/Y/Z minimum and maximum in meters in the base
frame. Review the displayed values and type `SAVE`. Existing files are never
overwritten; use a new filename when recalibrating. Runtime adds the table
clearance from `controller/config.yaml`, not the one stored in the calibration
file: the stored value only records what was assumed when the file was written,
and a mismatch between the two is printed and logged at startup. A startup TCP outside the effective box is
reported as a warning rather than refused; until the TCP is back inside, motion
that would push it further out is dropped axis by axis while inward and
along-boundary motion still works. This checks the TCP only, not the swept
volume of every link or payload.
Joint-space return paths can leave the box between samples; use return only in
a cleared workspace and validate its path during commissioning.

## Run

```bash
./run.sh                        # Xbox teleoperation with ./workspace.json
./run.sh --session-seconds 300  # extra arguments are passed through
python test.py --controller xbox --device /dev/input/event9 --workspace workspace.json
python test.py --controller keyboard --workspace workspace.json
```

`run.sh` picks the gamepad by its stable `/dev/input/by-id` path, falls back to
the first attached joystick, and refuses to start without a calibration file.
Set `PYTHON=` or `DEVICE=` in the environment to override either choice.

For a short supervised hand-control trial without a measured workspace file,
use `--no-workspace-limits --session-seconds 120`. The session starts in fine
speed level; the SDK joint limits, the speed-scaled target-lead cap, feedback
checks, fault pause, and diagnostic logging still apply. This skips only the
software XYZ box and table-height limit. Start/ Ctrl-C can end it earlier. Do not use
the trial setting as a substitute for measuring the physical work area.

The program asks for Enter before enabling position mode, then waits for neutral
controls. Missing calibration is rejected before connecting. An input device
that is absent or disconnected pauses control; reconnect, release all motion
controls, then press A. An initial successful attachment needs only neutral input.
A control already held when the pad is attached, or when a dropped event frame
forces a resync, counts as having no press in this session: the return button
must be released and pressed again before its hold can trigger a return.

| Xbox input | Action |
| --- | --- |
| Left stick | Base X/Y translation |
| Right stick | Tool yaw / pitch |
| LT / RT | Down / up, proportional to trigger travel |
| LB / RB | Tool roll |
| D-pad left / right | Close / open gripper |
| D-pad up / down | Increase / decrease speed level |
| B / A | Pause / resume when neutral |
| X | Start/stop dataset episode (when a recorder is attached) |
| Y | Accept reviewed dataset episode |
| Hold View/Back for 1 second | Return to the configured home joints, preserving gripper opening |
| Start or Ctrl-C | Stop and exit, without returning or closing |

Grip force comes from `gripper_squeeze`: the closing command may run past
measured position by that distance, and in position control that error is the
force. Releasing the D-pad keeps the commanded opening rather than
resynchronising to feedback, so a grip is not given away, and a pause holds the
commanded opening for the same reason. Raise `gripper_squeeze` for a firmer
grip and lower it for a gentler one. X and Y publish recorder events and do not
affect standalone teleoperation.

Keyboard motion bindings are in `controller/config.yaml`; `p` pauses, `r`
resumes, `c` returns, and Esc/Ctrl-C exits. Terminal key release is inferred from
key-repeat timeout, so keyboard release is less precise than gamepad release.

Switching into position mode blocks while the arm settles, so the input backlog
that builds up during it is absorbed and resynchronised from the device rather
than reported as input loss; without that, every resume was immediately undone
by the dropped-event frame it had caused.

Return goes to the `home` joint angles in `controller/config.yaml`, which default
to all zeros, the arm's own neutral configuration; the angles are checked against
the SDK joint limits before the session starts. Without that section the target
is wherever the arm was when position control was enabled, which drifts lower
each session as the brakes let the arm settle. Return is asynchronous and
cancellable by B, exit, or motion input. Cancellation does not execute that
motion input and leaves control waiting for neutral, so releasing the controls
resumes; a completed return pauses instead and needs A.
SDK errors latch a fault. A requests one clear-error/mode recovery attempt only
with neutral controls, followed by another neutral observation before movement.
Recovery is offered before the cycle-time and workspace checks, so a pause or
fault raised at a boundary can always be cleared without restarting.
Lost/stale robot feedback terminates the session after an idle attempt.

## Configuration and Diagnostics

The `teleop` section in `controller/config.yaml` controls speeds, deadzones,
acceleration, timing, lead limits, gripper bounds and clearance. Defaults:

- 50 Hz control, 60 ms SDK goal time, 3 trajectory feasibility samples.
- Fine/normal/fast translation: 2/6/12 cm/s; rotation: 6/18/36 degrees/s.
- Gripper: 2 cm/s, may close 4 mm past feedback for grip force, and is clipped
  to both configured and SDK position limits.
- Target lead: at least 5 mm and 3 degrees, otherwise the selected speed times
  `lead_time` (80 ms), which is also the approximate stopping distance: 9.6 mm
  and 4.8 degrees at the fast level. No backlog at workspace bounds.
- Commanded velocity is sent as an SDK goal feedforward velocity so the servo
  does not plan a stop at every 20 ms waypoint; set `feedforward: 0` to disable.
- A cycle longer than 100 ms pauses without integrating the missed time.
- `clearance` (5 mm) above the calibrated table height sets the floor.
- `sag_allowance` (30 mm) is how close above the table a session may end before
  it warns; the arm settles downward when the brakes engage, so a session that
  ends near the table presses the tool into it. Measured sag on a WidowX AI
  follower ranged from 4 mm near the table to 22 mm with the arm extended at
  z = 0.20 m, so it depends on pose. `sag_settle` (0.5 s) is the wait before
  measuring it.

`--config` selects another config file; the parser is PyYAML, and unknown
sections or unknown `teleop` keys are rejected rather than ignored. Key bindings
are quoted so YAML does not read `on` or `no` as booleans. `--goal-time`,
`--control-period`, `--reset-goal-time`, `--gripper-min` and `--gripper-max`
override single settings for one run; everything else is edited in the config
file. The earlier step/sensitivity and yaw-follow flags, the `sensitivity`
config section and the `MotionCommand` input layer they fed have been removed.

Every shutdown measures the braking sag: the TCP height is sampled before the
idle command and again after `sag_settle`, printed, and recorded as an
`idle_sag` event. A failed connection is reported with its likely causes,
because the SDK message is only `Resource temporarily unavailable`. A second
read-only client connects without trouble; the failure was observed when
connecting while another process was actively controlling the arm.

JSONL logs default to `logs/teleop-<timestamp>.jsonl`; `--log` selects a new path.
Startup records settings, SDK version, limits and calibration. Each cycle records
input, timing, measured state, targets, sent commands, clipping reasons and state.
Exceptions include tracebacks. A full queue or write error stops the session
after a hold/idle attempt. Files are exclusive-create and never overwritten.

The repository is also installable as `trossen-ctrl`. `ControlSession` is the
shared 50 Hz owner used by both the standalone CLI and the LeRobot recorder; it
publishes ordered, timestamped `ControlTransition` objects without putting
camera or dataset I/O on the control thread.

SDK IK failures are logged without guessing whether the cause is endpoint,
path, singularity, or joint limits. This version does not search alternative
orientations or silently switch interpolation spaces. Network or driver failure
can prevent a stop command from reaching the arm; failed stop attempts are
reported explicitly. SDK calls themselves may block beyond the loop period.

## Hardware Acceptance (Not Yet Performed)

After calibration, start in fine mode and verify translation, tool rotation,
gripper, pause/resume and cancellable return. Repeat each chosen reachable path
10 times; compare the same previously problematic paths with historical logs.
Require no unexpected rotation, autonomous fault restart or SDK errors on the
chosen reachable paths. Measure release-to-hold timing and overshoot: the initial
normal-speed targets are one control cycle to issue hold, at most the configured
lead for that level (5 mm and 3 degrees at fine/normal, 9.6 mm and 3 degrees at
fast) of additional TCP motion. Also verify at the fast level that motion stops
when the stream stops, since goal feedforward velocities are now sent. Tune goal time/speed and rerun affected cases
if those targets are not met. These are acceptance targets, not verified claims.
