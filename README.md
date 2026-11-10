# XLeRobot

A stationary drink-serving robot running on Ubuntu Linux. It builds a live
"mental image" of its surroundings with an RPLIDAR A1M8, tracks faces with
metric depth on an Orbbec Gemini 335, fuses both sensors to find the
nearest person, and hands out a drink with two SO-101 arms (Feetech
STS3215 servos): ARM1 picks a can from the rack and passes it to ARM2,
which stands near the person and gives it out. A third SO-101 (the
guiding/leader arm) is used only during setup to teach the poses.

## Project layout

```
main.py                        Entry point: sensors + display + serve state machine
find_ports.py                  Detect lidar/arm ports, write settings.py
setup_arms.py                  Calibrate the arms + record the handoff poses
settings.py                    All user-tunable settings
examples/                      Original single-file hardware test scripts
xlerobot/
  config.py                    Internal wiring (re-exports settings.py)
  display.py                   Live camera + lidar-map windows (--show)
  perception/
    lidar_tracker.py           Lidar mental image: classified point cloud + blobs
    depth_camera.py            Face detection/tracking with metric depth
    fusion.py                  Lidar <-> camera cross-verified person target
  arm/
    controller.py              Arm backends: feetech (direct), lerobot, dry-run
    calibration.py             Full range-of-motion capture + range mapping
    teleop.py                  Leader -> follower mirroring
    can_rack.py                Recorded poses, can grid, inventory, 2-arm serve
```

## Hardware mapping (verified)

| Device                       | USB id      | Role                      |
|------------------------------|-------------|---------------------------|
| RPLIDAR A1M8 (CP2102)        | `10c4:ea60` | lidar                     |
| CH343 serial `...5B3D045687` | `1a86:55d3` | ARM1 (rack picker)        |
| CH343 serial `...5B3E091223` | `1a86:55d3` | ARM2 (giver, near person) |
| CH343 serial `...5AAF263132` | `1a86:55d3` | guiding / leader arm      |
| Orbbec Gemini 335            | `2bc5:0800` | depth camera              |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python find_ports.py     # detect devices and configure ports
```

`find_ports.py` probes every serial port for what is actually on it
(servo bus vs lidar), assigns the three arm roles (using saved settings, or
by asking you to wiggle specific arms), and writes stable
`/dev/serial/by-id/` paths into `settings.py`. Re-run it any time the
wiring changes. A port that shows "no response" usually means the servos
have no power or another program holds the port.

If an arm's servos are misconfigured (joints frozen with glowing LEDs,
calibration spans near 358 deg, motors straining against nothing), repair
them with `repair_arm.py`: `--unlock-limits` widens locked angle-limit
registers, `--center` re-zeroes the servos so the pose you hold becomes
the middle of their range (then delete that arm's calibration file and
re-run `setup_arms.py`).

The user must be in the `dialout` group for the serial ports. Note:
`lerobot` is only needed for the calibrated IK-serving path, and it installs
`opencv-python-headless`, which breaks the GUI feeds - keep it in a
separate venv if you need it.

## Running

```bash
.venv/bin/python main.py --show           # live camera + lidar windows
.venv/bin/python main.py                  # headless
.venv/bin/python main.py --backend dry-run  # no arm attached
.venv/bin/python main.py --refill         # after reloading the rack
```

With `--show`, two windows appear: the face-tracking camera feed and the
lidar map (front = up; blue walls, red moving people, yellow static
objects). Press `q` to quit, `h` to trigger a handout manually.
Standby detection is CAMERA-ONLY - the lidar motor stays stopped (the map
shows a "standby" banner) and only spins up during a serve, capped to
`SERVE_LIDAR_RANGE_M`, to track the person while the arms move. Serves
run in a background thread, so both feeds stay live during the motion.
The lidar reconnects and resets itself automatically if its serial stream
is wedged (a red "NO DATA" banner shows on the map until it recovers).

## Settings

Behaviour knobs live in **`settings.py`** (repo root) - edit and restart:

* `SERVE_DISTANCE_M` / `SERVE_MAX_BEARING_DEG` - how close (metres) and at
  what angle off straight-ahead (degrees) a person must be to get a drink
* `SERVE_STABLE_S`, `SERVE_COOLDOWN_S`, `HANDOVER_HOLD_S` - timing
* `CAN_ROWS` / `CAN_COLS` - rack layout; `GRIP_SQUEEZE_DEG` - per-arm extra
  squeeze past the hand-recorded on-can grip reference when clamping
* `STEP_PAUSE_S` - stillness pause between every serve step
* `AIM_GAIN` / `AIM_MAX_PAN_DEG` / `AIM_PAN_SIGN` - how the base servo
  swings the held-out can towards the person's fused lidar+camera bearing
* `CAMERA_WIDTH/HEIGHT/FPS`, `FACE_DETECT_WIDTH` - camera speed vs range
* hardware ports (stable `/dev/serial/by-id/` paths) and file locations
* lidar map size/range for the live display

Internal wiring (geometry, servo IDs, classifier tuning) stays in
`xlerobot/config.py`.

## Arm setup: calibrate + record the handoff poses

Set `CAN_ROWS`/`CAN_COLS` in `settings.py` to match your rack, power all
three arms, then:

```bash
.venv/bin/python setup_arms.py
```

1. **Calibration** - any arm without a saved calibration goes limp and you
   move every joint through its full range of motion; ranges are saved
   under `calibrations/` and reused on later runs (`--recalibrate` to redo).
2. **Gripper references** (hand-set, exact): for each follower only the
   gripper goes limp - you set the jaws fully OPEN, then snugly ON A REAL
   CAN. These two values drive all gripping; pose recordings never touch
   the gripper.
3. **ARM1 poses** (mirrors the guiding arm; every capture is
   settle-averaged after Enter, so hand wobble doesn't matter):
   * START pose - rest/return position
   * FIRST and LAST can of the TOP row (open jaws centred around the can)
   * FIRST and LAST can of the BOTTOM row
   * HANDOFF pose - grip a REAL can and hold it where ARM2 will take it
   ARM1 stays parked there holding the can.
4. **ARM2 poses**: START, RECEIVE (open jaws centred around the can ARM1
   is holding), FINAL giving-out pose.
5. Both arms return to start, the inventory resets to a full rack, and you
   can run a full **test handoff** (`setup_arms.py --test` any time).

Every other can position is interpolated from the four corners using the
recorded grid. Serving a can (the can is never free-floating - ARM2 clamps
it BEFORE ARM1 releases):

    ARM1: jaws to OPEN ref -> can -> clamp to GRIP ref + GRIP_SQUEEZE_DEG
          -> handoff pose
    ARM2: jaws to OPEN ref -> receive pose -> clamp  (both hold it)
    ARM1: release -> back to start
    ARM2: carry to final (base aimed at the person's bearing) -> wait
          RELEASE_DELAY_S -> release -> hold HANDOVER_HOLD_S -> back to start

Each handed-out can is remembered in `can_state.json` (survives restarts);
the next serve takes the next can, top row first, left to right. After
reloading the rack run `main.py --refill`.