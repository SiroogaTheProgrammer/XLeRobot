# Simple setup: one arm, one camera, one can

A minimal drink server living entirely in this folder, reusing the main
project's modules (camera face tracking, arm control, pose capture). No
lidar, no guiding arm, no rack - just one can on a fixed spot.

## Train (arm connected, a can on its spot)

```bash
.venv/bin/python simple/setup_arm.py
```

1. **Gripper references** - only the gripper goes limp; set the jaws
   fully OPEN by hand, then snugly ON THE CAN.
2. **Poses** - the whole arm goes limp (hold it!); pose it by hand and
   press Enter for each: START (rest), CAN (open jaws around the can),
   FINAL (held out to the person).
3. Optional test serve; re-test any time with `setup_arm.py --test`.

Saved to `simple_poses.json` in this folder.

## Run

```bash
.venv/bin/python simple/main.py --show
```

When a face stays within `SERVE_DISTANCE_M` and `+/-SERVE_MAX_BEARING_DEG`
for `SERVE_STABLE_S` (see `simple_settings.py`), the arm serves:
open -> can -> clamp -> final -> wait RELEASE_DELAY_S -> release -> hold
-> back to start. One can at a time: after the cooldown, put a new can on
the spot and it serves again. `q` quits, `h` serves manually.

The arm port defaults to the main project's ARM1; edit `ARM_PORT` in
`simple_settings.py` to change it.
