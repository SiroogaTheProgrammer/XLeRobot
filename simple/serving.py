"""Pose storage + the one serve motion of the simple setup.

Poses (recorded with setup_arm.py by posing the arm BY HAND):
    start  - rest/return pose
    can    - jaws around the one and only can spot
    final  - where the can is held out to the person
plus the hand-set gripper references {"open", "grip"}.

Serve: open -> can -> clamp (grip + squeeze) -> final -> wait -> release
-> hold -> back to start. Pose moves never command the gripper.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simple_settings as s  # noqa: E402
from xlerobot.arm.controller import ArmController  # noqa: E402

log = logging.getLogger(__name__)

POSE_NAMES = ("start", "can", "final")
Pose = dict[str, float]


def save_poses(poses: dict):
    data = {name: {j: round(float(v), 2) for j, v in pose.items()}
            for name, pose in poses.items()}
    tmp = s.POSES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, s.POSES_FILE)
    log.info("Saved simple poses to %s", s.POSES_FILE)


def load_poses() -> dict | None:
    if not os.path.exists(s.POSES_FILE):
        return None
    try:
        with open(s.POSES_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        poses = {name: {j: float(v) for j, v in data[name].items()}
                 for name in POSE_NAMES}
        poses["gripper"] = {k: float(data["gripper"][k])
                            for k in ("open", "grip")}
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        log.exception("Bad poses file %s - re-run setup_arm.py", s.POSES_FILE)
        return None
    return poses


def serve_once(arm: ArmController, poses: dict):
    """The complete give-out-one-can motion."""
    refs = poses["gripper"]
    open_deg, grip_deg = refs["open"], refs["grip"]
    direction = 1.0 if grip_deg >= open_deg else -1.0
    clamp_deg = grip_deg + direction * s.GRIP_SQUEEZE_DEG

    def nog(pose: Pose) -> Pose:
        return {j: v for j, v in pose.items() if j != "gripper"}

    def step(targets: Pose, steps: int | None = None):
        arm.move_joints(targets, **({} if steps is None else {"steps": steps}))
        time.sleep(s.STEP_PAUSE_S)

    log.info("Serving the can (gripper open=%.1f clamp=%.1f)",
             open_deg, clamp_deg)
    step({"gripper": open_deg}, 12)          # open jaws
    step(nog(poses["can"]))                  # reach the can
    step({"gripper": clamp_deg}, 12)         # grab it
    step(nog(poses["final"]))                # bring it to the person
    log.info("At the final pose - waiting %.1f s before releasing",
             s.RELEASE_DELAY_S)
    time.sleep(s.RELEASE_DELAY_S)
    step({"gripper": open_deg}, 12)          # let go
    log.info("Can offered - holding for %.1f s", s.HANDOVER_HOLD_S)
    time.sleep(s.HANDOVER_HOLD_S)
    step(nog(poses["start"]))                # retreat
    step({"gripper": grip_deg}, 12)          # rest gently closed
    log.info("Back at start - place a new can on the spot")
