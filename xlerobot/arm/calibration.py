"""Arm joint-range calibration ("full range of motion").

An arm is calibrated by letting the user move every joint through its full
range while we record the min/max position of each servo. Calibrations are
saved per arm under settings.CALIBRATION_DIR and used to:

* map leader-arm positions onto the follower arm during teleoperation
  (normalized range -> range, like lerobot does), and
* clamp every commanded position to the follower's safe range.
"""

from __future__ import annotations

import json
import logging
import os
import select
import sys
import time
from dataclasses import dataclass

from xlerobot import config
from xlerobot.arm.controller import ALL_JOINTS, ArmController

log = logging.getLogger(__name__)


def _enter_pressed() -> bool:
    """Non-blocking check for Enter on stdin (Linux)."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


@dataclass
class JointRange:
    min_deg: float
    max_deg: float

    @property
    def span(self) -> float:
        return self.max_deg - self.min_deg


class ArmCalibration:
    """Per-joint ranges of one arm, with range-mapping helpers."""

    def __init__(self, name: str, ranges: dict[str, JointRange]):
        self.name = name
        self.ranges = ranges

    # -- persistence ---------------------------------------------------------

    @staticmethod
    def path_for(name: str) -> str:
        return os.path.join(config.CALIBRATION_DIR, f"{name}.json")

    def save(self):
        os.makedirs(config.CALIBRATION_DIR, exist_ok=True)
        data = {joint: {"min_deg": r.min_deg, "max_deg": r.max_deg}
                for joint, r in self.ranges.items()}
        with open(self.path_for(self.name), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        log.info("Saved calibration for %s to %s",
                 self.name, self.path_for(self.name))

    @classmethod
    def load(cls, name: str) -> "ArmCalibration | None":
        path = cls.path_for(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            ranges = {joint: JointRange(float(v["min_deg"]), float(v["max_deg"]))
                      for joint, v in data.items()}
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            log.exception("Bad calibration file %s", path)
            return None
        return cls(name, ranges)

    # -- mapping -------------------------------------------------------------

    def clamp(self, joint: str, deg: float) -> float:
        r = self.ranges.get(joint)
        if r is None:
            return deg
        return min(max(deg, r.min_deg), r.max_deg)

    def normalize(self, joint: str, deg: float) -> float:
        """Position -> 0..1 within this arm's calibrated range."""
        r = self.ranges.get(joint)
        if r is None or r.span < 1e-6:
            return 0.5
        return min(max((deg - r.min_deg) / r.span, 0.0), 1.0)

    def denormalize(self, joint: str, norm: float) -> float:
        """0..1 -> position within this arm's calibrated range."""
        r = self.ranges[joint]
        return r.min_deg + min(max(norm, 0.0), 1.0) * r.span


def map_position(leader_cal: ArmCalibration, follower_cal: ArmCalibration,
                 joint: str, leader_deg: float) -> float:
    """Maps a leader joint position onto the follower's calibrated range."""
    return follower_cal.denormalize(
        joint, leader_cal.normalize(joint, leader_deg))


def capture_ranges(arm: ArmController, name: str) -> ArmCalibration:
    """Interactive full-range capture for one arm.

    Torque is released so the user can move every joint by hand; min/max per
    joint are tracked live until Enter is pressed. Torque is re-enabled
    afterwards.
    """
    arm.set_torque(False)
    print(f"\n  [{name}] torque OFF. Slowly move EVERY joint (including the "
          "gripper)\n  through its FULL range of motion. Press Enter when done.")
    pos = arm.positions()
    ranges = {j: JointRange(pos[j], pos[j]) for j in ALL_JOINTS}
    try:
        while not _enter_pressed():
            pos = arm.positions()
            for joint in ALL_JOINTS:
                r = ranges[joint]
                r.min_deg = min(r.min_deg, pos[joint])
                r.max_deg = max(r.max_deg, pos[joint])
            line = "  ".join(f"{j}:{ranges[j].span:5.1f}" for j in ALL_JOINTS)
            print(f"\r  spans(deg)  {line}   ", end="", flush=True)
            time.sleep(1.0 / config.TELEOP_HZ)
    finally:
        print()
        arm.set_torque(True)
    narrow = [j for j in ALL_JOINTS if ranges[j].span < 10.0]
    if narrow:
        print(f"  ! joints barely moved (span < 10 deg): {', '.join(narrow)}\n"
              "    re-run calibration if that was a mistake")
    wrapped = [j for j in ALL_JOINTS if ranges[j].span > 300.0]
    if wrapped:
        print(f"  ! IMPOSSIBLE spans (> 300 deg): {', '.join(wrapped)}\n"
              "    these joints cross the servo 0/4095 tick wrap. Re-zero the\n"
              f"    arm first:  python repair_arm.py --arm {name.split('_')[0]} --center\n"
              "    then recalibrate.")
    cal = ArmCalibration(name, ranges)
    cal.save()
    return cal
