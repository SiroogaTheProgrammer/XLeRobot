"""High-level arm motion sequences for serving drinks.

Builds on :mod:`xlerobot.arm.kinematics` (where the end-effector should be)
and :mod:`xlerobot.arm.controller` (how to get the joints there). Every
sequence starts and ends in a safe, known pose so sequences can be chained.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

from xlerobot import config
from xlerobot.arm.controller import ArmController
from xlerobot.arm.kinematics import JointAngles, inverse_kinematics

log = logging.getLogger(__name__)


def load_handout_gesture() -> tuple[dict[str, float], float]:
    """Returns (extend_deltas, gripper_open_delta) for the handout gesture.

    Prefers a gesture recorded with configure_arm.py (HANDOUT_GESTURE_FILE);
    falls back to the defaults in config.py.
    """
    path = config.HANDOUT_GESTURE_FILE
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            deltas = {str(k): float(v)
                      for k, v in data["extend_deltas"].items()}
            grip = float(data.get("gripper_open_delta",
                                  config.HANDOUT_GRIPPER_OPEN_DELTA))
            log.info("Loaded recorded handout gesture from %s", path)
            return deltas, grip
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            log.exception("Bad gesture file %s, using config defaults", path)
    return dict(config.HANDOUT_EXTEND_DELTAS), config.HANDOUT_GRIPPER_OPEN_DELTA

# A compact, safe resting pose (arm folded up, gripper level).
HOME_POSE = JointAngles(
    shoulder_pan=0.0,
    shoulder_lift=80.0,
    elbow_flex=-95.0,
    wrist_flex=15.0,
    wrist_roll=0.0,
)


class ArmMotions:
    """Coordinates the dual arms through complete serve sequences."""

    def __init__(self, arms: dict[str, ArmController]):
        self.arms = arms

    # -- basic poses ---------------------------------------------------------

    def home(self, side: str):
        arm = self.arms[side]
        if arm.backend == "feetech":
            # No calibrated zero on this backend: the current pose IS home.
            log.info("[%s arm] feetech backend: holding current pose as home", side)
            return
        arm.move_to(HOME_POSE)

    def home_all(self):
        for side in self.arms:
            self.home(side)

    # -- picking -------------------------------------------------------------

    def pick_bottle(self, slot: config.BottleSlot) -> bool:
        """Grabs the bottle in ``slot`` with that slot's arm using IK.

        Sequence: open gripper -> hover above slot -> descend -> close ->
        lift clear. Returns False (arm back home) if the slot is unreachable.
        """
        arm = self.arms[slot.arm]
        up = config.PICK_APPROACH_UP_M
        try:
            hover = inverse_kinematics(slot.x, slot.y, slot.z + up,
                                       pitch_deg=config.PICK_PITCH_DEG)
            grasp = inverse_kinematics(slot.x, slot.y, slot.z,
                                       pitch_deg=config.PICK_PITCH_DEG)
        except Exception:
            log.exception("Slot %s unreachable for %s arm", slot.slot_id, slot.arm)
            return False

        log.info("[%s arm] picking bottle from slot %s", slot.arm, slot.slot_id)
        arm.open_gripper()
        arm.move_to(hover)
        arm.move_to(grasp)
        arm.close_gripper()
        arm.move_to(hover)      # lift straight up, clear of the rack
        self.home(slot.arm)     # carry pose
        return True

    # -- handover ------------------------------------------------------------

    def _handover_target(self, side: str, bearing_deg: float,
                         distance_m: float) -> tuple[float, float, float]:
        """Point (arm frame) towards the person, clamped to the arm's reach."""
        # Person position in the robot frame (bearing positive = left = +y)
        b = math.radians(bearing_deg)
        px = distance_m * math.cos(b)
        py = distance_m * math.sin(b)

        ax, ay, _az = config.ARM_BASE_OFFSETS[side]
        dx, dy = px - ax, py - ay
        horiz = math.hypot(dx, dy)

        hz = config.HANDOVER_HEIGHT_M
        dz = hz - config.ARM_SHOULDER_HEIGHT_M
        max_horiz = math.sqrt(
            max((config.ARM_MAX_REACH_M * config.HANDOVER_REACH_FRAC) ** 2 - dz * dz,
                0.01))
        scale = min(1.0, max_horiz / max(horiz, 1e-6))
        return dx * scale, dy * scale, hz

    def extend_to_person(self, side: str, bearing_deg: float,
                         distance_m: float) -> bool:
        """Extends the (loaded) gripper towards the person's position."""
        x, y, z = self._handover_target(side, bearing_deg, distance_m)
        try:
            target = inverse_kinematics(x, y, z, pitch_deg=0.0)
        except Exception:
            log.exception("Handover target unreachable (bearing %.0f deg, %.2f m)",
                          bearing_deg, distance_m)
            return False
        log.info("[%s arm] extending to person at %.0f deg / %.2f m "
                 "(target %.2f, %.2f, %.2f)", side, bearing_deg, distance_m, x, y, z)
        arm = self.arms[side]
        arm.move_to(target)
        return True

    def release_and_retract(self, side: str,
                            hold_s: float = config.HANDOVER_HOLD_S):
        """Holds the bottle out, opens the gripper, then returns home."""
        arm = self.arms[side]
        log.info("[%s arm] holding bottle out for %.1f s", side, hold_s)
        time.sleep(hold_s)
        arm.open_gripper()
        time.sleep(0.5)
        self.home(side)

    # -- general handout (calibration-free) ------------------------------------

    def general_handout(self, side: str,
                        hold_s: float = config.HANDOVER_HOLD_S):
        """Generic "here's your drink" gesture, relative to the current pose.

        Extends the arm forward, opens the gripper to offer the drink, holds,
        then reverses exactly back to where it started. Because every move is
        a delta from the live servo positions, this works on the uncalibrated
        feetech backend - start the arm in its folded rest pose.
        """
        arm = self.arms[side]
        log.info("[%s arm] general handout gesture", side)
        extend_deltas, grip_delta = load_handout_gesture()
        arm.move_relative(extend_deltas)
        arm.move_relative({"gripper": grip_delta}, steps=12)
        log.info("[%s arm] offering drink for %.1f s", side, hold_s)
        time.sleep(hold_s)
        arm.move_relative({"gripper": -grip_delta}, steps=12)
        arm.move_relative({name: -delta for name, delta
                           in extend_deltas.items()})
        log.info("[%s arm] handout complete, back at rest", side)

    # -- full serve ----------------------------------------------------------

    def serve(self, slot: config.BottleSlot, bearing_deg: float,
              distance_m: float) -> bool:
        """Complete sequence: pick from slot, extend to person, release."""
        if not self.pick_bottle(slot):
            return False
        if not self.extend_to_person(slot.arm, bearing_deg, distance_m):
            self.home(slot.arm)
            return False
        self.release_and_retract(slot.arm)
        return True
