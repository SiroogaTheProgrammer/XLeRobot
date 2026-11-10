"""Forward / inverse kinematics for the SO-101 arm.

Model conventions (per arm, in that arm's base frame)
-----------------------------------------------------
* Origin at the base plate, directly under the shoulder_pan axis.
* x forward, y left, z up (metres). Angles in degrees.
* ``shoulder_pan``  : yaw of the whole arm, 0 = pointing along +x,
                      positive = towards +y (left).
* ``shoulder_lift`` : upper-arm angle above horizontal, positive = up.
* ``elbow_flex``    : forearm angle relative to the upper arm,
                      0 = straight, negative = elbow bends down.
* ``wrist_flex``    : gripper angle relative to the forearm.
* ``wrist_roll``    : roll about the gripper axis (not used by IK).
* ``pitch``         : resulting gripper angle from horizontal
                      (= shoulder_lift + elbow_flex + wrist_flex).

IMPORTANT: these model zeros must correspond to the arm's lerobot
calibration. If a joint moves the wrong way or is offset on the real
hardware, fix it with JOINT_SIGNS / JOINT_OFFSETS_DEG below rather than
changing the math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from xlerobot import config

# Per-joint corrections applied when converting model angles <-> servo
# commands. Tune on the real hardware after calibration.
JOINT_SIGNS = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": 1.0,
}
JOINT_OFFSETS_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}


class KinematicsError(Exception):
    """Raised when a target is unreachable or violates joint limits."""


@dataclass
class JointAngles:
    """Model-frame joint angles in degrees (see module docstring)."""
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "shoulder_pan": self.shoulder_pan,
            "shoulder_lift": self.shoulder_lift,
            "elbow_flex": self.elbow_flex,
            "wrist_flex": self.wrist_flex,
            "wrist_roll": self.wrist_roll,
        }

    def to_servo(self) -> dict[str, float]:
        """Model angles -> servo command angles (sign/offset corrected)."""
        return {name: JOINT_SIGNS[name] * angle + JOINT_OFFSETS_DEG[name]
                for name, angle in self.as_dict().items()}


def _check_limits(joints: JointAngles) -> None:
    for name, angle in joints.as_dict().items():
        lo, hi = config.ARM_JOINT_LIMITS_DEG[name]
        if not lo <= angle <= hi:
            raise KinematicsError(
                f"{name}={angle:.1f} deg outside limits [{lo}, {hi}]")


def inverse_kinematics(x: float, y: float, z: float,
                       pitch_deg: float = 0.0,
                       wrist_roll_deg: float = 0.0) -> JointAngles:
    """Solves for joint angles placing the gripper TIP at (x, y, z).

    The gripper approaches radially (pointing away from the arm base) at
    ``pitch_deg`` from horizontal (0 = level, negative = tilted down).
    Uses the elbow-up solution. Raises KinematicsError if unreachable.
    """
    l1 = config.ARM_UPPER_ARM_M
    l2 = config.ARM_FOREARM_M
    lg = config.ARM_GRIPPER_M

    pan = math.degrees(math.atan2(y, x))
    r = math.hypot(x, y)
    z_rel = z - config.ARM_SHOULDER_HEIGHT_M

    # Step back from the gripper tip to the wrist along the gripper direction
    pitch = math.radians(pitch_deg)
    rw = r - lg * math.cos(pitch)
    zw = z_rel - lg * math.sin(pitch)

    d_sq = rw * rw + zw * zw
    cos_e = (d_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not -1.0 <= cos_e <= 1.0:
        raise KinematicsError(
            f"target ({x:.3f}, {y:.3f}, {z:.3f}) out of reach "
            f"(wrist distance {math.sqrt(d_sq):.3f} m, "
            f"links {l1:.3f}+{l2:.3f} m)")
    e = math.acos(cos_e)   # interior elbow angle, 0 = straight

    # Elbow-up: shoulder lifted above the straight line to the wrist
    shoulder = math.atan2(zw, rw) + math.atan2(l2 * math.sin(e),
                                               l1 + l2 * math.cos(e))
    elbow = -e                                   # forearm bends down
    wrist = pitch - shoulder - elbow

    joints = JointAngles(
        shoulder_pan=pan,
        shoulder_lift=math.degrees(shoulder),
        elbow_flex=math.degrees(elbow),
        wrist_flex=math.degrees(wrist),
        wrist_roll=wrist_roll_deg,
    )
    _check_limits(joints)
    return joints


def forward_kinematics(joints: JointAngles) -> tuple[float, float, float]:
    """Gripper tip position (x, y, z) for the given model joint angles."""
    l1 = config.ARM_UPPER_ARM_M
    l2 = config.ARM_FOREARM_M
    lg = config.ARM_GRIPPER_M

    q_sh = math.radians(joints.shoulder_lift)
    q_el = math.radians(joints.elbow_flex)
    q_wr = math.radians(joints.wrist_flex)

    r = (l1 * math.cos(q_sh)
         + l2 * math.cos(q_sh + q_el)
         + lg * math.cos(q_sh + q_el + q_wr))
    z = (config.ARM_SHOULDER_HEIGHT_M
         + l1 * math.sin(q_sh)
         + l2 * math.sin(q_sh + q_el)
         + lg * math.sin(q_sh + q_el + q_wr))

    pan = math.radians(joints.shoulder_pan)
    return r * math.cos(pan), r * math.sin(pan), z
