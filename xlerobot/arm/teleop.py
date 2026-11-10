"""Teleoperation: mirror the guiding (leader) arm onto the follower.

Positions are mapped through both arms' calibrated joint ranges
(normalized range -> range), so the arms do not need identical geometry.
Used by setup_arms.py while recording the rack key poses.
"""

from __future__ import annotations

from xlerobot.arm.calibration import ArmCalibration, map_position
from xlerobot.arm.controller import ALL_JOINTS, ArmController


def teleop_step(leader: ArmController, follower: ArmController,
                leader_cal: ArmCalibration,
                follower_cal: ArmCalibration) -> dict[str, float]:
    """One mirror step: read the leader, map and send to the follower.

    Returns the follower-side positions that were commanded.
    """
    leader_pos = leader.positions()
    targets = {joint: map_position(leader_cal, follower_cal, joint,
                                   leader_pos[joint])
               for joint in ALL_JOINTS}
    follower.set_joints(targets)
    return targets
