#!/usr/bin/env python3
"""Arm setup for the two-arm handoff: calibrate all three arms, then record
the key poses of both followers by guiding the leader arm.

Roles:
    LEADER - the guiding arm you move by hand (never moves on its own)
    ARM1   - stands at the rack, picks cans
    ARM2   - stands near the person, receives the can from ARM1, hands it over

Workflow (run from the repo root):

    .venv/bin/python setup_arms.py

    1. All three arms are connected (ports from find_ports.py).
    2. CALIBRATION - any arm without a saved calibration is calibrated by
       moving its joints through their full range (--recalibrate redoes all).
    3. GRIPPER REFERENCES (hand-set, exact): for each follower, only the
       gripper goes limp and you physically set the jaws fully OPEN, then
       snugly ON A REAL CAN. These two values drive all gripping - pose
       recordings never touch the gripper.
    4. ARM1 POSES (follower mirrors the guiding arm, Enter records; every
       pose is settle-averaged for precision):
         a. START pose
         b. 4 corner cans      (open jaws centred around each can)
         c. HANDOFF pose       (grip a real can and hold it out to ARM2)
       ARM1 then stays parked at the handoff holding the can.
    5. ARM2 POSES:
         d. START pose
         e. RECEIVE pose       (open jaws centred around the can ARM1 holds)
         f. FINAL giving-out pose
    6. Both arms return to their start poses, the can inventory is reset,
       and you can run a full test handoff.

Serve order guarantees the can is never dropped: ARM2 clamps it BEFORE
ARM1 releases.

Options:
    --recalibrate   redo calibration even if saved files exist
    --test          run one full two-arm serve with the saved poses, then
                    exit (does not mark the can as used - put it back)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from xlerobot import config
from xlerobot.arm.calibration import (
    ArmCalibration,
    _enter_pressed,
    capture_ranges,
    map_position,
)
from xlerobot.arm.can_rack import ArmPoses, CanInventory, serve_can
from xlerobot.arm.controller import ALL_JOINTS, ArmController
from xlerobot.arm.teleop import teleop_step

log = logging.getLogger("setup_arms")


def connect_arm_interactive(name: str, port: str) -> ArmController | None:
    """Connects with a fix-and-retry prompt (flaky servo power, cables...)."""
    while True:
        try:
            arm = ArmController(name, port, name, backend="feetech")
            arm.connect()
            return arm
        except Exception as exc:
            print(f"  ! {name} arm not reachable on {port}:\n    {exc}")
            answer = input("    check USB + SERVO POWER and press Enter to "
                           "retry (q aborts) ")
            if answer.strip().lower().startswith("q"):
                return None


def get_calibration(arm: ArmController, name: str,
                    recalibrate: bool) -> ArmCalibration:
    cal = None if recalibrate else ArmCalibration.load(name)
    if cal is not None:
        print(f"  [{name}] using saved calibration "
              f"({ArmCalibration.path_for(name)})")
        return cal
    return capture_ranges(arm, name)


def start_mirroring(leader: ArmController, follower: ArmController,
                    leader_cal: ArmCalibration,
                    follower_cal: ArmCalibration):
    """Puts the leader limp and eases the follower onto its mapped pose."""
    print(f"\n  {follower.side.upper()} will now mirror the guiding arm. It "
          "first moves to match\n  the guiding arm's current pose - keep the "
          "area clear!")
    input("  press Enter when ready... ")
    leader.set_torque(False)
    leader_pos = leader.positions()
    eased = {j: map_position(leader_cal, follower_cal, j, leader_pos[j])
             for j in ALL_JOINTS}
    follower.move_joints(eased)


def settle_read(arm: ArmController, settle_s: float = 0.4,
                samples: int = 5) -> dict[str, float]:
    """Precise pose capture: wait for the servos to settle, then average
    several reads so the recorded values are repeatable to ~0.1 deg."""
    time.sleep(settle_s)
    acc = {j: 0.0 for j in ALL_JOINTS}
    for _ in range(samples):
        pos = arm.positions()
        for j in ALL_JOINTS:
            acc[j] += pos[j]
        time.sleep(0.05)
    return {j: round(acc[j] / samples, 2) for j in ALL_JOINTS}


def capture_pose(prompt: str, leader: ArmController, follower: ArmController,
                 leader_cal: ArmCalibration,
                 follower_cal: ArmCalibration) -> dict[str, float]:
    """Mirrors the leader until Enter, then captures the follower pose.

    After Enter the mirroring stops (the follower holds and settles) and
    the pose is averaged over several reads - so it does not matter if your
    hand wobbles the guiding arm while reaching for the keyboard.
    """
    print(f"\n  >>> {prompt}")
    print("      (guide the arm, press Enter to record this pose)")
    period = 1.0 / config.TELEOP_HZ
    while not _enter_pressed():
        teleop_step(leader, follower, leader_cal, follower_cal)
        time.sleep(period)
    pose = settle_read(follower)
    print("      recorded: "
          + "  ".join(f"{j}={pose[j]:+.1f}" for j in ALL_JOINTS))
    return pose


def capture_gripper_refs(arm: ArmController, label: str) -> dict[str, float]:
    """Hand-set gripper references: exact, repeatable, teleop-free.

    Only the gripper servo is released; you physically place the jaws.
    This avoids the classic failure where releasing the guiding arm's
    trigger to press Enter records a garbage gripper value.
    """
    print(f"\n  {label} GRIPPER setup (only the gripper goes limp - set the "
          "jaws BY HAND)")
    while True:
        arm.set_torque(False, joints=("gripper",))
        input("  1) Open the jaws FULLY by hand, then press Enter... ")
        open_deg = settle_read(arm, settle_s=0.1)["gripper"]
        input("  2) Put a REAL CAN between the jaws and press them snugly "
              "onto it,\n     then press Enter... ")
        grip_deg = settle_read(arm, settle_s=0.1)["gripper"]
        arm.set_torque(True, joints=("gripper",))
        print(f"     open={open_deg:+.1f} deg, on-can grip={grip_deg:+.1f} deg "
              f"(travel {abs(open_deg - grip_deg):.1f} deg)")
        if abs(open_deg - grip_deg) >= 10.0:
            return {"open": open_deg, "grip": grip_deg}
        print("  ! open and grip are less than 10 deg apart - that cannot be "
              "right, try again")


def record_all_poses(leader: ArmController, arm1: ArmController,
                     arm2: ArmController, leader_cal: ArmCalibration,
                     cal1: ArmCalibration, cal2: ArmCalibration) -> ArmPoses:
    rows, cols = config.CAN_ROWS, config.CAN_COLS

    # ---- gripper references (hand-set, before any teleop) -------------------
    print("\nSTEP 3 - gripper references")
    arm1_grip = capture_gripper_refs(arm1, "ARM1")
    arm2_grip = capture_gripper_refs(arm2, "ARM2")

    # ---- ARM1: rack picker -------------------------------------------------
    print(f"\nSTEP 4 - ARM1 poses ({rows}x{cols} rack + handoff)")
    print("  NOTE: the gripper values of these poses are IGNORED during "
          "serving -\n  only the arm position matters. Use the leader's "
          "trigger freely.")
    start_mirroring(leader, arm1, leader_cal, cal1)
    cap1 = lambda p: capture_pose(p, leader, arm1, leader_cal, cal1)  # noqa: E731

    arm1_start = cap1("ARM1 START pose - rest/return position.")
    corners: dict[str, dict[str, float]] = {}
    corners["top_first"] = cap1(
        "FIRST can of the TOP row - open jaws centred around the can.")
    corners["top_last"] = (corners["top_first"] if cols == 1 else cap1(
        "LAST can of the TOP row - open jaws centred around the can."))
    if rows == 1:
        corners["bottom_first"] = corners["top_first"]
        corners["bottom_last"] = corners["top_last"]
    else:
        corners["bottom_first"] = cap1(
            "FIRST can of the BOTTOM row - open jaws centred around the can.")
        corners["bottom_last"] = (corners["bottom_first"] if cols == 1
                                  else cap1(
            "LAST can of the BOTTOM row - open jaws centred around the can."))
    handoff = cap1(
        "HANDOFF pose - GRIP A REAL CAN and hold it where ARM2 will take it.")

    print("\n  ARM1 stays parked at the handoff, holding the can.")

    # ---- ARM2: giver -------------------------------------------------------
    print("\nSTEP 5 - ARM2 poses (receive + final)")
    start_mirroring(leader, arm2, leader_cal, cal2)
    cap2 = lambda p: capture_pose(p, leader, arm2, leader_cal, cal2)  # noqa: E731

    arm2_start = cap2("ARM2 START pose - rest/return position.")
    receive = cap2(
        "RECEIVE pose - open jaws centred around the can that ARM1 is "
        "holding.")
    final = cap2("FINAL pose - where the can is held out to the person.")

    poses = ArmPoses(rows=rows, cols=cols,
                     arm1_start=arm1_start, corners=corners, handoff=handoff,
                     arm2_start=arm2_start, receive=receive, final=final,
                     arm1_grip=arm1_grip, arm2_grip=arm2_grip)
    poses.save()

    # ---- park both arms ----------------------------------------------------
    input("\n  Take the can OUT of ARM1's gripper, then press Enter to send "
          "both arms\n  to their start poses... ")
    arm2.move_joints({j: v for j, v in arm2_start.items()})
    arm1.move_joints({j: v for j, v in arm1_start.items()})
    return poses


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the three arms and record the handoff poses")
    parser.add_argument("--recalibrate", action="store_true",
                        help="redo calibration even if saved files exist")
    parser.add_argument("--test", action="store_true",
                        help="run one full two-arm serve, then exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s")

    # -- test-only mode --------------------------------------------------------
    if args.test:
        poses = ArmPoses.load()
        if poses is None:
            print(f"No saved poses ({config.ARM_POSES_FILE}) - record them first.")
            sys.exit(1)
        can = CanInventory(poses.rows, poses.cols).next_unused() or (0, 0)
        arm1 = connect_arm_interactive("arm1", config.ARM1_PORT)
        if arm1 is None:
            sys.exit(1)
        arm2 = connect_arm_interactive("arm2", config.ARM2_PORT)
        if arm2 is None:
            arm1.disconnect()
            sys.exit(1)
        try:
            print(f"Test-serving can {can} (not marked as used - put it back).")
            serve_can(arm1, arm2, poses, *can,
                      cal1=ArmCalibration.load("arm1"),
                      cal2=ArmCalibration.load("arm2"))
        finally:
            arm1.disconnect()
            arm2.disconnect()
        return

    # -- step 1: connect all three arms -----------------------------------------
    print("STEP 1 - connecting arms")
    arms: dict[str, ArmController] = {}
    try:
        for name, port in (("arm1", config.ARM1_PORT),
                           ("arm2", config.ARM2_PORT),
                           ("leader", config.LEADER_ARM_PORT)):
            arm = connect_arm_interactive(name, port)
            if arm is None:
                sys.exit(1)
            arms[name] = arm

        # -- step 2: calibration --------------------------------------------------
        print("\nSTEP 2 - calibration (full range of motion)")
        cal1 = get_calibration(arms["arm1"], "arm1", args.recalibrate)
        cal2 = get_calibration(arms["arm2"], "arm2", args.recalibrate)
        leader_cal = get_calibration(arms["leader"], "leader", args.recalibrate)

        # -- steps 3+4: record all poses --------------------------------------------
        poses = record_all_poses(arms["leader"], arms["arm1"], arms["arm2"],
                                 leader_cal, cal1, cal2)

        # -- step 6: fresh rack + optional test --------------------------------------
        inventory = CanInventory(poses.rows, poses.cols)
        inventory.reset()
        answer = input("\nTest the full handoff with the first can now (it is "
                       "NOT marked as used)? [y/N] ")
        if answer.strip().lower().startswith("y"):
            serve_can(arms["arm1"], arms["arm2"], poses, 0, 0,
                      cal1=cal1, cal2=cal2)
        print("\nDone. main.py will now serve cans through the two-arm "
              "handoff when it sees a person.")
    finally:
        for arm in arms.values():
            arm.disconnect()


if __name__ == "__main__":
    main()
