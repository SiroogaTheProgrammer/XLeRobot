#!/usr/bin/env python3
"""Train the simple one-arm setup - everything is posed BY HAND, no
guiding arm needed.

    .venv/bin/python simple/setup_arm.py

1. GRIPPER references: only the gripper goes limp; you set the jaws fully
   OPEN, then snugly ON THE CAN.
2. POSES: the whole arm goes limp; you pose it and press Enter for each:
     * START - rest/return pose (clear path to the can)
     * CAN   - open jaws centred around the can on its spot
     * FINAL - where the can is held out to the person
   Every capture settles and is averaged for precision.
3. The arm returns to START and you can run a test serve
   (also available any time with --test).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simple_settings as s  # noqa: E402
from serving import POSE_NAMES, load_poses, save_poses, serve_once  # noqa: E402
from setup_arms import capture_gripper_refs, settle_read  # noqa: E402  (reused)
from xlerobot.arm.controller import ArmController  # noqa: E402

POSE_PROMPTS = {
    "start": "START pose - rest/return position, with a clear path to the can",
    "can": "CAN pose - open jaws centred around the can on its spot",
    "final": "FINAL pose - where the can is held out to the person",
}


def capture_pose_by_hand(arm: ArmController, prompt: str) -> dict[str, float]:
    print(f"\n  >>> {prompt}")
    input("      pose the arm by hand, hold it, press Enter... ")
    pose = settle_read(arm, settle_s=0.1)
    print("      recorded: "
          + "  ".join(f"{j}={v:+.1f}" for j, v in pose.items()))
    return pose


def main():
    parser = argparse.ArgumentParser(description="Train the simple setup")
    parser.add_argument("--test", action="store_true",
                        help="run one serve with the saved poses, then exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s")

    arm = ArmController("arm", s.ARM_PORT, "arm", backend="feetech")
    try:
        arm.connect()
    except Exception as exc:
        print(f"Arm not reachable on {s.ARM_PORT}: {exc}")
        sys.exit(1)

    try:
        if args.test:
            poses = load_poses()
            if poses is None:
                print(f"No saved poses ({s.POSES_FILE}) - train first.")
                sys.exit(1)
            input("Test serve - put a can on its spot, press Enter... ")
            serve_once(arm, poses)
            return

        # 1) gripper references (only the gripper goes limp)
        print("STEP 1 - gripper references")
        grip = capture_gripper_refs(arm, "ARM")

        # 2) the three poses, posed by hand
        print("\nSTEP 2 - poses (the whole arm goes limp - hold it!)")
        arm.set_torque(False)
        poses: dict = {name: capture_pose_by_hand(arm, POSE_PROMPTS[name])
                       for name in POSE_NAMES}
        arm.set_torque(True)
        poses["gripper"] = grip
        save_poses(poses)

        # 3) back to start + optional test
        print("\n  Returning to the START pose.")
        arm.move_joints({j: v for j, v in poses["start"].items()
                         if j != "gripper"})
        arm.move_joints({"gripper": grip["grip"]}, steps=12)
        answer = input("\nRun a test serve now (put a can on its spot "
                       "first)? [y/N] ")
        if answer.strip().lower().startswith("y"):
            serve_once(arm, poses)
        print("\nDone. Run:  .venv/bin/python simple/main.py --show")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
