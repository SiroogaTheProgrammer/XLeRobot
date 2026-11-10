#!/usr/bin/env python3
"""Manually move one arm from the terminal (default: ARM1).

Usage:
    .venv/bin/python move_arm.py                # ARM1
    .venv/bin/python move_arm.py --arm arm2
    .venv/bin/python move_arm.py --arm leader
    .venv/bin/python move_arm.py --teleop       # mirror the leader right away

Commands:
    m                    MIRROR the guiding (leader) arm live - the arm
                         follows your hand until you press Enter
    p                    print live joint positions
    j JOINT DELTA        jog a joint by degrees, e.g.  j shoulder_pan -10
    s JOINT ANGLE        set a joint to an absolute angle, e.g.  s gripper -60
    o / c                gripper to its recorded OPEN / GRIP reference
                         (from arm_poses.json; falls back to +/-20 jogs)
    goto start|handoff|receive|final     move to a recorded pose
    goto can ROW COL     move to an interpolated can position (ARM1)
    t on | t off         torque on/off (off = move the arm by hand)
    q                    quit

Jogs are capped at 45 deg per command. Arm pose moves are clamped to the
arm's calibrated ranges; gripper reference moves are not (same rules as
serving). The gripper is never moved by `goto`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from xlerobot import config
from xlerobot.arm.calibration import ArmCalibration, _enter_pressed, map_position
from xlerobot.arm.can_rack import ArmPoses
from xlerobot.arm.controller import ALL_JOINTS, ArmController
from xlerobot.arm.teleop import teleop_step

ARM_PORTS = {
    "arm1": config.ARM1_PORT,
    "arm2": config.ARM2_PORT,
    "leader": config.LEADER_ARM_PORT,
}


def print_positions(arm: ArmController, settle: bool = False):
    if settle:
        time.sleep(0.5)   # let the servos reach the target before reading
    pos = arm.positions()
    print("  " + "  ".join(f"{j}={v:+8.2f}" for j, v in pos.items()))


def named_poses(arm_name: str, poses: ArmPoses | None):
    if poses is None:
        return {}
    if arm_name == "arm1":
        return {"start": poses.arm1_start, "handoff": poses.handoff}
    if arm_name == "arm2":
        return {"start": poses.arm2_start, "receive": poses.receive,
                "final": poses.final}
    return {}


def grip_refs(arm_name: str, poses: ArmPoses | None):
    if poses is None:
        return None
    return {"arm1": poses.arm1_grip, "arm2": poses.arm2_grip}.get(arm_name)


class LeaderMirror:
    """Lazily connected guiding arm used for free teleoperation."""

    def __init__(self, follower: ArmController,
                 follower_cal: ArmCalibration | None):
        self._follower = follower
        self._follower_cal = follower_cal
        self._leader: ArmController | None = None
        self._leader_cal: ArmCalibration | None = None

    def _ready(self) -> bool:
        if self._follower.side == "leader":
            print("! cannot mirror the leader onto itself")
            return False
        if self._follower_cal is None:
            print(f"! no calibration for {self._follower.side} "
                  "(run setup_arms.py first)")
            return False
        if self._leader is None:
            self._leader_cal = ArmCalibration.load("leader")
            if self._leader_cal is None:
                print("! no leader calibration (run setup_arms.py first)")
                return False
            try:
                leader = ArmController("leader", config.LEADER_ARM_PORT,
                                       "leader", backend="feetech")
                leader.connect()
            except Exception as exc:
                print(f"! leader arm not reachable: {exc}")
                return False
            self._leader = leader
        return True

    def mirror(self):
        """Mirrors the leader onto the follower until Enter is pressed."""
        if not self._ready():
            return
        print(f"  {self._follower.side.upper()} will ease to match the "
              "guiding arm's pose - keep the area clear!")
        input("  press Enter to start mirroring... ")
        self._leader.set_torque(False)
        leader_pos = self._leader.positions()
        eased = {j: map_position(self._leader_cal, self._follower_cal, j,
                                 leader_pos[j]) for j in ALL_JOINTS}
        self._follower.move_joints(eased)
        print("  MIRRORING - move the guiding arm freely; press Enter to stop")
        period = 1.0 / config.TELEOP_HZ
        while not _enter_pressed():
            teleop_step(self._leader, self._follower,
                        self._leader_cal, self._follower_cal)
            time.sleep(period)
        print("  mirroring stopped (follower holds its pose, leader is limp)")

    def close(self):
        if self._leader is not None:
            self._leader.set_torque(True)   # leave the leader holding itself
            self._leader.disconnect()
            self._leader = None


def repl(arm: ArmController, cal: ArmCalibration | None, poses: ArmPoses | None,
         teleop_now: bool = False):
    targets = named_poses(arm.side, poses)
    refs = grip_refs(arm.side, poses)
    mirror = LeaderMirror(arm, cal)
    print(__doc__.split("Commands:")[1])
    print_positions(arm)
    if teleop_now:
        mirror.mirror()

    def clamped(pose):
        pose = {j: v for j, v in pose.items() if j != "gripper"}
        if cal is None:
            return pose
        return {j: cal.clamp(j, v) for j, v in pose.items()}

    while True:
        try:
            line = input(f"[{arm.side}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, *rest = line.split()
        try:
            if cmd == "q":
                break
            elif cmd == "m":
                mirror.mirror()
                print_positions(arm)
            elif cmd == "p":
                print_positions(arm)
            elif cmd == "t":
                if rest[:1] not in (["on"], ["off"]):
                    print("! usage: t on | t off")
                    continue
                arm.set_torque(rest[0] == "on")
            elif cmd in ("j", "s"):
                if len(rest) != 2 or rest[0] not in ALL_JOINTS:
                    print(f"! usage: {cmd} JOINT VALUE   "
                          f"(joints: {', '.join(ALL_JOINTS)})")
                    continue
                joint, value = rest[0], float(rest[1])
                if cmd == "j":
                    if abs(value) > 45:
                        print("! refusing jogs larger than 45 degrees")
                        continue
                    arm.move_relative({joint: value}, steps=20)
                else:
                    current = arm.positions()[joint]
                    if abs(value - current) > 120:
                        print(f"! that is {abs(value-current):.0f} deg away - "
                              "refusing moves larger than 120 degrees")
                        continue
                    arm.move_joints({joint: value}, steps=30)
                print_positions(arm, settle=True)
            elif cmd in ("o", "c"):
                if refs is not None:
                    target = refs["open"] if cmd == "o" else refs["grip"]
                    arm.move_joints({"gripper": target}, steps=12)
                else:
                    delta = 20.0 if cmd == "o" else -20.0
                    arm.move_relative({"gripper": delta}, steps=12)
                print_positions(arm, settle=True)
            elif cmd == "goto":
                if rest[:1] == ["can"] and len(rest) == 3 and poses is not None:
                    pose = poses.can_pose(int(rest[1]), int(rest[2]))
                elif rest and rest[0] in targets:
                    pose = targets[rest[0]]
                else:
                    opts = list(targets) + (["can ROW COL"] if arm.side == "arm1"
                                            else [])
                    print(f"! usage: goto {' | '.join(opts) or '(no poses recorded)'}")
                    continue
                arm.move_joints(clamped(pose))
                print_positions(arm, settle=True)
            else:
                print("! unknown command (m, p, j, s, o, c, goto, t, q)")
        except Exception as exc:  # keep the tool alive on servo hiccups
            print(f"! error: {exc}")
    mirror.close()


def main():
    parser = argparse.ArgumentParser(description="Manually move one arm")
    parser.add_argument("--arm", default="arm1", choices=sorted(ARM_PORTS))
    parser.add_argument("--teleop", action="store_true",
                        help="start mirroring the guiding (leader) arm "
                             "immediately")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.teleop and args.arm == "leader":
        parser.error("--teleop mirrors the leader onto a follower; "
                     "use --arm arm1 or --arm arm2")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s")

    arm = ArmController(args.arm, ARM_PORTS[args.arm], args.arm,
                        backend="feetech")
    try:
        arm.connect()
    except Exception as exc:
        print(f"Could not connect to {args.arm} on {ARM_PORTS[args.arm]}: {exc}")
        sys.exit(1)
    try:
        repl(arm, ArmCalibration.load(args.arm), ArmPoses.load(),
             teleop_now=args.teleop)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
