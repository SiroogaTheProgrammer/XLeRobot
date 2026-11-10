#!/usr/bin/env python3
"""Repair misconfigured Feetech STS3215 servos on an SO-101 arm.

Fixes the two classic problems of an arm whose servos were never set up
for their current assembly:

* **Angle-limit registers** locked to a tiny window (the servo's LED glows
  but it refuses to move): ``--unlock-limits`` widens every servo to the
  full 0..4095 range. Safe travel is enforced by our own calibration
  clamps instead.
* **Wrong zero offset** (the joint's physical travel crosses the 0/4095
  tick wrap, which inflates calibration spans to ~358 deg and makes
  teleop command impossible targets): ``--center`` re-zeroes each servo
  so that the pose you hold the arm in becomes tick 2048 - the middle of
  the range. Hold the arm roughly in the MIDDLE of every joint's travel
  (gripper half open) when doing this.

Typical repair sequence for a new/misbehaving arm:

    .venv/bin/python repair_arm.py --arm arm2                  # inspect
    .venv/bin/python repair_arm.py --arm arm2 --unlock-limits
    .venv/bin/python repair_arm.py --arm arm2 --center         # hold middle!
    rm calibrations/arm2.json                                  # recalibrate
    .venv/bin/python setup_arms.py

Registers (STS3215): 9 = min angle limit, 11 = max angle limit,
40 = torque enable (writing 128 = re-zero current position to 2048),
55 = EEPROM lock.
"""

from __future__ import annotations

import argparse
import sys
import time

import scservo_sdk as scs

from xlerobot import config

REG_MIN_LIMIT = 9
REG_MAX_LIMIT = 11
REG_TORQUE = 40
REG_PRESENT_POS = 56
REG_LOCK = 55
CENTER_CMD = 128          # torque register value that re-zeroes to 2048

ARM_PORTS = {
    "arm1": config.ARM1_PORT,
    "arm2": config.ARM2_PORT,
    "leader": config.LEADER_ARM_PORT,
}
JOINTS = list(config.FEETECH_SERVO_IDS.items())


class Bus:
    def __init__(self, port: str):
        self.ph = scs.PortHandler(port)
        self.pk = scs.PacketHandler(0)
        if not (self.ph.openPort() and self.ph.setBaudRate(config.FEETECH_BAUD)):
            raise RuntimeError(f"cannot open {port}")

    def close(self):
        self.ph.closePort()

    def read2(self, sid: int, reg: int) -> int:
        val, res, _err = self.pk.read2ByteTxRx(self.ph, sid, reg)
        if res != scs.COMM_SUCCESS:
            raise RuntimeError(f"servo {sid}: read reg {reg} failed")
        return val

    def write1(self, sid: int, reg: int, val: int):
        for attempt in range(3):
            res, _err = self.pk.write1ByteTxRx(self.ph, sid, reg, val)
            if res == scs.COMM_SUCCESS:
                return
            time.sleep(0.05)
        raise RuntimeError(f"servo {sid}: write reg {reg} failed")

    def write2(self, sid: int, reg: int, val: int):
        for attempt in range(3):
            res, _err = self.pk.write2ByteTxRx(self.ph, sid, reg, val)
            if res == scs.COMM_SUCCESS:
                return
            time.sleep(0.05)
        raise RuntimeError(f"servo {sid}: write reg {reg} failed")


def inspect(bus: Bus):
    print(f"  {'joint':14s} {'ticks':>5s} {'deg':>8s}  limits")
    wrap_risk = []
    for name, sid in JOINTS:
        ticks = bus.read2(sid, REG_PRESENT_POS)
        lo = bus.read2(sid, REG_MIN_LIMIT)
        hi = bus.read2(sid, REG_MAX_LIMIT)
        deg = (ticks - 2048) / (4096 / 360.0)
        notes = []
        if hi - lo < 200:
            notes.append("LIMITS NEARLY LOCKED")
        if ticks < 300 or ticks > 3800:
            notes.append("near tick wrap - consider --center")
            wrap_risk.append(name)
        print(f"  {name:14s} {ticks:5d} {deg:+8.1f}  [{lo},{hi}] "
              + ("  <- " + ", ".join(notes) if notes else ""))
    return wrap_risk


def unlock_limits(bus: Bus):
    # EEPROM writes require torque to be disabled
    for _name, sid in JOINTS:
        bus.write1(sid, REG_TORQUE, 0)
    time.sleep(0.1)
    for name, sid in JOINTS:
        bus.write1(sid, REG_LOCK, 0)          # unlock EEPROM
        bus.write2(sid, REG_MIN_LIMIT, 0)
        bus.write2(sid, REG_MAX_LIMIT, 4095)
        bus.write1(sid, REG_LOCK, 1)          # lock again
        print(f"  {name:14s} limits -> [0, 4095]")
    for _name, sid in JOINTS:
        bus.write1(sid, REG_TORQUE, 1)        # hold the pose again


def center(bus: Bus):
    print("\n  Releasing torque. Hold the arm so EVERY joint is roughly in")
    print("  the MIDDLE of its physical travel (gripper half open).")
    for _name, sid in JOINTS:
        bus.write1(sid, REG_TORQUE, 0)
    input("  Hold it there and press Enter to re-zero all servos... ")
    for name, sid in JOINTS:
        bus.write1(sid, REG_LOCK, 0)
        bus.write1(sid, REG_TORQUE, CENTER_CMD)   # current position -> 2048
        bus.write1(sid, REG_LOCK, 1)
    time.sleep(0.3)
    ok = True
    for name, sid in JOINTS:
        ticks = bus.read2(sid, REG_PRESENT_POS)
        status = "ok" if abs(ticks - 2048) < 60 else "DID NOT RE-ZERO"
        ok &= status == "ok"
        print(f"  {name:14s} now at ticks {ticks} ({status})")
    for _name, sid in JOINTS:
        bus.write1(sid, REG_TORQUE, 1)            # hold the centred pose
    if ok:
        print("\n  Re-zeroed. Old calibrations/poses for this arm are now "
              "invalid -\n  delete its calibration file and re-run setup_arms.py.")
    else:
        print("\n  ! some servos did not re-zero - check their power/wiring "
              "and retry.")


def main():
    parser = argparse.ArgumentParser(description="Repair STS3215 servo config")
    parser.add_argument("--arm", default="arm2", choices=sorted(ARM_PORTS),
                        help="which arm to repair (default: %(default)s)")
    parser.add_argument("--unlock-limits", action="store_true",
                        help="widen all angle-limit registers to 0..4095")
    parser.add_argument("--center", action="store_true",
                        help="re-zero all servos: held pose becomes tick 2048")
    args = parser.parse_args()

    port = ARM_PORTS[args.arm]
    print(f"[{args.arm}] on {port}")
    try:
        bus = Bus(port)
    except RuntimeError as exc:
        print(f"! {exc}")
        sys.exit(1)
    try:
        print("\nBefore:")
        inspect(bus)
        if args.unlock_limits:
            print("\nWidening angle limits:")
            unlock_limits(bus)
        if args.center:
            center(bus)
        if args.unlock_limits or args.center:
            print("\nAfter:")
            inspect(bus)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
