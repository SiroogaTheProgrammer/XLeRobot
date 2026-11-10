#!/usr/bin/env python3
"""Reliable port discovery for XLeRobot - finds the lidar and both arms,
then writes the stable /dev/serial/by-id/ paths into settings.py.

What it does:

1. Scans /dev/serial/by-id/ (falls back to /dev/ttyACM* /dev/ttyUSB*).
2. Probes every port for what is REALLY on it:
     * Feetech servo bus (arm)  - pings servo IDs 1-6 at 1 Mbaud
     * RPLIDAR                  - asks for device info at 115200
3. Tells the two arms apart: if the known follower/leader port from
   settings.py is among them, roles are kept; otherwise you are asked to
   wiggle a joint on the GUIDING (leader) arm and the moving bus wins.
4. Shows the result and updates settings.py (asks first; --yes skips).

Usage:
    .venv/bin/python find_ports.py
    .venv/bin/python find_ports.py --yes      # no confirmation prompt
    .venv/bin/python find_ports.py --dry-run  # only report, never write
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "settings.py")
SERVO_IDS = (1, 2, 3, 4, 5, 6)
FEETECH_BAUD = 1_000_000
LIDAR_BAUD = 115_200
WIGGLE_THRESHOLD_DEG = 8.0
WIGGLE_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def candidate_ports() -> list[str]:
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    if by_id:
        return by_id
    return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def probe_feetech(port: str) -> list[int]:
    """Returns the servo IDs (1-6) that answer a ping on this port."""
    import scservo_sdk as scs
    ph = scs.PortHandler(port)
    pk = scs.PacketHandler(0)
    found: list[int] = []
    try:
        if not ph.openPort() or not ph.setBaudRate(FEETECH_BAUD):
            return []
        for sid in SERVO_IDS:
            _model, result, _err = pk.ping(ph, sid)
            if result == scs.COMM_SUCCESS:
                found.append(sid)
    except Exception:
        pass
    finally:
        try:
            ph.closePort()
        except Exception:
            pass
    return found


def probe_lidar(port: str) -> str | None:
    """Returns a lidar model string if an RPLIDAR answers on this port."""
    from rplidar import RPLidar
    lidar = None
    try:
        lidar = RPLidar(port, baudrate=LIDAR_BAUD, timeout=1)
        info = lidar.get_info()
        return str(info.get("model", "RPLIDAR"))
    except Exception:
        return None
    finally:
        if lidar is not None:
            for action in (lidar.stop, lidar.stop_motor, lidar.disconnect):
                try:
                    action()
                except Exception:
                    pass


def classify_ports() -> tuple[list[str], list[str], list[str]]:
    """Probes all candidates; returns (arm_ports, lidar_ports, unknown)."""
    arms, lidars, unknown = [], [], []
    for port in candidate_ports():
        print(f"  probing {port} ... ", end="", flush=True)
        servos = probe_feetech(port)
        if servos:
            print(f"ARM BUS (servos {servos})")
            arms.append(port)
            continue
        model = probe_lidar(port)
        if model is not None:
            print(f"LIDAR (model {model})")
            lidars.append(port)
            continue
        print("unknown / no response (is it powered?)")
        unknown.append(port)
    return arms, lidars, unknown


# ---------------------------------------------------------------------------
# Leader vs follower disambiguation
# ---------------------------------------------------------------------------

def _same_device(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return False


def read_all_positions(port: str) -> dict[int, int] | None:
    import scservo_sdk as scs
    ph = scs.PortHandler(port)
    pk = scs.PacketHandler(0)
    try:
        if not ph.openPort() or not ph.setBaudRate(FEETECH_BAUD):
            return None
        pos = {}
        for sid in SERVO_IDS:
            ticks, result, _err = pk.read2ByteTxRx(ph, sid, 56)
            if result != scs.COMM_SUCCESS:
                return None
            pos[sid] = ticks
        return pos
    except Exception:
        return None
    finally:
        try:
            ph.closePort()
        except Exception:
            pass


def wiggle_test(ports: list[str], role_desc: str) -> str | None:
    """Asks the user to move one specific arm; returns the port that moved."""
    print(f"\n  >>> WIGGLE any joint of the {role_desc} arm now <<<")
    base = {p: read_all_positions(p) for p in ports}
    if None in base.values():
        return None
    ticks_thresh = WIGGLE_THRESHOLD_DEG * 4096 / 360.0
    t0 = time.time()
    while time.time() - t0 < WIGGLE_TIMEOUT_S:
        for port in ports:
            now = read_all_positions(port)
            if now is None:
                continue
            moved = max(abs(now[s] - base[port][s]) for s in SERVO_IDS)
            if moved > ticks_thresh:
                print(f"  movement detected on {os.path.basename(port)}")
                return port
        time.sleep(0.1)
    print("  ! no movement detected within "
          f"{WIGGLE_TIMEOUT_S:.0f} s - cannot identify the {role_desc} arm")
    return None


ARM_ROLES = (
    ("LEADER_ARM_PORT", "GUIDING (leader)"),
    ("ARM1_PORT", "ARM1 (rack picker)"),
    ("ARM2_PORT", "ARM2 (giver, near the person)"),
)


def assign_arm_roles(arms: list[str], current: dict[str, str]) -> dict[str, str]:
    """Maps probed arm ports to the leader/arm1/arm2 roles."""
    assigned: dict[str, str] = {}
    remaining = list(arms)

    # Keep roles for ports already known from settings.py
    for role, _desc in ARM_ROLES:
        for port in remaining:
            if _same_device(port, current.get(role, "")):
                assigned[role] = port
                remaining.remove(port)
                break

    vacant = [(r, d) for r, d in ARM_ROLES if r not in assigned]
    while vacant and remaining:
        if len(remaining) == 1 and len(vacant) == 1:
            assigned[vacant[0][0]] = remaining.pop()
            vacant.clear()
            break
        if len(remaining) == 1:
            port = remaining[0]
            print(f"\n  One unassigned arm on {port}. Which one is it?")
            for i, (_r, desc) in enumerate(vacant, 1):
                print(f"    {i}. {desc}")
            answer = input("  number: ").strip()
            try:
                role, _ = vacant.pop(int(answer) - 1)
            except (ValueError, IndexError):
                break
            assigned[role] = remaining.pop()
            continue
        # Several unassigned ports: identify them one by one by wiggling
        role, desc = vacant.pop(0)
        port = wiggle_test(remaining, desc)
        if port is None:
            break
        assigned[role] = port
        remaining.remove(port)
    return assigned


# ---------------------------------------------------------------------------
# settings.py update
# ---------------------------------------------------------------------------

def read_current_settings() -> dict[str, str]:
    import settings
    return {name: getattr(settings, name, "")
            for name in ("LIDAR_PORT", "LEADER_ARM_PORT",
                         "ARM1_PORT", "ARM2_PORT")}


def update_settings(assignments: dict[str, str]) -> None:
    with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
        text = fh.read()
    for name, port in assignments.items():
        pattern = rf'^{name}\s*=.*$'
        replacement = f'{name} = "{port}"  # set by find_ports.py'
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.M)
        if n == 0:
            text += f"\n{replacement}\n"
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\nUpdated {SETTINGS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Detect and configure ports")
    parser.add_argument("--yes", action="store_true",
                        help="write settings.py without asking")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report, never write")
    args = parser.parse_args()

    print("Scanning serial ports (close main.py / setup_arms.py first -\n"
          "a port in use by another program shows as 'no response'):\n")
    arms, lidars, unknown = classify_ports()

    current = read_current_settings()
    assignments: dict[str, str] = {}

    if len(lidars) == 1:
        assignments["LIDAR_PORT"] = lidars[0]
    elif len(lidars) > 1:
        print(f"\n! multiple lidars found ({lidars}) - keeping current setting")

    if arms:
        assignments.update(assign_arm_roles(arms, current))
    else:
        print("\n! no arm bus found - check USB and SERVO POWER "
              "(unpowered servos do not answer pings)")

    print("\nResult:")
    for name in ("LIDAR_PORT", "LEADER_ARM_PORT", "ARM1_PORT", "ARM2_PORT"):
        if name in assignments:
            mark = ("unchanged" if _same_device(assignments[name], current[name])
                    else "NEW")
            print(f"  {name:18s} -> {assignments[name]}  [{mark}]")
        else:
            print(f"  {name:18s} -> (not detected, keeping: {current[name]})")

    if not assignments or args.dry_run:
        return
    if not args.yes:
        answer = input("\nWrite these into settings.py? [Y/n] ").strip().lower()
        if answer.startswith("n"):
            return
    update_settings(assignments)


if __name__ == "__main__":
    main()
