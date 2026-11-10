"""Can rack with two-arm handoff: recorded poses, grid, inventory, serving.

Two follower arms share the work:

* **ARM1** stands at the rack and picks cans.
* **ARM2** stands near the person, receives the can from ARM1 and hands it
  over.

setup_arms.py records (via the guiding arm) these FOLLOWER poses:

ARM1: ``start``   - rest pose
      4 corners   - open jaws centred around the first/last can of the
                    top and bottom rows
      ``handoff`` - where it presents the can to ARM2
ARM2: ``start``   - rest pose
      ``receive`` - open jaws centred around the can held by ARM1
      ``final``   - the giving-out pose in front of the person

plus two HAND-SET gripper references per arm ("open" = jaws fully open,
"grip" = jaws snugly on a real can). All gripping uses these references;
the gripper values inside the poses are ignored, because live-teleop
gripper readings are unreliable (the guiding trigger gets released when
reaching for the keyboard).

Every other can position is bilinearly interpolated (joint space) between
the four corners using the grid size recorded with the poses.

Serve sequence (the can is always held by at least one gripper):

    ARM1: open -> can -> clamp -> handoff pose
    ARM2: open -> receive pose -> clamp        (now both hold it)
    ARM1: release -> back to start             (ARM2 holds it)
    ARM2: carry to final (aimed at the person) -> release -> hold -> start

Which cans were already handed out is persisted in CAN_STATE_FILE.
"""

from __future__ import annotations

import json
import logging
import os
import time

from xlerobot import config
from xlerobot.arm.calibration import ArmCalibration
from xlerobot.arm.controller import ArmController

log = logging.getLogger(__name__)

CORNER_KEYS = ("top_first", "top_last", "bottom_first", "bottom_last")

Pose = dict[str, float]      # joint name -> degrees


class ArmPoses:
    """All recorded key poses for both arms plus the rack grid size.

    ``arm1_grip`` / ``arm2_grip`` are the explicit gripper references,
    recorded by physically setting the jaws (not teleop, which is noisy):
    ``{"open": deg, "grip": deg}`` where "open" = jaws fully open and
    "grip" = jaws snugly on a real can. During serving, pose moves NEVER
    command the gripper - only these references do.
    """

    def __init__(self, rows: int, cols: int,
                 arm1_start: Pose, corners: dict[str, Pose], handoff: Pose,
                 arm2_start: Pose, receive: Pose, final: Pose,
                 arm1_grip: dict[str, float], arm2_grip: dict[str, float]):
        self.rows = rows
        self.cols = cols
        self.arm1_start = arm1_start
        self.corners = corners
        self.handoff = handoff
        self.arm2_start = arm2_start
        self.receive = receive
        self.final = final
        self.arm1_grip = arm1_grip
        self.arm2_grip = arm2_grip

    # -- persistence ---------------------------------------------------------

    @staticmethod
    def _rounded(pose: Pose) -> Pose:
        return {j: round(float(v), 2) for j, v in pose.items()}

    def save(self, path: str = config.ARM_POSES_FILE):
        data = {
            "rows": self.rows, "cols": self.cols,
            "arm1": {"start": self._rounded(self.arm1_start),
                     "corners": {c: self._rounded(p)
                                 for c, p in self.corners.items()},
                     "handoff": self._rounded(self.handoff),
                     "gripper": self._rounded(self.arm1_grip)},
            "arm2": {"start": self._rounded(self.arm2_start),
                     "receive": self._rounded(self.receive),
                     "final": self._rounded(self.final),
                     "gripper": self._rounded(self.arm2_grip)},
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
        log.info("Saved two-arm poses (%dx%d rack) to %s",
                 self.rows, self.cols, path)

    @classmethod
    def load(cls, path: str = config.ARM_POSES_FILE) -> "ArmPoses | None":
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if ("arm1" not in data or "arm2" not in data
                    or "gripper" not in data.get("arm1", {})
                    or "gripper" not in data.get("arm2", {})):
                log.warning("%s is from an older setup (no explicit gripper "
                            "references) - re-run setup_arms.py", path)
                return None
            pose = lambda d: {k: float(v) for k, v in d.items()}  # noqa: E731
            poses = cls(
                rows=int(data["rows"]), cols=int(data["cols"]),
                arm1_start=pose(data["arm1"]["start"]),
                corners={c: pose(data["arm1"]["corners"][c])
                         for c in CORNER_KEYS},
                handoff=pose(data["arm1"]["handoff"]),
                arm2_start=pose(data["arm2"]["start"]),
                receive=pose(data["arm2"]["receive"]),
                final=pose(data["arm2"]["final"]),
                arm1_grip=pose(data["arm1"]["gripper"]),
                arm2_grip=pose(data["arm2"]["gripper"]),
            )
            for name, grip in (("arm1", poses.arm1_grip),
                               ("arm2", poses.arm2_grip)):
                if abs(grip["open"] - grip["grip"]) < 5.0:
                    log.warning("%s gripper references are suspiciously close "
                                "(open=%.1f grip=%.1f) - re-record them",
                                name, grip["open"], grip["grip"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            log.exception("Bad arm poses file %s", path)
            return None
        if (poses.rows, poses.cols) != (config.CAN_ROWS, config.CAN_COLS):
            log.warning("Rack in settings.py is %dx%d but poses were recorded "
                        "for %dx%d - using the recorded grid. Re-run "
                        "setup_arms.py after changing the rack size.",
                        config.CAN_ROWS, config.CAN_COLS, poses.rows, poses.cols)
        return poses

    # -- grid ----------------------------------------------------------------

    def can_pose(self, row: int, col: int) -> Pose:
        """Joint-space bilinear interpolation between the 4 corner poses.

        row 0 = the recorded TOP row, col 0 = the recorded FIRST can.
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"can ({row},{col}) outside {self.rows}x{self.cols}")
        r = row / (self.rows - 1) if self.rows > 1 else 0.0
        c = col / (self.cols - 1) if self.cols > 1 else 0.0
        tf, tl = self.corners["top_first"], self.corners["top_last"]
        bf, bl = self.corners["bottom_first"], self.corners["bottom_last"]
        return {j: ((1 - r) * (1 - c) * tf[j] + (1 - r) * c * tl[j]
                    + r * (1 - c) * bf[j] + r * c * bl[j])
                for j in tf}


class CanInventory:
    """Remembers which cans of the rack were already handed out."""

    def __init__(self, rows: int = config.CAN_ROWS, cols: int = config.CAN_COLS,
                 path: str = config.CAN_STATE_FILE):
        self.rows = rows
        self.cols = cols
        self._path = path
        self._used: set[tuple[int, int]] = set()
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._used = {(int(r), int(c)) for r, c in data.get("used", [])}
        except (OSError, ValueError, json.JSONDecodeError):
            log.exception("Bad can state file %s, starting fresh", self._path)

    def _save(self):
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"used": sorted(self._used)}, fh)
        os.replace(tmp, self._path)

    def remaining(self) -> int:
        return self.rows * self.cols - len(self._used)

    def next_unused(self) -> tuple[int, int] | None:
        """Next can to serve: top row first, left to right."""
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) not in self._used:
                    return row, col
        return None

    def mark_used(self, row: int, col: int):
        self._used.add((row, col))
        self._save()
        log.info("Can (%d,%d) handed out - %d of %d left",
                 row, col, self.remaining(), self.rows * self.cols)

    def reset(self):
        self._used.clear()
        self._save()
        log.info("Can inventory reset: %d cans available", self.rows * self.cols)


def grip_targets(grip_refs: dict[str, float], arm_name: str) -> tuple[float, float]:
    """Returns (open_deg, clamp_deg) from an arm's explicit gripper refs.

    ``clamp_deg`` is the hand-recorded on-can position plus that arm's
    extra squeeze (GRIP_SQUEEZE_DEG[arm_name]) in the closing direction,
    so the servo holds the can firmly without stalling at full force.
    """
    open_deg = grip_refs["open"]
    grip_deg = grip_refs["grip"]
    squeeze = config.GRIP_SQUEEZE_DEG[arm_name]
    direction = 1.0 if grip_deg >= open_deg else -1.0
    return open_deg, grip_deg + direction * squeeze


def aim_pan_offset(bearing_deg: float | None) -> float:
    """Base-rotation offset towards the person's fused bearing.

    The recorded final pose points at the camera centre (bearing 0); the
    offset swings the bottom servo towards where the person actually is,
    limited to AIM_MAX_PAN_DEG. Sign convention is fixed by AIM_PAN_SIGN.
    """
    if bearing_deg is None:
        return 0.0
    offset = config.AIM_GAIN * bearing_deg
    offset = max(-config.AIM_MAX_PAN_DEG, min(config.AIM_MAX_PAN_DEG, offset))
    return config.AIM_PAN_SIGN * offset


def _clamped(cal: ArmCalibration | None, pose: Pose) -> Pose:
    if cal is None:
        return dict(pose)
    return {j: cal.clamp(j, v) for j, v in pose.items()}


def serve_can(arm1: ArmController, arm2: ArmController, poses: ArmPoses,
              row: int, col: int,
              cal1: ArmCalibration | None = None,
              cal2: ArmCalibration | None = None,
              hold_s: float = config.HANDOVER_HOLD_S,
              aim_bearing_deg: float | None = None,
              aim_bearing_fn=None):
    """Full two-arm handout of the can at (row, col).

    The can is never free-floating: ARM2 clamps it BEFORE ARM1 releases.
    Aiming: ``aim_bearing_fn`` (if given) is called right before ARM2
    extends, so the aim uses the person's CURRENT position even if they
    moved during the pick/handoff; otherwise the static ``aim_bearing_deg``
    is used (fused bearing, 0 = camera centre, positive = left).
    """
    can = poses.can_pose(row, col)
    open1, grip1 = grip_targets(poses.arm1_grip, "arm1")
    open2, grip2 = grip_targets(poses.arm2_grip, "arm2")
    pause = config.STEP_PAUSE_S

    def nog(pose: Pose) -> Pose:
        """Pose without the gripper: arm moves NEVER command the gripper."""
        return {j: v for j, v in pose.items() if j != "gripper"}

    def step(arm, cal, targets: Pose, steps: int | None = None,
             clamp: bool = True):
        """One serve step followed by a stillness pause.

        Gripper steps pass clamp=False: their targets come from the
        hand-set references (+ squeeze) and may legitimately go past the
        range the calibration sweep happened to reach - clamping them
        silently weakens the grip.
        """
        kwargs = {} if steps is None else {"steps": steps}
        arm.move_joints(_clamped(cal, targets) if clamp else dict(targets),
                        **kwargs)
        time.sleep(pause)

    # --- ARM1 picks the can and presents it at the handoff pose ----------
    log.info("Serving can (%d,%d): ARM1 picking (gripper open=%.1f grip=%.1f)",
             row, col, open1, grip1)
    step(arm1, cal1, {"gripper": open1}, steps=12, clamp=False)
    step(arm1, cal1, nog(can))
    step(arm1, cal1, {"gripper": grip1}, steps=12, clamp=False)
    step(arm1, cal1, nog(poses.handoff))

    # --- ARM2 clamps the can FIRST, then ARM1 releases --------------------
    log.info("Handoff: ARM2 receiving (gripper open=%.1f grip=%.1f)",
             open2, grip2)
    step(arm2, cal2, {"gripper": open2}, steps=12, clamp=False)
    step(arm2, cal2, nog(poses.receive))
    step(arm2, cal2, {"gripper": grip2}, steps=12, clamp=False)
    step(arm1, cal1, {"gripper": open1}, steps=12, clamp=False)
    step(arm1, cal1, nog(poses.arm1_start))              # clear the way
    # Rest with the jaws gently closed (the on-can position, no squeeze)
    step(arm1, cal1, {"gripper": poses.arm1_grip["grip"]}, steps=12,
         clamp=False)

    # --- ARM2 delivers to the person --------------------------------------
    carry = nog(poses.final)
    if aim_bearing_fn is not None:
        aim_bearing_deg = aim_bearing_fn()   # freshest position, right now
    pan = aim_pan_offset(aim_bearing_deg)
    if pan:
        carry["shoulder_pan"] = carry["shoulder_pan"] + pan
        log.info("Aiming at person: bearing %+.1f deg -> base offset %+.1f deg",
                 aim_bearing_deg, pan)
    step(arm2, cal2, carry)
    log.info("At the final pose - waiting %.1f s before releasing",
             config.RELEASE_DELAY_S)
    time.sleep(config.RELEASE_DELAY_S)                   # let the person reach
    step(arm2, cal2, {"gripper": open2}, steps=12, clamp=False)
    log.info("Can offered - holding for %.1f s", hold_s)
    time.sleep(hold_s)

    step(arm2, cal2, nog(poses.arm2_start))
    step(arm2, cal2, {"gripper": poses.arm2_grip["grip"]}, steps=12,
         clamp=False)
    log.info("Both arms back at start")
