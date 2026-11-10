"""Low-level SO-101 arm controller with three interchangeable backends.

* ``feetech`` - drives the Feetech STS3215 servos directly over serial via
  ``feetech-servo-sdk``. Needs no calibration: positions are read from the
  servos and motions are expressed relative to the current pose, so it is
  the right backend for canned gestures (e.g. the general drink handout).
* ``lerobot`` - full lerobot SO101Follower (requires ``lerobot[feetech]``
  and a one-time ``lerobot-calibrate``). Needed for absolute IK motions.
  NOTE: lerobot installs opencv-python-headless, which conflicts with the
  GUI OpenCV used for the live feeds - install it in a separate venv.
* ``dry-run`` - logs moves instead of sending them; no hardware required.
"""

from __future__ import annotations

import logging
import time

from xlerobot import config
from xlerobot.arm.kinematics import JointAngles

log = logging.getLogger(__name__)

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll")
ALL_JOINTS = (*JOINT_NAMES, "gripper")


class _FeetechBus:
    """Minimal STS3215 bus: ping, read positions, write goal positions."""

    REG_TORQUE_ENABLE = 40
    REG_GOAL_ACC = 41
    REG_GOAL_POSITION = 42
    REG_PRESENT_POSITION = 56
    TICKS_PER_DEG = 4096 / 360.0
    CENTER_TICKS = 2048       # assembly-standard centre position

    def __init__(self, port: str):
        import scservo_sdk as scs
        self._scs = scs
        self._port = scs.PortHandler(port)
        self._packet = scs.PacketHandler(0)   # STS/SMS protocol end = 0
        if not self._port.openPort():
            raise RuntimeError(f"could not open serial port {port}")
        if not self._port.setBaudRate(config.FEETECH_BAUD):
            raise RuntimeError(f"could not set baud rate on {port}")

    def close(self):
        self._port.closePort()

    def ping(self, servo_id: int) -> bool:
        """True if the servo answers. Hardware status bits (voltage,
        overheat...) are logged as warnings but do not fail the ping -
        e.g. leader-arm servos may flag a voltage error while working fine."""
        _model, result, error = self._packet.ping(self._port, servo_id)
        if result != self._scs.COMM_SUCCESS:
            return False
        if error != 0:
            log.warning("servo %d answers but reports a hardware status "
                        "flag: %s (check its power supply voltage)",
                        servo_id,
                        self._packet.getRxPacketError(error).strip() or hex(error))
        return True

    def set_torque(self, servo_id: int, enabled: bool):
        self._packet.write1ByteTxRx(self._port, servo_id,
                                    self.REG_TORQUE_ENABLE, int(enabled))

    def set_goal_acc(self, servo_id: int, acc: int):
        self._packet.write1ByteTxRx(self._port, servo_id,
                                    self.REG_GOAL_ACC, acc)

    def read_position_deg(self, servo_id: int) -> float:
        ticks, result, _error = self._packet.read2ByteTxRx(
            self._port, servo_id, self.REG_PRESENT_POSITION)
        # Hardware status bits (e.g. voltage flag) are reported via ping();
        # only an actual communication failure invalidates the data.
        if result != self._scs.COMM_SUCCESS:
            raise RuntimeError(
                f"read position failed for servo {servo_id}: "
                f"{self._packet.getTxRxResult(result)}")
        return (ticks - self.CENTER_TICKS) / self.TICKS_PER_DEG

    def write_position_deg(self, servo_id: int, deg: float):
        ticks = int(round(deg * self.TICKS_PER_DEG + self.CENTER_TICKS))
        ticks = max(0, min(4095, ticks))
        self._packet.write2ByteTxRx(self._port, servo_id,
                                    self.REG_GOAL_POSITION, ticks)


class ArmController:
    """One SO-101 arm. ``side`` is "left" or "right"."""

    def __init__(self, side: str, port: str, arm_id: str,
                 backend: str = config.ARM_BACKEND):
        if backend not in ("feetech", "lerobot", "dry-run"):
            raise ValueError(f"unknown arm backend {backend!r}")
        self.side = side
        self.port = port
        self.arm_id = arm_id
        self.backend = backend
        self._robot = None      # lerobot backend
        self._bus: _FeetechBus | None = None
        # Simulated joint state (degrees) used in dry-run and as the
        # interpolation start before the first observation.
        self._state: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self._state["gripper"] = config.GRIPPER_OPEN

    # -- lifecycle ---------------------------------------------------------

    def connect(self):
        if self.backend == "dry-run":
            log.info("[%s arm] dry-run: skipping hardware connect", self.side)
            return
        if self.backend == "feetech":
            # Transient failures (just-plugged adapter, servo power settling)
            # are common on first access - retry a few times before giving up.
            last_missing: list[str] = []
            for attempt in range(3):
                if attempt:
                    time.sleep(0.5)
                try:
                    self._bus = _FeetechBus(self.port)
                except RuntimeError:
                    if attempt == 2:
                        raise
                    continue
                last_missing = [name for name, sid
                                in config.FEETECH_SERVO_IDS.items()
                                if not self._bus.ping(sid)]
                if not last_missing:
                    break
                log.warning("[%s arm] servos not answering on %s (attempt "
                            "%d/3): %s", self.side, self.port, attempt + 1,
                            ", ".join(last_missing))
                self._bus.close()
                self._bus = None
            if self._bus is None:
                raise RuntimeError(
                    f"[{self.side} arm] servos not responding on {self.port} "
                    f"after 3 attempts: {', '.join(last_missing)} "
                    "(check servo power; run find_ports.py to verify ports)")
            self._state = self._read_positions()
            for sid in config.FEETECH_SERVO_IDS.values():
                self._bus.set_goal_acc(sid, config.FEETECH_GOAL_ACC)
                self._bus.set_torque(sid, True)   # hold the current pose
            log.info("[%s arm] feetech backend connected on %s, pose %s",
                     self.side, self.port,
                     {k: round(v, 1) for k, v in self._state.items()})
            return
        from lerobot.robots.so101_follower import (  # lazy import
            SO101Follower,
            SO101FollowerConfig,
        )
        robot_config = SO101FollowerConfig(
            port=self.port, id=self.arm_id, use_degrees=True)
        self._robot = SO101Follower(robot_config)
        self._robot.connect()
        self._state = self._read_positions()
        log.info("[%s arm] lerobot backend connected on %s", self.side, self.port)

    def disconnect(self):
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
        if self._bus is not None:
            # Torque stays enabled so the arm keeps holding its pose.
            self._bus.close()
            self._bus = None
        log.info("[%s arm] disconnected", self.side)

    @property
    def connected(self) -> bool:
        return (self.backend == "dry-run" or self._robot is not None
                or self._bus is not None)

    # -- state -------------------------------------------------------------

    def _read_positions(self) -> dict[str, float]:
        if self._bus is not None:
            return {name: self._bus.read_position_deg(sid)
                    for name, sid in config.FEETECH_SERVO_IDS.items()}
        obs = self._robot.get_observation()
        return {name: float(obs[f"{name}.pos"]) for name in ALL_JOINTS}

    def positions(self) -> dict[str, float]:
        """Current joint positions in degrees (simulated in dry-run)."""
        if self._robot is not None or self._bus is not None:
            self._state = self._read_positions()
        return dict(self._state)

    # -- motion ------------------------------------------------------------

    def _send(self, targets: dict[str, float]):
        if self._bus is not None:
            for name, pos in targets.items():
                self._bus.write_position_deg(config.FEETECH_SERVO_IDS[name], pos)
        elif self._robot is not None:
            self._robot.send_action({f"{name}.pos": pos
                                     for name, pos in targets.items()})
        self._state.update(targets)

    def set_joints(self, targets: dict[str, float]):
        """Immediately sends raw joint targets (no interpolation).

        Used for high-rate streaming (teleoperation, motion replay) where
        the caller controls the timing.
        """
        self._send({name: pos for name, pos in targets.items()
                    if name in self._state})

    def move_joints(self, targets: dict[str, float],
                    steps: int = config.ARM_MOVE_STEPS,
                    step_delay: float = config.ARM_STEP_DELAY_S):
        """Linearly interpolates from the current pose to ``targets``."""
        start = self.positions()
        targets = {name: pos for name, pos in targets.items() if name in start}
        if not targets:
            return
        log.debug("[%s arm] move -> %s",
                  self.side, {k: round(v, 1) for k, v in targets.items()})
        for i in range(1, steps + 1):
            t = i / steps
            frame = {name: start[name] + (pos - start[name]) * t
                     for name, pos in targets.items()}
            self._send(frame)
            if self.backend != "dry-run":
                time.sleep(step_delay)

    def move_relative(self, deltas: dict[str, float], **kwargs):
        """Moves each joint by a delta (degrees) from its CURRENT position.

        Works on every backend and needs no calibration, which makes it the
        safe primitive for canned gestures.
        """
        start = self.positions()
        targets = {name: start[name] + delta
                   for name, delta in deltas.items() if name in start}
        self.move_joints(targets, **kwargs)

    def move_to(self, joints: JointAngles, **kwargs):
        """Moves to a kinematics solution (applies servo sign/offset map).

        Only meaningful on a calibrated arm (lerobot backend) or in dry-run.
        """
        if self.backend == "feetech":
            log.warning("[%s arm] absolute IK move on the uncalibrated "
                        "feetech backend - verify the pose carefully", self.side)
        self.move_joints(joints.to_servo(), **kwargs)

    def set_torque(self, enabled: bool, joints: tuple[str, ...] | None = None):
        """Enables/disables torque (feetech backend).

        ``joints`` limits the change to specific joints (e.g. only the
        gripper, so the rest of the arm keeps holding its pose while the
        jaws are set by hand); None affects every servo.
        """
        if self._bus is None:
            log.info("[%s arm] set_torque ignored (backend %s)",
                     self.side, self.backend)
            return
        names = joints if joints is not None else tuple(config.FEETECH_SERVO_IDS)
        for name in names:
            self._bus.set_torque(config.FEETECH_SERVO_IDS[name], enabled)
        log.info("[%s arm] torque %s (%s)", self.side,
                 "ON" if enabled else "OFF",
                 "all joints" if joints is None else ", ".join(names))

    def open_gripper(self):
        log.debug("[%s arm] open gripper", self.side)
        self.move_joints({"gripper": config.GRIPPER_OPEN}, steps=10)

    def close_gripper(self):
        log.debug("[%s arm] close gripper", self.side)
        self.move_joints({"gripper": config.GRIPPER_CLOSED}, steps=10)


def make_arms(backend: str = config.ARM_BACKEND,
              sides: tuple[str, ...] = config.ACTIVE_ARMS) -> dict[str, ArmController]:
    """Builds controllers for the arms that are actually attached."""
    ports = {"left": (config.LEFT_ARM_PORT, config.LEFT_ARM_ID),
             "right": (config.RIGHT_ARM_PORT, config.RIGHT_ARM_ID)}
    return {side: ArmController(side, *ports[side], backend=backend)
            for side in sides}
