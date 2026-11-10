#!/usr/bin/env python3
"""XLeRobot drink-serving robot - main entry point.

Wires everything together:

* ``perception.lidar_tracker``  - the lidar's live mental image + person blobs
* ``perception.depth_camera``   - face detection/tracking with metric depth
* ``perception.fusion``         - cross-verified "nearest person" estimate
* ``display``                   - camera + lidar feeds on screen (--show)
* ``arm.can_rack``              - recorded poses, inventory, two-arm serving

Behaviour: in STANDBY the robot watches with the CAMERA ONLY (the lidar
motor is stopped). When a face stays inside the serve window
(SERVE_DISTANCE_M / SERVE_MAX_BEARING_DEG in settings.py) for
SERVE_STABLE_S, a serve starts in a background thread - so the camera and
lidar feeds keep running - and the LIDAR SPINS UP, capped to
SERVE_LIDAR_RANGE_M, to track the person during the arm motion:

    ARM1 picks the can and presents it at the handoff pose; ARM2 clamps it
    BEFORE ARM1 releases (it can never fall), then carries it to the final
    pose - aimed at the person's CURRENT fused bearing, re-read right
    before extending - waits RELEASE_DELAY_S, releases, holds, and both
    arms return to their start poses. The lidar then goes back to standby.

Each handed-out can is remembered in CAN_STATE_FILE across restarts.

Usage:
    python main.py --show            # live camera + lidar windows (q quits,
                                     # h triggers a manual handout)
    python main.py                   # headless
    python main.py --backend dry-run # no arm hardware needed
    python main.py --refill          # rack reloaded: mark all cans available
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

from xlerobot import config
from xlerobot.arm.calibration import ArmCalibration
from xlerobot.arm.can_rack import ArmPoses, CanInventory, serve_can
from xlerobot.arm.controller import ArmController
from xlerobot.display import LiveDisplay
from xlerobot.perception import fusion
from xlerobot.perception.depth_camera import FaceTracker
from xlerobot.perception.lidar_tracker import LidarTracker

log = logging.getLogger("main")


class DrinkServer:
    """Top-level state machine: STANDBY (camera-only) -> SERVING (lidar on,
    arms moving in a background thread) -> COOLDOWN -> STANDBY."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # Standby detection is camera-only; the lidar only spins while the
        # arms move, with its range capped to SERVE_LIDAR_RANGE_M.
        self.lidar = LidarTracker(active=False)
        self.camera = FaceTracker()
        self.arm1 = ArmController("arm1", config.ARM1_PORT, "arm1",
                                  backend=args.backend)
        self.arm2 = ArmController("arm2", config.ARM2_PORT, "arm2",
                                  backend=args.backend)
        self.display = LiveDisplay() if args.show else None
        self._target_since: float | None = None
        self._cooldown_until = 0.0
        self._serve_thread: threading.Thread | None = None
        # Rack poses + can inventory configured with setup_arms.py
        self.arm_poses = ArmPoses.load()
        self.cans = (CanInventory(self.arm_poses.rows, self.arm_poses.cols)
                     if self.arm_poses is not None else None)
        self.cal1 = ArmCalibration.load("arm1")
        self.cal2 = ArmCalibration.load("arm2")

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        log.info("Starting sensors and connecting arms...")
        self.lidar.set_max_range(config.SERVE_LIDAR_RANGE_M)
        self.lidar.start()      # stays in standby until a serve begins
        self.camera.start()
        self.arm1.connect()
        self.arm2.connect()
        if self.arm_poses is not None:
            log.info("Ready: %dx%d rack, %d cans remaining (two-arm handoff)",
                     self.arm_poses.rows, self.arm_poses.cols,
                     self.cans.remaining())
        else:
            log.warning("No two-arm poses (%s) - run setup_arms.py first; "
                        "serving is DISABLED", config.ARM_POSES_FILE)

    def shutdown(self):
        log.info("Shutting down...")
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=60.0)
        if self.display:
            self.display.close()
        self.lidar.stop()
        self.camera.stop()
        for arm in (self.arm1, self.arm2):
            try:
                arm.disconnect()
            except Exception:
                log.exception("[%s] error during disconnect", arm.side)

    # -- main loop -----------------------------------------------------------

    @property
    def serving(self) -> bool:
        return self._serve_thread is not None and self._serve_thread.is_alive()

    def run(self):
        self.start()
        try:
            while True:
                # STANDBY: the camera alone decides who gets a drink.
                # While serving, the (active) lidar joins the fusion.
                lidar_dets = self.lidar.detections() if self.serving else []
                target = fusion.fuse(self.camera.faces(), lidar_dets)

                if self.display:
                    bearings, dist, kinds = self.lidar.point_cloud()
                    key = self.display.update(
                        self.camera.latest_frame(), bearings, dist, kinds,
                        self.lidar.detections(), target,
                        lidar_age_s=self.lidar.last_scan_age(),
                        lidar_standby=not self.lidar.is_active())
                    if key == "q":
                        break
                    if key == "h":
                        self._start_serve(target)
                        continue

                if self._should_serve(target):
                    log.info("Serving person %s at %.2f m (%.0f deg, %s)",
                             target.side, target.distance_m,
                             target.bearing_deg, target.confidence)
                    self._start_serve(target)
                time.sleep(0.02 if self.display else 0.1)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            self.shutdown()

    def _should_serve(self, target: fusion.PersonTarget | None) -> bool:
        """True once a person has stayed inside the serve window long enough.

        The serve window is settings.py's SERVE_DISTANCE_M (range) and
        SERVE_MAX_BEARING_DEG (angle off straight-ahead).
        """
        if (self.serving or time.time() < self._cooldown_until
                or self.arm_poses is None):
            return False
        in_window = (target is not None
                     and target.distance_m <= config.SERVE_DISTANCE_M
                     and abs(target.bearing_deg) <= config.SERVE_MAX_BEARING_DEG)
        if not in_window:
            self._target_since = None
            return False
        if self._target_since is None:
            self._target_since = time.time()
            log.info("Person spotted %s at %.2f m (%.0f deg, %s) - waiting "
                     "for them to stay...", target.side, target.distance_m,
                     target.bearing_deg, target.confidence)
            return False
        return time.time() - self._target_since >= config.SERVE_STABLE_S

    # -- serving (background thread so the feeds never freeze) ----------------

    def _start_serve(self, target: fusion.PersonTarget | None):
        if self.serving:
            log.info("Already serving - ignoring trigger")
            return
        if self.arm_poses is None:
            log.warning("No two-arm poses recorded - run setup_arms.py")
            return
        can = self.cans.next_unused()
        if can is None:
            log.warning("Rack is empty - refill it and run main.py --refill")
            return
        self._target_since = None
        initial_bearing = target.bearing_deg if target is not None else None
        self._serve_thread = threading.Thread(
            target=self._serve_worker, args=(can, initial_bearing),
            name="serve", daemon=True)
        self._serve_thread.start()

    def _live_bearing(self, fallback: float | None):
        """Freshest fused person bearing (lidar is active during serves)."""
        target = fusion.fuse(self.camera.faces(), self.lidar.detections())
        return target.bearing_deg if target is not None else fallback

    def _serve_worker(self, can: tuple[int, int],
                      initial_bearing: float | None):
        # Lidar on for the whole arm motion (short range, tracks the person)
        self.lidar.set_max_range(config.SERVE_LIDAR_RANGE_M)
        self.lidar.set_active(True)
        try:
            serve_can(self.arm1, self.arm2, self.arm_poses, *can,
                      cal1=self.cal1, cal2=self.cal2,
                      aim_bearing_fn=lambda: self._live_bearing(initial_bearing))
            self.cans.mark_used(*can)
        except Exception:
            log.exception("Serve failed")
        finally:
            self.lidar.set_active(False)
            self._cooldown_until = time.time() + config.SERVE_COOLDOWN_S
            log.info("Cooling down for %.0f s", config.SERVE_COOLDOWN_S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XLeRobot drink-serving robot")
    parser.add_argument("--show", action="store_true",
                        help="show the camera and lidar feeds on screen")
    parser.add_argument("--backend", default=config.ARM_BACKEND,
                        choices=["feetech", "lerobot", "dry-run"],
                        help="arm control backend (default: %(default)s)")
    parser.add_argument("--refill", action="store_true",
                        help="rack was reloaded: mark all cans available "
                             "and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug logging")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.refill:
        CanInventory().reset()
        return
    DrinkServer(args).run()


if __name__ == "__main__":
    main()
