#!/usr/bin/env python3
"""Simple drink server: ONE arm, ONE camera, ONE can at a time.

When the camera sees a face inside the serve window (distance + angle,
see simple_settings.py) continuously for SERVE_STABLE_S, the arm picks
the can from its trained spot and hands it out. After the cooldown a
human puts a new can on the spot and it can serve again.

Usage:
    .venv/bin/python simple/main.py --show   # with the camera window
    .venv/bin/python simple/main.py          # headless
    (in the window: q quits, h triggers a serve manually)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

import simple_settings as s  # noqa: E402
from serving import load_poses, serve_once  # noqa: E402
from xlerobot.arm.controller import ArmController  # noqa: E402
from xlerobot.perception import fusion  # noqa: E402
from xlerobot.perception.depth_camera import FaceTracker  # noqa: E402

log = logging.getLogger("simple")

WINDOW = "Simple drink server (q quits, h serves)"


class SimpleServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.camera = FaceTracker()
        self.arm = ArmController("arm", s.ARM_PORT, "arm",
                                 backend=args.backend)
        self.poses = load_poses()
        self._target_since: float | None = None
        self._cooldown_until = 0.0
        self._serve_thread: threading.Thread | None = None

    @property
    def serving(self) -> bool:
        return self._serve_thread is not None and self._serve_thread.is_alive()

    def run(self):
        if self.poses is None:
            log.error("No trained poses (%s) - run simple/setup_arm.py first",
                      s.POSES_FILE)
            return
        log.info("Starting camera and connecting the arm...")
        self.camera.start()
        self.arm.connect()
        log.info("Ready - serve window: <= %.1f m, +/- %.0f deg, %.1f s stable",
                 s.SERVE_DISTANCE_M, s.SERVE_MAX_BEARING_DEG, s.SERVE_STABLE_S)
        try:
            while True:
                target = fusion.fuse(self.camera.faces(), [])  # camera only
                if self.args.show:
                    frame = self.camera.latest_frame()
                    if frame is not None:
                        cv2.imshow(WINDOW, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("h"):
                        self._start_serve()
                        continue
                if self._should_serve(target):
                    log.info("Person at %.2f m (%.0f deg) - serving",
                             target.distance_m, target.bearing_deg)
                    self._start_serve()
                time.sleep(0.02 if self.args.show else 0.1)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            if self._serve_thread is not None:
                self._serve_thread.join(timeout=60.0)
            if self.args.show:
                cv2.destroyAllWindows()
            self.camera.stop()
            self.arm.disconnect()

    def _should_serve(self, target) -> bool:
        if self.serving or time.time() < self._cooldown_until:
            return False
        in_window = (target is not None
                     and target.distance_m is not None
                     and target.distance_m <= s.SERVE_DISTANCE_M
                     and abs(target.bearing_deg) <= s.SERVE_MAX_BEARING_DEG)
        if not in_window:
            self._target_since = None
            return False
        if self._target_since is None:
            self._target_since = time.time()
            log.info("Person spotted at %.2f m (%.0f deg) - waiting...",
                     target.distance_m, target.bearing_deg)
            return False
        return time.time() - self._target_since >= s.SERVE_STABLE_S

    def _start_serve(self):
        if self.serving:
            log.info("Already serving - ignoring trigger")
            return
        self._target_since = None
        self._serve_thread = threading.Thread(target=self._serve_worker,
                                              name="serve", daemon=True)
        self._serve_thread.start()

    def _serve_worker(self):
        try:
            serve_once(self.arm, self.poses)
        except Exception:
            log.exception("Serve failed")
        finally:
            self._cooldown_until = time.time() + s.SERVE_COOLDOWN_S
            log.info("Cooling down %.0f s - put a new can on the spot",
                     s.SERVE_COOLDOWN_S)


def main():
    parser = argparse.ArgumentParser(description="Simple one-arm drink server")
    parser.add_argument("--show", action="store_true",
                        help="show the camera window")
    parser.add_argument("--backend", default="feetech",
                        choices=["feetech", "dry-run"],
                        help="arm backend (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    SimpleServer(args).run()


if __name__ == "__main__":
    main()
