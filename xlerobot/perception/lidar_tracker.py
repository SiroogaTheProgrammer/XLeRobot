"""Headless lidar person/object tracker.

This reuses the proven scan-buffer + shape/motion classifier from
``examples/lidar.py`` (same USB port, baud rate and tuning), but instead of
drawing a matplotlib plot it runs in a background thread and publishes a
thread-safe list of :class:`LidarDetection` objects. Its job in the system:

* report the nearest things around the robot, and
* flag person-sized moving blobs so the depth camera's face detections can
  be cross-checked against them (see ``perception/fusion.py``).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from rplidar import RPLidar

from xlerobot import config

log = logging.getLogger(__name__)

N_BINS = int(round(360 / config.LIDAR_ANGLE_RES))


def _normalize_bearing(deg: float) -> float:
    """Wrap any angle into [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class LidarDetection:
    """One clustered blob seen by the lidar, in robot-frame polar coords."""
    kind: str            # "wall" | "person" | "static" | "unknown"
    bearing_deg: float   # 0 = ahead, positive = left (see config docstring)
    distance_m: float    # range to the blob centroid
    width_m: float       # extent along the blob's main axis
    timestamp: float


class ScanBuffer:
    """Keeps the freshest distance per angle bin, plus how long each bin has
    been stable (the basis for static-vs-moving decisions)."""

    def __init__(self):
        self.dist = np.full(N_BINS, np.nan)
        self.age = np.full(N_BINS, config.LIDAR_MAX_AGE + 1, dtype=int)
        self.stable = np.zeros(N_BINS, dtype=int)

    def add_scan(self, scan, max_mm: float = config.LIDAR_D_MAX):
        self.age += 1
        for _quality, angle, distance in scan:
            if config.LIDAR_D_MIN <= distance <= max_mm:
                b = int(angle / config.LIDAR_ANGLE_RES) % N_BINS
                old = self.dist[b]
                if np.isfinite(old) and abs(old - distance) < config.LIDAR_STATIC_TOL:
                    # Same surface: smooth the jitter and bump stability
                    self.dist[b] = 0.6 * distance + 0.4 * old
                    self.stable[b] += 1
                else:
                    self.dist[b] = distance
                    self.stable[b] = 0
                self.age[b] = 0
        # Expire stale bins so moving objects don't leave trails
        expired = self.age > config.LIDAR_MAX_AGE
        self.dist[expired] = np.nan
        self.stable[expired] = 0

    def points(self):
        """Returns (angle_deg_raw, dist_mm, is_static) of live bins."""
        idx = np.nonzero(np.isfinite(self.dist))[0]
        angles = idx * config.LIDAR_ANGLE_RES
        return angles, self.dist[idx], self.stable[idx] >= config.LIDAR_STATIC_FRAMES


def _split_clusters(pts, dist):
    """Clusters consecutive points, with range-adaptive gap and 360 wrap."""
    n = len(pts)
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    thresholds = config.LIDAR_CLUSTER_GAP + 0.03 * np.minimum(dist[:-1], dist[1:])
    breaks = np.nonzero(gaps > thresholds)[0] + 1
    clusters = np.split(np.arange(n), breaks)
    if len(clusters) > 1:
        d_wrap = np.linalg.norm(pts[clusters[-1][-1]] - pts[clusters[0][0]])
        if d_wrap < config.LIDAR_CLUSTER_GAP + 0.03 * dist[0]:
            clusters[0] = np.concatenate((clusters[-1], clusters[0]))
            clusters.pop()
    return clusters


class SceneClassifier:
    """Combines blob shape, per-bin stability and motion tracking over time:

      * long + straight                      -> wall
      * compact blob that has actually moved -> person
      * anything that just sits there        -> static object (box, chair...)
    """

    def __init__(self):
        self.tracks = []   # [{'pos', 'anchor', 'last_seen', 'person_until'}]
        self.frame = 0

    def _update_track(self, centroid):
        best, best_d = None, config.LIDAR_TRACK_MATCH
        for tr in self.tracks:
            d = np.linalg.norm(tr["pos"] - centroid)
            if d < best_d:
                best, best_d = tr, d
        if best is None:
            best = {"pos": centroid, "anchor": centroid.copy(),
                    "last_seen": self.frame, "person_until": -1}
            self.tracks.append(best)
        best["pos"] = centroid
        best["last_seen"] = self.frame
        # The anchor trails slowly behind the blob. A blob that wanders away
        # from its anchor is genuinely moving; sensor jitter never gets far.
        best["anchor"] = 0.97 * best["anchor"] + 0.03 * centroid
        if np.linalg.norm(centroid - best["anchor"]) > config.LIDAR_MOVE_THRESH:
            best["person_until"] = self.frame + config.LIDAR_PERSON_HOLD
        return self.frame <= best["person_until"]

    def classify(self, angles_deg, dist_mm, static):
        """Returns (detections, point_kinds): one LidarDetection per clustered
        blob, plus the per-point classification used for visualization."""
        self.frame += 1
        now = time.time()
        detections: list[LidarDetection] = []
        point_kinds = np.array(["unknown"] * len(angles_deg), dtype=object)
        if len(angles_deg) >= 3:
            theta = np.radians(angles_deg)
            pts = np.column_stack((dist_mm * np.cos(theta), dist_mm * np.sin(theta)))
            for cl in _split_clusters(pts, dist_mm):
                if len(cl) < 3:
                    continue
                p = pts[cl]
                centroid = p.mean(axis=0)
                centered = p - centroid
                # PCA: main direction + perpendicular spread
                _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
                along = centered @ vt[0]
                across = centered @ vt[1]
                length = along.max() - along.min()
                resid = np.abs(across).max()
                static_frac = static[cl].mean()

                person_sized = (config.LIDAR_PERSON_MIN_W <= length <= config.LIDAR_PERSON_MAX_W
                                and len(cl) >= config.LIDAR_PERSON_MIN_PTS)
                moving = (self._update_track(centroid)
                          if length <= config.LIDAR_PERSON_MAX_W * 1.3 else False)

                if length >= config.LIDAR_WALL_MIN_LEN and resid <= config.LIDAR_WALL_MAX_RESID:
                    kind = "wall"
                elif person_sized and moving:
                    kind = "person"
                elif static_frac >= 0.6:
                    kind = "static"
                else:
                    kind = "unknown"
                point_kinds[cl] = kind

                raw_angle = float(np.degrees(np.arctan2(centroid[1], centroid[0]))) % 360.0
                bearing = _normalize_bearing(
                    config.LIDAR_ANGLE_SIGN * raw_angle + config.LIDAR_YAW_OFFSET_DEG)
                detections.append(LidarDetection(
                    kind=kind,
                    bearing_deg=bearing,
                    distance_m=float(np.linalg.norm(centroid)) / 1000.0,
                    width_m=float(length) / 1000.0,
                    timestamp=now,
                ))
        self.tracks = [t for t in self.tracks
                       if self.frame - t["last_seen"] <= config.LIDAR_TRACK_TIMEOUT]
        return detections, point_kinds


class LidarTracker:
    """Background thread that owns the RPLidar and publishes detections."""

    def __init__(self, port: str = config.LIDAR_PORT,
                 baudrate: int = config.LIDAR_BAUD_RATE,
                 active: bool = True):
        self._port = port
        self._baudrate = baudrate
        self._lock = threading.Lock()
        self._detections: list[LidarDetection] = []
        # Point cloud "mental image": (bearing_deg, dist_mm, kind) per point
        self._cloud: tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.empty(0), np.empty(0), np.empty(0, dtype=object))
        self._stop = threading.Event()
        self._active = threading.Event()   # scanning on/off (motor control)
        if active:
            self._active.set()
        self._max_range_mm = float(config.LIDAR_D_MAX)
        self._thread: threading.Thread | None = None
        self._last_scan = 0.0     # time of the last successful scan

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lidar", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def set_active(self, active: bool):
        """Starts/stops scanning (and the spinning motor) on the fly."""
        if active and not self._active.is_set():
            log.info("Lidar activated (range %.1f m)",
                     self._max_range_mm / 1000.0)
            self._active.set()
        elif not active and self._active.is_set():
            log.info("Lidar deactivated (standby)")
            self._active.clear()

    def is_active(self) -> bool:
        return self._active.is_set()

    def set_max_range(self, range_m: float):
        """Limits how far the lidar looks (applies from the next scan)."""
        self._max_range_mm = min(float(range_m) * 1000.0,
                                 float(config.LIDAR_D_MAX))

    def _run(self):
        """Wait-for-active / connect / stream / reconnect loop.

        While inactive the sensor is disconnected and its motor stopped.
        The RPLidar serial stream is easily left in a wedged state (e.g. by
        a previous process that was killed mid-scan), which makes the next
        connection fail or raise mid-stream. Instead of dying - and leaving
        an empty radar - we reset the sensor and reconnect until stopped.
        """
        while not self._stop.is_set():
            if not self._active.is_set():
                self._stop.wait(0.2)
                continue
            lidar = None
            failed = False
            try:
                lidar = RPLidar(self._port, baudrate=self._baudrate)
                buffer = ScanBuffer()
                classifier = SceneClassifier()
                # max_buf_meas raised so slow frames don't abort the stream
                for scan in lidar.iter_scans(max_buf_meas=3000):
                    if self._stop.is_set() or not self._active.is_set():
                        break
                    buffer.add_scan(scan, self._max_range_mm)
                    angles, dist, static = buffer.points()
                    detections, kinds = classifier.classify(angles, dist, static)
                    bearings = np.array([
                        _normalize_bearing(config.LIDAR_ANGLE_SIGN * a
                                           + config.LIDAR_YAW_OFFSET_DEG)
                        for a in angles])
                    with self._lock:
                        self._detections = detections
                        self._cloud = (bearings, dist.copy(), kinds)
                        self._last_scan = time.time()
            except Exception as exc:
                failed = True
                log.warning("Lidar error: %s - resetting and reconnecting", exc)
            finally:
                if lidar is not None:
                    for action in (lidar.stop, lidar.stop_motor, lidar.disconnect):
                        try:
                            action()
                        except Exception:
                            pass
            if self._stop.is_set():
                break
            if not failed:
                continue   # deactivated on purpose - go back to waiting
            # Reset clears a wedged sensor left mid-scan by a dead process
            try:
                lidar = RPLidar(self._port, baudrate=self._baudrate)
                lidar.reset()
                lidar.disconnect()
            except Exception as exc:
                log.warning("Lidar reset failed: %s", exc)
            self._stop.wait(2.0)

    # -- queries -----------------------------------------------------------

    def detections(self) -> list[LidarDetection]:
        """Latest blob list (fresh copy, safe to iterate)."""
        with self._lock:
            return list(self._detections)

    def point_cloud(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Latest classified point cloud: (bearing_deg, dist_mm, kind) arrays.

        Bearings follow the robot convention (0 = ahead, positive = left),
        so this is the lidar's live "mental image" of the surroundings.
        """
        with self._lock:
            return self._cloud

    def _fresh(self) -> list[LidarDetection]:
        cutoff = time.time() - config.SENSOR_STALE_S
        return [d for d in self.detections() if d.timestamp >= cutoff]

    def last_scan_age(self) -> float:
        """Seconds since the last successful scan (inf before the first)."""
        with self._lock:
            last = self._last_scan
        return time.time() - last if last else float("inf")

    def nearest_object(self) -> LidarDetection | None:
        """Nearest non-wall blob of any kind."""
        candidates = [d for d in self._fresh() if d.kind != "wall"]
        return min(candidates, key=lambda d: d.distance_m, default=None)

    def nearest_person(self) -> LidarDetection | None:
        """Nearest blob the classifier currently believes is a person."""
        candidates = [d for d in self._fresh() if d.kind == "person"]
        return min(candidates, key=lambda d: d.distance_m, default=None)
