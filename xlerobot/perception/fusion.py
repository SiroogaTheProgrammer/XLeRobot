"""Lidar + depth-camera fusion.

The lidar assists the depth camera: the camera's face detection (with its
depth reading) is the primary "a person is here" signal, and the lidar is
used to verify it and to refine where the person is standing relative to
the robot. Confidence levels:

* ``confirmed`` - a face and a lidar person blob agree on bearing and range.
* ``camera``    - a face with valid depth, but no matching lidar blob
                  (e.g. the person is in the lidar's blind zone).
* ``lidar``     - a moving person-sized lidar blob with no visible face
                  (e.g. the person faces away from the camera).

All bearings follow the project convention: 0 = ahead, positive = left.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from xlerobot import config
from xlerobot.perception.depth_camera import FaceDetection
from xlerobot.perception.lidar_tracker import LidarDetection


@dataclass
class PersonTarget:
    """The fused best estimate of the nearest person."""
    bearing_deg: float
    distance_m: float
    confidence: str            # "confirmed" | "camera" | "lidar"
    face: FaceDetection | None
    lidar: LidarDetection | None
    timestamp: float

    @property
    def side(self) -> str:
        """Rough relative position in front of the robot."""
        if self.bearing_deg > 10.0:
            return "left"
        if self.bearing_deg < -10.0:
            return "right"
        return "center"


def _match(face: FaceDetection, blobs: list[LidarDetection]) -> LidarDetection | None:
    """Finds the lidar blob that best corroborates this face, if any."""
    best, best_err = None, None
    for blob in blobs:
        if blob.kind == "wall":
            continue
        bearing_err = abs(face.bearing_deg - blob.bearing_deg)
        if bearing_err > config.FUSION_BEARING_TOL_DEG:
            continue
        if face.distance_m is not None:
            dist_err = abs(face.distance_m - blob.distance_m)
            if dist_err > config.FUSION_DIST_TOL_M:
                continue
        else:
            dist_err = 0.0
        err = bearing_err / config.FUSION_BEARING_TOL_DEG + dist_err / config.FUSION_DIST_TOL_M
        if best_err is None or err < best_err:
            best, best_err = blob, err
    return best


def fuse(faces: list[FaceDetection],
         lidar_blobs: list[LidarDetection]) -> PersonTarget | None:
    """Returns the nearest fused person target, or None if nobody is around."""
    now = time.time()
    cutoff = now - config.SENSOR_STALE_S
    faces = [f for f in faces if f.timestamp >= cutoff]
    lidar_blobs = [b for b in lidar_blobs if b.timestamp >= cutoff]

    targets: list[PersonTarget] = []
    matched_blobs: set[int] = set()

    for face in faces:
        blob = _match(face, lidar_blobs)
        if blob is not None:
            matched_blobs.add(id(blob))
            # The camera depth verifies the lidar range; average them for the
            # final estimate, trust the camera for bearing (faces are precise).
            distance = (blob.distance_m if face.distance_m is None
                        else (face.distance_m + blob.distance_m) / 2.0)
            targets.append(PersonTarget(
                bearing_deg=face.bearing_deg, distance_m=distance,
                confidence="confirmed", face=face, lidar=blob, timestamp=now))
        elif face.distance_m is not None:
            targets.append(PersonTarget(
                bearing_deg=face.bearing_deg, distance_m=face.distance_m,
                confidence="camera", face=face, lidar=None, timestamp=now))

    for blob in lidar_blobs:
        if blob.kind == "person" and id(blob) not in matched_blobs:
            targets.append(PersonTarget(
                bearing_deg=blob.bearing_deg, distance_m=blob.distance_m,
                confidence="lidar", face=None, lidar=blob, timestamp=now))

    if not targets:
        return None
    # Prefer higher-confidence targets, then the nearest one.
    rank = {"confirmed": 0, "camera": 1, "lidar": 2}
    return min(targets, key=lambda t: (rank[t.confidence], t.distance_m))
