"""Live on-screen visualization: camera feed + lidar "mental image".

Both windows are rendered with OpenCV from the MAIN thread (GUI calls from
worker threads are unreliable on Linux). The lidar map is a top-down view
with the robot at the centre and its front pointing up; point colours match
the original examples/lidar.py legend:

    blue   wall        red   person (moving)
    yellow static obj  gray  unknown
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from xlerobot import config
from xlerobot.perception.fusion import PersonTarget
from xlerobot.perception.lidar_tracker import LidarDetection

# BGR colours, same palette as examples/lidar.py
KIND_COLORS = {
    "wall": (180, 119, 31),      # #1f77b4 blue
    "person": (51, 51, 255),     # #ff3333 red
    "static": (11, 134, 184),    # #b8860b dark yellow
    "unknown": (153, 153, 153),  # #999999 gray
}

WIN_CAMERA = "XLeRobot - Camera (face tracking)"
WIN_LIDAR = "XLeRobot - Lidar map (front = up)"


def _to_px(bearing_deg: float, dist_mm: float,
           center: int, scale: float) -> tuple[int, int]:
    """Robot polar coords -> image pixel (front = up, left = left)."""
    b = math.radians(bearing_deg)
    x_fwd = dist_mm * math.cos(b)     # robot x (forward)
    y_left = dist_mm * math.sin(b)    # robot y (left)
    px = int(round(center - y_left * scale))
    py = int(round(center - x_fwd * scale))
    return px, py


def render_lidar_map(bearings_deg: np.ndarray,
                     dist_mm: np.ndarray,
                     kinds: np.ndarray,
                     detections: list[LidarDetection],
                     target: PersonTarget | None = None,
                     data_age_s: float = 0.0,
                     standby: bool = False) -> np.ndarray:
    """Draws the lidar's current mental image of the surroundings."""
    size = config.LIDAR_VIEW_SIZE_PX
    center = size // 2
    max_mm = config.LIDAR_VIEW_RANGE_M * 1000.0
    scale = (center - 20) / max_mm

    img = np.full((size, size, 3), 20, dtype=np.uint8)

    # Range rings every metre + front direction marker
    for ring_m in range(1, int(config.LIDAR_VIEW_RANGE_M) + 1):
        r = int(ring_m * 1000.0 * scale)
        cv2.circle(img, (center, center), r, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(img, f"{ring_m} m", (center + 4, center - r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
    cv2.arrowedLine(img, (center, center), (center, center - 36),
                    (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(img, (center, center), 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Classified point cloud
    for bearing, dist, kind in zip(bearings_deg, dist_mm, kinds):
        if dist > max_mm:
            continue
        px, py = _to_px(bearing, dist, center, scale)
        if 0 <= px < size and 0 <= py < size:
            img[py, px] = KIND_COLORS[kind]
            cv2.circle(img, (px, py), 1, KIND_COLORS[kind], -1)

    # Annotate person/static blobs with their range
    for det in detections:
        if det.kind not in ("person", "static") or det.distance_m * 1000 > max_mm:
            continue
        px, py = _to_px(det.bearing_deg, det.distance_m * 1000.0, center, scale)
        cv2.circle(img, (px, py), 10, KIND_COLORS[det.kind], 1, cv2.LINE_AA)
        cv2.putText(img, f"{det.kind} {det.distance_m:.2f} m",
                    (px + 12, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    KIND_COLORS[det.kind], 1, cv2.LINE_AA)

    # Fused person target (camera + lidar agreement)
    if target is not None and target.distance_m * 1000 <= max_mm:
        px, py = _to_px(target.bearing_deg, target.distance_m * 1000.0,
                        center, scale)
        cv2.drawMarker(img, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(img, f"TARGET {target.confidence} {target.distance_m:.2f} m",
                    (px + 14, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

    # Legend + help
    y = 22
    for kind, color in KIND_COLORS.items():
        cv2.circle(img, (16, y - 4), 4, color, -1)
        cv2.putText(img, kind, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
    cv2.putText(img, "q: quit   h: handout gesture", (12, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # Health banner: standby is intentional; staleness while active is not
    if standby:
        cv2.rectangle(img, (0, size // 2 - 24), (size, size // 2 + 10),
                      (60, 45, 20), -1)
        cv2.putText(img, "LIDAR: standby (camera-only detection)",
                    (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (230, 180, 100), 2, cv2.LINE_AA)
    elif data_age_s > 2.0:
        msg = ("LIDAR: waiting for data..." if data_age_s == float("inf")
               else f"LIDAR: NO DATA for {data_age_s:.0f} s (reconnecting...)")
        cv2.rectangle(img, (0, size // 2 - 24), (size, size // 2 + 10),
                      (0, 0, 90), -1)
        cv2.putText(img, msg, (20, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
    return img


class LiveDisplay:
    """Shows the camera and lidar windows; call update() from the main loop."""

    def __init__(self):
        cv2.namedWindow(WIN_CAMERA, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(WIN_LIDAR, cv2.WINDOW_AUTOSIZE)

    def update(self, camera_frame, bearings, dist, kinds,
               detections, target, lidar_age_s: float = 0.0,
               lidar_standby: bool = False) -> str | None:
        """Redraws both windows; returns the pressed key (if any)."""
        if camera_frame is None:
            camera_frame = np.full((360, 640, 3), 20, dtype=np.uint8)
            cv2.putText(camera_frame, "CAMERA: waiting for frames...",
                        (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (80, 80, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN_CAMERA, camera_frame)
        cv2.imshow(WIN_LIDAR,
                   render_lidar_map(bearings, dist, kinds, detections,
                                    target, data_age_s=lidar_age_s,
                                    standby=lidar_standby))
        key = cv2.waitKey(1) & 0xFF
        return chr(key) if key != 255 else None

    def close(self):
        cv2.destroyAllWindows()
