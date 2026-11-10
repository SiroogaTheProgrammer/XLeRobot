"""Headless depth-camera face tracker.

Reuses the proven Orbbec pipeline setup, D2C alignment, YuNet face detection
and median-depth sampling from ``examples/depth_camera.py``, but runs in a
background thread and publishes :class:`FaceDetection` objects instead of
drawing a window (an optional preview window is available for debugging).

Each detection carries a robot-frame bearing (derived from the face's pixel
position and the camera's horizontal FOV) and a metric distance (from the
aligned depth frame), so it can be cross-checked against the lidar in
``perception/fusion.py``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

from xlerobot import config

log = logging.getLogger(__name__)


def download_face_detector_model() -> str:
    model_path = config.FACE_MODEL_PATH
    if not os.path.exists(model_path):
        log.info("Downloading YuNet face detection model...")
        urllib.request.urlretrieve(config.FACE_MODEL_URL, model_path)
    return model_path


def get_face_distance_mm(depth_data, depth_scale, box):
    """Median depth (in mm) over the central region of a face bounding box."""
    if depth_data is None:
        return None
    x, y, w, h = box
    # Sample the central 50% of the box to avoid background pixels at the edges
    cx0 = max(x + w // 4, 0)
    cy0 = max(y + h // 4, 0)
    cx1 = min(x + (3 * w) // 4, depth_data.shape[1])
    cy1 = min(y + (3 * h) // 4, depth_data.shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    region = depth_data[cy0:cy1, cx0:cx1]
    valid = region[region > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


@dataclass
class FaceDetection:
    """One detected face, localized in the robot frame."""
    bearing_deg: float        # 0 = ahead, positive = left
    distance_m: float | None  # None when no valid depth under the face
    box: tuple[int, int, int, int]   # x, y, w, h in the colour image
    score: float
    timestamp: float


class FaceTracker:
    """Background thread that owns the Orbbec pipeline and publishes faces.

    Also publishes the latest annotated colour frame (``latest_frame()``) so
    the main thread can display the live feed; OpenCV windows must be driven
    from the main thread on Linux.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._faces: list[FaceDetection] = []
        self._frame = None          # latest annotated BGR frame
        self._fps = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self):
        pipeline = Pipeline()
        ob_config = Config()
        try:
            color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = self._pick_profile(
                color_profile_list, OBFormat.RGB, "colour")
            ob_config.enable_stream(color_profile)

            depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = self._pick_profile(
                depth_profile_list, OBFormat.Y16, "depth")
            ob_config.enable_stream(depth_profile)

            # Only deliver frame sets that contain both color and depth
            ob_config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

            pipeline.start(ob_config)
        except Exception:
            log.exception("Failed to start the Orbbec pipeline")
            return

        # Align depth frames into the color camera's viewpoint (D2C)
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        model_path = download_face_detector_model()
        face_detector = cv2.FaceDetectorYN.create(
            model_path, "", (640, 480),
            config.FACE_SCORE_THRESHOLD, config.FACE_NMS_THRESHOLD, config.FACE_TOP_K,
        )

        log.info("Depth camera face tracking active")
        fps_t0, fps_n = time.monotonic(), 0
        try:
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(100)
                if not frames:
                    continue

                # Project depth into the color camera view so pixels match 1:1
                aligned = align_filter.process(frames)
                if aligned:
                    frames = aligned

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                depth_frame = frames.get_depth_frame()
                depth_data = None
                depth_scale = 1.0
                if depth_frame:
                    try:
                        depth_data = np.frombuffer(
                            depth_frame.get_data(), dtype=np.uint16
                        ).reshape((depth_frame.get_height(), depth_frame.get_width()))
                        depth_scale = depth_frame.get_depth_scale()
                    except ValueError:
                        depth_data = None

                w = color_frame.get_width()
                h = color_frame.get_height()
                data = np.asanyarray(color_frame.get_data())

                if color_frame.get_format() == OBFormat.RGB:
                    frame = cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
                elif color_frame.get_format() == OBFormat.MJPG:
                    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                else:
                    frame = data.reshape((h, w, 3))

                # Distance sampling requires depth aligned to color resolution
                if depth_data is not None and depth_data.shape != frame.shape[:2]:
                    depth_data = None

                # Detect on a downscaled copy (much cheaper), scale boxes back
                scale = min(1.0, config.FACE_DETECT_WIDTH / frame.shape[1])
                if scale < 1.0:
                    small = cv2.resize(
                        frame, (int(frame.shape[1] * scale),
                                int(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA)
                else:
                    small = frame
                face_detector.setInputSize((small.shape[1], small.shape[0]))
                _, faces = face_detector.detect(small)

                fps_n += 1
                now = time.monotonic()
                if now - fps_t0 >= 2.0:
                    self._fps = fps_n / (now - fps_t0)
                    fps_t0, fps_n = now, 0

                self._publish(faces, scale, frame, depth_data, depth_scale)
        except Exception:
            log.exception("Camera thread crashed")
        finally:
            pipeline.stop()

    @staticmethod
    def _pick_profile(profile_list, fmt, label):
        """Requested size/fps from settings, with graceful fallback."""
        for width, height in ((config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                              (config.CAMERA_WIDTH, 0), (0, 0)):
            try:
                profile = profile_list.get_video_stream_profile(
                    width, height, fmt, config.CAMERA_FPS)
                log.info("%s stream: %dx%d@%d", label, profile.get_width(),
                         profile.get_height(), profile.get_fps())
                return profile
            except Exception:
                continue
        profile = profile_list.get_default_video_stream_profile()
        log.info("%s stream: default %dx%d@%d", label, profile.get_width(),
                 profile.get_height(), profile.get_fps())
        return profile

    def _publish(self, faces, scale, frame, depth_data, depth_scale):
        now = time.time()
        width = frame.shape[1]
        detections: list[FaceDetection] = []
        if faces is not None:
            for face in faces:
                # Detection ran on the downscaled copy - scale the box back
                box = tuple(int(v / scale) for v in face[:4])
                score = float(face[-1])
                face_cx = box[0] + box[2] / 2.0
                # Pixel offset from image centre -> bearing. Faces right of
                # centre are to the robot's right, i.e. a negative bearing.
                bearing = (-(face_cx / width - 0.5) * config.CAMERA_HFOV_DEG
                           + config.CAMERA_YAW_OFFSET_DEG)
                distance_mm = get_face_distance_mm(depth_data, depth_scale, box)
                detections.append(FaceDetection(
                    bearing_deg=bearing,
                    distance_m=distance_mm / 1000.0 if distance_mm is not None else None,
                    box=box,
                    score=score,
                    timestamp=now,
                ))
        with self._lock:
            self._faces = detections
            self._frame = self._annotate(frame, detections, self._fps)

    @staticmethod
    def _annotate(frame, detections, fps: float = 0.0):
        for det in detections:
            x, y, w, h = det.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = (f"{det.distance_m:.2f} m @ {det.bearing_deg:+.0f} deg"
                     if det.distance_m is not None else "Face (no depth)")
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if fps:
            cv2.putText(frame, f"{fps:.0f} fps", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1,
                        cv2.LINE_AA)
        return frame

    # -- queries -----------------------------------------------------------

    def fps(self) -> float:
        """Processing rate of the camera thread (frames per second)."""
        return self._fps

    def latest_frame(self):
        """Latest annotated BGR frame, or None before the first frame."""
        with self._lock:
            return self._frame

    def faces(self) -> list[FaceDetection]:
        """Latest face list (fresh copy, safe to iterate)."""
        with self._lock:
            return list(self._faces)

    def nearest_face(self) -> FaceDetection | None:
        """Nearest face with valid depth, ignoring stale frames."""
        cutoff = time.time() - config.SENSOR_STALE_S
        candidates = [f for f in self.faces()
                      if f.timestamp >= cutoff and f.distance_m is not None]
        return min(candidates, key=lambda f: f.distance_m, default=None)
