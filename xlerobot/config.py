"""Central configuration for the XLeRobot drink-serving robot.

User-tunable behaviour knobs (serve distance/angle, ports, files...) live
in the top-level ``settings.py`` and are re-exported here; everything else
in this file is internal wiring and geometry.

Coordinate conventions used across the whole project
----------------------------------------------------
Robot frame:  x forward (metres), y left, z up. Origin on the floor at the
              centre of the robot base.
Bearing:      degrees, 0 = straight ahead, positive = to the robot's LEFT,
              negative = to the robot's RIGHT. Both the lidar and the camera
              report person bearings in this convention so they can be fused.
"""

from dataclasses import dataclass

from settings import (  # noqa: F401  (re-exported for the rest of the code)
    AIM_GAIN,
    AIM_MAX_PAN_DEG,
    AIM_PAN_SIGN,
    ARM1_PORT,
    ARM2_PORT,
    ARM_POSES_FILE,
    CALIBRATION_DIR,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CAN_COLS,
    CAN_ROWS,
    CAN_STATE_FILE,
    FACE_DETECT_WIDTH,
    GRIP_SQUEEZE_DEG,
    HANDOVER_HOLD_S,
    LEADER_ARM_PORT,
    LIDAR_PORT,
    LIDAR_VIEW_RANGE_M,
    LIDAR_VIEW_SIZE_PX,
    RELEASE_DELAY_S,
    SERVE_COOLDOWN_S,
    SERVE_DISTANCE_M,
    SERVE_LIDAR_RANGE_M,
    SERVE_MAX_BEARING_DEG,
    SERVE_STABLE_S,
    STEP_PAUSE_S,
    TELEOP_HZ,
)

# --------------------------------------------------------------------------
# Lidar (RPLIDAR A1M8) - values taken from the verified examples/lidar.py
# --------------------------------------------------------------------------
LIDAR_BAUD_RATE = 115200
LIDAR_D_MAX = 12000          # max usable distance (mm)
LIDAR_D_MIN = 100            # ignore readings closer than this (mm)

# The lidar reports raw angles increasing clockwise, with 0 at its own
# "front". These two values map raw lidar angles into robot bearings:
#   bearing = LIDAR_ANGLE_SIGN * raw_angle + LIDAR_YAW_OFFSET_DEG
# Adjust LIDAR_YAW_OFFSET_DEG if the lidar is mounted rotated on the base.
LIDAR_ANGLE_SIGN = -1
LIDAR_YAW_OFFSET_DEG = 0.0

# Scan buffer / classifier tuning (identical to examples/lidar.py)
LIDAR_ANGLE_RES = 0.45       # degrees per bin
LIDAR_MAX_AGE = 10           # frames a bin survives without a fresh reading
LIDAR_CLUSTER_GAP = 180      # base gap (mm) that splits two clusters
LIDAR_WALL_MIN_LEN = 900     # walls are long...
LIDAR_WALL_MAX_RESID = 60    # ...and straight (max deviation from fitted line)
LIDAR_PERSON_MIN_W = 80      # person-sized blob width (mm)
LIDAR_PERSON_MAX_W = 700
LIDAR_PERSON_MIN_PTS = 4
LIDAR_STATIC_TOL = 120       # mm a reading may wobble and still be "same surface"
LIDAR_STATIC_FRAMES = 6      # frames a bin must stay stable to count as static
LIDAR_MOVE_THRESH = 130      # mm a blob must drift to count as moving
LIDAR_PERSON_HOLD = 40       # frames a blob stays "person" after it stops moving
LIDAR_TRACK_MATCH = 500      # max centroid jump (mm/frame) to keep a track
LIDAR_TRACK_TIMEOUT = 15     # frames before a lost track is dropped

# --------------------------------------------------------------------------
# Depth camera (Orbbec Gemini 330) - setup mirrors examples/depth_camera.py
# --------------------------------------------------------------------------
CAMERA_HFOV_DEG = 90.0       # horizontal field of view of the colour stream
CAMERA_YAW_OFFSET_DEG = 0.0  # adjust if the camera is mounted rotated
FACE_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
FACE_SCORE_THRESHOLD = 0.9
FACE_NMS_THRESHOLD = 0.3
FACE_TOP_K = 5000

# --------------------------------------------------------------------------
# Sensor fusion
# --------------------------------------------------------------------------
FUSION_BEARING_TOL_DEG = 12.0   # max bearing mismatch between lidar & camera
FUSION_DIST_TOL_M = 0.45        # max range mismatch between lidar & camera
SENSOR_STALE_S = 0.8            # ignore sensor readings older than this

# --------------------------------------------------------------------------
# Serving behaviour: SERVE_* and HANDOVER_HOLD_S come from settings.py
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Arms (2x SO-101 follower, Feetech STS3215 servos over USB serial)
# --------------------------------------------------------------------------
# Which arms are physically attached, and how to drive them:
#   "feetech" - direct servo control, no calibration needed (safe for the
#               relative "general handout" gesture). Default.
#   "lerobot" - lerobot SO101Follower, required for absolute IK serving
#               (needs `pip install lerobot[feetech]` + lerobot-calibrate).
#   "dry-run" - log motions instead of sending them.
ACTIVE_ARMS = ("left",)        # the single connected arm (CH343 USB serial)
ARM_BACKEND = "feetech"

LEFT_ARM_PORT = ARM1_PORT
RIGHT_ARM_PORT = ARM2_PORT
LEFT_ARM_ID = "left_follower"
RIGHT_ARM_ID = "right_follower"

# Direct feetech backend
FEETECH_BAUD = 1_000_000
FEETECH_GOAL_ACC = 30          # gentle acceleration profile
FEETECH_SERVO_IDS = {          # standard SO-101 daisy-chain order
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# "General handout" gesture: joint deltas (degrees) applied RELATIVE to the
# arm's current pose, then reversed, so it is calibration-free. Start the
# arm in its folded rest pose. Flip a sign if a joint moves the wrong way.
HANDOUT_EXTEND_DELTAS = {
    "shoulder_lift": -30.0,    # lean the upper arm forward
    "elbow_flex": 40.0,        # unfold the forearm
    "wrist_flex": -10.0,       # keep the gripper roughly level
}
HANDOUT_GRIPPER_OPEN_DELTA = 30.0   # how far the gripper opens to offer

# A gesture recorded with configure_arm.py overrides the defaults above.
HANDOUT_GESTURE_FILE = "handout_gesture.json"

# SO-101 link lengths (metres)
ARM_UPPER_ARM_M = 0.1159     # shoulder -> elbow
ARM_FOREARM_M = 0.1350       # elbow -> wrist
ARM_GRIPPER_M = 0.108        # wrist -> gripper tip
ARM_SHOULDER_HEIGHT_M = 0.12  # arm base plate -> shoulder_lift axis

# Where each arm base sits in the robot frame (x forward, y left, z up).
ARM_BASE_OFFSETS = {
    "left": (0.0, 0.11, 0.85),    # metres; z = tabletop height of the arm base
    "right": (0.0, -0.11, 0.85),
}

ARM_MAX_REACH_M = ARM_UPPER_ARM_M + ARM_FOREARM_M + ARM_GRIPPER_M
ARM_JOINT_LIMITS_DEG = {      # conservative software limits
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-110.0, 110.0),
}
GRIPPER_OPEN = 50.0           # gripper joint command, degrees
GRIPPER_CLOSED = 2.0
ARM_MOVE_STEPS = 40           # interpolation steps per motion
ARM_STEP_DELAY_S = 0.03       # delay between interpolation steps

# Handover geometry
HANDOVER_HEIGHT_M = 0.12      # bottle height above the arm base during handover
HANDOVER_REACH_FRAC = 0.90    # how far towards max reach the arm extends
PICK_APPROACH_UP_M = 0.06     # vertical clearance above a slot before/after grab
PICK_PITCH_DEG = 0.0          # gripper pitch when grabbing a bottle

# --------------------------------------------------------------------------
# Live display (--show): LIDAR_VIEW_* come from settings.py
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Bottle slots
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BottleSlot:
    """A fixed rack position, expressed in the frame of the arm that serves it."""
    slot_id: str
    arm: str        # "left" or "right"
    x: float        # metres, forward of that arm's base
    y: float        # metres, left of that arm's base
    z: float        # metres, above that arm's base plate

# Measure these on the real rack and update. z is the grab height of the
# bottle neck/can body relative to the arm base plate. NOTE: for a level
# grab (PICK_PITCH_DEG = 0) at z = 0.05 the slot must sit roughly
# 0.26 - 0.34 m (radially) from that arm's base, otherwise IK rejects it.
BOTTLE_SLOTS = [
    BottleSlot("L1", "left", 0.26, 0.10, 0.05),
    BottleSlot("L2", "left", 0.32, 0.10, 0.05),
    BottleSlot("R1", "right", 0.26, -0.10, 0.05),
    BottleSlot("R2", "right", 0.32, -0.10, 0.05),
]

BOTTLE_STATE_FILE = "bottle_state.json"
