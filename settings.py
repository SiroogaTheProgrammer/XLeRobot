"""XLeRobot user settings - edit these to tune the robot's behaviour.

This file holds every knob you are expected to touch. The rest of the
wiring (geometry, servo IDs, classifier tuning...) lives in
xlerobot/config.py, which imports this file.
"""

# ---------------------------------------------------------------------------
# When is a person "close enough" to get a drink?
# ---------------------------------------------------------------------------
SERVE_DISTANCE_M = 1.2        # person must be within this range (metres)
SERVE_MAX_BEARING_DEG = 35.0  # ...and within +/- this angle from straight
                              # ahead (0 = dead centre of the camera)
SERVE_STABLE_S = 2.0          # ...continuously for this many seconds
SERVE_LIDAR_RANGE_M = 2.0     # lidar range while serving (it only runs
                              # during the arm motion to track the person;
                              # standby detection is camera-only)
SERVE_COOLDOWN_S = 7.0       # pause after a serve before serving again
RELEASE_DELAY_S = 1.5         # ARM2 holds the can at the final pose this
                              # long BEFORE opening the gripper
HANDOVER_HOLD_S = 3.0         # how long the arm holds the drink out

# ---------------------------------------------------------------------------
# Hardware ports (check with:  ls -l /dev/serial/by-id/ )
# by-id paths are stable no matter the order devices are plugged in.
# ---------------------------------------------------------------------------
LIDAR_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"  # set by find_ports.py
# ARM1 picks cans from the rack; ARM2 stands near the person and hands over.
ARM1_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D045687-if00"  # set by find_ports.py
ARM2_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E091223-if00"  # set by find_ports.py
LEADER_ARM_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF262866-if00"  # set by find_ports.py

# ---------------------------------------------------------------------------
# Camera performance
# ---------------------------------------------------------------------------
CAMERA_WIDTH = 640            # requested colour stream size / rate; falls
CAMERA_HEIGHT = 480           # back to the camera default if unavailable
CAMERA_FPS = 60
FACE_DETECT_WIDTH = 320       # face detection runs on a copy downscaled to
                              # this width (smaller = faster, shorter range)

# ---------------------------------------------------------------------------
# Handover aiming
# ---------------------------------------------------------------------------
# The recorded FINAL pose points at the camera centre. When serving, the
# base (shoulder_pan) is rotated towards where the fused lidar+camera
# target actually stands:  offset = AIM_PAN_SIGN * AIM_GAIN * bearing.
AIM_GAIN = 1.0                # degrees of base rotation per degree of bearing
AIM_MAX_PAN_DEG = 25.0        # max swing away from the recorded final pose
AIM_PAN_SIGN = 1              # flip to -1 if the arm aims the wrong way

# ---------------------------------------------------------------------------
# Can rack (configured with setup_arms.py by recording the 4 corner cans)
# ---------------------------------------------------------------------------
CAN_ROWS = 2                  # how many rows of cans in the rack
CAN_COLS = 4                  # how many cans per row
GRIP_SQUEEZE_DEG = {          # extra squeeze past the hand-recorded on-can
    "arm1": 12.0,             # grip reference, per arm, in the closing
    "arm2": 4.0,              # direction (raise a value to clamp harder)
}
STEP_PAUSE_S = 1.5            # stillness pause between every serve step
ARM_POSES_FILE = "arm_poses.json"   # start / corner / final poses
CAN_STATE_FILE = "can_state.json"   # which cans were already handed out

# ---------------------------------------------------------------------------
# Arm configuration files (created by setup_arms.py)
# ---------------------------------------------------------------------------
CALIBRATION_DIR = "calibrations"       # per-arm joint range files
TELEOP_HZ = 30                         # teleop / recording sample rate

# ---------------------------------------------------------------------------
# Live display (--show)
# ---------------------------------------------------------------------------
LIDAR_VIEW_RANGE_M = 4.0      # radius shown in the lidar map (metres)
LIDAR_VIEW_SIZE_PX = 700      # lidar map window size (square, pixels)
