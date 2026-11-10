"""Settings for the SIMPLE one-arm / one-camera setup.

(Deliberately named simple_settings.py, not settings.py, so it never
shadows the main project's settings.py that xlerobot.config imports.)
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import settings as _root  # the main project's settings (ports etc.)

# The one arm of this setup (defaults to the main project's ARM1)
ARM_PORT = _root.ARM1_PORT

# When is a person close enough to get the can? (camera-only detection)
SERVE_DISTANCE_M = 1.2        # metres from the camera
SERVE_MAX_BEARING_DEG = 35.0  # +/- degrees off the camera centre
SERVE_STABLE_S = 2.0          # must stay in the window this long
SERVE_COOLDOWN_S = 15.0       # pause after a serve (time to put a new can)

# Motion timing
STEP_PAUSE_S = 1.0            # stillness pause between every serve step
RELEASE_DELAY_S = 3.0         # hold the can at the final pose before opening
HANDOVER_HOLD_S = 4.0         # keep the open gripper there afterwards
GRIP_SQUEEZE_DEG = 12.0       # extra squeeze past the hand-set on-can grip

# Where the trained poses are stored (inside this folder)
POSES_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "simple_poses.json")
