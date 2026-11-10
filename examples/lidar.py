import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from rplidar import RPLidar

# Configuration
PORT_NAME = '/dev/ttyUSB0'
BAUD_RATE = 115200
D_MAX = 12000  # Max distance in mm
D_MIN = 100    # Ignore readings closer than this (mm)

# Point density: one slot per angle bin. New readings overwrite the old
# value in the same bin, so points accumulate across scans (dense image)
# without ever stacking multiple "layers" at the same angle.
ANGLE_RES = 0.45                       # degrees per bin
N_BINS = int(round(360 / ANGLE_RES))
MAX_AGE = 10                           # frames a bin survives without a fresh reading

# Shape tuning (all in mm)
CLUSTER_GAP = 180      # base gap that splits two clusters
WALL_MIN_LEN = 900     # walls are long...
WALL_MAX_RESID = 60    # ...and straight (max deviation from fitted line)
PERSON_MIN_W = 80      # person-sized blob width
PERSON_MAX_W = 700
PERSON_MIN_PTS = 4

# Temporal tuning
STATIC_TOL = 120       # mm a reading may wobble and still be the "same surface"
STATIC_FRAMES = 6      # frames a bin must stay stable to count as static
MOVE_THRESH = 130      # mm a blob must drift from its resting spot to count as moving
PERSON_HOLD = 40       # frames a blob stays "person" after it stops moving
TRACK_MATCH = 500      # max centroid jump (mm/frame) to keep following the same blob
TRACK_TIMEOUT = 15     # frames before a lost track is dropped

COLOR_WALL = '#1f77b4'     # blue
COLOR_PERSON = '#ff3333'   # red
COLOR_STATIC = '#b8860b'   # dark yellow: box / furniture / standing object
COLOR_UNKNOWN = '#999999'  # gray


class ScanBuffer:
    """Keeps the freshest distance per angle bin, plus how long each
    bin has been stable (the basis for static-vs-moving decisions)."""

    def __init__(self):
        self.dist = np.full(N_BINS, np.nan)
        self.age = np.full(N_BINS, MAX_AGE + 1, dtype=int)
        self.stable = np.zeros(N_BINS, dtype=int)  # consecutive stable frames

    def add_scan(self, scan):
        self.age += 1
        for _quality, angle, distance in scan:
            if D_MIN <= distance <= D_MAX:
                b = int(angle / ANGLE_RES) % N_BINS
                old = self.dist[b]
                if np.isfinite(old) and abs(old - distance) < STATIC_TOL:
                    # Same surface: smooth the jitter and bump stability
                    self.dist[b] = 0.6 * distance + 0.4 * old
                    self.stable[b] += 1
                else:
                    self.dist[b] = distance
                    self.stable[b] = 0
                self.age[b] = 0
        # Expire stale bins so moving objects don't leave trails
        expired = self.age > MAX_AGE
        self.dist[expired] = np.nan
        self.stable[expired] = 0

    def points(self):
        """Returns (theta_rad, dist_mm, is_static) of live bins, sorted by angle."""
        idx = np.nonzero(np.isfinite(self.dist))[0]
        theta = np.radians(idx * ANGLE_RES)
        return theta, self.dist[idx], self.stable[idx] >= STATIC_FRAMES


def _split_clusters(pts, dist):
    """Clusters consecutive points, with range-adaptive gap and 360° wrap."""
    n = len(pts)
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    thresholds = CLUSTER_GAP + 0.03 * np.minimum(dist[:-1], dist[1:])
    breaks = np.nonzero(gaps > thresholds)[0] + 1
    clusters = np.split(np.arange(n), breaks)
    if len(clusters) > 1:
        d_wrap = np.linalg.norm(pts[clusters[-1][-1]] - pts[clusters[0][0]])
        if d_wrap < CLUSTER_GAP + 0.03 * dist[0]:
            clusters[0] = np.concatenate((clusters[-1], clusters[0]))
            clusters.pop()
    return clusters


class SceneClassifier:
    """Combines shape (long/straight vs. compact/rounded), per-bin
    stability, and blob tracking over time:

      * long + straight                      -> wall
      * compact blob that has actually moved -> person
      * anything that just sits there        -> static object (box, chair...)
    """

    def __init__(self):
        self.tracks = []   # [{'pos', 'anchor', 'last_seen', 'person_until'}]
        self.frame = 0

    def _update_track(self, centroid):
        """Finds/creates the track for this blob; returns True if the blob
        counts as a person (moved recently)."""
        best, best_d = None, TRACK_MATCH
        for tr in self.tracks:
            d = np.linalg.norm(tr['pos'] - centroid)
            if d < best_d:
                best, best_d = tr, d
        if best is None:
            best = {'pos': centroid, 'anchor': centroid.copy(),
                    'last_seen': self.frame, 'person_until': -1}
            self.tracks.append(best)
        best['pos'] = centroid
        best['last_seen'] = self.frame
        # The anchor trails slowly behind the blob. A blob that wanders away
        # from its anchor is genuinely moving; sensor jitter never gets far.
        best['anchor'] = 0.97 * best['anchor'] + 0.03 * centroid
        if np.linalg.norm(centroid - best['anchor']) > MOVE_THRESH:
            best['person_until'] = self.frame + PERSON_HOLD
        return self.frame <= best['person_until']

    def classify(self, theta, dist, static):
        self.frame += 1
        n = len(theta)
        colors = np.array([COLOR_UNKNOWN] * n, dtype=object)
        if n >= 3:
            pts = np.column_stack((dist * np.cos(theta), dist * np.sin(theta)))
            for cl in _split_clusters(pts, dist):
                if len(cl) < 3:
                    continue
                p = pts[cl]
                centroid = p.mean(axis=0)
                centered = p - centroid
                # PCA: main direction + perpendicular spread
                _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
                along = centered @ vt[0]
                across = centered @ vt[1]
                length = along.max() - along.min()   # extent of the cluster
                resid = np.abs(across).max()         # deviation from a straight line
                static_frac = static[cl].mean()

                person_sized = (PERSON_MIN_W <= length <= PERSON_MAX_W
                                and len(cl) >= PERSON_MIN_PTS)
                # Only compact blobs are worth tracking for motion
                moving = (self._update_track(centroid)
                          if length <= PERSON_MAX_W * 1.3 else False)

                if length >= WALL_MIN_LEN and resid <= WALL_MAX_RESID:
                    colors[cl] = COLOR_WALL          # long + straight = wall
                elif person_sized and moving:
                    colors[cl] = COLOR_PERSON        # compact + actually moving
                elif static_frac >= 0.6:
                    colors[cl] = COLOR_STATIC        # parked = box/furniture
        # Forget blobs that vanished
        self.tracks = [t for t in self.tracks
                       if self.frame - t['last_seen'] <= TRACK_TIMEOUT]
        return colors


def update_line(_num, iterator, line, buffer, classifier):
    scan = next(iterator)
    buffer.add_scan(scan)
    theta, dist, static = buffer.points()
    if theta.size > 0:
        line.set_offsets(np.column_stack((theta, dist)))
        line.set_color(classifier.classify(theta, dist, static))
    return line,


def run():
    lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE)

    # Setup the polar plot window
    fig = plt.figure(figsize=(17, 17))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_rmax(D_MAX)
    # Mirror left/right so objects on the left appear on the left
    # (keeps front/back orientation unchanged)
    ax.set_theta_direction(-1)
    ax.set_title("RPLIDAR A1M8 Live Scan", va='bottom')

    line = ax.scatter([], [], s=6, lw=0)
    ax.legend(handles=[
        plt.Line2D([], [], color=COLOR_WALL, marker='o', ls='', label='Wall'),
        plt.Line2D([], [], color=COLOR_PERSON, marker='o', ls='', label='Person (moving)'),
        plt.Line2D([], [], color=COLOR_STATIC, marker='o', ls='', label='Static object'),
        plt.Line2D([], [], color=COLOR_UNKNOWN, marker='o', ls='', label='Unknown'),
    ], loc='upper right')

    buffer = ScanBuffer()
    classifier = SceneClassifier()

    try:
        # Start scanning iterator
        iterator = lidar.iter_scans()

        # Animate the scatter plot using animation.FuncAnimation
        ani = animation.FuncAnimation(
            fig, update_line, fargs=(iterator, line, buffer, classifier),
            interval=50, blit=False, cache_frame_data=False
        )
        plt.show()

    except KeyboardInterrupt:
        print("Stopping visualization...")
    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()


if __name__ == '__main__':
    run()
