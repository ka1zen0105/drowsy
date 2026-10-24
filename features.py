"""Pure-math facial-geometry features: EAR, MAR, and 3-D head-pose estimation.

Every function in this module is a pure computation over NumPy arrays — no I/O,
no camera, no MediaPipe.  This makes the module trivially testable with
synthetic landmark coordinates.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe Face Mesh 468-point topology)
# ---------------------------------------------------------------------------

# Eyes — 6-point contour per Soukupová & Čech (2016)
# Order: outer corner, upper-1, upper-2, inner corner, lower-2, lower-1
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Inner-lip landmarks for MAR (mouth width + 3 vertical spans)
LIP_LEFT = 61       # left mouth corner (outer)
LIP_RIGHT = 291     # right mouth corner (outer)
LIP_UPPER_L = 81    # upper inner lip — left
LIP_LOWER_L = 178   # lower inner lip — left
LIP_UPPER_C = 13    # upper inner lip — center
LIP_LOWER_C = 14    # lower inner lip — center
LIP_UPPER_R = 311   # upper inner lip — right
LIP_LOWER_R = 402   # lower inner lip — right

MOUTH_IDX = [LIP_LEFT, LIP_RIGHT, LIP_UPPER_L, LIP_LOWER_L,
             LIP_UPPER_C, LIP_LOWER_C, LIP_UPPER_R, LIP_LOWER_R]

# Head-pose key points (6-point canonical model)
POSE_IDX = [1, 152, 33, 263, 61, 291]
#            │   │    │    │    │    └ right mouth corner
#            │   │    │    │    └ left mouth corner
#            │   │    │    └ right eye outer corner
#            │   │    └ left eye outer corner
#            │   └ chin
#            └ nose tip

# Anthropometric 3-D model in millimetres (generic adult face)
_MODEL_3D = np.array([
    (   0.0,    0.0,    0.0),   # nose tip
    (   0.0, -330.0,  -65.0),   # chin
    (-225.0,  170.0, -135.0),   # left eye outer
    ( 225.0,  170.0, -135.0),   # right eye outer
    (-150.0, -150.0, -125.0),   # left mouth corner
    ( 150.0, -150.0, -125.0),   # right mouth corner
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature computations
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """2-D Euclidean distance between two points (uses only x, y)."""
    return float(np.linalg.norm(a[:2] - b[:2]))


def compute_ear(eye_pts: np.ndarray) -> float:
    """Eye Aspect Ratio (Soukupová & Čech, 2016).

    Args:
        eye_pts: shape (6, 2+) in order
            [outer, upper_1, upper_2, inner, lower_2, lower_1]

    Returns:
        EAR value.  Open eye ≈ 0.25–0.35; closed ≈ 0.02–0.08.
    """
    if eye_pts is None or len(eye_pts) < 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_pts[:6]
    v1 = _dist(p2, p6)
    v2 = _dist(p3, p5)
    h  = _dist(p1, p4)
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_ear_avg(landmarks_px: np.ndarray) -> Tuple[float, float, float]:
    """Compute left, right, and average EAR from a full landmark array.

    Args:
        landmarks_px: shape (N, 2+).

    Returns:
        (ear_left, ear_right, ear_avg)
    """
    ear_l = compute_ear(landmarks_px[LEFT_EYE_IDX])
    ear_r = compute_ear(landmarks_px[RIGHT_EYE_IDX])
    return ear_l, ear_r, (ear_l + ear_r) / 2.0


def compute_mar(landmarks_px: np.ndarray) -> float:
    """Mouth Aspect Ratio — inner-lip opening relative to mouth width.

    MAR = (d(81,178) + d(13,14) + d(311,402)) / (2 · d(61,291))

    Args:
        landmarks_px: shape (N, 2+).

    Returns:
        MAR value.  Resting ≈ 0.1–0.25; yawning ≈ 0.5–0.9.
    """
    try:
        d_h = _dist(landmarks_px[LIP_LEFT], landmarks_px[LIP_RIGHT])
        if d_h < 1e-6:
            return 0.0
        d1 = _dist(landmarks_px[LIP_UPPER_L], landmarks_px[LIP_LOWER_L])
        d2 = _dist(landmarks_px[LIP_UPPER_C], landmarks_px[LIP_LOWER_C])
        d3 = _dist(landmarks_px[LIP_UPPER_R], landmarks_px[LIP_LOWER_R])
        return (d1 + d2 + d3) / (2.0 * d_h)
    except (IndexError, TypeError):
        return 0.0


def estimate_head_pose(
    landmarks_px: np.ndarray,
    frame_shape: Tuple[int, int],
    camera_matrix: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate Euler head-pose angles via OpenCV ``solvePnP``.

    Args:
        landmarks_px: full pixel landmark array, shape (N, 2+).
        frame_shape:  (height, width).
        camera_matrix: 3×3 intrinsic matrix; estimated from frame dims when None.

    Returns:
        (pitch_deg, yaw_deg, roll_deg, rvec, tvec).
        Negative pitch → head tilted forward / down.
    """
    h, w = frame_shape[:2]
    pts_2d = np.array([landmarks_px[i][:2] for i in POSE_IDX], dtype=np.float64)

    if camera_matrix is None:
        f = float(w)
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array([[f, 0, cx],
                                   [0, f, cy],
                                   [0, 0,  1]], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_3D, pts_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])
    yaw   = float(angles[1])
    roll  = float(angles[2])
    return pitch, yaw, roll, rvec, tvec
