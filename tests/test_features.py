"""Unit tests for EAR, MAR, and solvePnP head-pose estimation (features.py)."""

import numpy as np
import pytest

from features import (
    compute_ear, compute_ear_avg, compute_mar, estimate_head_pose,
    LEFT_EYE_IDX, RIGHT_EYE_IDX,
    LIP_LEFT, LIP_RIGHT, LIP_UPPER_L, LIP_LOWER_L,
    LIP_UPPER_C, LIP_LOWER_C, LIP_UPPER_R, LIP_LOWER_R,
)


# ---------------------------------------------------------------------------
# EAR tests
# ---------------------------------------------------------------------------

class TestEAR:
    """Eye Aspect Ratio (EAR) calculations."""

    def test_open_eye(self):
        """Open eye → EAR ≈ 0.30, above typical 0.22 threshold."""
        # [outer, upper_1, upper_2, inner, lower_2, lower_1]
        # Horizontal width = 100, vertical height = 30 each
        pts = np.array([
            [0, 50],    # outer
            [30, 35],   # upper_1
            [70, 35],   # upper_2
            [100, 50],  # inner
            [70, 65],   # lower_2
            [30, 65],   # lower_1
        ], dtype=np.float32)
        ear = compute_ear(pts)
        assert ear == pytest.approx(0.30, rel=1e-3)
        assert ear > 0.22, "Open eye must be above typical closed threshold"

    def test_closed_eye(self):
        """Closed eye → EAR ≈ 0.04, well below threshold."""
        pts = np.array([
            [0, 50],    # outer
            [30, 48],   # upper_1
            [70, 48],   # upper_2
            [100, 50],  # inner
            [70, 52],   # lower_2
            [30, 52],   # lower_1
        ], dtype=np.float32)
        ear = compute_ear(pts)
        assert ear == pytest.approx(0.04, rel=1e-3)
        assert ear < 0.22, "Closed eye must be below typical threshold"

    def test_half_closed(self):
        """Half-closed eye → intermediate EAR."""
        pts = np.array([
            [0, 50],
            [30, 42],
            [70, 42],
            [100, 50],
            [70, 58],
            [30, 58],
        ], dtype=np.float32)
        ear = compute_ear(pts)
        assert 0.10 < ear < 0.25

    def test_degenerate_empty(self):
        assert compute_ear(np.array([])) == 0.0

    def test_degenerate_zero_width(self):
        pts = np.zeros((6, 2), dtype=np.float32)
        assert compute_ear(pts) == 0.0

    def test_compute_ear_avg_symmetric(self):
        """Average EAR with both eyes identical → same as single-eye EAR."""
        # Create a 468-point landmark array with known eye landmarks
        lm = np.zeros((468, 2), dtype=np.float32)
        open_eye = np.array([
            [0, 50], [30, 35], [70, 35], [100, 50], [70, 65], [30, 65],
        ], dtype=np.float32)
        for i, idx in enumerate(LEFT_EYE_IDX):
            lm[idx] = open_eye[i]
        for i, idx in enumerate(RIGHT_EYE_IDX):
            lm[idx] = open_eye[i] + [200, 0]  # offset right eye

        l, r, avg = compute_ear_avg(lm)
        assert l == pytest.approx(0.30, rel=1e-3)
        assert r == pytest.approx(0.30, rel=1e-3)
        assert avg == pytest.approx(0.30, rel=1e-3)


# ---------------------------------------------------------------------------
# MAR tests
# ---------------------------------------------------------------------------

class TestMAR:
    """Mouth Aspect Ratio (MAR) calculations."""

    def test_closed_mouth(self):
        """Closed mouth → MAR ≈ 0.13, well below yawn threshold."""
        lm = np.zeros((468, 2), dtype=np.float32)
        lm[LIP_LEFT]    = [50,  100]
        lm[LIP_RIGHT]   = [150, 100]
        lm[LIP_UPPER_L] = [75,  96]
        lm[LIP_LOWER_L] = [75,  104]
        lm[LIP_UPPER_C] = [100, 95]
        lm[LIP_LOWER_C] = [100, 105]
        lm[LIP_UPPER_R] = [125, 96]
        lm[LIP_LOWER_R] = [125, 104]
        mar = compute_mar(lm)
        assert mar < 0.35, "Closed mouth must be well below yawn threshold"
        assert mar == pytest.approx(0.13, rel=0.05)

    def test_yawning_mouth(self):
        """Wide-open yawning mouth → MAR ≈ 0.65, above threshold."""
        lm = np.zeros((468, 2), dtype=np.float32)
        lm[LIP_LEFT]    = [50,  100]
        lm[LIP_RIGHT]   = [150, 100]
        lm[LIP_UPPER_L] = [75,  80]
        lm[LIP_LOWER_L] = [75,  120]
        lm[LIP_UPPER_C] = [100, 75]
        lm[LIP_LOWER_C] = [100, 125]
        lm[LIP_UPPER_R] = [125, 80]
        lm[LIP_LOWER_R] = [125, 120]
        mar = compute_mar(lm)
        assert mar > 0.50, "Yawning must exceed threshold"
        assert mar == pytest.approx(0.65, rel=0.05)

    def test_talking_mouth(self):
        """Moderate opening (speech) → MAR between closed and yawning."""
        lm = np.zeros((468, 2), dtype=np.float32)
        lm[LIP_LEFT]    = [50,  100]
        lm[LIP_RIGHT]   = [150, 100]
        lm[LIP_UPPER_L] = [75,  92]
        lm[LIP_LOWER_L] = [75,  108]
        lm[LIP_UPPER_C] = [100, 90]
        lm[LIP_LOWER_C] = [100, 110]
        lm[LIP_UPPER_R] = [125, 92]
        lm[LIP_LOWER_R] = [125, 108]
        mar = compute_mar(lm)
        assert 0.13 < mar < 0.50


# ---------------------------------------------------------------------------
# Head-pose tests
# ---------------------------------------------------------------------------

class TestHeadPose:
    """solvePnP head-pose estimation."""

    def test_frontal_face_small_angles(self):
        """Symmetrical centred frontal face → pitch/yaw/roll near zero."""
        lm = np.zeros((468, 2), dtype=np.float32)
        lm[1]   = [320, 240]   # nose
        lm[152] = [320, 360]   # chin
        lm[33]  = [240, 180]   # left eye outer
        lm[263] = [400, 180]   # right eye outer
        lm[61]  = [270, 300]   # left mouth
        lm[291] = [370, 300]   # right mouth

        pitch, yaw, roll, rvec, tvec = estimate_head_pose(lm, (480, 640))
        assert rvec is not None
        assert abs(yaw)  < 15.0, f"Yaw too large for frontal face: {yaw}"
        assert abs(roll) < 15.0, f"Roll too large for frontal face: {roll}"

    def test_pose_returns_none_on_failure(self):
        """Degenerate landmark set → solvePnP may fail gracefully."""
        lm = np.zeros((468, 2), dtype=np.float32)
        # All points at origin — solvePnP should still not crash
        pitch, yaw, roll, rvec, tvec = estimate_head_pose(lm, (480, 640))
        # It may succeed or fail; either way it must not raise
        assert isinstance(pitch, float)
