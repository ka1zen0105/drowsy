"""Sensor-fusion engine: PERCLOS, three drowsiness flags, 2-of-3 voting, and
microsleep override.

Runs in the Fusion thread.  Consumes ``FeatureRecord`` from the bounded queue,
maintains sliding-window statistics and flag timers, and sets/clears
``alarm_event`` and ``fault_event`` on a ``SharedState`` object.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional

from config import AppConfig, CalibrationValues, DetectionConfig, FusionConfig
from vision import FeatureRecord


# ---------------------------------------------------------------------------
# System states
# ---------------------------------------------------------------------------

class SystemState(str, Enum):
    STARTUP      = "STARTUP"
    SELF_TEST    = "SELF_TEST"
    CALIBRATING  = "CALIBRATING"
    MONITORING   = "MONITORING"
    ALARM        = "ALARM"
    FAULT        = "FAULT"
    SHUTDOWN     = "SHUTDOWN"


# ---------------------------------------------------------------------------
# Shared state — thread-safe via lock + events
# ---------------------------------------------------------------------------

class SharedState:
    """Central shared state container accessed by all threads under a lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.alarm_event  = threading.Event()
        self.fault_event  = threading.Event()
        self.shutdown_event = threading.Event()
        self.ack_event    = threading.Event()

        # Timestamps updated by the vision thread
        self.last_frame_time: float = time.time()
        self.last_face_time: float  = time.time()

        # Latest feature snapshot (for display / logging)
        self.ear_avg: float = 0.30
        self.mar: float     = 0.0
        self.pitch: float   = 0.0
        self.perclos: float = 0.0
        self.eye_closure_duration: float = 0.0

        # Flags
        self.eye_flag: bool  = False
        self.yawn_flag: bool = False
        self.head_flag: bool = False
        self.microsleep: bool = False

        # State machine
        self.system_state: SystemState = SystemState.STARTUP

    def update_frame_time(self) -> None:
        with self.lock:
            self.last_frame_time = time.time()

    def update_face_time(self) -> None:
        with self.lock:
            self.last_face_time = time.time()


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

class FusionEngine:
    """Temporal fusion of EAR, MAR, and pitch into drowsiness flags.

    Decision rules
    ──────────────
    • **Normal**: < 2 flags active.
    • **Drowsiness alarm**: ≥ 2 of {eye_flag, yawn_flag, head_flag} active.
    • **Microsleep override**: continuous eye closure ≥ 3 s → alarm *regardless*
      of other flags.

    The 1.5 s eye_flag and 3 s microsleep override are separate thresholds.

    Alarm clears when the ack button is pressed **or** all flags have been
    inactive for ≥ 2 s continuously.
    """

    def __init__(self, det: DetectionConfig, fus: FusionConfig,
                 cal: CalibrationValues):
        self.det = det
        self.fus = fus
        self.cal = cal

        # Timers (accumulate continuous duration in seconds)
        self._eye_closure_start: Optional[float] = None
        self._yawn_start: Optional[float] = None
        self._nod_start: Optional[float] = None

        # PERCLOS sliding window: (timestamp, is_closed)
        self._perclos_buf: Deque[tuple] = deque()

        # Flags
        self.eye_flag  = False
        self.yawn_flag = False
        self.head_flag = False
        self.microsleep = False
        self.perclos: float = 0.0
        self.eye_closure_duration: float = 0.0

        # Auto-clear tracking
        self._all_clear_since: Optional[float] = None

        # Ack cooldown
        self._ack_until: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, rec: FeatureRecord) -> bool:
        """Process one FeatureRecord and update flags.

        Args:
            rec: feature record (face_found may be False).

        Returns:
            True if the drowsiness alarm should be active after this update.
        """
        now = rec.timestamp

        if not rec.face_found:
            # Do NOT reset timers — absence of face is NOT evidence of
            # an awake driver.  The watchdog handles this separately.
            return self._evaluate_alarm(now)

        # ── Eye closure ──────────────────────────────────────────────
        is_closed = rec.ear_avg < self.cal.ear_closed_threshold
        self._perclos_buf.append((now, is_closed))
        self._trim_perclos(now)
        self.perclos = self._compute_perclos(now)

        if is_closed:
            if self._eye_closure_start is None:
                self._eye_closure_start = now
            self.eye_closure_duration = now - self._eye_closure_start
        else:
            self._eye_closure_start = None
            self.eye_closure_duration = 0.0

        self.eye_flag = self.eye_closure_duration >= self.det.ear_closure_duration_sec
        self.microsleep = self.eye_closure_duration >= self.det.microsleep_duration_sec

        # ── Yawning ──────────────────────────────────────────────────
        is_yawning = rec.mar > self.cal.mar_yawn_threshold
        if is_yawning:
            if self._yawn_start is None:
                self._yawn_start = now
            yawn_dur = now - self._yawn_start
        else:
            self._yawn_start = None
            yawn_dur = 0.0
        self.yawn_flag = yawn_dur >= self.det.yawn_duration_sec

        # ── Head nod ─────────────────────────────────────────────────
        pitch_dev = self.cal.pitch_baseline - rec.pitch_deg  # positive when head dips forward
        is_nodding = pitch_dev >= self.det.pitch_deviation_deg
        if is_nodding:
            if self._nod_start is None:
                self._nod_start = now
            nod_dur = now - self._nod_start
        else:
            self._nod_start = None
            nod_dur = 0.0
        self.head_flag = nod_dur >= self.det.nod_duration_sec

        return self._evaluate_alarm(now)

    def acknowledge(self, now: Optional[float] = None) -> None:
        """Called when the driver presses the ack button."""
        now = now or time.time()
        self._ack_until = now + self.fus.ack_cooldown_sec

    def reset(self) -> None:
        """Full reset (e.g. after recalibration)."""
        self._eye_closure_start = None
        self._yawn_start = None
        self._nod_start = None
        self._perclos_buf.clear()
        self.eye_flag = self.yawn_flag = self.head_flag = self.microsleep = False
        self.perclos = 0.0
        self.eye_closure_duration = 0.0
        self._all_clear_since = None
        self._ack_until = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_alarm(self, now: float) -> bool:
        """Apply voting rule + microsleep override + ack cooldown."""
        if now < self._ack_until:
            self._all_clear_since = None
            return False

        active = sum([self.eye_flag, self.yawn_flag, self.head_flag])

        # Microsleep override
        if self.microsleep:
            self._all_clear_since = None
            return True

        # 2-of-3 voting
        if active >= self.fus.voting_threshold:
            self._all_clear_since = None
            return True

        # Auto-clear after all flags inactive for alarm_clear_duration_sec
        if active == 0 and not self.microsleep:
            if self._all_clear_since is None:
                self._all_clear_since = now
            elif (now - self._all_clear_since) >= self.fus.alarm_clear_duration_sec:
                return False
        else:
            self._all_clear_since = None

        return False

    def _trim_perclos(self, now: float) -> None:
        cutoff = now - self.det.perclos_window_sec
        while self._perclos_buf and self._perclos_buf[0][0] < cutoff:
            self._perclos_buf.popleft()

    def _compute_perclos(self, now: float) -> float:
        if len(self._perclos_buf) < 2:
            return 0.0
        closed_time = 0.0
        total_time  = 0.0
        cutoff = now - self.det.perclos_window_sec
        items = list(self._perclos_buf)
        for i in range(1, len(items)):
            t_prev, closed_prev = items[i - 1]
            t_curr, _ = items[i]
            t0 = max(t_prev, cutoff)
            dt = t_curr - t0
            if dt <= 0:
                continue
            total_time += dt
            if closed_prev:
                closed_time += dt
        return closed_time / total_time if total_time > 0.1 else 0.0
