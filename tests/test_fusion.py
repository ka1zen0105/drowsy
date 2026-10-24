"""Unit tests for sensor fusion, voting logic, and microsleep override (fusion.py)."""

import time
import pytest

from config import CalibrationValues, DetectionConfig, FusionConfig
from fusion import FusionEngine
from vision import FeatureRecord


@pytest.fixture
def test_configs():
    det = DetectionConfig(
        ear_closure_duration_sec=1.5,
        microsleep_duration_sec=3.0,
        yawn_duration_sec=1.5,
        nod_duration_sec=1.0,
        pitch_deviation_deg=15.0,
        perclos_window_sec=30.0,
    )
    fus = FusionConfig(
        voting_threshold=2,
        alarm_clear_duration_sec=2.0,
        ack_cooldown_sec=5.0,
    )
    cal = CalibrationValues(
        ear_open_baseline=0.30,
        ear_closed_threshold=0.22,
        mar_resting_baseline=0.20,
        mar_yawn_threshold=0.50,
        pitch_baseline=0.0,
        perclos_baseline=0.02,
        perclos_threshold=0.15,
    )
    return det, fus, cal


def make_record(t: float, ear: float = 0.30, mar: float = 0.20,
                pitch: float = 0.0, face_found: bool = True) -> FeatureRecord:
    return FeatureRecord(
        timestamp=t,
        frame_id=1,
        face_found=face_found,
        ear_avg=ear,
        mar=mar,
        pitch_deg=pitch,
    )


def test_normal_alert_driver(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # 5 seconds of nominal alert driving
    alarm = False
    for step in range(50):
        rec = make_record(t=step * 0.1, ear=0.32, mar=0.18, pitch=0.0)
        alarm = engine.update(rec)

    assert not alarm
    assert not engine.eye_flag
    assert not engine.yawn_flag
    assert not engine.head_flag
    assert not engine.microsleep


def test_isolated_blink_does_not_alarm(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # Eyes closed for 0.3 s (typical blink)
    for step in range(10):
        rec = make_record(t=step * 0.1, ear=0.30)
        alarm = engine.update(rec)
        assert not alarm

    for step in range(10, 13):
        rec = make_record(t=step * 0.1, ear=0.15)  # eyes closed
        alarm = engine.update(rec)
        assert not alarm

    for step in range(13, 25):
        rec = make_record(t=step * 0.1, ear=0.30)  # eyes open again
        alarm = engine.update(rec)
        assert not alarm

    assert not engine.eye_flag
    assert not engine.microsleep


def test_isolated_eye_closure_1_5s_sets_flag_but_no_alarm(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # Eyes closed for 2.0 s (< 3.0s microsleep, but >= 1.5s eye_flag)
    alarm = False
    for step in range(21):
        t = step * 0.1
        rec = make_record(t=t, ear=0.15, mar=0.18, pitch=0.0)
        alarm = engine.update(rec)

    assert engine.eye_flag, "eye_flag should be True after >= 1.5s"
    assert not engine.microsleep, "microsleep should be False (< 3.0s)"
    assert not engine.yawn_flag
    assert not engine.head_flag
    assert not alarm, "Single flag should not trigger alarm under 2-of-3 voting"


def test_microsleep_override_triggers_alarm(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # Eyes closed for 3.2 s alone (no yawning, no nodding)
    alarm = False
    for step in range(33):
        t = step * 0.1
        rec = make_record(t=t, ear=0.15, mar=0.18, pitch=0.0)
        alarm = engine.update(rec)

    assert engine.eye_flag
    assert engine.microsleep
    assert not engine.yawn_flag
    assert not engine.head_flag
    assert alarm, "Microsleep (>= 3.0s) MUST trigger alarm regardless of other flags"


def test_two_of_three_voting_yawn_and_head_nod(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # Driver yawning (mar=0.60) AND head nodding forward (pitch=-20.0 deg -> dev = 0 - (-20) = 20 > 15)
    # Eyes remain open (ear=0.30)
    alarm = False
    for step in range(18):  # 1.8 seconds (>= 1.5s yawn and >= 1.0s nod)
        t = step * 0.1
        rec = make_record(t=t, ear=0.30, mar=0.60, pitch=-20.0)
        alarm = engine.update(rec)

    assert not engine.eye_flag
    assert engine.yawn_flag
    assert engine.head_flag
    assert alarm, "2 of 3 flags (yawn + head nod) must trigger alarm"


def test_two_of_three_voting_eye_and_head_nod(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # Eyes closed (1.6s) AND head nodding (1.6s), mouth closed
    alarm = False
    for step in range(17):
        t = step * 0.1
        rec = make_record(t=t, ear=0.15, mar=0.15, pitch=-20.0)
        alarm = engine.update(rec)

    assert engine.eye_flag
    assert not engine.yawn_flag
    assert engine.head_flag
    assert not engine.microsleep
    assert alarm, "2 of 3 flags (eye closure + head nod) must trigger alarm"


def test_acknowledge_button_silences_alarm_with_cooldown(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # Trigger alarm via microsleep
    for step in range(35):
        t = step * 0.1
        rec = make_record(t=t, ear=0.15)
        alarm = engine.update(rec)
    assert alarm

    # Driver acknowledges at t=3.5
    engine.acknowledge(now=3.5)

    # Next update at t=3.6 during cooldown
    rec = make_record(t=3.6, ear=0.15)
    alarm = engine.update(rec)
    assert not alarm, "Alarm must be silenced during ack cooldown"

    # Beyond cooldown (t > 3.5 + 5.0 = 8.5) with eyes still closed
    for step in range(37, 88):
        t = step * 0.1
        rec = make_record(t=t, ear=0.15)
        alarm = engine.update(rec)
    assert alarm, "Alarm should re-trigger after cooldown if condition persists"


def test_auto_clear_after_recovery(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # Trigger alarm via yawn + nod
    for step in range(18):
        t = step * 0.1
        rec = make_record(t=t, ear=0.30, mar=0.60, pitch=-20.0)
        engine.update(rec)
    assert engine._evaluate_alarm(1.8)

    # Driver recovers (eyes open, mouth closed, head upright)
    # auto-clear duration is 2.0s
    t_start = 1.9
    for step in range(19):  # 1.9s: not yet cleared
        t = t_start + step * 0.1
        rec = make_record(t=t, ear=0.30, mar=0.15, pitch=0.0)
        alarm = engine.update(rec)

    # At t = 1.9 + 2.0 = 3.9s: flags have been inactive for >= 2.0s
    rec = make_record(t=4.0, ear=0.30, mar=0.15, pitch=0.0)
    alarm = engine.update(rec)
    assert not alarm, "Alarm must auto-clear after 2.0s of inactive flags"


def test_perclos_calculation(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # 10 seconds total: 5 seconds closed, 5 seconds open
    for step in range(50):
        t = step * 0.1
        rec = make_record(t=t, ear=0.10)  # closed
        engine.update(rec)

    for step in range(50, 100):
        t = step * 0.1
        rec = make_record(t=t, ear=0.30)  # open
        engine.update(rec)

    # PERCLOS over this 10-second period should be approximately 0.50
    assert 0.45 <= engine.perclos <= 0.55


def test_missing_face_does_not_clear_timers(test_configs):
    det, fus, cal = test_configs
    engine = FusionEngine(det, fus, cal)

    # Eyes closed for 1.0s
    for step in range(10):
        t = step * 0.1
        rec = make_record(t=t, ear=0.15)
        engine.update(rec)
    assert engine._eye_closure_start is not None

    # Face temporarily lost for 2 frames
    rec_lost1 = make_record(t=1.1, face_found=False)
    rec_lost2 = make_record(t=1.2, face_found=False)
    engine.update(rec_lost1)
    engine.update(rec_lost2)

    # Timer should not have been cleared!
    assert engine._eye_closure_start is not None
