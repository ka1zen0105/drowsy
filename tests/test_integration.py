"""Integration tests for logging, configuration, and watchdog fault logic."""

import csv
import tempfile
import time
from pathlib import Path

import pytest

from config import (
    AppConfig, CalibrationValues, LoggingConfig, WatchdogConfig,
    load_config, save_calibration,
)
from fusion import SharedState, SystemState
from logger import CSVLogger, TelemetryRow, TELEMETRY_HEADER, EVENT_HEADER


def test_config_load_and_calibration_save_roundtrip(tmp_path):
    cal_file = tmp_path / "test_cal.yaml"
    cal = CalibrationValues(
        ear_open_baseline=0.312,
        ear_closed_threshold=0.234,
        mar_resting_baseline=0.198,
        mar_yawn_threshold=0.348,
        pitch_baseline=-2.5,
        perclos_baseline=0.03,
        perclos_threshold=0.18,
    )
    save_calibration(cal, cal_file)

    cfg = load_config(calibration_path=cal_file)
    assert abs(cfg.cal.ear_open_baseline - 0.312) < 1e-4
    assert abs(cfg.cal.ear_closed_threshold - 0.234) < 1e-4
    assert abs(cfg.cal.mar_resting_baseline - 0.198) < 1e-4
    assert abs(cfg.cal.mar_yawn_threshold - 0.348) < 1e-4
    assert abs(cfg.cal.pitch_baseline - (-2.5)) < 1e-4


def test_csv_logger_writes_headers_and_rows(tmp_path):
    log_cfg = LoggingConfig(
        directory=str(tmp_path),
        telemetry_file="telemetry.csv",
        events_file="events.csv",
        telemetry_interval_sec=0.0,
    )
    logger = CSVLogger(log_cfg)

    # Log telemetry row
    row = TelemetryRow(
        timestamp=time.time(),
        frame_id=42,
        face_found=True,
        ear_avg=0.285,
        mar=0.175,
        pitch_deg=1.2,
        yaw_deg=-0.5,
        roll_deg=0.1,
        perclos=0.02,
        eye_flag=False,
        yawn_flag=False,
        head_flag=False,
        microsleep=False,
        eye_closure_dur=0.0,
        system_state="MONITORING",
        alarm_active=False,
        fault_active=False,
    )
    logger.log_telemetry(row)
    logger.log_event("ALARM_TRIGGERED", "Yawn + Head nod", ear=0.285, mar=0.55)
    logger.close()

    # Verify telemetry CSV
    tel_file = tmp_path / "telemetry.csv"
    assert tel_file.exists()
    with open(tel_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == TELEMETRY_HEADER
        assert len(reader) >= 2
        assert reader[1][2] == "42"  # frame_id
        assert reader[1][3] == "1"   # face_found

    # Verify events CSV
    evt_file = tmp_path / "events.csv"
    assert evt_file.exists()
    with open(evt_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == EVENT_HEADER
        # Includes SYSTEM_START, ALARM_TRIGGERED, and SYSTEM_STOP
        event_types = [r[2] for r in reader[1:]]
        assert "SYSTEM_START" in event_types
        assert "ALARM_TRIGGERED" in event_types
        assert "SYSTEM_STOP" in event_types


def test_watchdog_detection_logic():
    """Verify watchdog timeout condition logic."""
    shared = SharedState()
    watch_cfg = WatchdogConfig(
        frame_timeout_sec=0.5,
        face_timeout_sec=2.0,
    )

    now = time.time()
    shared.last_frame_time = now - 0.1
    shared.last_face_time = now - 0.2

    # Under normal condition: frame and face fresh
    is_frame_stalled = (now - shared.last_frame_time) > watch_cfg.frame_timeout_sec
    is_face_lost = (now - shared.last_face_time) > watch_cfg.face_timeout_sec
    assert not is_frame_stalled
    assert not is_face_lost

    # Frame stalled > 0.5s
    shared.last_frame_time = now - 0.6
    is_frame_stalled = (now - shared.last_frame_time) > watch_cfg.frame_timeout_sec
    assert is_frame_stalled

    # Face lost > 2.0s
    shared.last_face_time = now - 2.5
    is_face_lost = (now - shared.last_face_time) > watch_cfg.face_timeout_sec
    assert is_face_lost
