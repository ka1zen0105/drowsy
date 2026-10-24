"""Configuration loader: YAML parsing, typed dataclasses, and calibration merge.

Loads config.yaml, optionally merges calibration.yaml overrides, and exposes
typed Python dataclasses for every subsystem so the rest of the code never
handles raw dicts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Dataclasses — one per config.yaml section
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    source: Union[int, str] = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class GPIOConfig:
    buzzer_pin: int = 17
    green_led_pin: int = 27
    red_led_pin: int = 22
    ack_button_pin: int = 23
    simulation_mode: bool = False


@dataclass
class DetectionConfig:
    # EAR
    ear_closed_ratio: float = 0.75
    ear_closure_duration_sec: float = 1.5
    microsleep_duration_sec: float = 3.0
    # PERCLOS
    perclos_window_sec: float = 30.0
    perclos_threshold: float = 0.15
    perclos_baseline_margin: float = 0.05
    # MAR
    mar_threshold_offset: float = 0.15
    yawn_duration_sec: float = 1.5
    # Head pitch
    pitch_deviation_deg: float = 15.0
    nod_duration_sec: float = 1.0


@dataclass
class FusionConfig:
    voting_threshold: int = 2
    alarm_clear_duration_sec: float = 2.0
    ack_cooldown_sec: float = 3.0


@dataclass
class CalibrationConfig:
    duration_sec: float = 45.0
    output_file: str = "calibration.yaml"


@dataclass
class WatchdogConfig:
    frame_timeout_sec: float = 0.5
    face_timeout_sec: float = 2.0


@dataclass
class LoggingConfig:
    directory: str = "logs"
    telemetry_file: str = "telemetry.csv"
    events_file: str = "events.csv"
    telemetry_interval_sec: float = 0.1


@dataclass
class UIConfig:
    show_gui: bool = True
    window_name: str = "Drowsiness Detection — Raspberry Pi 4"


@dataclass
class RTSchedulingConfig:
    enabled: bool = False
    alert_priority: int = 80
    watchdog_priority: int = 60
    fusion_priority: int = 50
    alert_cpu_affinity: List[int] = field(default_factory=lambda: [3])
    fusion_cpu_affinity: List[int] = field(default_factory=lambda: [3])
    vision_cpu_affinity: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class CalibrationValues:
    """Runtime calibration results — loaded from calibration.yaml or computed."""
    ear_open_baseline: float = 0.30
    ear_closed_threshold: float = 0.225
    mar_resting_baseline: float = 0.18
    mar_yawn_threshold: float = 0.33
    pitch_baseline: float = 0.0
    perclos_baseline: float = 0.02
    perclos_threshold: float = 0.15


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    gpio: GPIOConfig = field(default_factory=GPIOConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    calibration_cfg: CalibrationConfig = field(default_factory=CalibrationConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    rt: RTSchedulingConfig = field(default_factory=RTSchedulingConfig)
    cal: CalibrationValues = field(default_factory=CalibrationValues)
    model_path: str = "models/face_landmarker.task"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_dict(obj: Any, data: Dict[str, Any]) -> None:
    """Set attributes on *obj* from *data*, ignoring unknown keys."""
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    calibration_path: Optional[Union[str, Path]] = None,
) -> AppConfig:
    """Load configuration from YAML and optionally merge calibration overrides.

    Args:
        config_path: Path to config.yaml.  Searches common locations when None.
        calibration_path: Path to calibration.yaml.  When None, uses the path
            recorded inside the loaded config.

    Returns:
        Fully resolved ``AppConfig``.
    """
    # Locate config.yaml
    if config_path is None:
        for candidate in ("config.yaml", "config/config.yaml"):
            if Path(candidate).exists():
                config_path = candidate
                break

    raw = _load_yaml(config_path) if config_path else {}
    cfg = AppConfig()

    section_map = {
        "camera": cfg.camera,
        "gpio": cfg.gpio,
        "detection": cfg.detection,
        "fusion": cfg.fusion,
        "calibration": cfg.calibration_cfg,
        "watchdog": cfg.watchdog,
        "logging": cfg.logging,
        "ui": cfg.ui,
        "rt_scheduling": cfg.rt,
    }
    for section, obj in section_map.items():
        if section in raw:
            _apply_dict(obj, raw[section])

    # Merge calibration values
    if calibration_path is None:
        calibration_path = cfg.calibration_cfg.output_file
    cal_raw = _load_yaml(calibration_path)
    if cal_raw:
        _apply_dict(cfg.cal, cal_raw)

    return cfg


def save_calibration(cal: CalibrationValues, path: Union[str, Path]) -> None:
    """Persist calibration results to YAML."""
    data = {
        "ear_open_baseline": round(float(cal.ear_open_baseline), 4),
        "ear_closed_threshold": round(float(cal.ear_closed_threshold), 4),
        "mar_resting_baseline": round(float(cal.mar_resting_baseline), 4),
        "mar_yawn_threshold": round(float(cal.mar_yawn_threshold), 4),
        "pitch_baseline": round(float(cal.pitch_baseline), 2),
        "perclos_baseline": round(float(cal.perclos_baseline), 4),
        "perclos_threshold": round(float(cal.perclos_threshold), 4),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
