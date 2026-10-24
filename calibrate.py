"""Driver calibration: collect 30–60 s of alert-driver data and establish
personal baselines for open-eye EAR, resting MAR, upright head pitch, and
baseline PERCLOS.

Can be run standalone (``python calibrate.py``) or called from ``main.py``.
"""

from __future__ import annotations

import statistics
import time
from typing import List, Optional, Union

import numpy as np

from config import (
    AppConfig, CalibrationValues, load_config, save_calibration,
)
from vision import CameraReader, FaceMeshDetector, FaceMeshDetector as _FM, process_frame


def run_calibration(
    cfg: AppConfig,
    camera: Optional[CameraReader] = None,
    detector: Optional[FaceMeshDetector] = None,
    show_gui: bool = True,
) -> CalibrationValues:
    """Collect calibration samples and compute baselines.

    The driver should sit alert, eyes open, mouth closed, head upright for the
    full calibration duration.

    Args:
        cfg: Loaded application configuration.
        camera: Optional pre-opened camera (created internally if None).
        detector: Optional pre-created detector (created internally if None).
        show_gui: Whether to show a live OpenCV preview during calibration.

    Returns:
        ``CalibrationValues`` populated with personal baselines.
    """
    own_camera = camera is None
    own_detector = detector is None

    if own_camera:
        camera = CameraReader(
            source=cfg.camera.source,
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
        ).start()

    if own_detector:
        detector = FaceMeshDetector(model_path=cfg.model_path)

    ear_samples: List[float] = []
    mar_samples: List[float] = []
    pitch_samples: List[float] = []
    blink_times: List[float] = []  # timestamps of detected blinks for PERCLOS

    duration = cfg.calibration_cfg.duration_sec
    start_time = time.time()
    frame_id = 0
    last_closed = False

    print(f"\n{'='*60}")
    print("  DRIVER CALIBRATION")
    print(f"  Please sit alert, eyes open, mouth closed, head upright.")
    print(f"  Duration: {duration:.0f} seconds")
    print(f"{'='*60}\n")

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        show_gui = False

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        frame_id += 1
        rec = process_frame(detector, frame, frame_id)

        if rec.face_found:
            ear_samples.append(rec.ear_avg)
            mar_samples.append(rec.mar)
            pitch_samples.append(rec.pitch_deg)

            # Rough blink tracking for PERCLOS baseline
            # Use a generous threshold (0.20) for blink detection during calibration
            is_closed = rec.ear_avg < 0.20
            if is_closed and not last_closed:
                blink_times.append(rec.timestamp)
            last_closed = is_closed

        # Progress display
        if frame_id % 30 == 0:
            remaining = max(0, duration - elapsed)
            pct = min(100, elapsed / duration * 100)
            print(f"  Calibrating … {pct:5.1f}%  ({remaining:.0f}s remaining)  "
                  f"samples={len(ear_samples)}", end="\r")

        if show_gui and has_cv2:
            txt = f"CALIBRATING: {elapsed:.0f}/{duration:.0f}s"
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            if rec.face_found:
                cv2.putText(frame, f"EAR={rec.ear_avg:.3f}  MAR={rec.mar:.3f}  "
                            f"Pitch={rec.pitch_deg:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1)
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print("\n[Calibrate] Aborted by user.")
                break

    print()  # newline after \r progress

    if show_gui and has_cv2:
        cv2.destroyWindow("Calibration")

    if own_camera:
        camera.stop()
    if own_detector:
        detector.close()

    # ── Compute baselines ────────────────────────────────────────────
    if len(ear_samples) < 30:
        print("[Calibrate] WARNING: fewer than 30 EAR samples — using defaults.")
        return CalibrationValues()

    ear_baseline = statistics.median(ear_samples)
    mar_baseline = statistics.median(mar_samples)
    pitch_baseline = statistics.median(pitch_samples)

    # PERCLOS baseline: fraction of calibration time spent blinking
    total_time = duration
    # Estimate ~0.3 s per blink
    blink_closed_time = len(blink_times) * 0.3
    perclos_baseline = blink_closed_time / total_time if total_time > 0 else 0.0

    # Derive thresholds
    ear_closed_threshold = ear_baseline * cfg.detection.ear_closed_ratio
    mar_yawn_threshold = mar_baseline + cfg.detection.mar_threshold_offset
    perclos_threshold = max(
        cfg.detection.perclos_threshold,
        perclos_baseline + cfg.detection.perclos_baseline_margin,
    )

    cal = CalibrationValues(
        ear_open_baseline=ear_baseline,
        ear_closed_threshold=ear_closed_threshold,
        mar_resting_baseline=mar_baseline,
        mar_yawn_threshold=mar_yawn_threshold,
        pitch_baseline=pitch_baseline,
        perclos_baseline=perclos_baseline,
        perclos_threshold=perclos_threshold,
    )

    print(f"\n{'─'*60}")
    print(f"  Calibration Results ({len(ear_samples)} samples)")
    print(f"{'─'*60}")
    print(f"  EAR open baseline:      {cal.ear_open_baseline:.4f}")
    print(f"  EAR closed threshold:   {cal.ear_closed_threshold:.4f}")
    print(f"  MAR resting baseline:   {cal.mar_resting_baseline:.4f}")
    print(f"  MAR yawn threshold:     {cal.mar_yawn_threshold:.4f}")
    print(f"  Pitch baseline:         {cal.pitch_baseline:.2f}°")
    print(f"  PERCLOS baseline:       {cal.perclos_baseline:.4f}")
    print(f"  PERCLOS threshold:      {cal.perclos_threshold:.4f}")
    print(f"{'─'*60}\n")

    # Save to YAML
    save_calibration(cal, cfg.calibration_cfg.output_file)
    print(f"[Calibrate] Saved to '{cfg.calibration_cfg.output_file}'.")

    return cal


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run driver calibration.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override calibration duration (seconds)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run without OpenCV preview")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.duration:
        cfg.calibration_cfg.duration_sec = args.duration

    run_calibration(cfg, show_gui=not args.no_gui)
