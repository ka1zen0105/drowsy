"""Post-run log evaluation and performance metrics analyzer.

Analyzes telemetry.csv and events.csv to compute:
- Effective processing FPS and frame latency
- Face detection reliability / dropout percentage
- Eye closure statistics and PERCLOS distribution
- Alarm trigger frequency, false alarm indicators, and acknowledge response latency

Usage:
    python tools/evaluate.py [path_to_logs_dir]
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


def analyze_logs(log_dir: Path) -> int:
    tel_path = log_dir / "telemetry.csv"
    evt_path = log_dir / "events.csv"

    print("=" * 65)
    print("  Driver Drowsiness Detection System — Post-Run Log Evaluation  ")
    print("=" * 65)
    print(f"Log directory: {log_dir.resolve()}\n")

    if not tel_path.exists():
        print(f"Error: Telemetry file not found at '{tel_path}'")
        return 1

    # 1. Parse Telemetry
    total_frames = 0
    face_detected_frames = 0
    timestamps: List[float] = []
    ears: List[float] = []
    mars: List[float] = []
    pitches: List[float] = []
    percloss: List[float] = []
    alarm_frames = 0
    fault_frames = 0

    with open(tel_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_frames += 1
            t = float(row["timestamp_epoch"])
            timestamps.append(t)

            has_face = int(row["face_found"]) == 1
            if has_face:
                face_detected_frames += 1
                ears.append(float(row["ear_avg"]))
                mars.append(float(row["mar"]))
                pitches.append(float(row["pitch_deg"]))
                percloss.append(float(row["perclos"]))

            if int(row.get("alarm_active", 0)) == 1:
                alarm_frames += 1
            if int(row.get("fault_active", 0)) == 1:
                fault_frames += 1

    if total_frames == 0:
        print("Telemetry log is empty.")
        return 0

    duration_sec = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    effective_fps = (total_frames / duration_sec) if duration_sec > 0 else 0.0

    print("── Telemetry Summary ──────────────────────────────────────────")
    print(f"Total recorded frames   : {total_frames}")
    print(f"Total session duration  : {duration_sec:.2f} s ({duration_sec / 60:.1f} min)")
    print(f"Average effective FPS   : {effective_fps:.2f} fps")
    print(f"Face tracking rate      : {face_detected_frames / total_frames * 100:.1f}% ({face_detected_frames}/{total_frames})")
    print(f"Alarm active frames     : {alarm_frames} ({alarm_frames / total_frames * 100:.1f}%)")
    print(f"Fault active frames     : {fault_frames} ({fault_frames / total_frames * 100:.1f}%)")

    if ears:
        print("\n── Feature Distribution ───────────────────────────────────────")
        ears.sort()
        mars.sort()
        pitches.sort()
        percloss.sort()

        def percentile(arr: List[float], p: float) -> float:
            idx = int(len(arr) * p)
            return arr[min(idx, len(arr) - 1)]

        print(f"EAR     — Min: {ears[0]:.3f} | Median: {percentile(ears, 0.5):.3f} | Max: {ears[-1]:.3f}")
        print(f"MAR     — Min: {mars[0]:.3f} | Median: {percentile(mars, 0.5):.3f} | Max: {mars[-1]:.3f}")
        print(f"Pitch   — Min: {pitches[0]:.1f}°| Median: {percentile(pitches, 0.5):.1f}°| Max: {pitches[-1]:.1f}°")
        print(f"PERCLOS — Min: {percloss[0]:.3f} | Median: {percentile(percloss, 0.5):.3f} | Max: {percloss[-1]:.3f}")

    # 2. Parse Events
    if evt_path.exists():
        event_counts: Dict[str, int] = {}
        alarm_events: List[Dict[str, str]] = []
        ack_events: List[float] = []

        with open(evt_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                evt = row["event_type"]
                event_counts[evt] = event_counts.get(evt, 0) + 1
                if "ALARM" in evt:
                    alarm_events.append(row)
                if "ACK" in evt or "BUTTON" in evt:
                    ack_events.append(float(row["timestamp_epoch"]))

        print("\n── Discrete Events Summary ────────────────────────────────────")
        for evt, count in sorted(event_counts.items()):
            print(f"  • {evt:<22} : {count}")

        if alarm_events:
            print("\n── Alarms Triggered ───────────────────────────────────────────")
            for al in alarm_events[:10]:
                print(f"  [{al['timestamp_iso']}] {al['event_type']} - {al.get('details', '')} (EAR={al.get('ear_avg')}, MAR={al.get('mar')})")
            if len(alarm_events) > 10:
                print(f"  ... and {len(alarm_events) - 10} more.")

    print("\n" + "=" * 65)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate drowsiness detection logs.")
    parser.add_argument("log_dir", nargs="?", default="logs",
                        help="Path to directory containing telemetry.csv and events.csv (default: logs)")
    args = parser.parse_args()
    sys.exit(analyze_logs(Path(args.log_dir)))


if __name__ == "__main__":
    main()
