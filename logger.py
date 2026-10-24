"""Thread-safe CSV logger for telemetry metrics and discrete events.

Runs a background writer thread that drains an unbounded queue so that the
vision and fusion paths are never blocked by file I/O.
"""

from __future__ import annotations

import csv
import time
import threading
import queue
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

from config import LoggingConfig


# ---------------------------------------------------------------------------
# Log record types
# ---------------------------------------------------------------------------

@dataclass
class TelemetryRow:
    timestamp: float
    frame_id: int
    face_found: bool
    ear_avg: float
    mar: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    perclos: float
    eye_flag: bool
    yawn_flag: bool
    head_flag: bool
    microsleep: bool
    eye_closure_dur: float
    system_state: str
    alarm_active: bool
    fault_active: bool


@dataclass
class EventRow:
    timestamp: float
    event_type: str
    details: str
    ear_avg: float = 0.0
    mar: float = 0.0
    pitch_deg: float = 0.0
    perclos: float = 0.0


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

TELEMETRY_HEADER = [
    "timestamp_iso", "timestamp_epoch", "frame_id", "face_found",
    "ear_avg", "mar", "pitch_deg", "yaw_deg", "roll_deg",
    "perclos", "eye_flag", "yawn_flag", "head_flag", "microsleep",
    "eye_closure_dur_s", "system_state", "alarm_active", "fault_active",
]

EVENT_HEADER = [
    "timestamp_iso", "timestamp_epoch", "event_type",
    "ear_avg", "mar", "pitch_deg", "perclos", "details",
]


class CSVLogger:
    """Asynchronous, thread-safe CSV logger.

    Two files are written:

    * **telemetry.csv** — periodic per-frame measurements.
    * **events.csv** — discrete state-change records (alarm, fault, ack, …).
    """

    def __init__(self, config: LoggingConfig):
        self.cfg = config
        self._log_dir = Path(config.directory)
        self._tel_path = self._log_dir / config.telemetry_file
        self._evt_path = self._log_dir / config.events_file
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_tel_time: float = 0.0

        self._init_files()
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop,
                                        daemon=True, name="CSVLogger")
        self._thread.start()
        self.log_event("SYSTEM_START", "Logger initialised.")

    def _init_files(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        if not self._tel_path.exists():
            self._write_header(self._tel_path, TELEMETRY_HEADER)
        if not self._evt_path.exists():
            self._write_header(self._evt_path, EVENT_HEADER)

    @staticmethod
    def _write_header(path: Path, header: List[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    # ── public API ────────────────────────────────────────────────────

    def log_telemetry(self, row: TelemetryRow) -> None:
        """Enqueue a telemetry row (respects sample interval)."""
        if not self._running:
            return
        now = row.timestamp
        if now - self._last_tel_time < self.cfg.telemetry_interval_sec:
            return
        self._last_tel_time = now
        self._queue.put(("tel", row))

    def log_event(self, event_type: str, details: str,
                  ear: float = 0.0, mar: float = 0.0,
                  pitch: float = 0.0, perclos: float = 0.0) -> None:
        """Enqueue a discrete event."""
        if not self._running:
            return
        self._queue.put(("evt", EventRow(
            timestamp=time.time(), event_type=event_type,
            details=details, ear_avg=ear, mar=mar,
            pitch_deg=pitch, perclos=perclos,
        )))

    def close(self) -> None:
        if not self._running:
            return
        self.log_event("SYSTEM_STOP", "Logger shutdown.")
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        print(f"[Logger] CSV files saved to '{self._log_dir}'.")

    # ── background writer ─────────────────────────────────────────────

    def _writer_loop(self) -> None:
        tel_buf: List[List[Any]] = []
        evt_buf: List[List[Any]] = []
        last_flush = time.time()

        while self._running or not self._queue.empty():
            try:
                kind, rec = self._queue.get(timeout=0.25)
                if kind == "tel":
                    tel_buf.append(self._tel_to_row(rec))
                else:
                    evt_buf.append(self._evt_to_row(rec))
                self._queue.task_done()
            except queue.Empty:
                pass

            now = time.time()
            if (now - last_flush >= 1.0) or len(tel_buf) >= 30 or len(evt_buf) >= 5:
                self._flush(tel_buf, evt_buf)
                tel_buf.clear(); evt_buf.clear()
                last_flush = now

        # Final drain
        while not self._queue.empty():
            try:
                kind, rec = self._queue.get_nowait()
                (tel_buf if kind == "tel" else evt_buf).append(
                    self._tel_to_row(rec) if kind == "tel" else self._evt_to_row(rec))
            except queue.Empty:
                break
        self._flush(tel_buf, evt_buf)

    def _flush(self, tel: List, evt: List) -> None:
        if tel:
            try:
                with open(self._tel_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(tel)
            except OSError as exc:
                print(f"[Logger] telemetry write error: {exc}")
        if evt:
            try:
                with open(self._evt_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(evt)
            except OSError as exc:
                print(f"[Logger] events write error: {exc}")

    @staticmethod
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _tel_to_row(self, r: TelemetryRow) -> list:
        return [
            self._iso(r.timestamp), f"{r.timestamp:.3f}", r.frame_id,
            int(r.face_found),
            f"{r.ear_avg:.4f}", f"{r.mar:.4f}",
            f"{r.pitch_deg:.2f}", f"{r.yaw_deg:.2f}", f"{r.roll_deg:.2f}",
            f"{r.perclos:.4f}",
            int(r.eye_flag), int(r.yawn_flag), int(r.head_flag),
            int(r.microsleep), f"{r.eye_closure_dur:.2f}",
            r.system_state, int(r.alarm_active), int(r.fault_active),
        ]

    def _evt_to_row(self, r: EventRow) -> list:
        return [
            self._iso(r.timestamp), f"{r.timestamp:.3f}", r.event_type,
            f"{r.ear_avg:.4f}", f"{r.mar:.4f}",
            f"{r.pitch_deg:.2f}", f"{r.perclos:.4f}", r.details,
        ]
