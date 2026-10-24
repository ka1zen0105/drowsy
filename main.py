"""Main application: thread orchestrator, startup sequence, shutdown handler.

Implements five cooperating threads (Vision, Fusion, Alert, Watchdog, Logging)
with bounded and unbounded queues, shared-state events, and safe cleanup.

Usage
─────
    python main.py                          # default camera, GPIO auto-detect
    python main.py --simulate              # mock GPIO on laptop
    python main.py --simulate --no-gui     # headless simulation
    python main.py --skip-calibration      # skip calibration (use saved or defaults)
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import threading
import queue
from typing import Optional

import cv2

from config import AppConfig, load_config
from vision import CameraReader, FaceMeshDetector, FeatureRecord, process_frame
from fusion import FusionEngine, SharedState, SystemState
from alert import AlertController, MockAlertController, create_alert_controller
from logger import CSVLogger, TelemetryRow
from calibrate import run_calibration


# ---------------------------------------------------------------------------
# Bounded queue helper — drops oldest when full
# ---------------------------------------------------------------------------

def _put_dropping(q: queue.Queue, item) -> None:
    """Put *item* on *q*, dropping the oldest entry if full."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


# ---------------------------------------------------------------------------
# Optional Linux real-time scheduling helpers
# ---------------------------------------------------------------------------

def _try_set_rt(thread_name: str, priority: int) -> None:
    """Attempt to apply SCHED_FIFO to the *current* thread (Linux only)."""
    if sys.platform != "linux":
        return
    try:
        import ctypes, ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

        class _SchedParam(ctypes.Structure):
            _fields_ = [("sched_priority", ctypes.c_int)]

        SCHED_FIFO = 1
        param = _SchedParam(sched_priority=priority)
        tid = libc.syscall(186)  # SYS_gettid on aarch64 / x86_64
        ret = libc.sched_setscheduler(tid, SCHED_FIFO, ctypes.byref(param))
        if ret == 0:
            print(f"  [{thread_name}] SCHED_FIFO priority {priority} applied.")
        else:
            print(f"  [{thread_name}] SCHED_FIFO failed (need root).")
    except Exception as exc:
        print(f"  [{thread_name}] RT scheduling unavailable: {exc}")


def _try_set_affinity(thread_name: str, cpus: list) -> None:
    if sys.platform != "linux":
        return
    try:
        os.sched_setaffinity(0, cpus)
        print(f"  [{thread_name}] CPU affinity set to {cpus}.")
    except Exception as exc:
        print(f"  [{thread_name}] CPU affinity failed: {exc}")


# ---------------------------------------------------------------------------
# Application class
# ---------------------------------------------------------------------------

class DrowsinessApp:
    """Top-level application coordinating all threads."""

    def __init__(self, cfg: AppConfig, *,
                 simulate: bool = False,
                 headless: bool = False,
                 skip_cal: bool = False):
        self.cfg = cfg
        self.simulate = simulate or cfg.gpio.simulation_mode
        self.headless = headless or (not cfg.ui.show_gui)
        self.skip_cal = skip_cal

        if self.simulate:
            self.cfg.gpio.simulation_mode = True

        self.state = SharedState()
        self.feature_q: queue.Queue[FeatureRecord] = queue.Queue(maxsize=5)
        self.camera: Optional[CameraReader] = None
        self.detector: Optional[FaceMeshDetector] = None
        self.fusion: Optional[FusionEngine] = None
        self.alert_ctrl: Optional[AlertController] = None
        self.csv_logger: Optional[CSVLogger] = None
        self._frame_id = 0

    # ── Startup sequence ──────────────────────────────────────────────

    def run(self) -> None:
        self._install_signal_handlers()
        print("=" * 65)
        print("  DRIVER DROWSINESS DETECTION SYSTEM")
        print("  Target: Raspberry Pi 4 Model B · Pi OS 64-bit · Python 3.11")
        print("=" * 65)
        print(f"  Camera:      {self.cfg.camera.source}")
        print(f"  Simulation:  {self.simulate}")
        print(f"  Headless:    {self.headless}")
        print("-" * 65)

        try:
            # 1. GPIO + self-test
            self.state.system_state = SystemState.STARTUP
            self.alert_ctrl = create_alert_controller(self.cfg.gpio)
            self.state.system_state = SystemState.SELF_TEST
            self.alert_ctrl.self_test()

            # 2. Camera
            self.camera = CameraReader(
                source=self.cfg.camera.source,
                width=self.cfg.camera.width,
                height=self.cfg.camera.height,
                fps=self.cfg.camera.fps,
            ).start()
            print("[Main] Camera opened.")

            # 3. MediaPipe
            self.detector = FaceMeshDetector(model_path=self.cfg.model_path)

            # 4. CSV logger
            self.csv_logger = CSVLogger(self.cfg.logging)

            # 5. Calibration
            if not self.skip_cal:
                self.state.system_state = SystemState.CALIBRATING
                cal = run_calibration(
                    self.cfg, camera=self.camera, detector=self.detector,
                    show_gui=not self.headless,
                )
                self.cfg.cal = cal
            else:
                print("[Main] Calibration skipped — using saved / default values.")

            # 6. Fusion engine
            self.fusion = FusionEngine(
                self.cfg.detection, self.cfg.fusion, self.cfg.cal,
            )

            # 7. Wire ack button
            self.alert_ctrl.register_ack_callback(self._on_ack)

            # 8. Launch threads
            self.state.system_state = SystemState.MONITORING
            self.alert_ctrl.set_normal()
            self._launch_threads()

        except Exception as exc:
            print(f"[FATAL] Startup failed: {exc}")
            import traceback; traceback.print_exc()
            self._shutdown()

    # ── Signal handling ───────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        print(f"\n[Main] Signal {signum} received — shutting down …")
        self.state.shutdown_event.set()

    # ── Ack button callback ──────────────────────────────────────────

    def _on_ack(self) -> None:
        now = time.time()
        self.state.ack_event.set()
        if self.fusion:
            self.fusion.acknowledge(now)
        self.state.alarm_event.clear()
        if self.csv_logger:
            self.csv_logger.log_event(
                "ACK_BUTTON", "Alarm acknowledged by driver.",
                ear=self.state.ear_avg, mar=self.state.mar,
                pitch=self.state.pitch, perclos=self.state.perclos,
            )
        print("[Main] Alarm acknowledged.")

    # ── Thread launcher ──────────────────────────────────────────────

    def _launch_threads(self) -> None:
        threads = [
            threading.Thread(target=self._vision_loop, name="Vision", daemon=True),
            threading.Thread(target=self._fusion_loop, name="Fusion", daemon=True),
            threading.Thread(target=self._alert_loop, name="Alert", daemon=True),
            threading.Thread(target=self._watchdog_loop, name="Watchdog", daemon=True),
        ]
        for t in threads:
            t.start()

        print("[Main] All threads started. Monitoring …")
        if not self.headless:
            print("  Keys: [SPACE/S] Ack · [Q/ESC] Quit")

        # Main thread runs the GUI event loop (or blocks until shutdown)
        try:
            if self.headless:
                self._headless_wait()
            else:
                self._gui_loop()
        except Exception as exc:
            print(f"[Main] Error in main loop: {exc}")
        finally:
            self._shutdown()

    # ── Vision thread ─────────────────────────────────────────────────

    def _vision_loop(self) -> None:
        if self.cfg.rt.enabled:
            _try_set_affinity("Vision", self.cfg.rt.vision_cpu_affinity)

        while not self.state.shutdown_event.is_set():
            ok, frame = self.camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            self.state.update_frame_time()
            self._frame_id += 1
            rec = process_frame(self.detector, frame, self._frame_id)

            if rec.face_found:
                self.state.update_face_time()

            _put_dropping(self.feature_q, rec)

    # ── Fusion thread ─────────────────────────────────────────────────

    def _fusion_loop(self) -> None:
        if self.cfg.rt.enabled:
            _try_set_rt("Fusion", self.cfg.rt.fusion_priority)
            _try_set_affinity("Fusion", self.cfg.rt.fusion_cpu_affinity)

        prev_alarm = False
        while not self.state.shutdown_event.is_set():
            try:
                rec = self.feature_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Skip if in fault state (watchdog owns that)
            if self.state.fault_event.is_set():
                continue

            alarm = self.fusion.update(rec)

            # Update shared state snapshot
            with self.state.lock:
                self.state.ear_avg = rec.ear_avg
                self.state.mar = rec.mar
                self.state.pitch = rec.pitch_deg
                self.state.perclos = self.fusion.perclos
                self.state.eye_flag = self.fusion.eye_flag
                self.state.yawn_flag = self.fusion.yawn_flag
                self.state.head_flag = self.fusion.head_flag
                self.state.microsleep = self.fusion.microsleep
                self.state.eye_closure_duration = self.fusion.eye_closure_duration

            if alarm and not prev_alarm:
                self.state.alarm_event.set()
                self.state.system_state = SystemState.ALARM
                reason = self._alarm_reason()
                if self.csv_logger:
                    self.csv_logger.log_event(
                        "ALARM_ON", reason,
                        ear=rec.ear_avg, mar=rec.mar,
                        pitch=rec.pitch_deg, perclos=self.fusion.perclos,
                    )
                print(f"\n[ALARM] {reason}")
            elif not alarm and prev_alarm:
                self.state.alarm_event.clear()
                if not self.state.fault_event.is_set():
                    self.state.system_state = SystemState.MONITORING
                if self.csv_logger:
                    self.csv_logger.log_event("ALARM_OFF", "Alarm cleared.")
                print("[Main] Alarm cleared.")

            prev_alarm = alarm

            # Telemetry logging
            if self.csv_logger:
                self.csv_logger.log_telemetry(TelemetryRow(
                    timestamp=rec.timestamp,
                    frame_id=rec.frame_id,
                    face_found=rec.face_found,
                    ear_avg=rec.ear_avg,
                    mar=rec.mar,
                    pitch_deg=rec.pitch_deg,
                    yaw_deg=rec.yaw_deg,
                    roll_deg=rec.roll_deg,
                    perclos=self.fusion.perclos,
                    eye_flag=self.fusion.eye_flag,
                    yawn_flag=self.fusion.yawn_flag,
                    head_flag=self.fusion.head_flag,
                    microsleep=self.fusion.microsleep,
                    eye_closure_dur=self.fusion.eye_closure_duration,
                    system_state=self.state.system_state.value,
                    alarm_active=self.state.alarm_event.is_set(),
                    fault_active=self.state.fault_event.is_set(),
                ))

    def _alarm_reason(self) -> str:
        parts = []
        if self.fusion.microsleep:
            parts.append(f"MICROSLEEP (eyes closed {self.fusion.eye_closure_duration:.1f}s ≥ 3.0s)")
        else:
            flags = []
            if self.fusion.eye_flag:  flags.append("EYE_CLOSURE")
            if self.fusion.yawn_flag: flags.append("YAWN")
            if self.fusion.head_flag: flags.append("HEAD_NOD")
            parts.append(f"Voting: {'+'.join(flags)} ({len(flags)}/3 flags)")
        return "; ".join(parts)

    # ── Alert thread ──────────────────────────────────────────────────

    def _alert_loop(self) -> None:
        if self.cfg.rt.enabled:
            _try_set_rt("Alert", self.cfg.rt.alert_priority)
            _try_set_affinity("Alert", self.cfg.rt.alert_cpu_affinity)

        while not self.state.shutdown_event.is_set():
            if self.state.fault_event.is_set():
                self.alert_ctrl.set_fault()
            elif self.state.alarm_event.is_set():
                self.alert_ctrl.set_alarm()
            else:
                self.alert_ctrl.set_normal()
            time.sleep(0.05)  # 50 ms polling — keeps alert-path latency ≤ 50 ms

    # ── Watchdog thread ───────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        if self.cfg.rt.enabled:
            _try_set_rt("Watchdog", self.cfg.rt.watchdog_priority)

        was_fault = False
        while not self.state.shutdown_event.is_set():
            now = time.time()
            with self.state.lock:
                frame_age = now - self.state.last_frame_time
                face_age  = now - self.state.last_face_time

            frame_stale = frame_age > self.cfg.watchdog.frame_timeout_sec
            face_lost   = face_age  > self.cfg.watchdog.face_timeout_sec

            if frame_stale or face_lost:
                if not self.state.fault_event.is_set():
                    reason = ("no frame" if frame_stale
                              else f"no face for {face_age:.1f}s")
                    self.state.fault_event.set()
                    self.state.system_state = SystemState.FAULT
                    if self.csv_logger:
                        self.csv_logger.log_event(
                            "FAULT_ON", f"Watchdog: {reason}",
                        )
                    print(f"\n[FAULT] Watchdog: {reason}")
                    was_fault = True
            else:
                if self.state.fault_event.is_set():
                    self.state.fault_event.clear()
                    if not self.state.alarm_event.is_set():
                        self.state.system_state = SystemState.MONITORING
                    if was_fault and self.csv_logger:
                        self.csv_logger.log_event("FAULT_OFF", "Watchdog: recovered.")
                    print("[Main] Fault recovered — monitoring resumed.")
                    was_fault = False

            time.sleep(0.1)

    # ── GUI loop (runs in main thread) ─────────────────────────────────

    def _gui_loop(self) -> None:
        while not self.state.shutdown_event.is_set():
            ok, frame = self.camera.read()
            if ok and frame is not None:
                self._draw_hud(frame)
                cv2.imshow(self.cfg.ui.window_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                self.state.shutdown_event.set()
                break
            elif key in (ord('s'), ord('S'), ord(' ')):
                self.alert_ctrl.simulate_ack()

    def _draw_hud(self, frame) -> None:
        """Minimal HUD overlay — keeps main loop lightweight."""
        h, w = frame.shape[:2]
        with self.state.lock:
            ear = self.state.ear_avg
            mar = self.state.mar
            pitch = self.state.pitch
            perclos = self.state.perclos
            eye_f = self.state.eye_flag
            yawn_f = self.state.yawn_flag
            head_f = self.state.head_flag
            ms = self.state.microsleep
            ss = self.state.system_state

        # Banner
        if ss == SystemState.ALARM:
            cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 200), -1)
            cv2.putText(frame, "!! DROWSINESS ALARM !!", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        elif ss == SystemState.FAULT:
            cv2.rectangle(frame, (0, 0), (w, 36), (0, 120, 255), -1)
            cv2.putText(frame, "FAULT: Camera / Face Lost", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (0, 0), (w, 36), (40, 130, 40), -1)
            cv2.putText(frame, "MONITORING — Driver Alert", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

        # Metrics
        y0 = 60
        lines = [
            f"EAR: {ear:.3f}  {'[CLOSED]' if eye_f else ''}",
            f"MAR: {mar:.3f}  {'[YAWN]' if yawn_f else ''}",
            f"Pitch: {pitch:.1f} deg  {'[NOD]' if head_f else ''}",
            f"PERCLOS: {perclos:.3f}",
            f"Microsleep: {'YES' if ms else 'No'}",
            f"Flags: E={int(eye_f)} Y={int(yawn_f)} H={int(head_f)}  "
            f"({sum([eye_f,yawn_f,head_f])}/3)",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (10, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # GPIO status
        gc = (0, 255, 0) if self.alert_ctrl.green_on else (50, 50, 50)
        rc = (0, 0, 255) if self.alert_ctrl.red_on else (50, 50, 50)
        cv2.circle(frame, (w - 40, 55), 12, gc, -1)
        cv2.circle(frame, (w - 40, 85), 12, rc, -1)
        cv2.putText(frame, "G", (w - 46, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "R", (w - 46, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        bz_txt = "BZ:ON" if self.alert_ctrl.buzzer_on else "BZ:off"
        cv2.putText(frame, bz_txt, (w - 65, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Bottom bar
        cv2.putText(frame, "[SPACE/S] Ack  [Q] Quit", (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # ── Headless wait ─────────────────────────────────────────────────

    def _headless_wait(self) -> None:
        tick = 0
        while not self.state.shutdown_event.is_set():
            time.sleep(1.0)
            tick += 1
            if tick % 5 == 0:
                with self.state.lock:
                    print(f"[Headless] EAR={self.state.ear_avg:.3f} "
                          f"MAR={self.state.mar:.3f} "
                          f"Pitch={self.state.pitch:.1f} "
                          f"PERCLOS={self.state.perclos:.3f} "
                          f"State={self.state.system_state.value}")

    # ── Shutdown ──────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        print("\n[Main] Shutting down …")
        self.state.shutdown_event.set()
        self.state.system_state = SystemState.SHUTDOWN

        if self.camera:
            self.camera.stop()
        if self.alert_ctrl:
            self.alert_ctrl.cleanup()
        if self.csv_logger:
            self.csv_logger.close()
        if self.detector:
            self.detector.close()
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[Main] Shutdown complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System — Raspberry Pi 4",
    )
    p.add_argument("--simulate", action="store_true",
                   help="Use mock GPIO (no Raspberry Pi hardware needed).")
    p.add_argument("--no-gui", action="store_true",
                   help="Headless mode (no OpenCV window).")
    p.add_argument("--skip-calibration", action="store_true",
                   help="Skip calibration; use saved or default values.")
    p.add_argument("--camera", default=None,
                   help="Camera source override (index, path, or video file).")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.simulate:
        cfg.gpio.simulation_mode = True
    if args.no_gui:
        cfg.ui.show_gui = False
    if args.camera is not None:
        cfg.camera.source = (int(args.camera) if args.camera.isdigit()
                             else args.camera)

    app = DrowsinessApp(
        cfg,
        simulate=args.simulate,
        headless=args.no_gui,
        skip_cal=args.skip_calibration,
    )

    try:
        app.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
