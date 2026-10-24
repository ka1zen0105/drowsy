# Testing & Verification Guide

This guide describes how to validate the **Driver Drowsiness Detection System** both on a development machine (Windows / macOS) using Mock GPIO and on a physical **Raspberry Pi 4 Model B**.

---

## 1. Automated Test Suite

Run the unit and integration tests using pytest:

```bash
python -m pytest tests/ -v
```

### Test Scope:
- **`test_features.py`**: Validates pure geometric calculations for EAR (open, closed, half-open, degenerate), MAR (resting, talking, yawning), and head pitch estimation using `solvePnP`.
- **`test_fusion.py`**: Verifies temporal voting logic (2-of-3 flags), 3-second microsleep override, isolated blinks, button acknowledgement with cooldown, and auto-clearing after 2.0s recovery.
- **`test_alert.py`**: Verifies `MockAlertController` state transitions (`set_normal`, `set_alarm`, `set_fault`, `all_off`) and hardware self-test sequences.
- **`test_integration.py`**: Verifies YAML config loading, calibration file persistence round-trips, asynchronous CSV logger writing, and watchdog timeout detection.

---

## 2. Testing on Development Laptop (Simulation Mode)

On non-Linux platforms (Windows/macOS), the system automatically activates `MockAlertController`.

### Step 2.1 — Run Calibration
```bash
python calibrate.py --duration 30
```
- Sit in front of your webcam, alert, with eyes open, mouth closed, and looking forward.
- Progress will be logged in the console.
- At completion, personal baseline values are written to `calibration.yaml`.

### Step 2.2 — Run Real-Time Detection
```bash
python main.py
```

### Step 2.3 — Test Scenarios in Real-Time:

| Scenario | Action | Expected Output |
|:---|:---|:---|
| **Normal Alertness** | Keep eyes open, looking forward | Green LED active, system state `MONITORING`, no buzzer |
| **Normal Blinking** | Blink normally (< 0.5 s) | EAR drops temporarily, but NO flag or alarm triggers |
| **Isolated Eye Closure** | Close eyes for 1.5 s – 2.5 s | `eye_flag` sets to `True`, but NO alarm (only 1 of 3 flags) |
| **Microsleep Override** | Close eyes continuously for ≥ 3.0 s | System switches to `ALARM`! Red LED and continuous buzzer sound |
| **Yawn + Head Nod** | Yawn (open mouth wide ≥ 1.5s) while nodding head down | 2 of 3 flags active → `ALARM` triggers! |
| **Push Button Ack** | Press `a` key in GUI or simulate ack | Alarm silences immediately and enters cooldown |
| **Auto-Recovery** | Return to alert posture for ≥ 2.0 s | Alarm automatically clears and returns to `MONITORING` |
| **Face Loss / Obstruction** | Cover camera lens or turn head away for > 2 s | Watchdog trips → System enters `FAULT`! Red LED blinks rapidly, buzzer beeps short pulses |

---

## 3. Testing on Physical Raspberry Pi 4

### Step 3.1 — Verify GPIO Hardware Wiring
1. Ensure the Raspberry Pi is powered off before connecting wires.
2. Connect Active Buzzer (GPIO 17), Green LED (GPIO 27), Red LED (GPIO 22), and Push Button (GPIO 23) according to [wiring.md](wiring.md).
3. Power on the Pi and open a terminal.

### Step 3.2 — Test GPIO Self-Test
Run a quick diagnostic to test the buzzer and LEDs:
```bash
python -c "from config import load_config; from alert import create_alert_controller; ctrl = create_alert_controller(load_config().gpio); ctrl.self_test(); ctrl.cleanup()"
```
- The Green LED will turn on for 0.3s.
- The Red LED will turn on for 0.3s.
- The Buzzer will beep for 0.15s.

### Step 3.3 — Run Headless / Vehicle Mode
On a vehicle dashboard without an attached display:
```bash
python main.py --headless
```

---

## 4. Performance & Telemetry Evaluation

After running a session, evaluate the recorded logs:

```bash
python tools/evaluate.py logs/
```

This outputs:
- Effective frame processing rate (FPS)
- Percentage of frames where face tracking was maintained
- Minimum, median, and maximum EAR, MAR, and pitch distributions
- Total number of alarms and acknowledgements recorded in `events.csv`
