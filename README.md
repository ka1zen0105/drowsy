# Real-Time Driver Drowsiness Detection System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Target Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204%20Model%20B-red.svg)](https://www.raspberrypi.com/)
[![OS](https://img.shields.io/badge/OS-Raspberry%20Pi%20OS%2064--bit%20(Bookworm)-green.svg)](https://www.raspberrypi.com/software/)
[![Tests](https://img.shields.io/badge/Tests-32%20passing-brightgreen.svg)](tests/)

A robust, multi-threaded Driver Drowsiness Detection System designed for embedded deployment on the **Raspberry Pi 4 Model B** running **Raspberry Pi OS 64-bit (Debian 12 / Bookworm)**. The system monitors facial landmarks in real-time, calculates physiological indices of fatigue, fuses indicators using voting and microsleep overrides, and controls automotive-grade alert hardware with watchdog fail-safes.

---

## Architecture Overview

```
                                      ┌────────────────────────────────┐
                                      │         USB Webcam             │
                                      └──────────────┬─────────────────┘
                                                     │ Frames
                                                     ▼
┌──────────────────────┐              ┌────────────────────────────────┐
│   Watchdog Thread    │              │         Vision Thread          │
│ • Frame stall > 0.5s │◄─────────────┤ • MediaPipe Face Mesh (Task)   │
│ • Face lost > 2.0s   │  Timestamps  │ • EAR, MAR, solvePnP Head Pose │
└──────────┬───────────┘              └──────────────┬─────────────────┘
           │                                         │ FeatureRecord
           │ Fault Event                             │ (Queue, maxsize=5)
           ▼                                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                            Fusion Thread                             │
│ • 30-second sliding-window PERCLOS                                   │
│ • 3 Independent Flags: Eye (≥1.5s), Yawn (≥1.5s), Nodding (≥1.0s)   │
│ • Decision Rules:                                                    │
│     - 2-of-3 Voting Threshold                                        │
│     - 3.0s Continuous Microsleep Override                            │
│ • Auto-clear (2.0s recovery) & Ack Button Cooldown                   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ Alarm / Fault Events
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                             Alert Thread                             │
│ • Active Buzzer (GPIO 17)                                            │
│ • Safe Green LED (GPIO 27)                                           │
│ • Alert Red LED (GPIO 22)                                            │
│ • Acknowledge Push Button (GPIO 23)                                  │
│ • Normal, Alarm, and Pulsed Fault Patterns                           │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Shared Telemetry & Events
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                            Logging Thread                            │
│ • Asynchronous thread-safe CSV writer                                │
│ • telemetry.csv (frame metrics, timestamps, states)                  │
│ • events.csv (discrete triggers, faults, acks)                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

1. **Facial Geometry Extraction (`features.py`)**:
   - **Eye Aspect Ratio (EAR)**: Computes 6-point Euclidean landmark ratios for both eyes.
   - **Mouth Aspect Ratio (MAR)**: Monitors vertical mouth opening relative to horizontal width.
   - **Head Pitch & Pose**: Uses OpenCV `solvePnP` with a 3D canonical facial model to estimate pitch (degrees), yaw, and roll without relying on uncalibrated 2D approximations.
2. **Personalized Baseline Calibration (`calibrate.py`)**:
   - Gathers 30–60 seconds of alert-driver data to establish individual baselines for open-eye EAR, resting MAR, upright head pitch, and baseline PERCLOS.
   - Saves parameters to `calibration.yaml` and merges them dynamically at startup.
3. **Robust Fusion Logic (`fusion.py`)**:
   - **3 Independent Flags**:
     - `eye_flag`: Smoothed EAR < calibrated closed threshold for continuous $\ge 1.5\,\text{s}$.
     - `yawn_flag`: MAR > calibrated yawn threshold for continuous $\ge 1.5\,\text{s}$.
     - `head_flag`: Forward head pitch deviation $\ge 15^\circ$ for continuous $\ge 1.0\,\text{s}$.
   - **2-of-3 Voting**: Prevents false alarms from an isolated yawn, blink, or gear-shift glance.
   - **Microsleep Override**: Continuous eye closure $\ge 3.0\,\text{s}$ triggers an emergency alarm immediately, bypassing the voting threshold.
   - **Auto-Recovery**: Automatically clears alarms after $2.0\,\text{s}$ of alert posture.
4. **Hardware Controller & Fail-Safe Watchdog (`alert.py`, `main.py`)**:
   - Controls physical GPIO using `gpiozero` and `lgpio` on Raspberry Pi.
   - Includes automatic `MockAlertController` simulation when running on development laptops (Windows/macOS).
   - **Watchdog Thread**: Detects camera stalls ($>0.5\,\text{s}$) or face disappearances ($>2.0\,\text{s}$) and transitions to `FAULT` state (distinct pulsing red LED and short buzzer beeps). Absence of face is never mistaken for an awake driver.
5. **Thread-Safe Telemetry & Evaluation (`logger.py`, `tools/evaluate.py`)**:
   - Asynchronous CSV writer drains an unbounded queue so computer-vision operations are never blocked by disk I/O.
   - Evaluation tool calculates effective FPS, face tracking reliability, latency, and alarm statistics.

---

## Hardware Wiring (Raspberry Pi 4)

| Component | BCM Pin | Physical Pin | Purpose / Wiring |
|:---|:---|:---|:---|
| **Active Buzzer** | **GPIO 17** | Pin 11 | Alert tone. Signal to GPIO 17, VCC to 5V (Pin 2), GND to GND (Pin 6). |
| **Green LED** | **GPIO 27** | Pin 13 | Normal/safe indicator. Connect in series with 220 Ω resistor to GND. |
| **Red LED** | **GPIO 22** | Pin 15 | Alert/fault indicator. Connect in series with 220 Ω resistor to GND. |
| **Push Button** | **GPIO 23** | Pin 16 | Alarm silence / ack. Connect between GPIO 23 and GND (internal pull-up). |

*See [docs/wiring.md](docs/wiring.md) for full electrical schematics and current-limiting calculations.*

---

## Directory Structure

```
c:/drowsy/
├── config.yaml               # Configurable thresholds, GPIO pins, and timings
├── config.py                 # YAML parser, dataclasses, calibration merger
├── features.py               # Pure math: EAR, MAR, solvePnP head pose
├── vision.py                 # MediaPipe Face Mesh pipeline & video capture
├── fusion.py                 # Temporal voting engine, PERCLOS, microsleep override
├── alert.py                  # gpiozero GPIO driver + MockGPIO simulation
├── logger.py                 # Asynchronous background CSV telemetry writer
├── calibrate.py              # Personal baseline calibration utility
├── main.py                   # 5-thread orchestrator, startup & shutdown handler
├── requirements.txt          # Python package requirements
├── .gitignore                # Git exclusions
├── README.md                 # Project documentation
├── models/
│   └── face_landmarker.task  # Pre-trained MediaPipe face landmark model
├── tests/
│   ├── test_features.py      # Feature calculation unit tests
│   ├── test_fusion.py        # Voting logic & microsleep tests
│   ├── test_alert.py         # Hardware state transition tests
│   └── test_integration.py   # Config, logger, and watchdog tests
├── docs/
│   ├── wiring.md             # GPIO wiring diagrams & safety specifications
│   └── testing.md            # Hardware and simulation testing instructions
└── tools/
    └── evaluate.py           # Post-run CSV log evaluation tool
```

---

## Getting Started

### 1. Installation

#### On Windows / Development Laptop:
```bash
git clone https://github.com/your-username/drowsiness-detection.git
cd drowsiness-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### On Raspberry Pi 4 Model B (Raspberry Pi OS 64-bit Bookworm):
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-gpiozero python3-lgpio libcamera-dev
git clone https://github.com/your-username/drowsiness-detection.git
cd drowsiness-detection
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run Personal Calibration

Before starting a drive, calibrate to establish personal baseline parameters:

```bash
python calibrate.py --duration 30
```
- Sit naturally in front of the camera with your normal resting expression.
- Baselines are calculated and saved to `calibration.yaml`.

---

### 3. Run Real-Time Detection

#### GUI Mode (Development & Testing):
```bash
python main.py
```
- Interactive controls in GUI window:
  - Press `a`: Simulate acknowledge button press.
  - Press `q` or `ESC`: Safe shutdown.

#### Headless Mode (Vehicle Deployment):
```bash
python main.py --headless
```

---

### 4. Running the Test Suite

Execute all 32 unit and integration tests:

```bash
python -m pytest tests/ -v
```

---

### 5. Post-Run Telemetry Analysis

Evaluate performance metrics from a completed trip:

```bash
python tools/evaluate.py logs/
```

Displays:
- Effective processing FPS
- Percentage of frames with face tracking
- Feature distributions (EAR, MAR, Pitch, PERCLOS percentiles)
- Discrete alarm and acknowledgement events

---

## License

This project is open-source under the MIT License.
