# Raspberry Pi 4 GPIO Wiring & Hardware Specification

This document details the exact hardware connection scheme, electrical safety considerations, and pinout for the **Driver Drowsiness Detection System** on a **Raspberry Pi 4 Model B**.

---

## 1. Pin Assignment Summary

| Component | BCM Pin | Physical Pin | Direction | Signal Type | Electrical Notes |
|:---|:---|:---|:---|:---|:---|
| **Active Buzzer** | **GPIO 17** | **Pin 11** | Output | Digital (HIGH = Alarm) | Connect via onboard transistor module or NPN S8050 |
| **Green LED (Safe)** | **GPIO 27** | **Pin 13** | Output | Digital (HIGH = ON) | In series with 220 Ω – 330 Ω resistor to GND |
| **Red LED (Alert)** | **GPIO 22** | **Pin 15** | Output | Digital (HIGH = ON) | In series with 220 Ω – 330 Ω resistor to GND |
| **Ack Push Button** | **GPIO 23** | **Pin 16** | Input | Digital (LOW = Pressed)| Connect to GND; Internal pull-up enabled in software |
| **VCC (5V Rail)** | — | **Pin 2 or 4** | Power | +5.0 V DC | Powers buzzer module VCC |
| **GND** | — | **Pin 6, 9, 14, 20, 25, 30, 34, or 39** | Power | Ground | Common reference ground |

---

## 2. Raspberry Pi 4 Model B 40-Pin Header Layout

```
                  3.3V Power (Pin 1)  [○] [○]  5.0V Power (Pin 2) ── Buzzer VCC
                      GPIO 2 (Pin 3)  [○] [○]  5.0V Power (Pin 4)
                      GPIO 3 (Pin 5)  [○] [●]  Ground (Pin 6) ────── Buzzer / LED GND
                      GPIO 4 (Pin 7)  [○] [○]  GPIO 14 (Pin 8)
                     Ground (Pin 9)   [○] [○]  GPIO 15 (Pin 10)
  Buzzer I/O ◄────  GPIO 17 (Pin 11)  [●] [○]  GPIO 18 (Pin 12)
  Green LED  ◄────  GPIO 27 (Pin 13)  [●] [●]  Ground (Pin 14) ───── Common GND
  Red LED    ◄────  GPIO 22 (Pin 15)  [●] [●]  GPIO 23 (Pin 16) ──── Ack Button
                  3.3V Power (Pin 17) [○] [○]  GPIO 24 (Pin 18)
                     GPIO 10 (Pin 19) [○] [●]  Ground (Pin 20) ───── Button GND
```

---

## 3. Circuit Schematics & Electrical Calculation

### A. Red and Green LEDs
- **Forward Voltage ($V_f$)**: ~2.0 V (Red/Green standard 5mm LEDs)
- **Desired Forward Current ($I_f$)**: 6–10 mA (well below the maximum 16 mA GPIO limit)
- **GPIO Voltage ($V_{OH}$)**: 3.3 V

$$R = \frac{V_{GPIO} - V_f}{I_f} = \frac{3.3\,\text{V} - 2.0\,\text{V}}{0.008\,\text{A}} \approx 162.5\,\Omega \implies \text{Use standard } 220\,\Omega \text{ or } 330\,\Omega \text{ resistor.}$$

Wiring:
```
RPi GPIO 27 ────[ 220 Ω ]────(+) LED Anode | ( - ) Cathode ──── GND
RPi GPIO 22 ────[ 220 Ω ]────(+) LED Anode | ( - ) Cathode ──── GND
```

### B. Active Buzzer Module (3-Pin)
A standard 3-pin Active Buzzer module contains an onboard NPN transistor (e.g., S8050) and flyback diode.
- **VCC**: Connect to Physical Pin 2 or 4 (+5V)
- **GND**: Connect to Physical Pin 6 (Ground)
- **I/O (Signal)**: Connect to Physical Pin 11 (GPIO 17)

*Caution:* Never connect a raw 5V buzzer coil directly to a GPIO pin without a transistor driver; Raspberry Pi GPIO pins tolerate a maximum of 3.3V and up to 16mA per pin.

### C. Acknowledge Push Button
- One terminal connects to **GPIO 23 (Pin 16)**.
- The opposite terminal connects to **GND (Pin 20)**.
- The software configures the internal pull-up resistor (`pull_up=True` in `gpiozero.Button`), holding the pin HIGH (+3.3V) at rest. When pressed, the pin is pulled to GND (LOW).
- Software debounce time is set to `0.2s` (`bounce_time=0.2`).

---

## 4. Hardware Verification & Troubleshooting

1. **LEDs don't light up**: Check polarity (long leg is Anode `+`, flat edge is Cathode `-`).
2. **Buzzer clicking instead of buzzing**: Ensure an *active* buzzer is used (produces sound with continuous DC voltage), not a *passive* buzzer (which requires PWM frequency generation).
3. **Ghost button presses**: Ensure the button connects directly between GPIO 23 and Ground. Never connect 5V or 3.3V to the button terminal when pull-up is active.
