"""GPIO hardware controller: buzzer, LEDs, acknowledge button.

Provides two implementations sharing a common interface:

* ``HardwareAlertController`` — real GPIO via *gpiozero* + *lgpio* on Raspberry Pi.
* ``MockAlertController``     — simulation for Windows / macOS development.

The factory ``create_alert_controller()`` picks the right one automatically.
"""

from __future__ import annotations

import abc
import sys
import threading
import time
from typing import Callable, Optional

from config import GPIOConfig


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AlertController(abc.ABC):
    """Interface for buzzer / LED / button hardware."""

    @abc.abstractmethod
    def self_test(self) -> None:
        """Quick power-on self-test: blink LEDs and beep buzzer."""

    @abc.abstractmethod
    def set_normal(self) -> None:
        """Monitoring state — green LED on, red LED off, buzzer off."""

    @abc.abstractmethod
    def set_alarm(self) -> None:
        """Drowsiness alarm — red LED solid, buzzer continuous."""

    @abc.abstractmethod
    def set_fault(self) -> None:
        """Fault state — red LED blinking, buzzer short beeps."""

    @abc.abstractmethod
    def all_off(self) -> None:
        """Silence everything (shutdown path)."""

    @abc.abstractmethod
    def register_ack_callback(self, cb: Callable[[], None]) -> None:
        """Register callback fired when acknowledge button is pressed."""

    @abc.abstractmethod
    def simulate_ack(self) -> None:
        """Software-triggered ack (keyboard shortcut in GUI)."""

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Release all GPIO resources."""

    @property
    @abc.abstractmethod
    def green_on(self) -> bool: ...

    @property
    @abc.abstractmethod
    def red_on(self) -> bool: ...

    @property
    @abc.abstractmethod
    def buzzer_on(self) -> bool: ...


# ---------------------------------------------------------------------------
# Mock controller (Windows / macOS / CI)
# ---------------------------------------------------------------------------

class MockAlertController(AlertController):
    """Software mock — logs state changes, no hardware required."""

    def __init__(self, config: GPIOConfig):
        self.cfg = config
        self._green = False
        self._red = False
        self._buzzer = False
        self._ack_cb: Optional[Callable[[], None]] = None
        self._fault_thread: Optional[threading.Thread] = None
        self._fault_running = False
        self._current_mode = "off"
        print("[MockGPIO] Simulation controller initialised "
              f"(Buzzer={config.buzzer_pin}, Green={config.green_led_pin}, "
              f"Red={config.red_led_pin}, Button={config.ack_button_pin}).")

    @property
    def green_on(self) -> bool:
        return self._green

    @property
    def red_on(self) -> bool:
        return self._red

    @property
    def buzzer_on(self) -> bool:
        return self._buzzer

    def self_test(self) -> None:
        print("[MockGPIO] Self-test: Green ON …")
        self._green = True
        time.sleep(0.3)
        self._green = False
        print("[MockGPIO] Self-test: Red ON …")
        self._red = True
        time.sleep(0.3)
        self._red = False
        print("[MockGPIO] Self-test: Buzzer beep …")
        self._buzzer = True
        time.sleep(0.15)
        self._buzzer = False
        print("[MockGPIO] Self-test complete.")

    def set_normal(self) -> None:
        if self._current_mode == "normal":
            return
        self._stop_fault_pattern()
        self._green = True
        self._red = False
        self._buzzer = False
        self._current_mode = "normal"

    def set_alarm(self) -> None:
        if self._current_mode == "alarm":
            return
        self._stop_fault_pattern()
        self._green = False
        self._red = True
        self._buzzer = True
        self._current_mode = "alarm"

    def set_fault(self) -> None:
        if self._current_mode == "fault":
            return
        self._stop_fault_pattern()
        self._green = False
        self._fault_running = True
        self._fault_thread = threading.Thread(target=self._fault_loop,
                                              daemon=True, name="MockFault")
        self._fault_thread.start()
        self._current_mode = "fault"

    def _fault_loop(self) -> None:
        """Blink red LED (0.25 s on/off) while keeping buzzer silent."""
        self._buzzer = False
        while self._fault_running:
            self._red = not self._red
            time.sleep(0.25)

    def _stop_fault_pattern(self) -> None:
        self._fault_running = False
        if self._fault_thread and self._fault_thread.is_alive():
            self._fault_thread.join(timeout=1.0)
        self._fault_thread = None

    def all_off(self) -> None:
        self._stop_fault_pattern()
        self._green = False
        self._red = False
        self._buzzer = False
        self._current_mode = "off"

    def register_ack_callback(self, cb: Callable[[], None]) -> None:
        self._ack_cb = cb

    def simulate_ack(self) -> None:
        print("[MockGPIO] ▶ Acknowledge button pressed (simulated).")
        if self._ack_cb:
            self._ack_cb()

    def cleanup(self) -> None:
        self.all_off()
        print("[MockGPIO] Cleaned up.")


# ---------------------------------------------------------------------------
# Real Raspberry Pi controller (gpiozero + lgpio)
# ---------------------------------------------------------------------------

class HardwareAlertController(AlertController):
    """Production GPIO controller using *gpiozero* with the *lgpio* pin factory.

    Requires ``gpiozero`` and ``lgpio`` packages on Raspberry Pi OS 64-bit
    (Bookworm / Debian 12).
    """

    def __init__(self, config: GPIOConfig):
        self.cfg = config
        self._ack_cb: Optional[Callable[[], None]] = None
        self._fault_thread: Optional[threading.Thread] = None
        self._fault_running = False
        self._current_mode = "off"

        try:
            from gpiozero import LED, Buzzer, Button
            from gpiozero import Device
            # Try lgpio pin factory (Bookworm default)
            try:
                from gpiozero.pins.lgpio import LGPIOFactory
                Device.pin_factory = LGPIOFactory()
            except Exception:
                pass  # Fall back to default (rpigpio / pigpio)

            self._green_led = LED(config.green_led_pin)
            self._red_led   = LED(config.red_led_pin)
            active_high     = getattr(config, "buzzer_active_high", True)
            self._buzzer    = Buzzer(config.buzzer_pin, active_high=active_high)
            self._button    = Button(config.ack_button_pin, pull_up=True,
                                      bounce_time=0.2)
            self._button.when_pressed = self._on_button
        except Exception as exc:
            raise RuntimeError(f"GPIO hardware init failed: {exc}")

        print(f"[GPIO] Hardware initialised — Buzzer={config.buzzer_pin}, "
              f"Green={config.green_led_pin}, Red={config.red_led_pin}, "
              f"Button={config.ack_button_pin}.")

    # -- properties ---------------------------------------------------------

    @property
    def green_on(self) -> bool:
        return bool(self._green_led.is_lit)

    @property
    def red_on(self) -> bool:
        return bool(self._red_led.is_lit)

    @property
    def buzzer_on(self) -> bool:
        return bool(self._buzzer.is_active)

    # -- modes --------------------------------------------------------------

    def self_test(self) -> None:
        self._green_led.on();  time.sleep(0.3); self._green_led.off()
        self._red_led.on();    time.sleep(0.3); self._red_led.off()
        self._buzzer.on();     time.sleep(0.15); self._buzzer.off()
        print("[GPIO] Self-test complete.")

    def set_normal(self) -> None:
        if self._current_mode == "normal":
            return
        self._stop_fault_pattern()
        self._green_led.on()
        self._red_led.off()
        self._buzzer.off()
        self._current_mode = "normal"

    def set_alarm(self) -> None:
        if self._current_mode == "alarm":
            return
        self._stop_fault_pattern()
        self._green_led.off()
        self._red_led.on()
        self._buzzer.on()
        self._current_mode = "alarm"

    def set_fault(self) -> None:
        if self._current_mode == "fault":
            return
        self._stop_fault_pattern()
        self._green_led.off()
        self._buzzer.off()
        self._fault_running = True
        self._fault_thread = threading.Thread(target=self._fault_loop,
                                              daemon=True, name="FaultBlink")
        self._fault_thread.start()
        self._current_mode = "fault"

    def _fault_loop(self) -> None:
        """Blink red LED during fault state while keeping buzzer silent."""
        while self._fault_running:
            self._red_led.toggle()
            time.sleep(0.25)

    def _stop_fault_pattern(self) -> None:
        self._fault_running = False
        if self._fault_thread and self._fault_thread.is_alive():
            self._fault_thread.join(timeout=1.0)
        self._fault_thread = None

    def all_off(self) -> None:
        self._stop_fault_pattern()
        self._green_led.off()
        self._red_led.off()
        self._buzzer.off()
        self._current_mode = "off"

    def _on_button(self) -> None:
        print("[GPIO] ▶ Acknowledge button pressed.")
        if self._ack_cb:
            self._ack_cb()

    def register_ack_callback(self, cb: Callable[[], None]) -> None:
        self._ack_cb = cb

    def simulate_ack(self) -> None:
        self._on_button()

    def cleanup(self) -> None:
        self.all_off()
        try:
            self._green_led.close()
            self._red_led.close()
            self._buzzer.close()
            self._button.close()
        except Exception:
            pass
        print("[GPIO] Hardware released.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _is_raspberry_pi() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "r", errors="ignore") as f:
                if "raspberry pi" in f.read().lower():
                    return True
        except OSError:
            pass
    return False


def create_alert_controller(config: GPIOConfig) -> AlertController:
    """Auto-detect platform and return the appropriate controller."""
    if config.simulation_mode:
        return MockAlertController(config)
    if _is_raspberry_pi():
        try:
            return HardwareAlertController(config)
        except RuntimeError as exc:
            print(f"[Alert] Hardware init failed ({exc}); falling back to mock.")
            return MockAlertController(config)
    print("[Alert] Not running on Raspberry Pi — using mock controller.")
    return MockAlertController(config)
