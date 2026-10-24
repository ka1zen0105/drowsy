"""Unit tests for MockAlertController and alert state transitions (alert.py)."""

import time
import pytest

from alert import MockAlertController, create_alert_controller
from config import GPIOConfig


@pytest.fixture
def gpio_config():
    return GPIOConfig(
        buzzer_pin=17,
        green_led_pin=27,
        red_led_pin=22,
        ack_button_pin=23,
        simulation_mode=True,
    )


def test_initial_state(gpio_config):
    ctrl = MockAlertController(gpio_config)
    assert not ctrl.green_on
    assert not ctrl.red_on
    assert not ctrl.buzzer_on


def test_set_normal(gpio_config):
    ctrl = MockAlertController(gpio_config)
    ctrl.set_normal()
    assert ctrl.green_on
    assert not ctrl.red_on
    assert not ctrl.buzzer_on


def test_set_alarm(gpio_config):
    ctrl = MockAlertController(gpio_config)
    ctrl.set_normal()
    assert ctrl.green_on

    ctrl.set_alarm()
    assert not ctrl.green_on
    assert ctrl.red_on
    assert ctrl.buzzer_on


def test_set_fault(gpio_config):
    ctrl = MockAlertController(gpio_config)
    ctrl.set_fault()
    assert not ctrl.green_on
    # Give fault pattern a moment to tick
    time.sleep(0.3)
    # The fault pattern toggles red LED and beeps buzzer periodically
    assert ctrl._fault_running
    ctrl.all_off()
    assert not ctrl._fault_running
    assert not ctrl.green_on
    assert not ctrl.red_on
    assert not ctrl.buzzer_on


def test_transition_alarm_to_normal(gpio_config):
    ctrl = MockAlertController(gpio_config)
    ctrl.set_alarm()
    assert ctrl.red_on
    assert ctrl.buzzer_on

    ctrl.set_normal()
    assert ctrl.green_on
    assert not ctrl.red_on
    assert not ctrl.buzzer_on


def test_all_off_and_cleanup(gpio_config):
    ctrl = MockAlertController(gpio_config)
    ctrl.set_alarm()
    ctrl.cleanup()
    assert not ctrl.green_on
    assert not ctrl.red_on
    assert not ctrl.buzzer_on


def test_ack_callback(gpio_config):
    ctrl = MockAlertController(gpio_config)
    called = []

    def on_ack():
        called.append(True)

    ctrl.register_ack_callback(on_ack)
    ctrl.simulate_ack()
    assert len(called) == 1


def test_factory_creates_mock_when_simulation_mode(gpio_config):
    ctrl = create_alert_controller(gpio_config)
    assert isinstance(ctrl, MockAlertController)
