import os
import sys
import types
import unittest
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "serial" not in sys.modules:
    serial_module = types.ModuleType("serial")
    serial_module.PARITY_NONE = "N"
    serial_module.SerialException = Exception
    serial_module.Serial = lambda *args, **kwargs: None
    serial_tools_module = types.ModuleType("serial.tools")
    list_ports_module = types.ModuleType("serial.tools.list_ports")
    list_ports_module.comports = lambda: []
    serial_tools_module.list_ports = list_ports_module
    sys.modules["serial"] = serial_module
    sys.modules["serial.tools"] = serial_tools_module
    sys.modules["serial.tools.list_ports"] = list_ports_module

from utils.BrushlessPtz import BrushlessGimbal


class BrushlessGimbalSerialTests(unittest.TestCase):
    def test_init_opens_pitch_and_yaw_serial_ports_once(self):
        fake_pitch_serial = mock.Mock()
        fake_yaw_serial = mock.Mock()

        with mock.patch(
            "utils.BrushlessPtz.serial.Serial",
            side_effect=[fake_pitch_serial, fake_yaw_serial],
        ) as serial_ctor:
            gimbal = BrushlessGimbal(
                pitch_port="/dev/ttyUSB0",
                yaw_port="/dev/ttyUSB1",
                pitch_addr=1,
                yaw_addr=1,
                baudrate=115200,
                home_wait_sec=0,
            )

        self.assertEqual(2, serial_ctor.call_count)
        self.assertEqual("/dev/ttyUSB0", serial_ctor.call_args_list[0].kwargs["port"])
        self.assertEqual("/dev/ttyUSB1", serial_ctor.call_args_list[1].kwargs["port"])
        self.assertIs(gimbal.pitch.ser, fake_pitch_serial)
        self.assertIs(gimbal.yaw.ser, fake_yaw_serial)

    def test_home_lock_homes_then_enters_speed_mode_and_stops(self):
        class Axis:
            def __init__(self, name):
                self.name = name
                self.calls = []

            def close_loop(self):
                self.calls.append("close_loop")
                return True

            def set_position_mode(self):
                self.calls.append("position_mode")
                return True

            def set_abs_position(self, angle_deg):
                self.calls.append(("abs_position", angle_deg))
                return True

            def set_speed_mode(self):
                self.calls.append("speed_mode")
                return True

            def set_speed(self, rpm):
                self.calls.append(("speed", rpm))
                return True

        gimbal = BrushlessGimbal.__new__(BrushlessGimbal)
        gimbal.pitch = Axis("pitch")
        gimbal.yaw = Axis("yaw")
        gimbal.home_pitch_deg = 0.0
        gimbal.home_yaw_deg = 0.0
        gimbal.home_wait_sec = 0
        gimbal.last_pitch_speed = None
        gimbal.last_yaw_speed = None

        gimbal.home_lock()

        expected = [
            "close_loop",
            "position_mode",
            ("abs_position", 0.0),
            "speed_mode",
            ("speed", 0.0),
        ]
        self.assertEqual(expected, gimbal.pitch.calls)
        self.assertEqual(expected, gimbal.yaw.calls)
        self.assertEqual(0.0, gimbal.last_pitch_speed)
        self.assertEqual(0.0, gimbal.last_yaw_speed)


if __name__ == "__main__":
    unittest.main()
