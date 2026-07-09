import logging
import struct
import time

import serial
from serial.tools import list_ports


def crc16_modbus(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return struct.pack("<H", crc)


def _format_available_serial_ports() -> str:
    ports = list(list_ports.comports())
    if not ports:
        return "none"
    return ", ".join(f"{port.device} ({port.description})" for port in ports)


class JC2804Serial:
    WRITE_SINGLE = 0x06
    WRITE_MULTI = 0x10

    REG_CONTROL_MODE = 0x60
    REG_SET_SPEED = 0x21
    REG_ABS_POS = 0x23
    REG_CLOSE_LOOP = 0xA2

    MODE_SPEED = 1
    MODE_TRAP = 2

    def __init__(self, name: str, port: str, slave_addr: int, baudrate=115200, timeout=0.5):
        self.name = name
        self.port = port
        self.addr = slave_addr
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=8,
                parity=serial.PARITY_NONE,
                stopbits=1,
                timeout=timeout,
            )
        except serial.SerialException as exc:
            available_ports = _format_available_serial_ports()
            raise serial.SerialException(
                f"could not open {name} serial port {port!r}: {exc}. "
                f"Available ports: {available_ports}. "
                "Check whether another process already has this USB serial port open."
            ) from exc
        logging.info("%s serial opened: %s addr=%s baudrate=%s", self.name, port, slave_addr, baudrate)

    def _check_crc(self, reply: bytes) -> bool:
        if len(reply) < 3:
            return False
        recv_crc = struct.unpack("<H", reply[-2:])[0]
        calc_crc = struct.unpack("<H", crc16_modbus(reply[:-2]))[0]
        return recv_crc == calc_crc

    def _send_pdu(self, pdu: bytes, reply_len=8) -> bytes:
        frame_no_crc = bytes([self.addr]) + pdu
        frame = frame_no_crc + crc16_modbus(frame_no_crc)

        self.ser.reset_input_buffer()
        self.ser.write(frame)
        self.ser.flush()
        time.sleep(0.03)

        reply = self.ser.read(reply_len)
        if len(reply) < 4:
            logging.warning("%s reply too short", self.name)
            return b""
        if not self._check_crc(reply):
            logging.warning("%s reply CRC check failed: %s", self.name, reply.hex(" ").upper())
            return b""
        return reply

    def write_single(self, reg: int, value: int) -> bool:
        pdu = struct.pack(">BHH", self.WRITE_SINGLE, reg, value & 0xFFFF)
        reply = self._send_pdu(pdu, reply_len=8)
        if len(reply) < 8:
            logging.error("%s write single failed reg=0x%04X", self.name, reg)
            return False
        return True

    def write_multi(self, reg: int, values) -> bool:
        num_regs = len(values)
        data = b"".join(struct.pack(">H", value & 0xFFFF) for value in values)
        pdu = struct.pack(">BHHB", self.WRITE_MULTI, reg, num_regs, num_regs * 2) + data
        reply = self._send_pdu(pdu, reply_len=8)
        if len(reply) < 8:
            logging.error("%s write multi failed reg=0x%04X values=%s", self.name, reg, values)
            return False
        return True

    def close_loop(self) -> bool:
        logging.info("%s enter closed-loop control", self.name)
        return self.write_single(self.REG_CLOSE_LOOP, 0x0001)

    def set_speed_mode(self) -> bool:
        logging.info("%s set speed mode", self.name)
        return self.write_single(self.REG_CONTROL_MODE, self.MODE_SPEED)

    def set_position_mode(self) -> bool:
        logging.info("%s set position mode", self.name)
        return self.write_single(self.REG_CONTROL_MODE, self.MODE_TRAP)

    def set_speed(self, rpm: float) -> bool:
        raw = int(float(rpm) * 100) & 0xFFFFFFFF
        high = (raw >> 16) & 0xFFFF
        low = raw & 0xFFFF
        return self.write_multi(self.REG_SET_SPEED, [high, low])

    def set_abs_position(self, angle_deg: float) -> bool:
        raw = int(float(angle_deg) * 100) & 0xFFFFFFFF
        high = (raw >> 16) & 0xFFFF
        low = raw & 0xFFFF
        logging.info("%s set absolute position %.2f deg raw=0x%08X", self.name, angle_deg, raw)
        return self.write_multi(self.REG_ABS_POS, [high, low])

    def stop(self):
        return self.set_speed(0.0)

    def close(self):
        self.ser.close()
        logging.info("%s serial closed", self.name)


class BrushlessGimbal:
    def __init__(
        self,
        pitch_port: str,
        yaw_port: str,
        pitch_addr: int = 1,
        yaw_addr: int = 1,
        baudrate: int = 115200,
        pitch_max_rpm: float = 20.0,
        yaw_max_rpm: float = 25.0,
        home_pitch_deg: float = 0.0,
        home_yaw_deg: float = 0.0,
        home_wait_sec: float = 3.0,
        command_wait_sec: float = 0.2,
    ):
        self.pitch = JC2804Serial("Pitch", pitch_port, pitch_addr, baudrate)
        self.yaw = JC2804Serial("Yaw", yaw_port, yaw_addr, baudrate)
        self.pitch_max_rpm = float(pitch_max_rpm)
        self.yaw_max_rpm = float(yaw_max_rpm)
        self.home_pitch_deg = float(home_pitch_deg)
        self.home_yaw_deg = float(home_yaw_deg)
        self.home_wait_sec = float(home_wait_sec)
        self.command_wait_sec = float(command_wait_sec)
        self.last_pitch_speed = None
        self.last_yaw_speed = None

    def home_lock(self):
        logging.info("Brushless PTZ homing and locking")
        self.pitch.close_loop()
        self.yaw.close_loop()

        self.pitch.set_position_mode()
        self.yaw.set_position_mode()
        self._sleep(getattr(self, "command_wait_sec", 0.0))

        self.pitch.set_abs_position(self.home_pitch_deg)
        self.yaw.set_abs_position(self.home_yaw_deg)
        self._sleep(getattr(self, "home_wait_sec", 0.0))

        self.last_pitch_speed = None
        self.last_yaw_speed = None
        self.pitch.set_speed_mode()
        self.yaw.set_speed_mode()
        self.stop()

    def set_speed(self, pitch_rpm: float = 0.0, yaw_rpm: float = 0.0):
        pitch_max_rpm = getattr(self, "pitch_max_rpm", 20.0)
        yaw_max_rpm = getattr(self, "yaw_max_rpm", 25.0)
        pitch_rpm = self._clip(float(pitch_rpm), -pitch_max_rpm, pitch_max_rpm)
        yaw_rpm = self._clip(float(yaw_rpm), -yaw_max_rpm, yaw_max_rpm)

        if self.last_pitch_speed == pitch_rpm and self.last_yaw_speed == yaw_rpm:
            return

        self.pitch.set_speed(pitch_rpm)
        self.yaw.set_speed(yaw_rpm)
        self.last_pitch_speed = pitch_rpm
        self.last_yaw_speed = yaw_rpm
        logging.info("Brushless PTZ speed pitch=%.2f rpm yaw=%.2f rpm", pitch_rpm, yaw_rpm)

    def stop(self):
        self.set_speed(0.0, 0.0)

    def close(self):
        try:
            self.stop()
        finally:
            self.pitch.close()
            self.yaw.close()

    def _sleep(self, seconds: float):
        if seconds > 0:
            time.sleep(seconds)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def __del__(self):
        try:
            if hasattr(self, "pitch") and hasattr(self, "yaw"):
                self.close()
        except Exception:
            pass
