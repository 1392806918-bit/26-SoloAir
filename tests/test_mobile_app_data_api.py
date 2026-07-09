import json
import os
import sys
import tempfile
import time
import types
import unittest
from io import BytesIO
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "cv2" not in sys.modules:
    dnn = types.SimpleNamespace(
        NMSBoxes=lambda boxes, confidences, conf_threshold, nms_threshold: list(range(len(boxes)))
    )
    sys.modules["cv2"] = types.SimpleNamespace(
        FONT_HERSHEY_SIMPLEX=0,
        MARKER_CROSS=0,
        circle=lambda *args, **kwargs: None,
        dnn=dnn,
        destroyAllWindows=lambda: None,
        drawMarker=lambda *args, **kwargs: None,
        imencode=lambda *args, **kwargs: (False, None),
        imwrite=lambda *args, **kwargs: False,
        putText=lambda *args, **kwargs: None,
        rectangle=lambda *args, **kwargs: None,
        resize=lambda frame, size: frame,
    )

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

from scripts import soloair_control_service
from services.mobile_control_api import MobileControlApi
from services.tracking_controller import TrackingController


class MobileAppDataApiTests(unittest.TestCase):
    def test_tracking_controller_exposes_latest_detections_for_app_overlay(self):
        class FakeYolo:
            classes = ["person"]

            def detect(self, frame):
                return [
                    [300, 220, 340, 260, 0.80, 0],
                    [500, 300, 620, 460, 0.99, 0],
                ]

        class FakePtz:
            def set_speed(self, pitch_speed=0, yaw_speed=0):
                pass

        controller = TrackingController.__new__(TrackingController)
        controller.enabled = True
        controller.yolo = FakeYolo()
        controller.ptz = FakePtz()
        controller.dead_zone_px = 25
        controller.pitch_max_rpm = 20.0
        controller.yaw_max_rpm = 25.0
        controller.no_target_timeout = 3.0
        controller.t_last_target = time.time()
        controller.has_returned_home_after_lost = False
        controller.target_class_id = 0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        controller.process_frame(frame)
        payload = controller.latest_detections()

        self.assertTrue(payload["enabled"])
        self.assertEqual({"width": 640, "height": 480}, payload["frame"])
        self.assertEqual(2, len(payload["detections"]))
        self.assertEqual(
            {
                "x1": 300,
                "y1": 220,
                "x2": 340,
                "y2": 260,
                "confidence": 0.8,
                "class_id": 0,
                "class_name": "person",
                "selected": True,
            },
            payload["detections"][0],
        )
        self.assertFalse(payload["detections"][1]["selected"])
        self.assertGreater(payload["updated_at"], 0)

    def test_control_service_reads_voice_logs_without_requiring_files_to_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            text_path = os.path.join(tmp, "text.txt")
            command_path = os.path.join(tmp, "command.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write("无人机起飞\n")

            service = soloair_control_service.SoloAirControlService.__new__(
                soloair_control_service.SoloAirControlService
            )
            service.voice_text_log_file = text_path
            service.voice_command_log_file = command_path

            payload = service.voice_logs()

        self.assertEqual("text.txt", payload["text"]["name"])
        self.assertEqual("无人机起飞\n", payload["text"]["content"])
        self.assertFalse(payload["text"]["missing"])
        self.assertEqual("command.txt", payload["commands"]["name"])
        self.assertEqual("", payload["commands"]["content"])
        self.assertTrue(payload["commands"]["missing"])

    def test_http_api_routes_tracking_and_voice_payloads(self):
        class FakeController:
            def latest_detections(self):
                return {"enabled": True, "frame": {"width": 640, "height": 480}, "detections": []}

            def voice_logs(self):
                return {
                    "text": {"name": "text.txt", "content": "hello", "missing": False},
                    "commands": {"name": "command.txt", "content": "cmd", "missing": False},
                }

            def voice_log_text(self, kind):
                return "hello" if kind == "text" else "cmd"

        class FakeSessionManager:
            def list_sessions(self):
                return []

        api = MobileControlApi("127.0.0.1", 8080, FakeController(), FakeSessionManager())

        detections = self._handle_get_json(api, "/api/tracking/detections")
        voice_logs = self._handle_get_json(api, "/api/voice/logs")
        command_text = self._handle_get_text(api, "/api/voice/command.txt")

        self.assertEqual({"width": 640, "height": 480}, detections["frame"])
        self.assertEqual("hello", voice_logs["text"]["content"])
        self.assertEqual("cmd", command_text)

    def test_http_tracking_target_forwards_profile_id_for_app_identity_switch(self):
        class FakeController:
            def __init__(self):
                self.calls = []

            def set_tracking_target(self, target=None, class_id=None, profile_id=None):
                self.calls.append({"target": target, "class_id": class_id, "profile_id": profile_id})
                return {"mode": "identity", "class_id": 0, "name": "person", "profile_id": profile_id}

        class FakeSessionManager:
            def list_sessions(self):
                return []

        controller = FakeController()
        api = MobileControlApi("127.0.0.1", 8080, controller, FakeSessionManager())
        handler = self._fake_handler("/api/tracking/target")
        body = json.dumps({"target": "person", "profile_id": "person_b"}).encode("utf-8")
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)

        api._handle_post(handler)
        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))

        self.assertEqual([{"target": "person", "class_id": None, "profile_id": "person_b"}], controller.calls)
        self.assertEqual(
            {"ok": True, "result": {"mode": "identity", "class_id": 0, "name": "person", "profile_id": "person_b"}},
            payload,
        )

    def _handle_get_json(self, api, path):
        handler = self._fake_handler(path)
        api._handle_get(handler)
        return json.loads(handler.wfile.getvalue().decode("utf-8"))

    def _handle_get_text(self, api, path):
        handler = self._fake_handler(path)
        api._handle_get(handler)
        return handler.wfile.getvalue().decode("utf-8")

    def _fake_handler(self, path):
        handler = mock.Mock()
        handler.path = path
        handler.wfile = BytesIO()
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        return handler


if __name__ == "__main__":
    unittest.main()
