import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(
        FONT_HERSHEY_SIMPLEX=0,
        MARKER_CROSS=0,
        circle=lambda *args, **kwargs: None,
        destroyAllWindows=lambda: None,
        drawMarker=lambda *args, **kwargs: None,
        imencode=lambda *args, **kwargs: (False, None),
        imread=lambda path: np.zeros((20, 20, 3), dtype=np.uint8),
        imwrite=lambda *args, **kwargs: False,
        putText=lambda *args, **kwargs: None,
        rectangle=lambda *args, **kwargs: None,
        resize=lambda frame, size: np.zeros((size[1], size[0], frame.shape[2]), dtype=frame.dtype),
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
from services.tracking_controller import TrackingController
from services.target_classes import resolve_target_class


class TrackingTargetSwitchTests(unittest.TestCase):
    def test_resolves_sports_ball_to_coco_class_id(self):
        target = resolve_target_class(target="sports ball")

        self.assertEqual({"class_id": 32, "name": "sports ball"}, target)

    def test_tracking_controller_switches_live_yolo_filter_to_sports_ball(self):
        class FakeYolo:
            classes = ["person"]

            def __init__(self):
                self.target_class_id = 0

        class FakePtz:
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1

        controller = TrackingController.__new__(TrackingController)
        controller.target_class_id = 0
        controller.yolo = FakeYolo()
        controller.ptz = FakePtz()

        result = controller.set_target_class(32)

        self.assertEqual({"mode": "class", "class_id": 32, "name": "sports ball", "profile_id": None}, result)
        self.assertEqual(32, controller.target_class_id)
        self.assertEqual(32, controller.yolo.target_class_id)
        self.assertEqual(1, controller.ptz.stop_calls)
        self.assertEqual(
            {"mode": "class", "class_id": 32, "name": "sports ball", "profile_id": None},
            controller.target_status(),
        )

    def test_tracking_controller_switches_to_identity_target_without_changing_to_face_box_tracking(self):
        class FakeYolo:
            classes = ["person"]

            def __init__(self):
                self.target_class_id = 32

        class FakePtz:
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1

        controller = TrackingController.__new__(TrackingController)
        controller.target_class_id = 32
        controller.yolo = FakeYolo()
        controller.ptz = FakePtz()

        result = controller.set_identity_target("person_b")

        self.assertEqual(
            {"mode": "identity", "class_id": 0, "name": "person", "profile_id": "person_b"},
            result,
        )
        self.assertEqual("identity", controller.target_mode)
        self.assertEqual("person_b", controller.active_profile_id)
        self.assertEqual(0, controller.target_class_id)
        self.assertEqual(0, controller.yolo.target_class_id)
        self.assertEqual(1, controller.ptz.stop_calls)

    def test_identity_selection_does_not_run_face_recognition_in_main_loop(self):
        class FakeRecognizer:
            calls = 0

            def identify_person_boxes(self, frame, person_detections):
                self.calls += 1
                raise AssertionError("face recognition must not run in the tracking loop")

        controller = TrackingController.__new__(TrackingController)
        controller.target_mode = "identity"
        controller.active_profile_id = "person_b"
        controller.face_recognizer = FakeRecognizer()
        controller.identity_tracks = []
        controller.identity_next_track_id = 1
        controller.identity_scan_in_progress = True
        controller.identity_by_detection_id = {}
        detections = [
            [10, 10, 80, 180, 0.90, 0],
            [200, 20, 280, 190, 0.88, 0],
            [300, 20, 340, 80, 0.70, 32],
        ]

        selected = controller._select_target(detections, 640, 480, frame=object())

        self.assertIsNone(selected)
        self.assertEqual(0, controller.face_recognizer.calls)

    def test_background_identity_result_selects_current_yolo_detection(self):
        class FakeRecognizer:
            def __init__(self):
                self.calls = 0

            def identify_person_boxes(self, frame, person_detections):
                self.calls += 1
                return [
                    {
                        "detection": person_detections[0],
                        "profile_id": "person_b",
                        "score": 0.91,
                        "face_box": [20, 20, 40, 40],
                    }
                ]

        recognizer = FakeRecognizer()
        controller = TrackingController.__new__(TrackingController)
        controller.target_mode = "identity"
        controller.active_profile_id = "person_b"
        controller.face_recognizer = recognizer
        controller.identity_tracks = []
        controller.identity_next_track_id = 1
        controller.identity_track_ttl_sec = 2.0
        controller.identity_scan_in_progress = True
        controller.identity_by_detection_id = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first_detections = [[100, 80, 220, 360, 0.90, 0]]

        with mock.patch("services.tracking_controller.time.time", return_value=100.0):
            first_selected = controller._select_target(first_detections, 640, 480, frame=frame)

        self.assertIsNone(first_selected)
        self.assertEqual(1, len(controller.identity_tracks))

        track_id = controller.identity_tracks[0]["track_id"]
        controller._run_identity_scan(
            frame,
            [{"track_id": track_id, "detection": first_detections[0]}],
            "person_b",
        )

        next_detections = [[112, 86, 232, 366, 0.89, 0]]
        with mock.patch("services.tracking_controller.time.time", return_value=100.2):
            next_selected = controller._select_target(next_detections, 640, 480, frame=frame)

        self.assertIs(next_detections[0], next_selected)
        self.assertEqual(1, recognizer.calls)
        self.assertEqual([112.0, 86.0, 232.0, 366.0], controller.identity_tracks[0]["bbox"])
        self.assertEqual("person_b", controller.identity_by_detection_id[id(next_detections[0])]["profile_id"])

    def test_identity_selection_keeps_confirmed_track_when_background_scan_temporarily_misses(self):
        class FakeRecognizer:
            def __init__(self):
                self.calls = 0

            def identify_person_boxes(self, frame, person_detections):
                self.calls += 1
                if self.calls == 1:
                    profile_id = "person_a"
                    score = 0.92
                else:
                    profile_id = None
                    score = 0.0
                return [
                    {
                        "detection": person_detections[0],
                        "profile_id": profile_id,
                        "score": score,
                        "face_box": [20, 20, 40, 40] if profile_id else None,
                    }
                ]

        recognizer = FakeRecognizer()
        controller = TrackingController.__new__(TrackingController)
        controller.target_mode = "identity"
        controller.active_profile_id = "person_a"
        controller.face_recognizer = recognizer
        controller.identity_tracks = []
        controller.identity_next_track_id = 1
        controller.identity_track_ttl_sec = 2.0
        controller.identity_scan_in_progress = True
        controller.identity_by_detection_id = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first_detections = [[100, 80, 220, 360, 0.90, 0]]

        with mock.patch("services.tracking_controller.time.time", return_value=100.0):
            first_selected = controller._select_target(first_detections, 640, 480, frame=frame)

        self.assertIsNone(first_selected)
        track_id = controller.identity_tracks[0]["track_id"]
        scan_items = [{"track_id": track_id, "detection": first_detections[0]}]
        controller._run_identity_scan(frame, scan_items, "person_a")
        controller._run_identity_scan(frame, scan_items, "person_a")

        next_detections = [[108, 88, 228, 368, 0.88, 0]]
        with mock.patch("services.tracking_controller.time.time", return_value=100.4):
            next_selected = controller._select_target(next_detections, 640, 480, frame=frame)

        self.assertIs(next_detections[0], next_selected)
        self.assertEqual(2, recognizer.calls)
        self.assertEqual("person_a", controller.identity_by_detection_id[id(next_detections[0])]["profile_id"])

    def test_identity_tracking_speed_uses_person_body_anchor(self):
        controller = TrackingController.__new__(TrackingController)
        controller.target_mode = "identity"
        controller.person_anchor_y_ratio = 0.28
        controller.identity_desired_y_ratio = 0.38
        controller.dead_zone_px = 0
        controller.pitch_max_rpm = 20.0
        controller.yaw_max_rpm = 20.0
        target = [100, 100, 300, 500, 0.9, 0]

        _pitch_speed, _yaw_speed, error_x, error_y, target_x, target_y = controller._calculate_tracking_speed(
            target,
            640,
            480,
        )

        self.assertEqual(200, target_x)
        self.assertEqual(212.0, target_y)
        self.assertEqual(-120, error_x)
        self.assertAlmostEqual(29.6, error_y)

    def test_control_service_switches_target_by_name(self):
        class FakeTracking:
            def __init__(self, **kwargs):
                self.enabled = False
                self.target_class_id = kwargs["target_class_id"]
                self.set_calls = []

            def set_target_class(self, class_id):
                self.target_class_id = class_id
                self.set_calls.append(class_id)
                return {"mode": "class", "class_id": class_id, "name": "sports ball", "profile_id": None}

            def target_status(self):
                return {"mode": "class", "class_id": self.target_class_id, "name": "sports ball", "profile_id": None}

        with mock.patch.object(soloair_control_service, "TrackingController", FakeTracking), \
             mock.patch.object(soloair_control_service, "SessionManager", mock.Mock()), \
             mock.patch.object(soloair_control_service, "RtspFileRecorder", mock.Mock()):
            service = soloair_control_service.SoloAirControlService()

        result = service.set_tracking_target(target="sports ball")

        self.assertEqual({"mode": "class", "class_id": 32, "name": "sports ball", "profile_id": None}, result)
        self.assertEqual([32], service.tracking.set_calls)
        self.assertEqual(
            {"mode": "class", "class_id": 32, "name": "sports ball", "profile_id": None},
            service.status()["target"],
        )

    def test_control_service_switches_identity_target_from_profile_id(self):
        class FakeTracking:
            def __init__(self, **kwargs):
                self.enabled = False
                self.identity_calls = []

            def set_identity_target(self, profile_id):
                self.identity_calls.append(profile_id)
                return {"mode": "identity", "class_id": 0, "name": "person", "profile_id": profile_id}

            def target_status(self):
                return {"mode": "identity", "class_id": 0, "name": "person", "profile_id": "person_a"}

        with mock.patch.object(soloair_control_service, "TrackingController", FakeTracking), \
             mock.patch.object(soloair_control_service, "SessionManager", mock.Mock()), \
             mock.patch.object(soloair_control_service, "RtspFileRecorder", mock.Mock()):
            service = soloair_control_service.SoloAirControlService()

        result = service.set_tracking_target(target="person", profile_id="person_a")

        self.assertEqual({"mode": "identity", "class_id": 0, "name": "person", "profile_id": "person_a"}, result)
        self.assertEqual(["person_a"], service.tracking.identity_calls)

    def test_control_service_attaches_face_recognizer_to_tracking_controller(self):
        created_recognizers = []
        tracking_kwargs = []

        class FakeRecognizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created_recognizers.append(self)

            def available_profiles(self):
                return ["person_a", "person_b"]

        class FakeTracking:
            def __init__(self, **kwargs):
                self.enabled = False
                self.face_recognizer = kwargs.get("face_recognizer")
                tracking_kwargs.append(kwargs)

            def target_status(self):
                return {"mode": "class", "class_id": 0, "name": "person", "profile_id": None}

        with mock.patch.object(soloair_control_service, "FACE_IDENTITY_ENABLED", True), \
             mock.patch.object(soloair_control_service, "FACE_DB_PATH", "/tmp/face_db.npz"), \
             mock.patch.object(soloair_control_service, "FaceIdentityRecognizer", FakeRecognizer), \
             mock.patch.object(soloair_control_service, "TrackingController", FakeTracking), \
             mock.patch.object(soloair_control_service, "SessionManager", mock.Mock()), \
             mock.patch.object(soloair_control_service, "RtspFileRecorder", mock.Mock()):
            service = soloair_control_service.SoloAirControlService()

        self.assertEqual(1, len(created_recognizers))
        self.assertIs(created_recognizers[0], service.tracking.face_recognizer)
        self.assertIs(created_recognizers[0], tracking_kwargs[0]["face_recognizer"])


if __name__ == "__main__":
    unittest.main()
