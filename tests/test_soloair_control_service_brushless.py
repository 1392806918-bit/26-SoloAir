import os
import sys
import types
import unittest
from unittest import mock

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
        imwrite=lambda *args, **kwargs: False,
        putText=lambda *args, **kwargs: None,
        rectangle=lambda *args, **kwargs: None,
        resize=lambda frame, size: frame,
    )

from scripts import soloair_control_service


class SoloAirBrushlessStartupTests(unittest.TestCase):
    def test_start_homes_gimbal_without_enabling_tracking(self):
        events = []

        class FakeTracking:
            def __init__(self, **kwargs):
                self.enabled = False
                self.kwargs = kwargs

            def home_lock(self):
                events.append("home_lock")

            def enable(self):
                events.append("enable")
                self.enabled = True

            def close(self):
                events.append("tracking_close")

        class FakeCamera:
            def __init__(self, **kwargs):
                events.append("camera")

            def stop(self):
                events.append("camera_stop")

        class FakeHttpApi:
            def __init__(self, *args):
                pass

            def start(self):
                events.append("http_start")

            def stop(self):
                events.append("http_stop")

        with mock.patch.object(soloair_control_service, "TrackingController", FakeTracking), \
             mock.patch.object(soloair_control_service, "CameraStream", FakeCamera), \
             mock.patch.object(soloair_control_service, "MobileControlApi", FakeHttpApi), \
             mock.patch.object(soloair_control_service, "SessionManager", mock.Mock()), \
             mock.patch.object(soloair_control_service, "RtspFileRecorder", mock.Mock()):
            service = soloair_control_service.SoloAirControlService()
            service._main_loop = lambda: events.append("main_loop")
            service.start()

        self.assertEqual(["home_lock", "camera", "http_start", "main_loop"], events)
        self.assertFalse(service.tracking.enabled)


if __name__ == "__main__":
    unittest.main()
