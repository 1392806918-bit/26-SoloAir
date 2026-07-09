import os
import sys
import time
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.tracking_controller import TrackingController


class TrackingControllerBrushlessTests(unittest.TestCase):
    def test_process_frame_tracks_nearest_center_target_with_dead_zone_speed(self):
        class FakeYolo:
            classes = ["person"]

            def detect(self, frame):
                return [
                    [310, 230, 330, 250, 0.80, 0],
                    [500, 300, 620, 460, 0.99, 0],
                ]

        class FakePtz:
            def __init__(self):
                self.speed_calls = []

            def set_speed(self, pitch_speed=0, yaw_speed=0):
                self.speed_calls.append((pitch_speed, yaw_speed))

        ptz = FakePtz()
        controller = TrackingController.__new__(TrackingController)
        controller.enabled = True
        controller.yolo = FakeYolo()
        controller.ptz = ptz
        controller.dead_zone_px = 25
        controller.pitch_max_rpm = 20.0
        controller.yaw_max_rpm = 25.0
        controller.no_target_timeout = 3.0
        controller.t_last_target = time.time()
        controller.has_returned_home_after_lost = False

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, detections = controller.process_frame(frame)

        self.assertEqual(2, len(detections))
        self.assertEqual([(0.0, 0.0)], ptz.speed_calls)

    def test_tracking_speed_zeroes_tiny_commands(self):
        controller = TrackingController.__new__(TrackingController)
        controller.target_mode = "class"
        controller.dead_zone_px = 0
        controller.pitch_max_rpm = 20.0
        controller.yaw_max_rpm = 20.0
        controller.min_tracking_speed_rpm = 0.5
        controller.speed_smoothing_alpha = 1.0
        controller.last_pitch_speed_cmd = None
        controller.last_yaw_speed_cmd = None
        target = [320, 238, 324, 242, 0.9, 0]

        pitch_speed, yaw_speed, error_x, error_y, _target_x, _target_y = controller._calculate_tracking_speed(
            target,
            640,
            480,
        )

        self.assertEqual(2, error_x)
        self.assertEqual(0, error_y)
        self.assertEqual(0.0, pitch_speed)
        self.assertEqual(0.0, yaw_speed)

    def test_tracking_speed_smooths_same_direction_commands(self):
        controller = TrackingController.__new__(TrackingController)
        controller.target_mode = "class"
        controller.dead_zone_px = 0
        controller.pitch_max_rpm = 20.0
        controller.yaw_max_rpm = 20.0
        controller.min_tracking_speed_rpm = 0.0
        controller.speed_smoothing_alpha = 0.25
        controller.last_pitch_speed_cmd = 0.0
        controller.last_yaw_speed_cmd = 10.0
        target = [620, 238, 660, 242, 0.9, 0]

        pitch_speed, yaw_speed, _error_x, _error_y, _target_x, _target_y = controller._calculate_tracking_speed(
            target,
            640,
            480,
        )

        self.assertEqual(0.0, pitch_speed)
        self.assertAlmostEqual(12.5, yaw_speed)


if __name__ == "__main__":
    unittest.main()
