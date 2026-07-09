import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config


class BrushlessConfigTests(unittest.TestCase):
    def test_uses_latest_brushless_camera_serial_and_model_settings(self):
        self.assertEqual("/dev/ttyCH340_1", config.PITCH_PORT)
        self.assertEqual("/dev/ttyCH340_2", config.YAW_PORT)
        self.assertEqual(1, config.PITCH_ADDR)
        self.assertEqual(1, config.YAW_ADDR)
        self.assertEqual(115200, config.BAUDRATE)
        self.assertEqual(
            "/dev/v4l/by-id/usb-LRCP_Technology_Co.__Ltd._LRCP_1080P-60fps_SN0001-video-index0",
            config.CAMERA_DEVICE,
        )
        self.assertEqual(21, config.CAMERA_INDEX)
        self.assertEqual(60, config.CAMERA_FPS)
        self.assertEqual("/home/elf/VsCode/无刷云台3/models/yolov5nu.rknn", config.MODEL_PATH)
        self.assertEqual("/home/elf/VsCode/无刷云台3/models/person.names", config.CLASS_PATH)
        self.assertEqual("/home/elf/VsCode/无刷云台3/models/face_db.npz", config.FACE_DB_PATH)
        self.assertEqual(0.50, config.PERSON_ANCHOR_Y_RATIO)
        self.assertEqual(0.50, config.IDENTITY_DESIRED_Y_RATIO)
        self.assertEqual(0.40, config.IDENTITY_RECOGNITION_INTERVAL_SEC)
        self.assertEqual(1.20, config.IDENTITY_CACHE_TTL_SEC)
        self.assertEqual(224, config.FACE_MAX_CROP_SIZE)
        self.assertEqual(0.30, config.TRACKING_SPEED_SMOOTHING_ALPHA)
        self.assertEqual(0.50, config.MIN_TRACKING_SPEED_RPM)


if __name__ == "__main__":
    unittest.main()
