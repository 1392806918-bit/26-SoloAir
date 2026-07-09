import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.Yolov5lite_infer import YOLOv5Lite


class YOLOv5LitePostprocessTests(unittest.TestCase):
    def make_detector(self):
        detector = YOLOv5Lite.__new__(YOLOv5Lite)
        detector.classes = ["person"]
        detector.confThreshold = 0.5
        detector.nmsThreshold = 0.5
        detector.target_class_id = 0
        return detector

    def test_batched_nms_output_filters_to_person_class(self):
        detector = self.make_detector()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        outs = np.array(
            [
                [
                    [50, 70, 150, 220, 0.90, 0],
                    [10, 20, 120, 180, 0.95, 1],
                    [30, 50, 60, 90, 0.40, 0],
                ]
            ],
            dtype=np.float32,
        )

        detections = detector.postprocess(frame, outs, (240, 320, 40, 0))

        self.assertEqual(1, len(detections))
        self.assertEqual([100, 60, 300, 360], detections[0][:4])
        self.assertAlmostEqual(0.90, detections[0][4], places=6)
        self.assertEqual(0, detections[0][5])

    def test_raw_yolo_output_decodes_person_class(self):
        detector = self.make_detector()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        outs = np.zeros((1, 84, 2100), dtype=np.float32)
        outs[0, 0:4, 0] = [100, 145, 100, 150]
        outs[0, 4, 0] = 0.90
        outs[0, 0:4, 1] = [65, 100, 110, 160]
        outs[0, 4, 1] = 0.10
        outs[0, 5, 1] = 0.95
        outs[0, 0:4, 2] = [45, 70, 30, 40]
        outs[0, 4, 2] = 0.40

        detections = detector.postprocess(frame, outs, (240, 320, 40, 0))

        self.assertEqual(1, len(detections))
        self.assertEqual([100, 60, 300, 360], detections[0][:4])
        self.assertAlmostEqual(0.90, detections[0][4], places=6)
        self.assertEqual(0, detections[0][5])


if __name__ == "__main__":
    unittest.main()
