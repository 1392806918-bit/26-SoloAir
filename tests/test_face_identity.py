import os
import sys
import tempfile
import types
import unittest
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
        imread=lambda path: np.zeros((20, 20, 3), dtype=np.uint8),
        imwrite=lambda *args, **kwargs: False,
        putText=lambda *args, **kwargs: None,
        rectangle=lambda *args, **kwargs: None,
        resize=lambda frame, size: np.zeros((size[1], size[0], frame.shape[2]), dtype=frame.dtype),
    )


class FakeFace:
    def __init__(self, embedding, bbox=(1, 2, 9, 10)):
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.bbox = np.asarray(bbox, dtype=np.float32)


class FaceIdentityTests(unittest.TestCase):
    def test_recognizer_matches_person_box_to_profile_from_face_db(self):
        from services.face_identity import FaceIdentityRecognizer

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "face_db.npz")
            np.savez(
                db_path,
                person_a=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                person_b=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            )

            class FakeApp:
                def get(self, image):
                    return [FakeFace([0.95, 0.05, 0.0], bbox=(3, 4, 13, 16))]

            recognizer = FaceIdentityRecognizer(db_path, threshold=0.7, app=FakeApp())
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            detection = [10, 20, 50, 90, 0.88, 0]

            identities = recognizer.identify_person_boxes(frame, [detection])

        self.assertEqual(1, len(identities))
        self.assertIs(detection, identities[0]["detection"])
        self.assertEqual("person_a", identities[0]["profile_id"])
        self.assertGreater(identities[0]["score"], 0.9)
        self.assertEqual([13, 24, 23, 36], identities[0]["face_box"])

    def test_recognizer_returns_unknown_when_best_score_is_below_threshold(self):
        from services.face_identity import FaceIdentityRecognizer

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "face_db.npz")
            np.savez(db_path, person_a=np.asarray([[1.0, 0.0]], dtype=np.float32))

            class FakeApp:
                def get(self, image):
                    return [FakeFace([0.0, 1.0], bbox=(0, 0, 4, 4))]

            recognizer = FaceIdentityRecognizer(db_path, threshold=0.5, app=FakeApp())
            frame = np.zeros((80, 80, 3), dtype=np.uint8)
            detection = [0, 0, 40, 80, 0.9, 0]

            identities = recognizer.identify_person_boxes(frame, [detection])

        self.assertEqual(1, len(identities))
        self.assertIsNone(identities[0]["profile_id"])
        self.assertLess(identities[0]["score"], 0.5)

    def test_recognizer_downsamples_large_person_crop_and_maps_face_box_to_frame(self):
        from services.face_identity import FaceIdentityRecognizer

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "face_db.npz")
            np.savez(db_path, person_a=np.asarray([[1.0, 0.0]], dtype=np.float32))

            class FakeApp:
                def __init__(self):
                    self.seen_shapes = []

                def get(self, image):
                    self.seen_shapes.append(image.shape)
                    return [FakeFace([1.0, 0.0], bbox=(10, 20, 60, 80))]

            app = FakeApp()
            recognizer = FaceIdentityRecognizer(
                db_path,
                threshold=0.5,
                crop_top_ratio=1.0,
                max_crop_size=200,
                app=app,
            )
            frame = np.zeros((500, 500, 3), dtype=np.uint8)
            detection = [50, 40, 450, 440, 0.9, 0]

            identities = recognizer.identify_person_boxes(frame, [detection])

        self.assertEqual([(200, 200, 3)], app.seen_shapes)
        self.assertEqual("person_a", identities[0]["profile_id"])
        self.assertEqual([70, 80, 170, 200], identities[0]["face_box"])

    def test_enroll_faces_writes_profile_embeddings_to_npz(self):
        from scripts import enroll_faces

        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "faces")
            os.makedirs(os.path.join(input_dir, "person_a"))
            os.makedirs(os.path.join(input_dir, "person_b"))
            for rel_path in (
                "person_a/a1.jpg",
                "person_a/a2.jpg",
                "person_b/b1.jpg",
            ):
                open(os.path.join(input_dir, rel_path), "wb").close()

            output_path = os.path.join(tmp, "face_db.npz")

            class FakeApp:
                def get(self, image):
                    value = float(image[0, 0, 0])
                    return [FakeFace([value, 1.0], bbox=(0, 0, 5, 5))]

            def fake_imread(path):
                image = np.zeros((10, 10, 3), dtype=np.uint8)
                image[0, 0, 0] = 2 if "person_a" in path else 4
                return image

            with mock.patch.object(enroll_faces.cv2, "imread", side_effect=fake_imread):
                summary = enroll_faces.enroll_faces(input_dir, output_path, app=FakeApp())

            with np.load(output_path) as db:
                person_a = db["person_a"].copy()
                person_b = db["person_b"].copy()

        self.assertEqual({"person_a": 2, "person_b": 1}, summary)
        self.assertEqual((2, 2), person_a.shape)
        self.assertEqual((1, 2), person_b.shape)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(person_a[0])), places=6)


if __name__ == "__main__":
    unittest.main()
