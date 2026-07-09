import os

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def normalize_embedding_matrix(embeddings):
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"embedding array must be 1D or 2D, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def load_face_database(db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"face database not found: {db_path}")

    data = np.load(db_path, allow_pickle=False)
    database = {}
    for profile_id in data.files:
        database[profile_id] = normalize_embedding_matrix(data[profile_id])

    if not database:
        raise ValueError(f"face database is empty: {db_path}")
    return database


def build_face_app(model_name="buffalo_s", providers=None, det_size=(320, 320)):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError("install insightface and onnxruntime to enable face identity recognition") from exc

    providers = providers or ["CPUExecutionProvider"]
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0, det_size=tuple(det_size))
    return app


class FaceIdentityRecognizer:
    def __init__(
        self,
        db_path,
        threshold=0.5,
        model_name="buffalo_s",
        providers=None,
        det_size=(320, 320),
        crop_top_ratio=0.65,
        max_crop_size=224,
        app=None,
    ):
        self.db_path = db_path
        self.threshold = float(threshold)
        self.crop_top_ratio = float(crop_top_ratio)
        self.max_crop_size = int(max_crop_size)
        self.database = load_face_database(db_path)
        self.app = app or build_face_app(model_name=model_name, providers=providers, det_size=det_size)

    def available_profiles(self):
        return sorted(self.database.keys())

    def match_embedding(self, embedding):
        embedding = normalize_embedding(embedding)
        best_profile_id = None
        best_score = -1.0

        for profile_id, profile_embeddings in self.database.items():
            scores = profile_embeddings @ embedding
            score = float(np.max(scores))
            if score > best_score:
                best_score = score
                best_profile_id = profile_id

        if best_score < self.threshold:
            return None, best_score
        return best_profile_id, best_score

    def identify_person_boxes(self, frame, person_detections):
        if frame is None:
            return []

        identities = []
        for detection in person_detections:
            crop_info = self._crop_person_face_region(frame, detection)
            if crop_info is None:
                identities.append(self._unknown_identity(detection))
                continue

            crop, offset_x, offset_y = crop_info
            recognition_crop, scale_x, scale_y = self._resize_crop_for_recognition(crop)
            faces = self.app.get(recognition_crop)
            if not faces:
                identities.append(self._unknown_identity(detection))
                continue

            face = max(faces, key=self._face_area)
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = face.embedding
            profile_id, score = self.match_embedding(embedding)
            identities.append(
                {
                    "detection": detection,
                    "profile_id": profile_id,
                    "score": score,
                    "face_box": self._offset_face_box(face.bbox, offset_x, offset_y, scale_x, scale_y),
                }
            )

        return identities

    def _crop_person_face_region(self, frame, detection):
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = detection[:4]
        x1 = max(0, min(int(x1), frame_w - 1))
        x2 = max(0, min(int(x2), frame_w))
        y1 = max(0, min(int(y1), frame_h - 1))
        y2 = max(0, min(int(y2), frame_h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop_y2 = y1 + int((y2 - y1) * self.crop_top_ratio)
        crop_y2 = max(y1 + 1, min(crop_y2, frame_h))
        crop = frame[y1:crop_y2, x1:x2]
        if crop.size == 0:
            return None
        return crop, x1, y1

    def _resize_crop_for_recognition(self, crop):
        max_crop_size = int(getattr(self, "max_crop_size", 0) or 0)
        h, w = crop.shape[:2]
        if max_crop_size <= 0 or max(h, w) <= max_crop_size:
            return crop, 1.0, 1.0

        scale = max_crop_size / float(max(h, w))
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (resized_w, resized_h))
        return resized, w / float(resized_w), h / float(resized_h)

    @staticmethod
    def _face_area(face):
        x1, y1, x2, y2 = face.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    @staticmethod
    def _offset_face_box(face_box, offset_x, offset_y, scale_x=1.0, scale_y=1.0):
        x1, y1, x2, y2 = face_box
        return [
            int(round(float(x1) * scale_x + offset_x)),
            int(round(float(y1) * scale_y + offset_y)),
            int(round(float(x2) * scale_x + offset_x)),
            int(round(float(y2) * scale_y + offset_y)),
        ]

    @staticmethod
    def _unknown_identity(detection):
        return {
            "detection": detection,
            "profile_id": None,
            "score": 0.0,
            "face_box": None,
        }
