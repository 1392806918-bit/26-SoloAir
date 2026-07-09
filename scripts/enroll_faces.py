#!/usr/bin/env python3
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from config import (
    FACE_DB_PATH,
    FACE_DET_SIZE,
    FACE_ENROLL_DIR,
    FACE_MODEL_NAME,
    FACE_PROVIDERS,
)
from services.face_identity import SUPPORTED_IMAGE_EXTENSIONS, build_face_app, normalize_embedding


def enroll_faces(input_dir, output_path, app=None):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"face input directory not found: {input_dir}")

    app = app or build_face_app(
        model_name=FACE_MODEL_NAME,
        providers=FACE_PROVIDERS,
        det_size=FACE_DET_SIZE,
    )
    database = {}
    summary = {}

    for profile_id in sorted(os.listdir(input_dir)):
        profile_dir = os.path.join(input_dir, profile_id)
        if not os.path.isdir(profile_dir):
            continue

        embeddings = []
        for image_path in _iter_image_files(profile_dir):
            image = cv2.imread(image_path)
            if image is None:
                print(f"skip unreadable image: {image_path}")
                continue

            faces = app.get(image)
            if not faces:
                print(f"skip image without detected face: {image_path}")
                continue

            face = max(faces, key=_face_area)
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = face.embedding
            embeddings.append(normalize_embedding(embedding))
            print(f"enrolled {profile_id}: {image_path}")

        if embeddings:
            database[profile_id] = np.stack(embeddings).astype(np.float32)
            summary[profile_id] = len(embeddings)

    if not database:
        raise RuntimeError(f"no valid faces found under {input_dir}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path, **database)
    print(f"saved face database: {output_path}")
    for profile_id, count in summary.items():
        print(f"  {profile_id}: {count} face embeddings")
    return summary


def _iter_image_files(profile_dir):
    for name in sorted(os.listdir(profile_dir)):
        if name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            yield os.path.join(profile_dir, name)


def _face_area(face):
    x1, y1, x2, y2 = face.bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def parse_args():
    parser = argparse.ArgumentParser(description="Enroll face images into SoloAir face_db.npz.")
    parser.add_argument("--input", default=FACE_ENROLL_DIR, help="Directory containing one subdirectory per profile.")
    parser.add_argument("--output", default=FACE_DB_PATH, help="Output .npz face database path.")
    return parser.parse_args()


def main():
    args = parse_args()
    enroll_faces(args.input, args.output)


if __name__ == "__main__":
    main()
