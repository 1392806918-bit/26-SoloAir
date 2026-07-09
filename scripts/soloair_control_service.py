#!/usr/bin/env python3
import logging
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2

from config import (
    BAUDRATE,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CLASS_PATH,
    CONTROL_VIDEO_SOURCE,
    CONF_THRES,
    DEAD_ZONE_PX,
    DISPLAY_WINDOW_NAME,
    ENABLE_LOCAL_DISPLAY,
    BALL_DESIRED_Y_RATIO,
    FACE_DB_PATH,
    FACE_DET_SIZE,
    FACE_IDENTITY_ENABLED,
    FACE_MAX_CROP_SIZE,
    FACE_MATCH_THRESHOLD,
    FACE_MODEL_NAME,
    FACE_PERSON_CROP_TOP_RATIO,
    FACE_PROVIDERS,
    FFMPEG_BIN,
    HTTP_HOST,
    HTTP_PORT,
    IDENTITY_CACHE_TTL_SEC,
    IDENTITY_DESIRED_Y_RATIO,
    IDENTITY_RECOGNITION_INTERVAL_SEC,
    IOU_THRES,
    LOST_RETURN_HOME_SEC,
    HOME_PITCH_DEG,
    HOME_YAW_DEG,
    MIN_TRACKING_SPEED_RPM,
    MODEL_PATH,
    PERSON_CLASS_ID,
    PERSON_ANCHOR_Y_RATIO,
    PITCH_ADDR,
    PITCH_MAX_RPM,
    PITCH_PORT,
    RTSP_LOCAL_URL,
    RTSP_NETWORK_URL,
    SESSION_ROOT,
    TRACKING_SPEED_SMOOTHING_ALPHA,
    YAW_ADDR,
    YAW_MAX_RPM,
    YAW_PORT,
)
from services.face_identity import FaceIdentityRecognizer
from services.mobile_control_api import MobileControlApi
from services.rtsp_file_recorder import RtspFileRecorder
from services.session_manager import SessionManager
from services.target_classes import resolve_target_class
from services.tracking_controller import TrackingController
from utils.Yolov5lite_infer import CameraStream


class SoloAirControlService:
    def __init__(self):
        self.session_manager = SessionManager(SESSION_ROOT)
        self.recorder = RtspFileRecorder(FFMPEG_BIN, RTSP_LOCAL_URL)
        self.voice_text_log_file = "/home/elf/model/text.txt"
        self.voice_command_log_file = "/home/elf/model/command.txt"
        self.face_recognizer = self._build_face_recognizer()
        self.tracking = TrackingController(
            pitch_port=PITCH_PORT,
            yaw_port=YAW_PORT,
            pitch_addr=PITCH_ADDR,
            yaw_addr=YAW_ADDR,
            baudrate=BAUDRATE,
            model_path=MODEL_PATH,
            class_path=CLASS_PATH,
            pitch_max_rpm=PITCH_MAX_RPM,
            yaw_max_rpm=YAW_MAX_RPM,
            dead_zone_px=DEAD_ZONE_PX,
            no_target_timeout=LOST_RETURN_HOME_SEC,
            home_pitch_deg=HOME_PITCH_DEG,
            home_yaw_deg=HOME_YAW_DEG,
            conf_threshold=CONF_THRES,
            nms_threshold=IOU_THRES,
            target_class_id=PERSON_CLASS_ID,
            face_recognizer=self.face_recognizer,
            person_anchor_y_ratio=PERSON_ANCHOR_Y_RATIO,
            identity_desired_y_ratio=IDENTITY_DESIRED_Y_RATIO,
            ball_desired_y_ratio=BALL_DESIRED_Y_RATIO,
            identity_recognition_interval_sec=IDENTITY_RECOGNITION_INTERVAL_SEC,
            identity_cache_ttl_sec=IDENTITY_CACHE_TTL_SEC,
            speed_smoothing_alpha=TRACKING_SPEED_SMOOTHING_ALPHA,
            min_tracking_speed_rpm=MIN_TRACKING_SPEED_RPM,
        )
        self.http_api = MobileControlApi(HTTP_HOST, HTTP_PORT, self, self.session_manager)
        self.camera = None
        self.last_raw_frame = None
        self.last_frame_shape = None
        self.last_frame_at = None
        self.running = False
        self._logged_first_frame = False

    def start(self):
        self.tracking.home_lock()
        logging.info("Starting control video source=%s", CONTROL_VIDEO_SOURCE)
        self.camera = CameraStream(
            src=CONTROL_VIDEO_SOURCE,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        )
        self.http_api.start()
        self.running = True
        logging.info("External RTSP URL: %s", RTSP_NETWORK_URL)
        logging.info("HTTP API: http://<elf2-ip>:%s/api/status", HTTP_PORT)
        self._main_loop()

    def stop(self):
        self.running = False
        try:
            self.stop_recording()
        except Exception:
            logging.exception("failed to stop recording")
        self.tracking.close()
        self.http_api.stop()
        if self.camera:
            self.camera.stop()
            self.camera = None
        cv2.destroyAllWindows()

    def start_recording(self):
        session_id = self.session_manager.start_session()
        if not self.recorder.is_recording:
            self.recorder.start(self.session_manager.video_path(), self.session_manager.temp_video_path())
            self.session_manager.write_event("record_start", {"rtsp_url": RTSP_LOCAL_URL})
        self.tracking.enable()
        return {"session_id": session_id, "recording": True, "tracking": True}

    def stop_recording(self):
        saved_path = None
        if self.recorder.is_recording:
            saved_path = self.recorder.stop()
            self.session_manager.write_event("record_stop", {"video_path": saved_path})
        self.tracking.disable()
        session_id = self.session_manager.current_session_id
        self.session_manager.finish_session()
        return {"session_id": session_id, "recording": False, "tracking": False, "video_path": saved_path}

    def save_snapshot(self, label="manual"):
        if self.last_raw_frame is None:
            raise RuntimeError("no video frame available")
        session_id = self.session_manager.start_session()
        path = self.session_manager.snapshot_path(label)
        ok = cv2.imwrite(path, self.last_raw_frame)
        if not ok:
            raise RuntimeError(f"failed to save snapshot: {path}")
        self.session_manager.write_event("snapshot", {"path": path, "label": label})
        return {"session_id": session_id, "path": path}

    def set_tracking_target(self, target=None, class_id=None, profile_id=None):
        if profile_id:
            if getattr(self.tracking, "face_recognizer", object()) is None:
                raise RuntimeError("face identity recognizer is not available; run scripts/enroll_faces.py first")
            return self.tracking.set_identity_target(profile_id)
        resolved = resolve_target_class(target=target, class_id=class_id)
        return self.tracking.set_target_class(resolved["class_id"])

    def latest_detections(self):
        return self.tracking.latest_detections()

    def voice_logs(self):
        return {
            "text": self._voice_log_entry("text.txt", self.voice_text_log_file),
            "commands": self._voice_log_entry("command.txt", self.voice_command_log_file),
        }

    def voice_log_text(self, kind):
        if kind == "text":
            return self._read_text_file(self.voice_text_log_file)
        if kind in ("command", "commands"):
            return self._read_text_file(self.voice_command_log_file)
        raise ValueError(f"unknown voice log kind: {kind}")

    def status(self):
        return {
            "camera": self.camera is not None,
            "video_source": CONTROL_VIDEO_SOURCE,
            "rtsp_url": RTSP_NETWORK_URL,
            "rtsp_local_url": RTSP_LOCAL_URL,
            "http_port": HTTP_PORT,
            "recording": self.recorder.is_recording,
            "tracking": self.tracking.enabled,
            "target": self.tracking.target_status(),
            "face_identity": self._face_identity_status(),
            "session_id": self.session_manager.current_session_id,
            "last_frame_at": self.last_frame_at,
            "last_frame_shape": self.last_frame_shape,
            "rtsp": {"mode": "external-system-gstreamer", "local_url": RTSP_LOCAL_URL},
        }

    def _build_face_recognizer(self):
        if not FACE_IDENTITY_ENABLED:
            return None
        try:
            recognizer = FaceIdentityRecognizer(
                db_path=FACE_DB_PATH,
                threshold=FACE_MATCH_THRESHOLD,
                model_name=FACE_MODEL_NAME,
                providers=FACE_PROVIDERS,
                det_size=FACE_DET_SIZE,
                crop_top_ratio=FACE_PERSON_CROP_TOP_RATIO,
                max_crop_size=FACE_MAX_CROP_SIZE,
            )
            logging.info("Face identity recognizer loaded: %s", FACE_DB_PATH)
            return recognizer
        except FileNotFoundError:
            logging.warning("Face database not found: %s; run scripts/enroll_faces.py first", FACE_DB_PATH)
            return None
        except ImportError as exc:
            logging.warning("Face identity dependencies are not installed: %s", exc)
            return None
        except Exception:
            logging.exception("failed to initialize face identity recognizer")
            return None

    def _face_identity_status(self):
        profiles = []
        if self.face_recognizer is not None:
            profiles = self.face_recognizer.available_profiles()
        return {
            "enabled": bool(FACE_IDENTITY_ENABLED),
            "available": self.face_recognizer is not None,
            "db_path": FACE_DB_PATH,
            "profiles": profiles,
        }

    def _voice_log_entry(self, name, path):
        missing = not os.path.exists(path)
        updated_at = os.path.getmtime(path) if not missing else None
        size = os.path.getsize(path) if not missing else 0
        return {
            "name": name,
            "content": self._read_text_file(path),
            "missing": missing,
            "size": size,
            "updated_at": updated_at,
        }

    def _read_text_file(self, path):
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def latest_frame_jpeg(self):
        if self.last_raw_frame is None:
            raise RuntimeError("no video frame available")
        ok, encoded = cv2.imencode(".jpg", self.last_raw_frame)
        if not ok:
            raise RuntimeError("failed to encode latest frame")
        return encoded.tobytes()

    def _main_loop(self):
        if ENABLE_LOCAL_DISPLAY:
            cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DISPLAY_WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

        while self.running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            if not self._logged_first_frame:
                logging.info("First control frame shape: %s", getattr(frame, "shape", None))
                self._logged_first_frame = True

            raw_frame = self._normalize_frame(frame)
            self.last_raw_frame = raw_frame
            self.last_frame_shape = list(raw_frame.shape)
            self.last_frame_at = time.time()

            display_frame, _ = self.tracking.process_frame(raw_frame)

            if ENABLE_LOCAL_DISPLAY:
                cv2.imshow(DISPLAY_WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logging.info("Quit requested from local display")
                    break

    def _normalize_frame(self, frame):
        if frame.shape[0] != CAMERA_HEIGHT or frame.shape[1] != CAMERA_WIDTH:
            logging.warning(
                "Control frame shape %s does not match configured %sx%s; resizing",
                frame.shape,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            )
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        if not frame.flags["C_CONTIGUOUS"]:
            frame = frame.copy()
        return frame


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [control] %(message)s")
    service = SoloAirControlService()
    try:
        service.start()
    except KeyboardInterrupt:
        logging.info("Interrupted")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
