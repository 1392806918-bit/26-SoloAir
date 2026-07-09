import logging
import threading
import time

import cv2

from services.target_classes import class_name_for_id
from utils.BrushlessPtz import BrushlessGimbal
from utils.Yolov5lite_infer import YOLOv5Lite


class TrackingController:
    def __init__(
        self,
        pitch_port,
        yaw_port,
        model_path,
        class_path,
        pitch_addr=1,
        yaw_addr=1,
        baudrate=115200,
        pitch_max_rpm=20.0,
        yaw_max_rpm=25.0,
        dead_zone_px=25,
        no_target_timeout=3.0,
        home_pitch_deg=0.0,
        home_yaw_deg=0.0,
        conf_threshold=0.35,
        nms_threshold=0.45,
        target_class_id=0,
        face_recognizer=None,
        person_anchor_y_ratio=0.28,
        identity_desired_y_ratio=0.38,
        ball_desired_y_ratio=0.50,
        identity_recognition_interval_sec=0.4,
        identity_cache_ttl_sec=1.2,
        speed_smoothing_alpha=1.0,
        min_tracking_speed_rpm=0.0,
    ):
        self.pitch_port = pitch_port
        self.yaw_port = yaw_port
        self.pitch_addr = pitch_addr
        self.yaw_addr = yaw_addr
        self.baudrate = baudrate
        self.model_path = model_path
        self.class_path = class_path
        self.pitch_max_rpm = float(pitch_max_rpm)
        self.yaw_max_rpm = float(yaw_max_rpm)
        self.dead_zone_px = int(dead_zone_px)
        self.no_target_timeout = float(no_target_timeout)
        self.home_pitch_deg = float(home_pitch_deg)
        self.home_yaw_deg = float(home_yaw_deg)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.target_class_id = target_class_id
        self.target_mode = "class"
        self.active_profile_id = None
        self.face_recognizer = face_recognizer
        self.identity_by_detection_id = {}
        self.person_anchor_y_ratio = float(person_anchor_y_ratio)
        self.identity_desired_y_ratio = float(identity_desired_y_ratio)
        self.ball_desired_y_ratio = float(ball_desired_y_ratio)
        self.identity_recognition_interval_sec = float(identity_recognition_interval_sec)
        self.identity_cache_ttl_sec = float(identity_cache_ttl_sec)
        self.identity_track_ttl_sec = float(identity_cache_ttl_sec)
        self.identity_last_scan_at = 0.0
        self.identity_target_cache = None
        self.identity_tracks = []
        self.identity_next_track_id = 1
        self.identity_scan_in_progress = False
        self.identity_scan_thread = None
        self.identity_lock = threading.Lock()
        self.speed_smoothing_alpha = float(speed_smoothing_alpha)
        self.min_tracking_speed_rpm = float(min_tracking_speed_rpm)
        self.last_pitch_speed_cmd = None
        self.last_yaw_speed_cmd = None

        self.enabled = False
        self.ptz = None
        self.yolo = None
        self.t_last_target = time.time()
        self.has_returned_home_after_lost = False
        self.latest_detection_payload = self._empty_detection_payload()

    def home_lock(self):
        self._ensure_ptz()
        self.ptz.home_lock()
        self.t_last_target = time.time()
        self.has_returned_home_after_lost = True
        logging.info("Brushless PTZ homed and locked")

    def enable(self):
        if self.enabled:
            return
        self._ensure_ptz()
        self._ensure_yolo()
        self.enabled = True
        self.t_last_target = time.time()
        self.has_returned_home_after_lost = False
        logging.info("Tracking enabled")

    def disable(self):
        if self.ptz is not None:
            self.ptz.stop()
        self.enabled = False
        logging.info("Tracking disabled")

    def set_target_class(self, target_class_id):
        target_class_id = int(target_class_id)
        target_name = class_name_for_id(target_class_id)
        self.target_mode = "class"
        self.active_profile_id = None
        self.target_class_id = target_class_id
        self.identity_by_detection_id = {}
        self.identity_target_cache = None
        self.identity_last_scan_at = 0.0
        self.identity_tracks = []
        self.identity_next_track_id = 1
        self.identity_scan_in_progress = False
        self._reset_speed_filter()
        self.t_last_target = time.time()
        self.has_returned_home_after_lost = False
        if self.yolo is not None:
            self.yolo.target_class_id = target_class_id
        if self.ptz is not None:
            self.ptz.stop()
        logging.info("Tracking target switched to %s (%s)", target_name, target_class_id)
        return self.target_status()

    def set_identity_target(self, profile_id):
        profile_id = str(profile_id).strip()
        if not profile_id:
            raise ValueError("profile_id is required for identity target")
        self.target_mode = "identity"
        self.active_profile_id = profile_id
        self.target_class_id = 0
        self.identity_by_detection_id = {}
        self.identity_target_cache = None
        self.identity_last_scan_at = 0.0
        self.identity_tracks = []
        self.identity_next_track_id = 1
        self.identity_scan_in_progress = False
        self._reset_speed_filter()
        self.t_last_target = time.time()
        self.has_returned_home_after_lost = False
        if self.yolo is not None:
            self.yolo.target_class_id = 0
        if self.ptz is not None:
            self.ptz.stop()
        logging.info("Tracking identity target switched to %s", profile_id)
        return self.target_status()

    def target_status(self):
        mode = getattr(self, "target_mode", "class")
        profile_id = getattr(self, "active_profile_id", None) if mode == "identity" else None
        return {
            "mode": mode,
            "class_id": int(self.target_class_id),
            "name": self._class_name(self.target_class_id),
            "profile_id": profile_id,
        }

    def latest_detections(self):
        return getattr(self, "latest_detection_payload", self._empty_detection_payload())

    def process_frame(self, raw_frame):
        display_frame = raw_frame.copy()
        if not self.enabled:
            self.latest_detection_payload = self._empty_detection_payload(raw_frame)
            return display_frame, []

        h, w = raw_frame.shape[:2]
        detections = self.yolo.detect(raw_frame)
        target = self._select_target(detections, w, h, raw_frame)

        if target is not None:
            self.t_last_target = time.time()
            self.has_returned_home_after_lost = False
            pitch_speed, yaw_speed, error_x, error_y, target_x, target_y = self._calculate_tracking_speed(
                target,
                w,
                h,
            )
            self.ptz.set_speed(pitch_speed, yaw_speed)
            self._draw_target(display_frame, target, error_x, error_y, pitch_speed, yaw_speed, target_x, target_y)
            logging.info(
                "tracking target=%s conf=%.2f offset=(%.0f,%.0f) speed=(pitch %.2f,yaw %.2f)",
                self._class_name(target[5]),
                target[4],
                error_x,
                error_y,
                pitch_speed,
                yaw_speed,
            )
        else:
            self.ptz.stop()
            self._reset_speed_filter()
            lost_time = time.time() - self.t_last_target
            if lost_time >= self.no_target_timeout and not self.has_returned_home_after_lost:
                logging.warning("Target lost for %.1fs; returning brushless PTZ home", lost_time)
                self.ptz.home_lock()
                self.has_returned_home_after_lost = True
            else:
                logging.info("No target")

        self._update_latest_detections(raw_frame, detections, target)
        self._draw_center(display_frame)
        return display_frame, detections

    def close(self):
        try:
            self.disable()
        except Exception:
            logging.exception("failed to disable tracking")
        if self.yolo is not None:
            self.yolo.release()
            self.yolo = None
        if self.ptz is not None:
            self.ptz.close()
            self.ptz = None

    def _ensure_ptz(self):
        if self.ptz is not None:
            return
        self.ptz = BrushlessGimbal(
            pitch_port=self.pitch_port,
            yaw_port=self.yaw_port,
            pitch_addr=self.pitch_addr,
            yaw_addr=self.yaw_addr,
            baudrate=self.baudrate,
            pitch_max_rpm=self.pitch_max_rpm,
            yaw_max_rpm=self.yaw_max_rpm,
            home_pitch_deg=self.home_pitch_deg,
            home_yaw_deg=self.home_yaw_deg,
        )

    def _ensure_yolo(self):
        if self.yolo is not None:
            return
        self.yolo = YOLOv5Lite(
            self.model_path,
            self.class_path,
            confThreshold=self.conf_threshold,
            nmsThreshold=self.nms_threshold,
            target_class_id=self.target_class_id,
        )

    def _select_target(self, detections, frame_w, frame_h, frame=None):
        if not detections:
            self.identity_by_detection_id = {}
            return None
        if getattr(self, "target_mode", "class") == "identity":
            return self._select_identity_target(frame, detections)
        center_x = frame_w / 2
        center_y = frame_h / 2
        return min(
            detections,
            key=lambda det: (((det[0] + det[2]) / 2 - center_x) ** 2 + ((det[1] + det[3]) / 2 - center_y) ** 2),
        )

    def _select_identity_target(self, frame, detections):
        self._ensure_identity_state()
        recognizer = getattr(self, "face_recognizer", None)
        profile_id = getattr(self, "active_profile_id", None)
        if recognizer is None or not profile_id:
            self.identity_by_detection_id = {}
            return None

        person_detections = [detection for detection in detections if int(detection[5]) == 0]
        if not person_detections:
            self.identity_by_detection_id = {}
            return None

        now = time.time()
        tracks = self._update_identity_tracks(person_detections, now)
        self._maybe_start_identity_scan(frame, tracks, profile_id, now)
        active_track = self._active_identity_track(profile_id, now)
        if active_track is not None:
            self._mark_track_identity(active_track)
            return active_track.get("detection")
        return None

    def _ensure_identity_state(self):
        if not hasattr(self, "identity_tracks"):
            self.identity_tracks = []
        if not hasattr(self, "identity_next_track_id"):
            self.identity_next_track_id = 1
        if not hasattr(self, "identity_scan_in_progress"):
            self.identity_scan_in_progress = False
        if not hasattr(self, "identity_last_scan_at"):
            self.identity_last_scan_at = 0.0
        if not hasattr(self, "identity_by_detection_id"):
            self.identity_by_detection_id = {}
        if not hasattr(self, "identity_lock"):
            self.identity_lock = threading.Lock()

    def _update_identity_tracks(self, person_detections, now):
        self._ensure_identity_state()
        ttl = max(0.0, float(getattr(self, "identity_track_ttl_sec", getattr(self, "identity_cache_ttl_sec", 1.2))))
        with self.identity_lock:
            live_tracks = [
                track
                for track in self.identity_tracks
                if now - float(track.get("last_seen_at", 0.0)) <= ttl
            ]
            unmatched_tracks = list(live_tracks)
            updated_tracks = []

            for detection in person_detections:
                track = self._match_identity_track(detection, unmatched_tracks)
                if track is None:
                    track = {
                        "track_id": self.identity_next_track_id,
                        "profile_id": None,
                        "score": 0.0,
                        "face_box": None,
                        "last_identity_at": 0.0,
                    }
                    self.identity_next_track_id += 1
                else:
                    unmatched_tracks.remove(track)

                track["bbox"] = [float(value) for value in detection[:4]]
                track["detection"] = detection
                track["last_seen_at"] = float(now)
                updated_tracks.append(track)

            for track in unmatched_tracks:
                track["detection"] = None
                updated_tracks.append(track)

            self.identity_tracks = updated_tracks
            self.identity_by_detection_id = {}
            for track in self.identity_tracks:
                if track.get("detection") is not None and track.get("profile_id") is not None:
                    self._mark_track_identity(track)
            return [track for track in self.identity_tracks if track.get("detection") is not None]

    def _match_identity_track(self, detection, tracks):
        if not tracks:
            return None

        best = max(tracks, key=lambda track: self._bbox_iou(track.get("bbox", []), detection[:4]))
        if self._bbox_iou(best.get("bbox", []), detection[:4]) > 0.0:
            return best

        best = min(tracks, key=lambda track: self._bbox_center_distance_sq(track.get("bbox", []), detection[:4]))
        bbox = best.get("bbox", detection[:4])
        max_distance = max(
            abs(float(bbox[2]) - float(bbox[0])),
            abs(float(bbox[3]) - float(bbox[1])),
            1.0,
        ) * 0.75
        if self._bbox_center_distance_sq(bbox, detection[:4]) <= max_distance * max_distance:
            return best
        return None

    def _maybe_start_identity_scan(self, frame, tracks, profile_id, now):
        if frame is None or not tracks or getattr(self, "face_recognizer", None) is None:
            return

        interval = max(0.0, float(getattr(self, "identity_recognition_interval_sec", 0.4)))
        if bool(getattr(self, "identity_scan_in_progress", False)):
            return
        if now - float(getattr(self, "identity_last_scan_at", 0.0) or 0.0) < interval:
            return

        scan_items = [
            {
                "track_id": track["track_id"],
                "detection": list(track["detection"]),
            }
            for track in tracks
            if track.get("detection") is not None
        ]
        if not scan_items:
            return

        frame_snapshot = frame.copy() if hasattr(frame, "copy") else frame
        self.identity_scan_in_progress = True
        self.identity_last_scan_at = now
        thread = threading.Thread(
            target=self._run_identity_scan,
            args=(frame_snapshot, scan_items, profile_id),
            daemon=True,
        )
        self.identity_scan_thread = thread
        thread.start()

    def _run_identity_scan(self, frame, scan_items, profile_id):
        self._ensure_identity_state()
        recognizer = getattr(self, "face_recognizer", None)
        if recognizer is None:
            self.identity_scan_in_progress = False
            return

        detections = [item["detection"] for item in scan_items]
        try:
            identities = recognizer.identify_person_boxes(frame, detections)
            now = time.time()
            self.identity_last_scan_at = now
            with self.identity_lock:
                for scan_item, identity in zip(scan_items, identities):
                    matched_profile = identity.get("profile_id")
                    if matched_profile is None:
                        continue
                    for track in self.identity_tracks:
                        if track.get("track_id") == scan_item.get("track_id"):
                            track["profile_id"] = matched_profile
                            track["score"] = float(identity.get("score", 0.0))
                            track["face_box"] = identity.get("face_box")
                            track["last_identity_at"] = float(now)
                            break
        except Exception:
            logging.exception("identity background scan failed for %s", profile_id)
        finally:
            self.identity_scan_in_progress = False

    def _active_identity_track(self, profile_id, now):
        ttl = max(0.0, float(getattr(self, "identity_track_ttl_sec", getattr(self, "identity_cache_ttl_sec", 1.2))))
        with self.identity_lock:
            for track in self.identity_tracks:
                if track.get("profile_id") != profile_id:
                    continue
                if track.get("detection") is None:
                    continue
                if now - float(track.get("last_identity_at", 0.0)) > ttl:
                    continue
                return track
        return None

    def _mark_track_identity(self, track):
        detection = track.get("detection")
        if detection is None:
            return
        self.identity_by_detection_id[id(detection)] = {
            "detection": detection,
            "profile_id": track.get("profile_id"),
            "score": track.get("score", 0.0),
            "face_box": track.get("face_box"),
            "track_id": track.get("track_id"),
        }

    @staticmethod
    def _bbox_iou(a, b):
        if len(a) < 4 or len(b) < 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0.0:
            return 0.0
        return inter_area / denom

    @staticmethod
    def _bbox_center_distance_sq(a, b):
        if len(a) < 4 or len(b) < 4:
            return float("inf")
        ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        return (acx - bcx) ** 2 + (acy - bcy) ** 2

    def _empty_detection_payload(self, frame=None):
        frame_info = {"width": 0, "height": 0}
        if frame is not None:
            h, w = frame.shape[:2]
            frame_info = {"width": int(w), "height": int(h)}
        return {
            "enabled": bool(getattr(self, "enabled", False)),
            "frame": frame_info,
            "detections": [],
            "updated_at": time.time(),
        }

    def _update_latest_detections(self, frame, detections, selected_target):
        h, w = frame.shape[:2]
        self.latest_detection_payload = {
            "enabled": bool(self.enabled),
            "frame": {"width": int(w), "height": int(h)},
            "detections": [
                self._serialize_detection(detection, detection is selected_target)
                for detection in detections
            ],
            "updated_at": time.time(),
        }

    def _serialize_detection(self, detection, selected):
        x1, y1, x2, y2, conf, cls_id = detection
        payload = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": round(float(conf), 4),
            "class_id": int(cls_id),
            "class_name": self._class_name(cls_id),
            "selected": bool(selected),
        }
        identity = getattr(self, "identity_by_detection_id", {}).get(id(detection))
        if identity is not None:
            payload["profile_id"] = identity.get("profile_id")
            payload["identity_score"] = round(float(identity.get("score", 0.0)), 4)
            payload["face_box"] = identity.get("face_box")
        return payload

    def _calculate_tracking_speed(self, target, frame_w, frame_h):
        x1, y1, x2, y2, _conf, _cls_id = target
        target_x, target_y, desired_x, desired_y = self._tracking_points_for_target(
            x1,
            y1,
            x2,
            y2,
            _cls_id,
            frame_w,
            frame_h,
        )
        error_x = target_x - desired_x
        error_y = target_y - desired_y

        if abs(error_x) < self.dead_zone_px:
            yaw_speed = 0.0
        else:
            yaw_speed = self.yaw_max_rpm * (error_x / (frame_w / 2))

        if abs(error_y) < self.dead_zone_px:
            pitch_speed = 0.0
        else:
            pitch_speed = self.pitch_max_rpm * (error_y / (frame_h / 2))

        yaw_speed = max(-self.yaw_max_rpm, min(self.yaw_max_rpm, yaw_speed))
        pitch_speed = max(-self.pitch_max_rpm, min(self.pitch_max_rpm, pitch_speed))
        pitch_speed, yaw_speed = self._filter_tracking_speed(pitch_speed, yaw_speed)
        return pitch_speed, yaw_speed, error_x, error_y, target_x, target_y

    def _filter_tracking_speed(self, pitch_speed, yaw_speed):
        pitch_speed = self._filter_axis_speed(pitch_speed, getattr(self, "last_pitch_speed_cmd", None))
        yaw_speed = self._filter_axis_speed(yaw_speed, getattr(self, "last_yaw_speed_cmd", None))
        self.last_pitch_speed_cmd = pitch_speed
        self.last_yaw_speed_cmd = yaw_speed
        return pitch_speed, yaw_speed

    def _filter_axis_speed(self, speed, last_speed):
        min_speed = max(0.0, float(getattr(self, "min_tracking_speed_rpm", 0.0)))
        if abs(speed) < min_speed:
            return 0.0

        alpha = max(0.0, min(1.0, float(getattr(self, "speed_smoothing_alpha", 1.0))))
        if last_speed is None or abs(float(last_speed)) < min_speed:
            return speed
        if speed * float(last_speed) <= 0.0:
            return speed
        return (1.0 - alpha) * float(last_speed) + alpha * speed

    def _reset_speed_filter(self):
        self.last_pitch_speed_cmd = None
        self.last_yaw_speed_cmd = None

    def _tracking_points_for_target(self, x1, y1, x2, y2, cls_id, frame_w, frame_h):
        target_x = (x1 + x2) / 2
        if getattr(self, "target_mode", "class") == "identity":
            person_anchor_y_ratio = getattr(self, "person_anchor_y_ratio", 0.28)
            desired_y_ratio = getattr(self, "identity_desired_y_ratio", 0.38)
            target_y = y1 + person_anchor_y_ratio * (y2 - y1)
        else:
            desired_y_ratio = getattr(self, "ball_desired_y_ratio", 0.50) if int(cls_id) == 32 else 0.50
            target_y = (y1 + y2) / 2

        return target_x, target_y, frame_w * 0.5, frame_h * desired_y_ratio

    def _draw_target(self, frame, target, error_x, error_y, pitch_speed, yaw_speed, target_x, target_y):
        x1, y1, x2, y2, conf, cls_id = target
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (int(target_x), int(target_y)), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"{self._class_name(cls_id)} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"err=({error_x:.1f},{error_y:.1f}) spd=({pitch_speed:.1f},{yaw_speed:.1f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    def _draw_center(self, frame):
        h, w = frame.shape[:2]
        cv2.drawMarker(
            frame,
            (w // 2, h // 2),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
        )
        cv2.rectangle(
            frame,
            (w // 2 - self.dead_zone_px, h // 2 - self.dead_zone_px),
            (w // 2 + self.dead_zone_px, h // 2 + self.dead_zone_px),
            (255, 0, 0),
            1,
        )

    def _class_name(self, cls_id):
        if self.yolo is None:
            try:
                return class_name_for_id(cls_id)
            except ValueError:
                return str(cls_id)
        if 0 <= int(cls_id) < len(self.yolo.classes):
            return self.yolo.classes[int(cls_id)]
        try:
            return class_name_for_id(cls_id)
        except ValueError:
            pass
        return str(cls_id)
