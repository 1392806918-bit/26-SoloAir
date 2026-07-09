import json
import os
from datetime import datetime


class SessionManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.current_session_id = None
        os.makedirs(self.root_dir, exist_ok=True)

    def start_session(self):
        if self.current_session_id is None:
            self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.session_dir, exist_ok=True)
            os.makedirs(self.screenshot_dir, exist_ok=True)
            self.write_event("session_start", {})
        return self.current_session_id

    def finish_session(self):
        if self.current_session_id:
            self.write_event("session_finish", {})
        self.current_session_id = None

    @property
    def session_dir(self):
        if self.current_session_id is None:
            raise RuntimeError("no active session")
        return os.path.join(self.root_dir, self.current_session_id)

    @property
    def screenshot_dir(self):
        return os.path.join(self.session_dir, "screenshots")

    def video_path(self):
        return os.path.join(self.session_dir, "video.mp4")

    def temp_video_path(self):
        return os.path.join(self.session_dir, "video.tmp.mp4")

    def snapshot_path(self, label="snapshot"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
        return os.path.join(self.screenshot_dir, f"{safe_label}_{timestamp}.jpg")

    def write_event(self, event, payload):
        os.makedirs(self.session_dir, exist_ok=True)
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "payload": payload,
        }
        with open(os.path.join(self.session_dir, "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def list_sessions(self):
        if not os.path.isdir(self.root_dir):
            return []
        sessions = []
        for session_id in sorted(os.listdir(self.root_dir), reverse=True):
            session_dir = os.path.join(self.root_dir, session_id)
            if not os.path.isdir(session_dir):
                continue
            sessions.append(
                {
                    "session_id": session_id,
                    "video": os.path.exists(os.path.join(session_dir, "video.mp4")),
                    "screenshots": self._list_screenshots(session_id),
                    "summary": os.path.exists(os.path.join(session_dir, "summary.txt")),
                }
            )
        return sessions

    def _list_screenshots(self, session_id):
        screenshot_dir = os.path.join(self.root_dir, session_id, "screenshots")
        if not os.path.isdir(screenshot_dir):
            return []
        return sorted(
            name for name in os.listdir(screenshot_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def resolve_file(self, relative_path):
        normalized = os.path.normpath(relative_path).replace("\\", os.sep)
        full_path = os.path.abspath(os.path.join(self.root_dir, normalized))
        root_path = os.path.abspath(self.root_dir)
        if not full_path.startswith(root_path + os.sep):
            raise ValueError("invalid file path")
        if not os.path.isfile(full_path):
            raise FileNotFoundError(full_path)
        return full_path
