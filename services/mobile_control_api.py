import json
import logging
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote


class MobileControlApi:
    def __init__(self, host, port, controller, session_manager):
        self.host = host
        self.port = int(port)
        self.controller = controller
        self.session_manager = session_manager
        self._server = None
        self._thread = None

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner._handle_get(self)

            def do_POST(self):
                owner._handle_post(self)

            def log_message(self, fmt, *args):
                logging.info("HTTP %s - %s", self.address_string(), fmt % args)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logging.info("HTTP control API started: http://%s:%s", self.host, self.port)

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _handle_get(self, handler):
        if handler.path == "/api/status":
            self._send_json(handler, self.controller.status())
            return
        if handler.path == "/api/files":
            self._send_json(handler, {"sessions": self.session_manager.list_sessions()})
            return
        if handler.path == "/api/tracking/detections":
            self._send_json(handler, self.controller.latest_detections())
            return
        if handler.path == "/api/voice/logs":
            self._send_json(handler, self.controller.voice_logs())
            return
        if handler.path == "/api/voice/text.txt":
            self._send_text(handler, self.controller.voice_log_text("text"))
            return
        if handler.path == "/api/voice/command.txt":
            self._send_text(handler, self.controller.voice_log_text("command"))
            return
        if handler.path == "/api/frame.jpg":
            self._send_jpeg(handler, self.controller.latest_frame_jpeg())
            return
        if handler.path.startswith("/files/"):
            relative = unquote(handler.path[len("/files/"):])
            self._send_file(handler, relative)
            return
        self._send_json(handler, {"error": "not found"}, status=404)

    def _handle_post(self, handler):
        try:
            if handler.path == "/api/record/start":
                result = self.controller.start_recording()
            elif handler.path == "/api/record/stop":
                result = self.controller.stop_recording()
            elif handler.path == "/api/snapshot":
                result = self.controller.save_snapshot("manual")
            elif handler.path == "/api/tracking/target":
                payload = self._read_json_body(handler)
                result = self.controller.set_tracking_target(
                    target=payload.get("target"),
                    class_id=payload.get("class_id"),
                    profile_id=payload.get("profile_id"),
                )
            else:
                self._send_json(handler, {"error": "not found"}, status=404)
                return
            self._send_json(handler, {"ok": True, "result": result})
        except ValueError as exc:
            logging.warning("HTTP command rejected: %s", exc)
            self._send_json(handler, {"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            logging.exception("HTTP command failed")
            self._send_json(handler, {"ok": False, "error": str(exc)}, status=500)

    def _read_json_body(self, handler):
        length = int(handler.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        data = handler.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def _send_json(self, handler, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(data)

    def _send_text(self, handler, text, status=200):
        data = text.encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "text/plain; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(data)

    def _send_file(self, handler, relative):
        try:
            file_path = self.session_manager.resolve_file(relative)
        except Exception as exc:
            self._send_json(handler, {"error": str(exc)}, status=404)
            return
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(os.path.getsize(file_path)))
        handler.end_headers()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                handler.wfile.write(chunk)

    def _send_jpeg(self, handler, data):
        handler.send_response(200)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(data)
