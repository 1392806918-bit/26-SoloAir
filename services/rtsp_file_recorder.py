import logging
import os
import subprocess
import time


class RtspFileRecorder:
    """Records the shared local RTSP stream to MP4."""

    def __init__(self, ffmpeg_bin, rtsp_url):
        self.ffmpeg_bin = ffmpeg_bin
        self.rtsp_url = rtsp_url
        self.process = None
        self.output_path = None
        self.temp_path = None

    @property
    def is_recording(self):
        return self.process is not None and self.process.poll() is None

    def start(self, output_path, temp_path):
        if self.is_recording:
            raise RuntimeError("recording is already running")
        self.output_path = output_path
        self.temp_path = temp_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.rtsp_url,
            "-map",
            "0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            temp_path,
        ]
        logging.info("Starting recorder: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    def stop(self, timeout=8):
        if self.process is None:
            return None
        process = self.process
        if process.stdin is None:
            return_code = process.poll()
            if return_code is None:
                process.terminate()
            self.process = None
            return None
        if process.poll() is None:
            try:
                process.stdin.write("q\n")
                process.stdin.flush()
            except Exception:
                pass
            deadline = time.time() + timeout
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()

        self.process = None
        output_path = self.output_path
        temp_path = self.temp_path
        self.output_path = None
        self.temp_path = None

        if temp_path and os.path.exists(temp_path):
            if os.path.getsize(temp_path) > 0:
                os.replace(temp_path, output_path)
                logging.info("Recording saved: %s", output_path)
                return output_path
            os.remove(temp_path)
        return None
