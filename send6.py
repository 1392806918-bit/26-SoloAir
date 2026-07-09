#!/usr/bin/env python3
import subprocess
import serial
import os
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime

class VoiceCommand:
    WAKE_WORD = "无人机"
    AUDIO_DEVICE = "usbmic"
    AUDIO_DEVICE_ENV = "SOLOAIR_VOICE_AUDIO_DEVICE"
    FLIGHT_SERIAL_PORT = "/dev/ttyCH340_3"
    FLIGHT_SERIAL_BAUDRATE = 115200
    SENTENCE_IDLE_SECONDS = 1.5
    PARTIAL_IDLE_SECONDS = 1.5
    COMMAND_COOLDOWN_SECONDS = 2.0
    ALSA_PLUGIN_DIR = "/usr/lib/aarch64-linux-gnu/alsa-lib"
    RECENT_FRAGMENT_COUNT = 3
    MAX_FRAGMENT_CHARS = 8
    MAX_JOINED_COMMAND_CHARS = 12
    MAX_COMMAND_SENTENCE_CHARS = 12

    def __init__(self):
        # 飞控串口配置 - 默认使用 USB-TTL
        try:
            self.ser = serial.Serial(
                self.FLIGHT_SERIAL_PORT,
                self.FLIGHT_SERIAL_BAUDRATE,
                timeout=1,
            )
            print(f"串口已打开: {self.ser.name}")
        except Exception as e:
            print(f"错误: 无法打开串口 {e}")
            sys.exit(1)   # 退出进程，让 systemd 重启
                
        self.audio_device = self._get_audio_device()

        # 模型路径 - 使用绝对路径
        model_base = os.path.expanduser("~/model/sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_base):
            print(f"错误: 模型路径不存在: {model_base}")
            model_base = "./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
            if not os.path.exists(model_base):
                print(f"错误: 模型路径也不存在: {model_base}")
                print("请确认模型文件位置")
                sys.exit(1)
        
        print(f"使用模型路径: {model_base}")
        
        # 正确的参数配置（平衡模式）
        self.cmd = self._build_sherpa_command(model_base)

        # 用于防止流式识别重复输出导致短时间内重复发同一条命令
        self.last_command_name = None
        self.last_command_at = 0.0
        
        # 确保记录文件所在的目录存在
        self.log_dir = "/home/elf/model"
        self.text_log_file = os.path.join(self.log_dir, "text.txt")
        self.command_log_file = os.path.join(self.log_dir, "command.txt")
        os.makedirs(self.log_dir, exist_ok=True)
        self._reset_log_files()

        self.command_rules = self.default_command_rules()
        self._reset_candidate_window()

    @classmethod
    def default_command_rules(cls):
        return [
            {
                "name": "起飞",
                "byte": 0x01,
                "keywords": ["起飞"],
            },
            {
                "name": "拉近景",
                "byte": 0x02,
                "keywords": ["近景", "前", "靠近", "拉进", "拉近", "前进", "向前"],
            },
            {
                "name": "拉远景",
                "byte": 0x03,
                "keywords": ["远景", "后", "向后", "后退", "拉远"],
            },
            {
                "name": "降落",
                "byte": 0x04,
                "keywords": ["降落"],
            },
            {
                "name": "向左",
                "byte": 0x05,
                "keywords": ["向左", "往左", "左移", "左"],
            },
            {
                "name": "向右",
                "byte": 0x06,
                "keywords": ["向右", "往右", "右移", "右"],
            },
            {
                "name": "向上",
                "byte": 0x07,
                "keywords": ["向上", "往上", "上升", "升高", "高一点"],
            },
            {
                "name": "向下",
                "byte": 0x08,
                "keywords": ["向下", "往下", "下降", "降低", "低一点"],
            },
        ]

    def _find_sherpa_alsa(self):
        env_bin = os.path.join(os.path.dirname(sys.executable), "sherpa-onnx-alsa")
        if os.path.exists(env_bin):
            return env_bin
        return shutil.which("sherpa-onnx-alsa") or "sherpa-onnx-alsa"

    def _get_audio_device(self):
        return os.environ.get(self.AUDIO_DEVICE_ENV, self.AUDIO_DEVICE)

    def _build_sherpa_command(self, model_base):
        sherpa_alsa = self._find_sherpa_alsa()
        audio_device = getattr(self, "audio_device", self._get_audio_device())
        return [
            sherpa_alsa,
            "--provider=rknn",
            f"--encoder={model_base}/encoder.rknn",
            f"--decoder={model_base}/decoder.rknn",
            f"--joiner={model_base}/joiner.rknn",
            f"--tokens={model_base}/tokens.txt",
            "--rule1-min-trailing-silence=0.3",
            "--rule2-min-trailing-silence=0.2",
            "--rule3-min-trailing-silence=0.1",
            "--rule3-min-utterance-length=3.0",
            audio_device,
        ]

    def _build_subprocess_env(self):
        env = os.environ.copy()
        env["ALSA_PLUGIN_DIR"] = self.ALSA_PLUGIN_DIR
        return env

    def _reset_candidate_window(self):
        self.recent_fragments = []
        self.best_candidate = None
        self.best_candidate_score = None

    def _clean_recognition_text(self, line):
        text = re.sub(r"^\d+:\s*", "", line.strip())
        return re.sub(r"\s+", "", text)

    def _action_keywords(self):
        keywords = []
        for rule in self.command_rules:
            keywords.extend(rule["keywords"])
        return keywords

    def _candidate_texts(self, clean_text):
        candidates = [clean_text]
        max_count = min(len(self.recent_fragments), self.RECENT_FRAGMENT_COUNT)
        for count in range(2, max_count + 1):
            fragments = self.recent_fragments[-count:]
            if not all(len(fragment) <= self.MAX_FRAGMENT_CHARS for fragment in fragments):
                continue
            joined = "".join(fragments)
            if len(joined) <= self.MAX_JOINED_COMMAND_CHARS:
                candidates.append(joined)

        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _candidate_command(self, text):
        if len(text) > self.MAX_COMMAND_SENTENCE_CHARS:
            return None
        return self._extract_command(text)

    def _is_wake_prefix(self, text):
        return text != self.WAKE_WORD and self.WAKE_WORD.startswith(text)

    def _action_text_after_wake(self, text):
        wake_index = text.find(self.WAKE_WORD)
        if wake_index < 0:
            return None
        return text[wake_index + len(self.WAKE_WORD):]

    def _has_partial_action_after_wake(self, text):
        action_text = self._action_text_after_wake(text)
        if not action_text:
            return action_text == ""
        return any(
            keyword.startswith(action_text) and keyword != action_text
            for keyword in self._action_keywords()
        )

    def _candidate_needs_more_text(self, text):
        if not text or self._candidate_command(text):
            return False
        if self._is_wake_prefix(text):
            return True
        if self.WAKE_WORD in text and self._has_partial_action_after_wake(text):
            return True
        if self.WAKE_WORD not in text and any(keyword in text for keyword in self._action_keywords()):
            return True
        return False

    def _candidate_idle_seconds(self):
        if self._candidate_needs_more_text(self.best_candidate):
            return self.PARTIAL_IDLE_SECONDS
        return self.SENTENCE_IDLE_SECONDS

    def _score_candidate(self, text):
        command = self._candidate_command(text)
        if command:
            return (4, -len(text))
        if self.WAKE_WORD in text and self._has_partial_action_after_wake(text):
            return (3, 1, -len(text))
        if self.WAKE_WORD in text:
            return (3, 0, -len(text))
        if any(keyword in text for keyword in self._action_keywords()):
            return (2, -len(text))
        return (1, -len(text))

    def _record_recognition_candidate(self, line):
        clean_text = self._clean_recognition_text(line)
        if not clean_text:
            return self.best_candidate

        self.recent_fragments.append(clean_text)
        self.recent_fragments = self.recent_fragments[-self.RECENT_FRAGMENT_COUNT:]

        for candidate in self._candidate_texts(clean_text):
            score = self._score_candidate(candidate)
            if self.best_candidate_score is None or score > self.best_candidate_score:
                self.best_candidate = candidate
                self.best_candidate_score = score

        return self.best_candidate

    def _reset_log_files(self):
        """程序启动时清空识别文本和命令日志。"""
        for log_file in [self.text_log_file, self.command_log_file]:
            try:
                with open(log_file, "w", encoding="utf-8"):
                    pass
                print(f"日志已清空: {log_file}")
            except Exception as e:
                print(f"警告: 清空日志失败 {log_file}: {e}")

    def _append_line(self, file_path, text):
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            print(f"警告: 写入文件失败 {file_path}: {e}")

    def _save_recognized_text(self, text):
        """保存确认完成的一句识别文本。"""
        self._append_line(self.text_log_file, text)
        print(f"记录识别文本: {text} -> {self.text_log_file}")

    def _save_command_log(self, source_text, command_name, command_byte, status, frame=None):
        """保存命令触发结果，包含时间、原文、命令和发送状态。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame_text = " ".join(f"{b:02X}" for b in frame) if frame else "-"
        byte_text = f"0x{command_byte:02X}" if command_byte is not None else "-"
        log_line = (
            f"{timestamp} | 原文: {source_text} | 命令: {command_name} | "
            f"字节: {byte_text} | 状态: {status} | 帧: {frame_text}"
        )
        self._append_line(self.command_log_file, log_line)
        print(f"记录命令: {log_line} -> {self.command_log_file}")

    def calculate_checksum(self, data):
        """计算和校验(SUM)和附加校验(ADD)"""
        sum_check = 0
        add_check = 0
        for byte in data:
            sum_check = (sum_check + byte) & 0xFF
            add_check = (add_check + sum_check) & 0xFF
        return sum_check, add_check

    def build_command_frame(self, command_byte):
        """构建完整的匿名通信协议帧"""
        HEAD = 0xAA
        D_ADDR = 0xFF
        ID = 0xF1      # 灵活格式帧
        LEN = 0x01
        DATA = command_byte
        
        checksum_data = [HEAD, D_ADDR, ID, LEN, DATA]
        sum_check, add_check = self.calculate_checksum(checksum_data)
        
        return bytes([HEAD, D_ADDR, ID, LEN, DATA, sum_check, add_check])

    def _extract_command(self, text):
        """只在唤醒词之后寻找动作关键词，返回最先出现的命令。"""
        wake_index = text.find(self.WAKE_WORD)
        if wake_index < 0:
            return None

        action_text = text[wake_index + len(self.WAKE_WORD):]
        matches = []
        for rule_index, rule in enumerate(self.command_rules):
            for keyword in rule["keywords"]:
                keyword_index = action_text.find(keyword)
                if keyword_index >= 0:
                    matches.append((keyword_index, rule_index, keyword, rule))

        if not matches:
            return None

        _, _, matched_keyword, rule = min(matches, key=lambda item: (item[0], item[1]))
        command = {
            "name": rule["name"],
            "keyword": matched_keyword,
        }
        if "byte" in rule:
            command["byte"] = rule["byte"]
        if "type" in rule:
            command["type"] = rule["type"]
        if "target" in rule:
            command["target"] = rule["target"]
        return command

    def send_command(self, command, source_text):
        """发送串口命令；串口不可用时也写入命令日志。"""
        now = time.monotonic()
        command_name = command["name"]
        command_byte = command["byte"]
        if (
            self.last_command_name == command_name and
            now - self.last_command_at < self.COMMAND_COOLDOWN_SECONDS
        ):
            status = f"跳过：重复命令冷却中({self.COMMAND_COOLDOWN_SECONDS:.1f}s)"
            print(f">>> {status}: {command_name}")
            self._save_command_log(source_text, command_name, command_byte, status)
            return

        frame = self.build_command_frame(command_byte)
        self.last_command_name = command_name
        self.last_command_at = now

        if not self.ser:
            status = "未发送：串口未打开"
            print(f">>> {status}: {command_name} -> 命令:0x{command_byte:02X}")
            self._save_command_log(source_text, command_name, command_byte, status, frame)
            return

        try:
            self.ser.write(frame)
            status = "已发送"
            print(
                f">>> 发送: {command_name} -> 命令:0x{command_byte:02X} -> "
                f"帧: {' '.join(f'{b:02X}' for b in frame)}"
            )
        except Exception as e:
            status = f"未发送：串口写入失败 {e}"
            print(f">>> {status}: {command_name}")

        self._save_command_log(source_text, command_name, command_byte, status, frame)

    def _handle_final_sentence(self, text):
        """确认一句话结束后，先写文本，再检测唤醒词和动作词。"""
        self._save_recognized_text(text)

        if self.WAKE_WORD not in text:
            print(f"未检测到唤醒词“{self.WAKE_WORD}”，不执行指令")
            return

        command = self._extract_command(text)
        if not command:
            print(f"检测到唤醒词“{self.WAKE_WORD}”，但没有动作关键词")
            return

        print(f"命中关键词: {command['keyword']} -> {command['name']}")
        self.send_command(command, text)

    def _is_recognition_line(self, line):
        if not line:
            return False
        ignored_prefixes = (
            "/",
            "OnlineRecognizerConfig",
            "Current sample rate:",
            "Recording started!",
            "Use recording device:",
            "Started! Please speak",
            "XRUN.",
            "Too many overruns.",
            "ALSA lib ",
            "Unable to open:",
            "Please use the command:",
            "arecord -l",
            "to list all available devices.",
            "For instance,",
            "if the output is:",
            "**** List of CAPTURE",
            "card ",
            "Subdevices:",
            "Subdevice #",
            "and if you want",
            "plughw:",
            "hw:",
        )
        if line.startswith(ignored_prefixes):
            return False
        return True

    def _is_visible_diagnostic_line(self, line):
        visible_prefixes = (
            "Current sample rate:",
            "Recording started!",
            "Use recording device:",
            "Started! Please speak",
            "XRUN.",
            "Too many overruns.",
            "ALSA lib ",
            "Unable to open:",
        )
        return line.startswith(visible_prefixes)

    def _read_process_output(self, process, output_queue):
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)
    
    def run(self):
        """运行识别"""
        print("="*50)
        print("语音命令系统启动")
        print(f"音频输入: {self.audio_device}")
        print(f"飞控串口: {self.FLIGHT_SERIAL_PORT} @ {self.FLIGHT_SERIAL_BAUDRATE}")
        print(f"唤醒词: {self.WAKE_WORD}")
        print(f"停顿确认: {self.SENTENCE_IDLE_SECONDS:.1f}s")
        print(f"同命令冷却: {self.COMMAND_COOLDOWN_SECONDS:.1f}s")
        print("关键词映射:")
        print("  起飞     -> 0x01")
        print("  前/近景  -> 0x02")
        print("  后/远景  -> 0x03")
        print("  降落     -> 0x04")
        print("  左       -> 0x05")
        print("  右       -> 0x06")
        print("  上       -> 0x07")
        print("  下       -> 0x08")
        print("="*50)
        
        process = subprocess.Popen(
            self.cmd,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=self._build_subprocess_env(),
        )
        output_queue = queue.Queue()
        reader_thread = threading.Thread(
            target=self._read_process_output,
            args=(process, output_queue),
            daemon=True,
        )
        reader_thread.start()

        last_update_at = None
        
        try:
            while True:
                try:
                    line = output_queue.get(timeout=0.1)
                    if line is None:
                        break

                    line = line.strip()
                    if self._is_recognition_line(line):
                        clean_text = self._clean_recognition_text(line)
                        best_candidate = self._record_recognition_candidate(line)
                        print(f"识别更新: {line} -> 清洗: {clean_text} -> 最佳候选: {best_candidate}")
                        last_update_at = time.monotonic()
                    elif self._is_visible_diagnostic_line(line):
                        print(f"识别诊断: {line}")
                except queue.Empty:
                    pass

                if (
                    self.best_candidate and
                    last_update_at and
                    time.monotonic() - last_update_at >= self._candidate_idle_seconds()
                ):
                    final_text = self.best_candidate
                    print(f"确认句子: {final_text}")
                    self._handle_final_sentence(final_text)
                    self._reset_candidate_window()
                    last_update_at = None

                if process.poll() is not None and output_queue.empty():
                    break

            if self.best_candidate:
                final_text = self.best_candidate
                print(f"确认句子: {final_text}")
                self._handle_final_sentence(final_text)
                self._reset_candidate_window()

            if process.poll() is not None:
                print(f"语音识别进程已退出，退出码: {process.returncode}")
        except KeyboardInterrupt:
            print("\n\n停止识别...")
            if self.best_candidate:
                final_text = self.best_candidate
                print(f"确认句子: {final_text}")
                self._handle_final_sentence(final_text)
                self._reset_candidate_window()
        finally:
            if process.poll() is None:
                process.terminate()
            if self.ser:
                self.ser.close()
                print("串口已关闭")

if __name__ == "__main__":
    vc = VoiceCommand()
    vc.run()
