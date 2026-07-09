import os
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VoiceCommandScopeTests(unittest.TestCase):
    def test_voice_script_no_longer_contains_tracking_target_switching(self):
        path = os.path.join(PROJECT_ROOT, "sounds_to_words1.py")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()

        self.assertNotIn("tracking_target", source)
        self.assertNotIn("TRACKING_TARGET_URL", source)
        self.assertNotIn("_send_tracking_target_command", source)
        self.assertNotIn("urllib.request", source)
        self.assertNotIn("urllib.error", source)


if __name__ == "__main__":
    unittest.main()
