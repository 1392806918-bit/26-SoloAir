import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.soloair_system_rtsp_server import build_audio_chain


class SystemRtspAudioChainTests(unittest.TestCase):
    def test_audio_chain_runs_webrtcdsp_before_aac_encoding(self):
        chain = build_audio_chain("usbmic", 48000, 1, 64000)

        denoise = (
            "webrtcdsp echo-cancel=false high-pass-filter=true "
            "noise-suppression=true noise-suppression-level=2 gain-control=false"
        )

        self.assertIn(denoise, chain)
        self.assertLess(chain.index("audio/x-raw,rate=48000,channels=1"), chain.index(denoise))
        self.assertLess(chain.index(denoise), chain.index("voaacenc bitrate=64000"))


if __name__ == "__main__":
    unittest.main()
