"""SoloAir runtime configuration for ELF2 brushless PTZ."""

# Brushless PTZ motor serial ports.
PITCH_PORT = "/dev/ttyCH340_1"
YAW_PORT = "/dev/ttyCH340_2"
PITCH_ADDR = 1
YAW_ADDR = 1
BAUDRATE = 115200

HOME_PITCH_DEG = 0.0
HOME_YAW_DEG = 0.0

# Legacy scripts may still import SERIAL_PORT; the brushless runtime uses the
# axis-specific USB ports above.
SERIAL_PORT = PITCH_PORT

# Camera settings.
#CAMERA_INDEX = 21
#CAMERA_DEVICE = "/dev/video22"
CAMERA_INDEX = 21
CAMERA_DEVICE = "/dev/v4l/by-id/usb-LRCP_Technology_Co.__Ltd._LRCP_1080P-60fps_SN0001-video-index0"
CAMERA_GST_FORMAT = "MJPEG"  # MJPEG or YUYV
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

# RKNN / YOLO model settings.
MODEL_PATH = "/home/elf/VsCode/无刷云台4/models/yolov5nu.rknn"
CLASS_PATH = "/home/elf/VsCode/无刷云台4/models/COCO.names"
PERSON_CLASS_ID = 0
CONF_THRES = 0.35
IOU_THRES = 0.45

# Face identity recognition settings. Enrollment photos are read from
# FACE_ENROLL_DIR and converted into FACE_DB_PATH by scripts/enroll_faces.py.
FACE_IDENTITY_ENABLED = True
FACE_ENROLL_DIR = "/home/elf/VsCode/无刷云台4/faces"
FACE_DB_PATH = "/home/elf/VsCode/无刷云台4/models/face_db.npz"
FACE_MODEL_NAME = "buffalo_s"
FACE_PROVIDERS = ["CPUExecutionProvider"]
FACE_DET_SIZE = (320, 320)
FACE_MATCH_THRESHOLD = 0.40
FACE_PERSON_CROP_TOP_RATIO = 0.65
FACE_MAX_CROP_SIZE = 224

# Identity mode still tracks the YOLO person box. Face recognition only chooses
# which person box belongs to the active profile.
PERSON_ANCHOR_Y_RATIO = 0.50
IDENTITY_DESIRED_Y_RATIO = 0.50
IDENTITY_RECOGNITION_INTERVAL_SEC = 0.50
IDENTITY_CACHE_TTL_SEC = 1.20
BALL_DESIRED_Y_RATIO = 0.50

# Brushless tracking control settings.
DEAD_ZONE_PX = 35        #DEAD_ZONE original = 25       
PITCH_MAX_RPM = 15.0     #PITCH original = 20 
YAW_MAX_RPM = 15.0       #YAW original = 25
TRACKING_SPEED_SMOOTHING_ALPHA = 0.30
MIN_TRACKING_SPEED_RPM = 0.50
LOST_RETURN_HOME_SEC = 3.0

# RTSP audio/video stream. The RTSP stream is produced by system Python/GStreamer,
# not by the conda control process.
RTSP_HOST = "0.0.0.0"
RTSP_PORT = 8554
RTSP_PATH = "/camera"
RTSP_LOCAL_URL = f"rtsp://127.0.0.1:{RTSP_PORT}{RTSP_PATH}"
RTSP_NETWORK_URL = f"rtsp://<elf2-ip>:{RTSP_PORT}{RTSP_PATH}"
CONTROL_VIDEO_SOURCE = RTSP_LOCAL_URL   

AUDIO_DEVICE = "usbmic"
AUDIO_RATE = 48000
AUDIO_CHANNELS = 1
AUDIO_BITRATE = 64000
ENABLE_RTSP_AUDIO = True
VIDEO_BITRATE = 1200000

# Mobile control API.
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080

# Session output directory.     
SESSION_ROOT = "/home/elf/soloair/sessions"

# Local debug display. Keep False unless you are checking YOLO boxes on ELF2.
ENABLE_LOCAL_DISPLAY = False
DISPLAY_WINDOW_NAME = "SoloAir Tracking"

# Recording uses ffmpeg to copy the local RTSP stream to MP4.
FFMPEG_BIN = "ffmpeg"

# Conda environment used by the control/tracking process in start_soloair_stack.sh.
CONDA_ENV_NAME = "sherpa_env"
CONDA_SH = "/home/elf/miniconda3/etc/profile.d/conda.sh"
