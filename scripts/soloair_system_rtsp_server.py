#!/usr/bin/env python3
import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    AUDIO_BITRATE,
    AUDIO_CHANNELS,
    AUDIO_DEVICE,
    AUDIO_RATE,
    CAMERA_DEVICE,
    CAMERA_FPS,
    CAMERA_GST_FORMAT,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    ENABLE_RTSP_AUDIO,
    RTSP_HOST,
    RTSP_PATH,
    RTSP_PORT,
    VIDEO_BITRATE,
)


def build_video_chain(device, width, height, fps, gst_format, bitrate):
    gst_format = gst_format.upper()
    if gst_format == "MJPEG":
        input_caps = f"image/jpeg,width={width},height={height},framerate={fps}/1"
        decode = "jpegdec ! videoconvert"
    elif gst_format == "YUYV":
        input_caps = f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1"
        decode = "videoconvert"
    else:
        raise ValueError("CAMERA_GST_FORMAT must be MJPEG or YUYV")

    return (
        f"v4l2src device={device} do-timestamp=true "
        f"! queue leaky=downstream max-size-buffers=2 "
        f"! {input_caps} "
        f"! {decode} "
        f"! video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 "
        f"! mpph264enc bps={bitrate} "
        f"! h264parse config-interval=1 "
        f"! rtph264pay name=pay0 pt=96 config-interval=1"
    )


def build_audio_chain(device, rate, channels, bitrate):
    denoise = (
        "webrtcdsp echo-cancel=false high-pass-filter=true "
        "noise-suppression=true noise-suppression-level=2 gain-control=false"
    )
    return (
        f"alsasrc device={device} do-timestamp=true "
        f"! queue leaky=downstream max-size-time=200000000 "
        f"! audioconvert ! audioresample "
        f"! audio/x-raw,rate={rate},channels={channels} "
        f"! {denoise} "
        f"! voaacenc bitrate={bitrate} "
        f"! aacparse "
        f"! rtpmp4gpay name=pay1 pt=97"
    )


def build_launch(args):
    video = build_video_chain(
        device=args.camera_device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        gst_format=args.camera_format,
        bitrate=args.video_bitrate,
    )
    if args.video_only:
        return f"( {video} )"

    audio = build_audio_chain(
        device=args.audio_device,
        rate=args.audio_rate,
        channels=args.audio_channels,
        bitrate=args.audio_bitrate,
    )
    return f"( {video} {audio} )"


def parse_args():
    parser = argparse.ArgumentParser(description="System GStreamer RTSP server for SoloAir.")
    parser.add_argument("--host", default=RTSP_HOST)
    parser.add_argument("--port", default=str(RTSP_PORT))
    parser.add_argument("--path", default=RTSP_PATH)
    parser.add_argument("--camera-device", default=CAMERA_DEVICE)
    parser.add_argument("--camera-format", default=CAMERA_GST_FORMAT, choices=("MJPEG", "YUYV"))
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=CAMERA_FPS)
    parser.add_argument("--video-bitrate", type=int, default=VIDEO_BITRATE)
    parser.add_argument("--audio-device", default=AUDIO_DEVICE)
    parser.add_argument("--audio-rate", type=int, default=AUDIO_RATE)
    parser.add_argument("--audio-channels", type=int, default=AUDIO_CHANNELS)
    parser.add_argument("--audio-bitrate", type=int, default=AUDIO_BITRATE)
    parser.add_argument("--video-only", action="store_true", default=not ENABLE_RTSP_AUDIO)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [system-rtsp] %(message)s")
    args = parse_args()
    path = args.path if args.path.startswith("/") else f"/{args.path}"

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import GLib, Gst, GstRtspServer

    Gst.init(None)
    launch = build_launch(args)
    logging.info("GStreamer runtime: %s", Gst.version_string())
    logging.info("RTSP launch pipeline: %s", launch)

    server = GstRtspServer.RTSPServer()
    server.set_address(args.host)
    server.set_service(str(args.port))

    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(launch)
    factory.set_shared(True)
    factory.set_latency(120)
    server.get_mount_points().add_factory(path, factory)
    server.attach(None)

    logging.info("RTSP server started: rtsp://<elf2-ip>:%s%s", args.port, path)
    GLib.MainLoop().run()


if __name__ == "__main__":
    main()
