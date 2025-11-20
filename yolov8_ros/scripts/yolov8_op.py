#!/usr/bin/env python3
import argparse
import os
import sys
import signal
from typing import Optional
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject  # noqa: E402


def generate_config_file(
    model_engine_path: str,
    num_classes: int = 1,
    threshold: float = 0.5,
    nms_threshold: float = 0.4,
    output_path: Optional[str] = None,
) -> str:
    """Generate a DeepStream nvinfer config file from model path."""
    if output_path is None:
        # Generate config path next to the engine file
        base_path = os.path.splitext(model_engine_path)[0]
        output_path = f"{base_path}_nvinfer_config.txt"

    config_content = f"""[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file={model_engine_path}
batch-size=1
process-mode=1
num-detected-classes={num_classes}
interval=0
gie-unique-id=1
output-blob-names=output0
network-mode=0

[class-attrs-all]
threshold={threshold}
nms-iou-threshold={nms_threshold}
topk=20
"""

    with open(output_path, "w") as f:
        f.write(config_content)

    print(f"Generated config file: {output_path}", file=sys.stderr)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 DeepStream/GStreamer NVMM pipeline"
    )
    parser.add_argument(
        "--device", type=str, default="/dev/video0", help="V4L2 device path"
    )
    parser.add_argument("--width", type=int, default=640, help="Input width")
    parser.add_argument("--height", type=int, default=480, help="Input height")
    parser.add_argument("--fps", type=int, default=30, help="Input FPS")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="nvinfer config-file-path (optional if --model-engine is provided)",
    )
    parser.add_argument(
        "--model-engine",
        type=str,
        default=None,
        help="Path to TensorRT engine file (will auto-generate config if --config not provided)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of classes in model (used when auto-generating config)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (used when auto-generating config)",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.4,
        help="NMS IoU threshold (used when auto-generating config)",
    )
    parser.add_argument(
        "--display", action="store_true", help="Enable display sink (nveglglessink)"
    )
    parser.add_argument(
        "--log-level", type=str, default="2", help="GStreamer log level (0-7)"
    )
    return parser.parse_args()


def check_deepstream_plugins() -> bool:
    """Check if DeepStream plugins are available."""
    plugin_paths = [
        "/opt/nvidia/deepstream/deepstream/lib/gst-plugins",
        "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/",
    ]

    for path in plugin_paths:
        if os.path.exists(path):
            nvstreammux_so = os.path.join(path, "libgstnvstreammux.so")
            if os.path.exists(nvstreammux_so):
                return True
    return False


def set_gst_env(log_level: str) -> None:
    """Set GStreamer environment variables and verify DeepStream plugins."""
    os.environ.setdefault("GST_DEBUG", log_level)

    # DeepStream plugin paths (try both with and without trailing slash)
    plugin_paths = [
        "/opt/nvidia/deepstream/deepstream/lib/gst-plugins",
        "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/",
    ]

    # Find the correct plugin path
    found_path = None
    for path in plugin_paths:
        if os.path.exists(path):
            nvstreammux_so = os.path.join(path, "libgstnvstreammux.so")
            if os.path.exists(nvstreammux_so):
                found_path = path
                break

    if found_path:
        # Append to existing path if set, otherwise set new
        existing_path = os.environ.get("GST_PLUGIN_PATH", "")
        if existing_path:
            os.environ["GST_PLUGIN_PATH"] = f"{existing_path}:{found_path}"
        else:
            os.environ["GST_PLUGIN_PATH"] = found_path

        existing_path_1_0 = os.environ.get("GST_PLUGIN_PATH_1_0", "")
        if existing_path_1_0:
            os.environ["GST_PLUGIN_PATH_1_0"] = f"{existing_path_1_0}:{found_path}"
        else:
            os.environ["GST_PLUGIN_PATH_1_0"] = found_path

        print(f"Using DeepStream plugins from: {found_path}", file=sys.stderr)
    else:
        print(
            "WARNING: DeepStream plugin path not found. Expected: /opt/nvidia/deepstream/deepstream/lib/gst-plugins",
            file=sys.stderr,
        )
        print(
            "Please verify DeepStream is installed correctly.",
            file=sys.stderr,
        )


def on_bus_message(bus, message, loop):
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        print("EOS received - stopping pipeline.")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"ERROR from {message.src.get_name()}: {err}", file=sys.stderr)
        if dbg:
            print(f"Debug details: {dbg}", file=sys.stderr)
        loop.quit()
    return True


def build_pipeline(args: argparse.Namespace) -> Gst.Pipeline:
    Gst.init(None)

    # Verify DeepStream plugins are available before building pipeline
    if not check_deepstream_plugins():
        raise RuntimeError(
            "DeepStream plugins not found. Please verify DeepStream installation:\n"
            "  - Check: ls /opt/nvidia/deepstream/deepstream/lib/gst-plugins/\n"
            "  - Verify: gst-inspect-1.0 nvstreammux"
        )

    pipeline = Gst.Pipeline.new("yolov8-nvmm-pipeline")
    if not pipeline:
        raise RuntimeError("Failed to create Gst.Pipeline")

    # Source: V4L2 (UVC RealSense color). Use DMABUF io-mode=2 when available.
    source = Gst.ElementFactory.make("v4l2src", "source")
    if not source:
        raise RuntimeError("Failed to create v4l2src (is GStreamer v4l2 installed?)")
    source.set_property("device", args.device)
    try:
        source.set_property("io-mode", 2)
    except Exception:
        pass

    # CPU format caps (commonly YUY2 for UVC). Adjust if your device differs.
    capsfilter_cpu = Gst.ElementFactory.make("capsfilter", "caps_cpu")
    if not capsfilter_cpu:
        raise RuntimeError("Failed to create capsfilter (cpu)")
    caps_cpu = Gst.Caps.from_string(
        f"video/x-raw,format=YUY2,width={args.width},height={args.height},framerate={args.fps}/1"
    )
    capsfilter_cpu.set_property("caps", caps_cpu)

    # Convert CPU->NVMM and to NV12
    # nvvidconv accepts CPU memory and can produce NVMM surfaces when requested in caps
    conv_to_nvmm = Gst.ElementFactory.make("nvvidconv", "conv_to_nvmm")
    if not conv_to_nvmm:
        raise RuntimeError("Failed to create nvvidconv (CPU->NVMM)")

    capsfilter_nvmm = Gst.ElementFactory.make("capsfilter", "caps_nvmm")
    if not capsfilter_nvmm:
        raise RuntimeError("Failed to create capsfilter (nvmm)")
    caps_nvmm = Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM),format=NV12,width={args.width},height={args.height},framerate={args.fps}/1"
    )
    capsfilter_nvmm.set_property("caps", caps_nvmm)

    # Stream muxer (DeepStream requires nvstreammux before nvinfer)
    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    if not streammux:
        plugin_path = os.environ.get("GST_PLUGIN_PATH", "not set")
        raise RuntimeError(
            f"Failed to create nvstreammux. GST_PLUGIN_PATH={plugin_path}\n"
            "Try: gst-inspect-1.0 nvstreammux\n"
            "Verify DeepStream plugins exist at: /opt/nvidia/deepstream/deepstream/lib/gst-plugins/"
        )
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", args.width)
    streammux.set_property("height", args.height)
    streammux.set_property("live-source", 1)
    # Keep NVMM
    try:
        streammux.set_property(
            "nvbuf-memory-type", 0
        )  # 0=NVBUF_MEM_DEFAULT (NVMM on Jetson)
    except Exception:
        pass

    # Primary inference (nvinfer) - uses TensorRT engine configured via config-file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
    if not pgie:
        raise RuntimeError("Failed to create nvinfer (primary)")
    pgie.set_property("config-file-path", args.config)

    # Convert for OSD and display (NVMM path)
    conv_post = Gst.ElementFactory.make("nvvideoconvert", "conv_post")
    if not conv_post:
        raise RuntimeError("Failed to create nvvideoconvert")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        raise RuntimeError("Failed to create nvdsosd")

    if args.display:
        sink = Gst.ElementFactory.make("nveglglessink", "sink")
        if not sink:
            raise RuntimeError("Failed to create nveglglessink (display)")
        sink.set_property("sync", False)
    else:
        sink = Gst.ElementFactory.make("fakesink", "sink")
        if not sink:
            raise RuntimeError("Failed to create fakesink (headless)")
        sink.set_property("sync", False)

    # Add elements
    for elem in (
        source,
        capsfilter_cpu,
        conv_to_nvmm,
        capsfilter_nvmm,
        streammux,
        pgie,
        conv_post,
        nvosd,
        sink,
    ):
        pipeline.add(elem)

    # Link linear part up to streammux via requested sink_0 pad
    if not Gst.Element.link(source, capsfilter_cpu):
        raise RuntimeError("Failed to link source -> caps_cpu")
    if not Gst.Element.link(capsfilter_cpu, conv_to_nvmm):
        raise RuntimeError("Failed to link caps_cpu -> nvvidconv")
    if not Gst.Element.link(conv_to_nvmm, capsfilter_nvmm):
        raise RuntimeError("Failed to link nvvidconv -> caps_nvmm")

    # Request sink pad on streammux and link caps_nvmm src pad to it
    src_pad = capsfilter_nvmm.get_static_pad("src")
    if not src_pad:
        raise RuntimeError("Failed to get src pad from caps_nvmm")
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        raise RuntimeError("Failed to get request pad sink_0 from streammux")
    if src_pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link caps_nvmm -> streammux.sink_0")

    # Link rest: streammux -> nvinfer -> nvvideoconvert -> nvdsosd -> sink
    if not Gst.Element.link(streammux, pgie):
        raise RuntimeError("Failed to link streammux -> nvinfer")
    if not Gst.Element.link(pgie, conv_post):
        raise RuntimeError("Failed to link nvinfer -> nvvideoconvert")
    if not Gst.Element.link(conv_post, nvosd):
        raise RuntimeError("Failed to link nvvideoconvert -> nvdsosd")
    if not Gst.Element.link(nvosd, sink):
        raise RuntimeError("Failed to link nvdsosd -> sink")

    return pipeline


def main() -> int:
    args = parse_args()

    # Auto-generate config file if not provided but model-engine is
    if args.config is None:
        if args.model_engine is None:
            print(
                "ERROR: Either --config or --model-engine must be provided",
                file=sys.stderr,
            )
            return 1
        if not os.path.exists(args.model_engine):
            print(
                f"ERROR: Model engine file not found: {args.model_engine}",
                file=sys.stderr,
            )
            return 1
        args.config = generate_config_file(
            args.model_engine,
            num_classes=args.num_classes,
            threshold=args.threshold,
            nms_threshold=args.nms_threshold,
        )
    elif not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1

    set_gst_env(args.log_level)

    # Check for DeepStream plugins before initializing GStreamer
    if not check_deepstream_plugins():
        print(
            "ERROR: DeepStream plugins not found. Cannot proceed.",
            file=sys.stderr,
        )
        print(
            "Please verify DeepStream installation:",
            file=sys.stderr,
        )
        print(
            "  ls /opt/nvidia/deepstream/deepstream/lib/gst-plugins/",
            file=sys.stderr,
        )
        return 1

    GObject.threads_init()
    Gst.init(None)

    try:
        pipeline = build_pipeline(args)
    except Exception as e:
        print(f"Pipeline build failed: {e}", file=sys.stderr)
        return 1

    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    def handle_sigint(signum, frame):
        try:
            pipeline.send_event(Gst.Event.new_eos())
        except Exception:
            loop.quit()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to set pipeline to PLAYING", file=sys.stderr)
        pipeline.set_state(Gst.State.NULL)
        return 1

    print("Pipeline running. Press Ctrl+C to stop.")
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
