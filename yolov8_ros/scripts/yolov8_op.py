#!/usr/bin/env python3
import argparse
import os
import sys
import signal
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject  # noqa: E402


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="YOLOv8 DeepStream/GStreamer NVMM pipeline")
	parser.add_argument("--device", type=str, default="/dev/video0", help="V4L2 device path")
	parser.add_argument("--width", type=int, default=640, help="Input width")
	parser.add_argument("--height", type=int, default=480, help="Input height")
	parser.add_argument("--fps", type=int, default=30, help="Input FPS")
	parser.add_argument("--config", type=str, required=True, help="nvinfer config-file-path")
	parser.add_argument("--display", action="store_true", help="Enable display sink (nveglglessink)")
	parser.add_argument("--log-level", type=str, default="2", help="GStreamer log level (0-7)")
	return parser.parse_args()


def set_gst_env(log_level: str) -> None:
	os.environ.setdefault("GST_DEBUG", log_level)
	os.environ.setdefault("GST_PLUGIN_PATH", "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/")
	os.environ.setdefault("GST_PLUGIN_PATH_1_0", "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/")


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
		raise RuntimeError("Failed to create nvstreammux")
	streammux.set_property("batch-size", 1)
	streammux.set_property("width", args.width)
	streammux.set_property("height", args.height)
	streammux.set_property("live-source", 1)
	# Keep NVMM
	try:
		streammux.set_property("nvbuf-memory-type", 0)  # 0=NVBUF_MEM_DEFAULT (NVMM on Jetson)
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
	set_gst_env(args.log_level)
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


