#!/usr/bin/env bash
# Wrapper script for 'docker run': https://docs.docker.com/engine/reference/commandline/run/

# Get the absolute path of this script
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Jetson
grep -aq "NVIDIA" /proc/device-tree/model 2>/dev/null
if [ $? -eq 0 ]; then
	# Check for V4L2 camera devices
	V4L2_DEVICES=""
	for i in {0..9}
	do
		if [ -a "/dev/video$i" ]; then
			V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
		fi
	done

	# Check for display monitor
	DISPLAY_DEVICE=""
	if [ -n "$DISPLAY" ]; then
		# give docker root user X11 permissions
		sudo xhost +si:localuser:root

		# enable SSH X11 forwarding inside container (https://stackoverflow.com/q/48235040)
		XAUTH=/tmp/.docker.xauth
		xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
		chmod 777 $XAUTH
		DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
	fi

	# Add Hugging Face token if it exists in the environment
	HUGGINGFACE=""
	if [ -n "$HUGGINGFACE_TOKEN" ]; then
		HUGGINGFACE="$HUGGINGFACE --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN"
	fi

	# Add interactive flag if this is not ran by a systemd service
	if [ -z "$INVOCATION_ID" ]; then
		INTERACTIVE="-it"
	else
		INTERACTIVE=""
	fi

	# Check if sudo is needed
	if [ $(id -u) -eq 0 ] || id -nG "$USER" | grep -qw "docker"; then
		SUDO=""
	else
		SUDO="sudo"
	fi

	# Check if default image is needed
	if [ $# -eq 0 ]; then
		DEFAULT_IMAGE="nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-realsense-wrapper"
	else
		DEFAULT_IMAGE=""
	fi

	# This file shows what Jetson board is running
	# /proc or /sys files aren't mountable into docker
	cat /proc/device-tree/model > /tmp/nv_jetson_model

	set -x
	$SUDO docker run --runtime nvidia --rm --privileged --network host \
		--volume /tmp/argus_socket:/tmp/argus_socket \
		--volume /etc/enctune.conf:/etc/enctune.conf \
		--volume /etc/nv_tegra_release:/etc/nv_tegra_release \
		--volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
		--volume /var/run/dbus:/var/run/dbus \
		--volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
		--volume /var/run/docker.sock:/var/run/docker.sock \
		--volume $ROOT/data:/data \
		--volume $ROOT/../../.ssh:/root/.ssh \
		--volume /home:/home \
		--volume /usr/local/bin:/usr/local/bin \
		--volume /dev:/dev \
		--device /dev/snd \
		--device /dev/bus/usb \
		--env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
		$INTERACTIVE \
		$DATA_VOLUME \
		$DISPLAY_DEVICE \
		$V4L2_DEVICES \
		$HUGGINGFACE \
		$DEFAULT_IMAGE \
		"$@"
# MacOS
else
	winpty docker run -it --rm --privileged --network host \
		--volume $ROOT/..:/home/ultraviolet/CV_Detection \
		--env DISPLAY=host.docker.internal:0.0 \
		nyuultraviolet/cv-dockerfile:macos
fi