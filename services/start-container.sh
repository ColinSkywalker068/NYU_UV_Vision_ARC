#!/usr/bin/env bash
# Wrapper script for 'docker run': https://docs.docker.com/engine/reference/commandline/run/

# Get the absolute path of this script
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$(cd "$ROOT/.." >/dev/null 2>&1 && pwd)"
IMAGE_NAME="$1"
CMD_ARGS=("${@:2}")

sanitize_name() {
	echo "$1" | tr '/:@' '-' | tr -cd '[:alnum:]_.-'
}

get_container_name() {
	local image_name="$1"
	if [ -n "$UV_CONTAINER_NAME" ]; then
		echo "$UV_CONTAINER_NAME"
	else
		echo "uv-$(sanitize_name "$image_name")"
	fi
}

container_exists() {
	local docker_cmd="$1"
	local container_name="$2"
	$docker_cmd container inspect "$container_name" >/dev/null 2>&1
}

container_running() {
	local docker_cmd="$1"
	local container_name="$2"
	[ "$($docker_cmd inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null)" = "true" ]
}

reuse_container() {
	local docker_cmd="$1"
	local container_name="$2"
	shift 2
	local exec_cmd=("$@")

	if [ ${#exec_cmd[@]} -eq 0 ]; then
		if container_running "$docker_cmd" "$container_name"; then
			$docker_cmd exec -it "$container_name" bash
		else
			$docker_cmd start -ai "$container_name"
		fi
	else
		if ! container_running "$docker_cmd" "$container_name"; then
			$docker_cmd start "$container_name" >/dev/null
		fi
		$docker_cmd exec -it "$container_name" "${exec_cmd[@]}"
	fi
}

get_pylon_mount_arg() {
	if [ -d "/opt/pylon" ]; then
		echo "--volume /opt/pylon:/opt/pylon:ro"
	fi
}

is_cplusplus_context() {
	local image_name="$1"
	if [[ "$image_name" == *"cplusplus"* ]]; then
		return 0
	fi
	if [ -n "${CV_STM_IMAGE:-}" ] || [ -n "${CV_STM_LAUNCH_PROFILE:-}" ]; then
		return 0
	fi
	return 1
}

get_cplusplus_env_args() {
	local image_name="$1"
	if ! is_cplusplus_context "$image_name"; then
		return
	fi
	local gst_plugin_path="/opt/nvidia/deepstream/deepstream-6.4/lib/gst-plugins:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/opt/pylon/lib/gstreamer-1.0:/opt/pylon/lib/gst-plugins"
	local ld_library_path="/opt/nvidia/deepstream/deepstream-6.4/lib:/opt/nvidia/deepstream/deepstream/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/opt/pylon/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/aarch64-linux-gnu"
	echo "--env CV_CPLUSPLUS_GST_PLUGIN_PATH=$gst_plugin_path --env CV_CPLUSPLUS_LD_LIBRARY_PATH=$ld_library_path --env GST_PLUGIN_PATH=$gst_plugin_path --env GST_PLUGIN_PATH_1_0=$gst_plugin_path --env LD_LIBRARY_PATH=$ld_library_path --env GST_REGISTRY=/tmp/gst-registry-deepstream.bin"
}

get_cv_stm_env_args() {
	local env_names=(
		CV_STM_REPO_ROOT
		CV_STM_IMAGE
		CV_STM_MODEL_PATH
		CV_STM_LAUNCH_PROFILE
		CV_STM_TARGET_COLOR
		CV_STM_USE_MODEL_COLOR_CLASSES
		CV_STM_ENABLE_COLOR_VERIFICATION
		CV_STM_CAMERA_TYPE
		CV_STM_DETECTOR_BACKEND
		CV_STM_ALLOW_NATIVE_TENSORRT_FALLBACK
		CV_STM_DEBUG_PUBLISH_IMAGE_RAW
		CV_STM_SAVE_TARGET_FRAMES
		CV_STM_TARGET_FRAME_SAVE_DIR
		CV_STM_TARGET_FRAME_SAVE_MAX
		CV_STM_TARGET_FRAME_JPEG_QUALITY
		CV_STM_SERIAL_PORT
		CV_STM_SERIAL_VERBOSE
		CV_STM_SHOW
		CV_STM_REQUIRE_SERIAL
		CV_STM_EXTRA_LAUNCH_ARGS
		CV_STM_BASLER_EXPOSURE_US
		CV_STM_BASLER_GAIN_DB
		CV_STM_BASLER_FPS
		CV_STM_BASLER_UNCAPPED_FPS
		CV_STM_BASLER_PIXEL_FORMAT
		CV_STM_BASLER_SOURCE_WIDTH
		CV_STM_BASLER_SOURCE_HEIGHT
		CV_STM_BASLER_USE_FULL_SENSOR
		CV_STM_BASLER_FLIP
		CV_STM_REALSENSE_WIDTH
		CV_STM_REALSENSE_HEIGHT
		CV_STM_REALSENSE_FPS
		CV_STM_REALSENSE_EXPOSURE
		CV_STM_REALSENSE_AUTO_EXPOSURE
		CV_STM_REALSENSE_USE_POLL
		CV_STM_REALSENSE_USE_MAX_NATIVE_FPS
		CV_STM_HEALTH_TIMEOUT_S
	)
	local name
	for name in "${env_names[@]}"; do
		if [ -n "${!name+x}" ]; then
			printf -- "--env %s " "$name"
		fi
	done
}

prepare_cplusplus_command() {
	local image_name="$1"
	if ! is_cplusplus_context "$image_name"; then
		return
	fi
	if [[ ${#CMD_ARGS[@]} -eq 0 || (${#CMD_ARGS[@]} -eq 1 && "${CMD_ARGS[0]}" == "bash") ]]; then
		CMD_ARGS=(bash -lc 'export GST_PLUGIN_PATH="$CV_CPLUSPLUS_GST_PLUGIN_PATH"; export GST_PLUGIN_PATH_1_0="$GST_PLUGIN_PATH"; export LD_LIBRARY_PATH="$CV_CPLUSPLUS_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"; export GST_REGISTRY=/tmp/gst-registry-deepstream.bin; rm -rf /root/.cache/gstreamer-1.0; exec bash -i')
	elif [[ ${#CMD_ARGS[@]} -ge 2 && "${CMD_ARGS[0]}" == "bash" && "${CMD_ARGS[1]}" == "-lc" ]]; then
		local user_cmd="${CMD_ARGS[*]:2}"
		CMD_ARGS=(bash -lc "export GST_PLUGIN_PATH=\"\$CV_CPLUSPLUS_GST_PLUGIN_PATH\"; export GST_PLUGIN_PATH_1_0=\"\$GST_PLUGIN_PATH\"; export LD_LIBRARY_PATH=\"\$CV_CPLUSPLUS_LD_LIBRARY_PATH:\${LD_LIBRARY_PATH:-}\"; export GST_REGISTRY=/tmp/gst-registry-deepstream.bin; rm -rf /root/.cache/gstreamer-1.0; $user_cmd")
	fi
}

should_run_power_script() {
	local image_name="$1"
	[[ "$image_name" == *"realsense"* ]]
}

run_power_script_if_needed() {
	local image_name="$1"
	if ! should_run_power_script "$image_name"; then
		return
	fi
	local power_script="$REPO_ROOT/power.sh"
	if [ ! -f "$power_script" ]; then
		echo "power.sh not found at $power_script, skipping power setup"
		return
	fi
	echo "Running power.sh before starting CV RealSense container..."
	bash "$power_script"
}

# Jetson
grep -aq "NVIDIA" /proc/device-tree/model 2>/dev/null
if [ $? -eq 0 ]; then
	run_power_script_if_needed "$IMAGE_NAME"
	prepare_cplusplus_command "$IMAGE_NAME"
	CONTAINER_NAME="$(get_container_name "$IMAGE_NAME")"
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
	DOCKER_CMD="${SUDO:+$SUDO }docker"
	PYLON_MOUNT="$(get_pylon_mount_arg)"
	CPP_ENV="$(get_cplusplus_env_args "$IMAGE_NAME")"
	CV_STM_ENV="$(get_cv_stm_env_args)"

	# This file shows what Jetson board is running
	# /proc or /sys files aren't mountable into docker
	if ! cat /proc/device-tree/model > /tmp/nv_jetson_model 2>/dev/null; then
		echo "Unable to write /tmp/nv_jetson_model; continuing without Jetson model bind mount" >&2
		rm -f /tmp/nv_jetson_model 2>/dev/null || true
	fi

	if container_exists "$DOCKER_CMD" "$CONTAINER_NAME"; then
		reuse_container "$DOCKER_CMD" "$CONTAINER_NAME" "${CMD_ARGS[@]}"
	else
		$DOCKER_CMD run --name "$CONTAINER_NAME" --runtime nvidia --privileged --network host \
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
			$PYLON_MOUNT \
			$CPP_ENV \
			$CV_STM_ENV \
			"$IMAGE_NAME" "${CMD_ARGS[@]}"
	fi
# Generic Linux (KR260, Ubuntu, etc.)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
	prepare_cplusplus_command "$IMAGE_NAME"
	CONTAINER_NAME="$(get_container_name "$IMAGE_NAME")"
	PYLON_MOUNT="$(get_pylon_mount_arg)"
	CPP_ENV="$(get_cplusplus_env_args "$IMAGE_NAME")"
	CV_STM_ENV="$(get_cv_stm_env_args)"
	if container_exists "docker" "$CONTAINER_NAME"; then
		reuse_container "docker" "$CONTAINER_NAME" "${CMD_ARGS[@]}"
	else
		docker run -it --name "$CONTAINER_NAME" --privileged --network host \
			--volume "$REPO_ROOT":/home/ultraviolet/CV_Detection \
			--volume /dev:/dev \
			--volume /home:/home \
			$PYLON_MOUNT \
			$CPP_ENV \
			$CV_STM_ENV \
			"$IMAGE_NAME" "${CMD_ARGS[@]}"
	fi
# MacOS
elif [[ "$OSTYPE" == "darwin"* ]]; then
	prepare_cplusplus_command "$IMAGE_NAME"
	CONTAINER_NAME="$(get_container_name "$IMAGE_NAME")"
	CPP_ENV="$(get_cplusplus_env_args "$IMAGE_NAME")"
	CV_STM_ENV="$(get_cv_stm_env_args)"
	if container_exists "docker" "$CONTAINER_NAME"; then
		reuse_container "docker" "$CONTAINER_NAME" "${CMD_ARGS[@]}"
	else
		docker run -it --name "$CONTAINER_NAME" --privileged --network host \
			--volume "$REPO_ROOT":/home/ultraviolet/CV_Detection \
			--env DISPLAY=host.docker.internal:0.0 \
			$CPP_ENV \
			$CV_STM_ENV \
			"$IMAGE_NAME" "${CMD_ARGS[@]}"
	fi
# Fallback (Windows)
else
	CV_STM_ENV="$(get_cv_stm_env_args)"
	if command -v winpty >/dev/null 2>&1; then
		winpty docker run -it --name "$(get_container_name "$IMAGE_NAME")" --privileged --network host \
			--volume "$REPO_ROOT":/home/ultraviolet/CV_Detection \
			$CV_STM_ENV \
			"$IMAGE_NAME" "${CMD_ARGS[@]}"
	else
		docker run -it --name "$(get_container_name "$IMAGE_NAME")" --privileged --network host \
			--volume "$REPO_ROOT":/home/ultraviolet/CV_Detection \
			$CV_STM_ENV \
			"$IMAGE_NAME" "${CMD_ARGS[@]}"
	fi
fi
