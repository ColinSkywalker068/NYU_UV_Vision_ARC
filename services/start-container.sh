#!/usr/bin/env bash
# Wrapper script for 'docker run': https://docs.docker.com/engine/reference/commandline/run/

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$ROOT/.." >/dev/null 2>&1 && pwd)"

if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
	echo "Usage: services/start-container.sh <image> [command...]" >&2
	return 2 2>/dev/null || exit 2
fi

IMAGE_NAME="$1"
CMD_ARGS=("${@:2}")
CONTAINER_NAME=""
RUN_ARGS=()
DOCKER_CMD=(docker)

is_dry_run() {
	case "${CV_DOCKER_DRY_RUN:-}" in
		1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
		*) return 1 ;;
	esac
}

print_command() {
	printf '[start-container] dry-run:'
	printf ' %q' "$@"
	printf '\n'
}

sanitize_name() {
	echo "$1" | tr '/:@' '-' | tr -cd '[:alnum:]_.-'
}

get_container_name() {
	local image_name="$1"
	if [ -n "${UV_CONTAINER_NAME:-}" ]; then
		echo "$UV_CONTAINER_NAME"
	else
		echo "uv-$(sanitize_name "$image_name")"
	fi
}

select_jetson_docker_cmd() {
	local current_user="${USER:-}"
	if [ -z "$current_user" ]; then
		current_user="$(id -un 2>/dev/null || true)"
	fi

	if [ "$(id -u)" -eq 0 ] || { [ -n "$current_user" ] && id -nG "$current_user" 2>/dev/null | grep -qw "docker"; }; then
		DOCKER_CMD=(docker)
	else
		DOCKER_CMD=(sudo docker)
	fi
}

container_exists() {
	local container_name="$1"
	"${DOCKER_CMD[@]}" container inspect "$container_name" >/dev/null 2>&1
}

container_running() {
	local container_name="$1"
	[ "$("${DOCKER_CMD[@]}" inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null)" = "true" ]
}

start_existing_container() {
	local container_name="$1"

	if container_running "$container_name"; then
		return 0
	fi

	if ! "${DOCKER_CMD[@]}" start "$container_name" >/dev/null; then
		echo "Failed to start existing container: ${container_name}" >&2
		echo "If this container is stale, remove it with: docker rm ${container_name}" >&2
		return 1
	fi

	if ! container_running "$container_name"; then
		echo "Container ${container_name} exited immediately after docker start." >&2
		echo "Its original command may have completed. Remove it with: docker rm ${container_name}" >&2
		echo "Then rerun uv docker to create a fresh container." >&2
		return 1
	fi
}

reuse_container() {
	local container_name="$1"
	shift
	local exec_cmd=("$@")
	local -a exec_flags=()

	if [ -z "${INVOCATION_ID:-}" ]; then
		exec_flags=(-it)
	fi

	start_existing_container "$container_name" || return $?

	if [ ${#exec_cmd[@]} -eq 0 ]; then
		"${DOCKER_CMD[@]}" exec "${exec_flags[@]}" "$container_name" bash
	else
		"${DOCKER_CMD[@]}" exec "${exec_flags[@]}" "$container_name" "${exec_cmd[@]}"
	fi
}

reset_run_args() {
	RUN_ARGS=(--name "$CONTAINER_NAME" --privileged --network host)
}

add_interactive_arg() {
	if [ -z "${INVOCATION_ID:-}" ]; then
		RUN_ARGS+=(-it)
	fi
}

add_volume_if_exists() {
	local source_path="$1"
	local target_path="$2"
	local mode="${3:-}"

	if [ ! -e "$source_path" ]; then
		return
	fi

	if [ -n "$mode" ]; then
		RUN_ARGS+=(--volume "${source_path}:${target_path}:${mode}")
	else
		RUN_ARGS+=(--volume "${source_path}:${target_path}")
	fi
}

add_device_if_exists() {
	local device_path="$1"
	if [ -e "$device_path" ]; then
		RUN_ARGS+=(--device "$device_path")
	fi
}

add_env_value() {
	local name="$1"
	local value="$2"
	RUN_ARGS+=(--env "${name}=${value}")
}

add_env_if_set() {
	local name="$1"
	if [ -n "${!name+x}" ]; then
		RUN_ARGS+=(--env "$name")
	fi
}

add_pylon_mount_if_available() {
	add_volume_if_exists /opt/pylon /opt/pylon ro
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

add_cplusplus_env_args() {
	local image_name="$1"
	if ! is_cplusplus_context "$image_name"; then
		return
	fi

	local gst_plugin_path="/opt/nvidia/deepstream/deepstream-6.4/lib/gst-plugins:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/opt/pylon/lib/gstreamer-1.0:/opt/pylon/lib/gst-plugins"
	local ld_library_path="/opt/nvidia/deepstream/deepstream-6.4/lib:/opt/nvidia/deepstream/deepstream/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/opt/pylon/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/aarch64-linux-gnu"

	add_env_value CV_CPLUSPLUS_GST_PLUGIN_PATH "$gst_plugin_path"
	add_env_value CV_CPLUSPLUS_LD_LIBRARY_PATH "$ld_library_path"
	add_env_value GST_PLUGIN_PATH "$gst_plugin_path"
	add_env_value GST_PLUGIN_PATH_1_0 "$gst_plugin_path"
	add_env_value LD_LIBRARY_PATH "$ld_library_path"
	add_env_value GST_REGISTRY /tmp/gst-registry-deepstream.bin
}

add_cv_stm_env_args() {
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
		CV_STM_PREVIEW_MAX_FPS
		CV_STM_REQUIRE_SERIAL
		CV_STM_EXTRA_LAUNCH_ARGS
		CV_STM_BASLER_EXPOSURE_US
		CV_STM_BASLER_GAIN_DB
		CV_STM_BASLER_FPS
		CV_STM_BASLER_UNCAPPED_FPS
		CV_STM_BASLER_PIXEL_FORMAT
		CV_STM_BASLER_SOURCE_WIDTH
		CV_STM_BASLER_SOURCE_HEIGHT
		CV_STM_DEEPSTREAM_PRE_MUX_RESIZE
		CV_STM_DEEPSTREAM_PRE_MUX_WIDTH
		CV_STM_DEEPSTREAM_PRE_MUX_HEIGHT
		CV_STM_DEEPSTREAM_PROFILE_STAGES
		CV_STM_DEEPSTREAM_PROFILE_STAGE_INTERVAL
		CV_STM_DEEPSTREAM_BAYER_CONVERSION_BACKEND
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
		ROS_DOMAIN_ID
		ROS_LOCALHOST_ONLY
		YOLO_CUDA_BAYER_SYNC_MEMCPY
	)
	local name
	for name in "${env_names[@]}"; do
		add_env_if_set "$name"
	done
}

prepare_cplusplus_command() {
	local image_name="$1"
	if ! is_cplusplus_context "$image_name"; then
		return
	fi

	local env_setup='export GST_PLUGIN_PATH="$CV_CPLUSPLUS_GST_PLUGIN_PATH"; export GST_PLUGIN_PATH_1_0="$GST_PLUGIN_PATH"; export LD_LIBRARY_PATH="$CV_CPLUSPLUS_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"; export GST_REGISTRY=/tmp/gst-registry-deepstream.bin; rm -rf /root/.cache/gstreamer-1.0'

	if [[ ${#CMD_ARGS[@]} -eq 0 || (${#CMD_ARGS[@]} -eq 1 && "${CMD_ARGS[0]}" == "bash") ]]; then
		CMD_ARGS=(bash -lc "${env_setup}; exec bash -i")
	elif [[ ${#CMD_ARGS[@]} -ge 3 && "${CMD_ARGS[0]}" == "bash" && "${CMD_ARGS[1]}" == "-lc" ]]; then
		CMD_ARGS=(bash -lc "${env_setup}; ${CMD_ARGS[2]}" "${CMD_ARGS[@]:3}")
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

	local power_script="$REPO_ROOT/services/power.sh"
	if [ ! -f "$power_script" ]; then
		echo "services/power.sh not found at $power_script, skipping power setup"
		return
	fi

	echo "Running services/power.sh before starting CV RealSense container..."
	bash "$power_script"
}

write_jetson_model_file() {
	local model_file="/tmp/cv_detection_nv_jetson_model"
	rm -rf "$model_file" 2>/dev/null || true
	if cat /proc/device-tree/model > "$model_file" 2>/dev/null; then
		printf '%s' "$model_file"
	else
		echo "Unable to write $model_file; continuing without Jetson model bind mount" >&2
		rm -f "$model_file" 2>/dev/null || true
	fi
}

add_display_args() {
	if [ -z "${DISPLAY:-}" ]; then
		return
	fi

	if ! is_dry_run; then
		sudo xhost +si:localuser:root
	fi

	local xauth="/tmp/.docker.xauth"
	add_env_value DISPLAY "$DISPLAY"
	add_volume_if_exists /tmp/.X11-unix /tmp/.X11-unix
	add_volume_if_exists "$xauth" "$xauth"
	add_env_value XAUTHORITY "$xauth"
}

start_or_reuse_container() {
	if is_dry_run; then
		print_command "${DOCKER_CMD[@]}" run "${RUN_ARGS[@]}" "$IMAGE_NAME" "${CMD_ARGS[@]}"
		return 0
	fi

	if container_exists "$CONTAINER_NAME"; then
		reuse_container "$CONTAINER_NAME" "${CMD_ARGS[@]}"
	else
		"${DOCKER_CMD[@]}" run "${RUN_ARGS[@]}" "$IMAGE_NAME" "${CMD_ARGS[@]}"
	fi
}

configure_common_unix_args() {
	reset_run_args
	add_interactive_arg
	add_cplusplus_env_args "$IMAGE_NAME"
	add_cv_stm_env_args
}

configure_jetson_args() {
	local jetson_model_file=""

	select_jetson_docker_cmd
	configure_common_unix_args

	RUN_ARGS+=(--runtime nvidia)
	add_volume_if_exists /tmp/argus_socket /tmp/argus_socket
	add_volume_if_exists /etc/enctune.conf /etc/enctune.conf ro
	add_volume_if_exists /etc/nv_tegra_release /etc/nv_tegra_release ro
	if ! is_dry_run; then
		jetson_model_file="$(write_jetson_model_file)"
		add_volume_if_exists "$jetson_model_file" /tmp/nv_jetson_model ro
	fi
	add_volume_if_exists /var/run/dbus /var/run/dbus
	add_volume_if_exists /var/run/avahi-daemon/socket /var/run/avahi-daemon/socket
	add_volume_if_exists /var/run/docker.sock /var/run/docker.sock
	add_volume_if_exists "$REPO_ROOT/data" /data
	add_volume_if_exists "$REPO_ROOT/../.ssh" /root/.ssh
	add_volume_if_exists /home /home
	add_volume_if_exists /usr/local/bin /usr/local/bin
	add_volume_if_exists /dev /dev
	add_device_if_exists /dev/snd
	for device in /dev/video{0..9}; do
		add_device_if_exists "$device"
	done
	add_env_value RMW_IMPLEMENTATION "${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
	add_env_if_set HUGGINGFACE_TOKEN
	add_pylon_mount_if_available
	add_display_args
}

configure_generic_linux_args() {
	DOCKER_CMD=(docker)
	configure_common_unix_args
	add_volume_if_exists "$REPO_ROOT" /home/ultraviolet/CV_Detection
	add_volume_if_exists /dev /dev
	add_volume_if_exists /home /home
	add_pylon_mount_if_available
}

configure_macos_args() {
	DOCKER_CMD=(docker)
	configure_common_unix_args
	add_volume_if_exists "$REPO_ROOT" /home/ultraviolet/CV_Detection
	add_env_value DISPLAY host.docker.internal:0.0
}

configure_windows_args() {
	if command -v winpty >/dev/null 2>&1; then
		DOCKER_CMD=(winpty docker)
	else
		DOCKER_CMD=(docker)
	fi
	reset_run_args
	add_interactive_arg
	add_cv_stm_env_args
	add_volume_if_exists "$REPO_ROOT" /home/ultraviolet/CV_Detection
}

CONTAINER_NAME="$(get_container_name "$IMAGE_NAME")"
prepare_cplusplus_command "$IMAGE_NAME"

if grep -aq "NVIDIA" /proc/device-tree/model 2>/dev/null; then
	if ! is_dry_run; then
		run_power_script_if_needed "$IMAGE_NAME"
	fi
	configure_jetson_args
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
	configure_generic_linux_args
elif [[ "$OSTYPE" == "darwin"* ]]; then
	configure_macos_args
else
	configure_windows_args
fi

start_or_reuse_container
