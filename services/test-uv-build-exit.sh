#!/usr/bin/env bash
# No Docker or ROS needed: exercise the real wrapper with boundary doubles.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT
mkdir -p "$test_dir/services" "$test_dir/workspace/yolo_cpp"
cp "$repo_root/services/uv.sh" "$test_dir/services/uv.sh"
touch "$test_dir/workspace/yolo_cpp/package.xml"
cat > "$test_dir/services/start-container.sh" <<'SH'
shift # image
"$@"
SH

# Limit ROS setup interception to /opt/ros; all other sources run normally.
source() {
  case "$1" in
    /opt/ros/*) return "$TEST_ROS_SETUP_STATUS" ;;
    *) builtin source "$@" ;;
  esac
}
colcon() { return "$TEST_BUILD_STATUS"; }
uv() { return "$TEST_BUILD_STATUS"; }
export -f source colcon uv
export CV_STM_ENV_FILE="$test_dir/absent.env"
cd "$test_dir/workspace"
for TEST_ROS_SETUP_STATUS in 0 1; do
  export TEST_ROS_SETUP_STATUS
  for TEST_BUILD_STATUS in 0 42; do
    export TEST_BUILD_STATUS
    for package in all yolo_cpp another_package; do
      args=(build)
      [[ "$package" = all ]] || args+=("$package")
      status=0
      bash "$test_dir/services/uv.sh" "${args[@]}" || status=$?
      if [[ "$status" != "$TEST_BUILD_STATUS" ]]; then
        echo "FAIL: ROS setup=$TEST_ROS_SETUP_STATUS package=$package expected=$TEST_BUILD_STATUS got=$status" >&2
        exit 1
      fi
    done
  done
done
echo 'PASS: direct and container-forwarded builds preserve success and failure status'
