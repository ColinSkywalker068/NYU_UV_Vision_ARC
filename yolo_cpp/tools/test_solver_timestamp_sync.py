"""Hardware-free camera/attitude timestamp interpolation check."""
import os
import signal
import subprocess
import time

os.environ["ROS_DOMAIN_ID"] = "91"
os.environ["ROS_LOCALHOST_ONLY"] = "1"

import rclpy
from cv_auto_aim_interfaces.msg import AimSolution, Target
from std_msgs.msg import Float32MultiArray


proc = subprocess.Popen(
    [
        "ros2",
        "run",
        "yolo_cpp",
        "aim_solver_node",
        "--ros-args",
        "-p",
        "use_3d_solution:=false",
        "-p",
        "output_mode:=absolute",
        "-p",
        "attitude_sync_tolerance_ms:=30.0",
    ],
    start_new_session=True,
)
rclpy.init()
node = rclpy.create_node("solver_timestamp_sync_test")
target_pub = node.create_publisher(Target, "tracker/target", 10)
attitude_pub = node.create_publisher(Float32MultiArray, "attitude_data", 10)
solutions = []
node.create_subscription(AimSolution, "aim_solution", solutions.append, 10)


def spin(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.01)


def centered_target(stamp) -> Target:
    target = Target()
    target.header.stamp = stamp.to_msg()
    target.has_target = True
    target.image_width = 640
    target.image_height = 480
    target.bbox_xyxy = [300, 220, 340, 260]
    return target


try:
    deadline = time.monotonic() + 15.0
    while target_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        spin(0.1)
    assert target_pub.get_subscription_count(), "solver did not subscribe"

    attitude_pub.publish(Float32MultiArray(data=[0.0, -0.2, 0.0]))
    spin(0.08)
    spin(0.04)
    capture_stamp = node.get_clock().now()
    spin(0.04)
    attitude_pub.publish(Float32MultiArray(data=[0.4, 0.2, 0.0]))
    spin(0.08)

    solutions.clear()
    target_pub.publish(centered_target(capture_stamp))
    spin(0.12)
    assert solutions and solutions[-1].has_solution, "interpolated target was rejected"
    assert 0.08 < solutions[-1].yaw < 0.32, (
        f"expected interpolated yaw, got {solutions[-1].yaw:.4f}"
    )
    assert -0.12 < solutions[-1].pitch < 0.12, (
        f"expected interpolated pitch, got {solutions[-1].pitch:.4f}"
    )

    # The latest attitude remains fresh, but a frame beyond the 30 ms edge
    # tolerance must fail closed instead of combining mismatched timestamps.
    spin(0.06)
    solutions.clear()
    target_pub.publish(centered_target(node.get_clock().now()))
    spin(0.12)
    assert solutions and not solutions[-1].has_solution, (
        "out-of-tolerance frame incorrectly used the latest attitude"
    )
    print("PASS: capture-time attitude interpolation and fail-closed edge tolerance")
finally:
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    node.destroy_node()
    rclpy.shutdown()
