"""Hardware-free sentry sign and attitude-expiry integration check."""
import os
import signal
import subprocess
import time

os.environ['ROS_DOMAIN_ID'] = '90'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
import rclpy
from std_msgs.msg import Float32MultiArray
from cv_auto_aim_interfaces.msg import Target, AimSolution

proc = subprocess.Popen(['ros2', 'run', 'yolo_cpp', 'aim_solver_node',
    '--ros-args', '-p', 'use_3d_solution:=false', '-p', 'output_mode:=absolute'],
    start_new_session=True)
rclpy.init()
node = rclpy.create_node('sentry_solver_test')
tp = node.create_publisher(Target, 'tracker/target', 10)
ap = node.create_publisher(Float32MultiArray, 'attitude_data', 10)
solutions = []
node.create_subscription(AimSolution, 'aim_solution', solutions.append, 10)


def spin(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)


try:
    end = time.monotonic() + 15
    while tp.get_subscription_count() == 0 and time.monotonic() < end:
        spin(0.1)
    assert tp.get_subscription_count()
    spin(0.5)
    t = Target()
    t.has_target = True
    t.image_width, t.image_height = 640, 480
    t.bbox_xyxy = [400, 100, 440, 140]  # Target right and up of center.
    tp.publish(t)
    spin(0.15)
    assert solutions and not solutions[-1].has_solution, 'absolute aim without attitude'
    ap.publish(Float32MultiArray(data=[0.4, -0.2, 0.0]))
    spin(0.05)
    tp.publish(t)
    spin(0.1)
    assert solutions[-1].has_solution and not solutions[-1].used_3d
    assert solutions[-1].yaw > 0.4 and solutions[-1].pitch > -0.2, 'wrong sentry signs'
    spin(0.3)
    tp.publish(t)
    spin(0.1)
    assert not solutions[-1].has_solution, 'stale attitude accepted'
    print('PASS: absolute requires attitude; 2D sentry right/up signs; 250 ms attitude expiry')
finally:
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    node.destroy_node()
    rclpy.shutdown()
