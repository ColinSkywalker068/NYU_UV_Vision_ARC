"""Run inside a sourced ROS workspace. Isolated domains; no hardware or TX."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SERVER = """
import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from cv_auto_aim_interfaces.msg import GimbalAttitude
rclpy.init()
node = Node('stm32_serial_bridge_node')
mode = sys.argv[1]
node.declare_parameter('receive_enable', mode != 'disabled')
node.declare_parameter('transmit_enable', False)
pub = node.create_publisher(GimbalAttitude, '/gimbal/attitude', 10)
def publish():
    if mode in ('missing', 'disabled'):
        return
    sample = GimbalAttitude()
    sample.valid = True
    sample.receive_stamp_ns = node.get_clock().now().nanoseconds
    if mode == 'stale':
        sample.receive_stamp_ns -= 10_000_000_000
    pub.publish(sample)
timer = node.create_timer(0.02, publish)
try:
    rclpy.spin(node)
except ExternalShutdownException:
    pass
"""


class TypeCLinkROS(unittest.TestCase):
    def test_live_and_unhealthy_links(self):
        checker = Path(__file__).with_name('check_typec_link.py')
        for domain, mode, expected in ((80, 'fresh', 0), (81, 'missing', 1),
                                      (82, 'stale', 1), (83, 'disabled', 0)):
            with self.subTest(mode=mode), tempfile.TemporaryFile(mode='w+t') as logs:
                env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1')
                server = subprocess.Popen([sys.executable, '-u', '-c', SERVER, mode],
                                          env=env, stdout=logs, stderr=logs)
                try:
                    result = subprocess.run([sys.executable, str(checker), '--timeout', '8'],
                                            env=env, capture_output=True, text=True, timeout=20)
                    self.assertIsNone(server.poll(), 'simulated ROS publisher exited')
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    expected_text = ('link OK' if mode == 'fresh' else
                                     'not requested' if mode == 'disabled' else 'link FAILED')
                    self.assertIn(expected_text, result.stdout)
                finally:
                    server.terminate()
                    server.wait(timeout=5)
                    logs.seek(0)
                    if server.returncode not in (0, -15):
                        print(logs.read(), file=sys.stderr)


if __name__ == '__main__':
    unittest.main()
