"""Hardware-free SP/ST integration check. Run in sourced ROS container."""
import os
import pty
import struct
import subprocess
import time

os.environ['ROS_DOMAIN_ID'] = '89'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
import rclpy
from std_msgs.msg import Bool, Float32MultiArray


def crc(data):
    value = 0xffff
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x8408 if value & 1 else 0)
    return struct.pack('<H', value)


master, slave = pty.openpty()
os.set_blocking(master, False)
port = os.ttyname(slave)
assert port.startswith('/dev/pts/')  # Never permit a physical serial device.
proc = subprocess.Popen([
    'ros2', 'run', 'yolo_cpp', 'stm32_serial_bridge', '--ros-args',
    '-p', 'serial_port:=' + port, '-p', 'transmit_enable:=true',
    '-p', 'command_output_mode:=absolute'], start_new_session=True)
rclpy.init()
node = rclpy.create_node('sp_pty_test')
attitudes = []
node.create_subscription(Float32MultiArray, 'attitude_data',
                         lambda m: attitudes.append(list(m.data)), 10)
op = node.create_publisher(Bool, 'override', 10)
ap = node.create_publisher(Float32MultiArray, 'bounding_box_angles', 10)


def spin(seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        rclpy.spin_once(node, timeout_sec=0.02)


def read_frames():
    data = b''
    while True:
        try:
            data += os.read(master, 4096)
        except BlockingIOError:
            break
    assert len(data) % 29 == 0, data
    frames = [data[i:i+29] for i in range(0, len(data), 29)]
    for frame in frames:
        assert frame[:2] == b'SP' and crc(frame[:-2]) == frame[-2:]
    return frames


try:
    deadline = time.monotonic() + 15
    while op.get_subscription_count() == 0 and time.monotonic() < deadline:
        spin(0.1)
    assert op.get_subscription_count(), 'bridge failed to start'
    spin(0.5)
    sp = b'SP\x01' + struct.pack('<9fH', 1, 0, 0, 0, 0.25, 0, -0.15, 0, 18, 0)
    os.write(master, sp + crc(sp))
    spin(0.3)
    assert attitudes and abs(attitudes[-1][0] - 0.25) < 1e-5
    count = len(attitudes)
    st = b'ST' + bytes(35)
    os.write(master, st + crc(st))
    spin(0.3)
    assert len(attitudes) == count, 'ST must not publish zero attitude'
    bad = sp + b'\x00\x00'
    os.write(master, bad)
    spin(0.2)
    assert len(attitudes) == count, 'bad CRC accepted'
    op.publish(Bool(data=True))
    spin(0.05)
    ap.publish(Float32MultiArray(data=[0.1, -0.2]))
    spin(0.1)
    frames = read_frames()
    aim = [f for f in frames if f[2] == 1]
    assert aim, 'no aim packet'
    values = struct.unpack('<6f', aim[-1][3:27])
    assert abs(values[0]-0.1) < 1e-5 and abs(values[3]+0.2) < 1e-5
    assert values[1:3] == (0, 0) and values[4:] == (0, 0)
    spin(0.4)
    assert any(f[2] == 0 for f in read_frames()), 'watchdog did not relinquish control'
    op.publish(Bool(data=True))
    spin(0.05)
    ap.publish(Float32MultiArray(data=[0.1, 0.2]))
    spin(0.08)
    read_frames()
    op.publish(Bool(data=False))
    spin(0.1)
    assert any(f[2] == 0 for f in read_frames()), 'target loss not sent'
    print('PASS: SP CRC/order/radians; SP feedback; ST isolation; bad CRC; stale stop; target loss')
finally:
    import signal
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    node.destroy_node()
    rclpy.shutdown()
    os.close(master)
    os.close(slave)
