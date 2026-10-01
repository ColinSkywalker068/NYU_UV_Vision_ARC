#!/usr/bin/env python3
"""Read-only check of decoded SP attitude; never opens USB or sends commands."""

import argparse
import math
import os
import time


def fresh_attitude(sample, now_ns, previous_stamp_ns):
    stamp = sample.receive_stamp_ns
    return (
        sample.valid
        and stamp > max(0, previous_stamp_ns)
        and 0 <= now_ns - stamp <= 500_000_000
        and all(math.isfinite(getattr(sample, key))
                for key in ("yaw", "pitch", "roll", "yaw_rate", "pitch_rate"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")

    import rclpy
    from rclpy.node import Node
    from rclpy.parameter_client import AsyncParameterClient
    from rcl_interfaces.msg import ParameterType
    from cv_auto_aim_interfaces.msg import GimbalAttitude

    rclpy.init()
    node = Node(f"cv_typec_link_check_{os.getpid()}")
    deadline = time.monotonic() + args.timeout
    client = AsyncParameterClient(node, "/stm32_serial_bridge_node")
    accepted = 0
    previous_stamp = 0

    def receive(sample):
        nonlocal accepted, previous_stamp
        if fresh_attitude(sample, node.get_clock().now().nanoseconds, previous_stamp):
            previous_stamp = sample.receive_stamp_ns
            accepted += 1

    subscription = node.create_subscription(GimbalAttitude, "/gimbal/attitude", receive, 10)
    try:
        while not client.services_are_ready() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not client.services_are_ready():
            print("[stm-stack] Type C link FAILED: bridge parameter service unavailable")
            return 1
        future = client.get_parameters(["receive_enable", "transmit_enable"])
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not future.done() or future.exception() is not None:
            print("[stm-stack] Type C link FAILED: could not read active RX/TX configuration")
            return 1
        values = future.result().values
        if len(values) != 2 or any(v.type != ParameterType.PARAMETER_BOOL for v in values):
            print("[stm-stack] Type C link FAILED: unexpected bridge parameter types")
            return 1
        if not any(v.bool_value for v in values):
            print("[stm-stack] Type C link not requested: active RX and TX are both disabled")
            return 0
        while accepted < 3 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if accepted >= 3:
            print("[stm-stack] Type C link OK: fresh, valid SP attitude samples received; "
                  "this does not verify motor response or MCU source-clock synchronization")
            return 0
        print("[stm-stack] Type C link FAILED: no fresh valid SP attitude stream. "
              "An existing USB port is not sufficient. Check cv logs, the USB data cable, "
              "serial_port, and installed Type C firmware. CV remains running; "
              "no commands, calibration, controller reset, or flash were performed by this check.")
        return 1
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
