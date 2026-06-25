#!/usr/bin/env python3
"""Behavior-tree communication adapter for RM handoff compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import rclpy
from auto_aim_interfaces.msg import Armor, Armors
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rm_decision_interfaces.msg import AllRobotHP, GameStatus, RobotControl, RobotStatus
from std_msgs.msg import String


@dataclass
class TimedValue:
    stamp_sec: float
    value: object


class BtCommAdapter(Node):
    def __init__(self) -> None:
        super().__init__("bt_comm_adapter")

        self.declare_parameter("autoaim_target_topic", "auto_aim_target_pos")
        self.declare_parameter("armors_topic", "/detector/armors")
        self.declare_parameter("input_cmd_vel_topic", "/cmd_vel_chassis")
        self.declare_parameter("output_cmd_vel_topic", "/cmd_vel_chassis_bt")
        self.declare_parameter("robot_control_topic", "/robot_control")
        self.declare_parameter("game_status_topic", "/game_status")
        self.declare_parameter("robot_status_topic", "/robot_status")
        self.declare_parameter("all_robot_hp_topic", "/all_robot_hp")

        self.declare_parameter("publish_detector_adapter", True)
        self.declare_parameter("publish_referee_fallback", True)
        self.declare_parameter("publish_cmd_vel_mux", True)

        self.declare_parameter("detector_rate_hz", 10.0)
        self.declare_parameter("cmd_vel_rate_hz", 30.0)
        self.declare_parameter("referee_rate_hz", 2.0)
        self.declare_parameter("detector_timeout_sec", 0.5)
        self.declare_parameter("cmd_vel_timeout_sec", 0.5)
        self.declare_parameter("robot_control_timeout_sec", 0.5)

        self.declare_parameter("default_game_progress", 4)
        self.declare_parameter("default_stage_remain_time", 180)
        self.declare_parameter("default_robot_id", 7)
        self.declare_parameter("default_current_hp", 600)
        self.declare_parameter("default_shooter_heat", 0)
        self.declare_parameter("default_team_color", 0)
        self.declare_parameter("default_is_attacked", False)

        self.autoaim_target_topic = str(
            self.get_parameter("autoaim_target_topic")
            .get_parameter_value()
            .string_value
        )
        self.armors_topic = str(
            self.get_parameter("armors_topic").get_parameter_value().string_value
        )
        self.input_cmd_vel_topic = str(
            self.get_parameter("input_cmd_vel_topic").get_parameter_value().string_value
        )
        self.output_cmd_vel_topic = str(
            self.get_parameter("output_cmd_vel_topic")
            .get_parameter_value()
            .string_value
        )
        self.robot_control_topic = str(
            self.get_parameter("robot_control_topic").get_parameter_value().string_value
        )
        self.game_status_topic = str(
            self.get_parameter("game_status_topic").get_parameter_value().string_value
        )
        self.robot_status_topic = str(
            self.get_parameter("robot_status_topic").get_parameter_value().string_value
        )
        self.all_robot_hp_topic = str(
            self.get_parameter("all_robot_hp_topic").get_parameter_value().string_value
        )

        self.publish_detector_adapter = (
            self.get_parameter("publish_detector_adapter")
            .get_parameter_value()
            .bool_value
        )
        self.publish_referee_fallback = (
            self.get_parameter("publish_referee_fallback")
            .get_parameter_value()
            .bool_value
        )
        self.publish_cmd_vel_mux = (
            self.get_parameter("publish_cmd_vel_mux").get_parameter_value().bool_value
        )

        self.detector_timeout_sec = (
            self.get_parameter("detector_timeout_sec")
            .get_parameter_value()
            .double_value
        )
        self.cmd_vel_timeout_sec = (
            self.get_parameter("cmd_vel_timeout_sec").get_parameter_value().double_value
        )
        self.robot_control_timeout_sec = (
            self.get_parameter("robot_control_timeout_sec")
            .get_parameter_value()
            .double_value
        )

        self.latest_target: TimedValue | None = None
        self.latest_cmd_vel: TimedValue | None = None
        self.latest_robot_control: TimedValue | None = None

        self.armors_pub = self.create_publisher(Armors, self.armors_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.output_cmd_vel_topic, 10)
        self.game_status_pub = self.create_publisher(
            GameStatus, self.game_status_topic, 10
        )
        self.robot_status_pub = self.create_publisher(
            RobotStatus, self.robot_status_topic, 10
        )
        self.all_robot_hp_pub = self.create_publisher(
            AllRobotHP, self.all_robot_hp_topic, 10
        )

        self.create_subscription(
            String, self.autoaim_target_topic, self.on_autoaim_target, 10
        )
        self.create_subscription(Twist, self.input_cmd_vel_topic, self.on_cmd_vel, 10)
        self.create_subscription(
            RobotControl, self.robot_control_topic, self.on_robot_control, 10
        )

        detector_period = 1.0 / max(
            1.0,
            self.get_parameter("detector_rate_hz").get_parameter_value().double_value,
        )
        cmd_period = 1.0 / max(
            1.0,
            self.get_parameter("cmd_vel_rate_hz").get_parameter_value().double_value,
        )
        referee_period = 1.0 / max(
            0.5,
            self.get_parameter("referee_rate_hz").get_parameter_value().double_value,
        )

        self.create_timer(detector_period, self.publish_armors)
        self.create_timer(cmd_period, self.publish_muxed_cmd_vel)
        self.create_timer(referee_period, self.publish_referee_fallbacks)

        self.get_logger().info(
            "BT comm adapter started: "
            f"vision {self.autoaim_target_topic} -> {self.armors_topic}, "
            f"{self.input_cmd_vel_topic} + {self.robot_control_topic} -> {self.output_cmd_vel_topic}"
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def is_fresh(self, timed_value: TimedValue | None, timeout_sec: float) -> bool:
        return (
            timed_value is not None
            and (self.now_sec() - timed_value.stamp_sec) <= timeout_sec
        )

    def on_autoaim_target(self, msg: String) -> None:
        try:
            parts = [part.strip() for part in msg.data.split(",")]
            if len(parts) != 4:
                raise ValueError(f"expected 4 fields, got {len(parts)}")

            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            target_id = float(parts[3])
            self.latest_target = TimedValue(self.now_sec(), (x, y, z, target_id))
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to parse auto_aim_target_pos '{msg.data}': {exc}"
            )

    def on_cmd_vel(self, msg: Twist) -> None:
        self.latest_cmd_vel = TimedValue(self.now_sec(), msg)

    def on_robot_control(self, msg: RobotControl) -> None:
        self.latest_robot_control = TimedValue(self.now_sec(), msg)

    def publish_armors(self) -> None:
        if not self.publish_detector_adapter:
            return
        if self.count_publishers(self.armors_topic) > 1:
            return

        msg = Armors()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gimbal"

        if self.is_fresh(self.latest_target, self.detector_timeout_sec):
            x, y, z, target_id = self.latest_target.value
            if any(abs(v) > 1e-6 for v in (x, y, z)) and abs(target_id) > 1e-6:
                armor = Armor()
                armor.number = str(int(round(target_id)))
                armor.type = "vision_adapter"
                armor.distance_to_image_center = 0.0
                armor.pose.position.x = float(x)
                armor.pose.position.y = float(y)
                armor.pose.position.z = float(z)
                armor.pose.orientation.w = 1.0
                msg.armors.append(armor)

        self.armors_pub.publish(msg)

    def publish_muxed_cmd_vel(self) -> None:
        if not self.publish_cmd_vel_mux:
            return

        msg = Twist()
        if self.is_fresh(self.latest_cmd_vel, self.cmd_vel_timeout_sec):
            msg.linear.x = float(self.latest_cmd_vel.value.linear.x)
            msg.linear.y = float(self.latest_cmd_vel.value.linear.y)
            msg.linear.z = float(self.latest_cmd_vel.value.linear.z)
            msg.angular.x = float(self.latest_cmd_vel.value.angular.x)
            msg.angular.y = float(self.latest_cmd_vel.value.angular.y)
            msg.angular.z = 0.0

        if self.is_fresh(self.latest_robot_control, self.robot_control_timeout_sec):
            msg.angular.z = float(self.latest_robot_control.value.chassis_spin_vel)

        self.cmd_vel_pub.publish(msg)

    def publish_referee_fallbacks(self) -> None:
        if not self.publish_referee_fallback:
            return

        if self.count_publishers(self.game_status_topic) <= 1:
            game_status = GameStatus()
            game_status.game_progress = int(
                self.get_parameter("default_game_progress")
                .get_parameter_value()
                .integer_value
            )
            game_status.stage_remain_time = int(
                self.get_parameter("default_stage_remain_time")
                .get_parameter_value()
                .integer_value
            )
            self.game_status_pub.publish(game_status)

        if self.count_publishers(self.robot_status_topic) <= 1:
            robot_status = RobotStatus()
            robot_status.robot_id = int(
                self.get_parameter("default_robot_id")
                .get_parameter_value()
                .integer_value
            )
            robot_status.current_hp = int(
                self.get_parameter("default_current_hp")
                .get_parameter_value()
                .integer_value
            )
            robot_status.shooter_heat = int(
                self.get_parameter("default_shooter_heat")
                .get_parameter_value()
                .integer_value
            )
            robot_status.team_color = bool(
                self.get_parameter("default_team_color")
                .get_parameter_value()
                .integer_value
            )
            robot_status.is_attacked = (
                self.get_parameter("default_is_attacked")
                .get_parameter_value()
                .bool_value
            )
            self.robot_status_pub.publish(robot_status)

        if self.count_publishers(self.all_robot_hp_topic) <= 1:
            all_robot_hp = AllRobotHP()
            all_robot_hp.red_1_robot_hp = 100
            all_robot_hp.red_2_robot_hp = 100
            all_robot_hp.red_3_robot_hp = 100
            all_robot_hp.red_4_robot_hp = 100
            all_robot_hp.red_5_robot_hp = 200
            all_robot_hp.red_7_robot_hp = 200
            all_robot_hp.red_outpost_hp = 20
            all_robot_hp.red_base_hp = 1000
            all_robot_hp.blue_1_robot_hp = 100
            all_robot_hp.blue_2_robot_hp = 100
            all_robot_hp.blue_3_robot_hp = 100
            all_robot_hp.blue_4_robot_hp = 200
            all_robot_hp.blue_5_robot_hp = 200
            all_robot_hp.blue_7_robot_hp = 200
            all_robot_hp.blue_outpost_hp = 1000
            all_robot_hp.blue_base_hp = 1000
            self.all_robot_hp_pub.publish(all_robot_hp)


def is_shutdown_noise(exc: Exception) -> bool:
    text = str(exc)
    return (
        "context is not valid" in text
        or "rcl_shutdown already called" in text
        or "failed to initialize wait set" in text
    )


def main() -> None:
    rclpy.init()
    node = BtCommAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if not is_shutdown_noise(exc):
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
