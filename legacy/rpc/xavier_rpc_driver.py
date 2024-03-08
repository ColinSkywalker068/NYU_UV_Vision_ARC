# Custom base implementation for Sentry

import time
from typing import Any, Dict, Optional
from viam.components.base import Base, Vector3
import socket
import sys
from dataclasses import dataclass
import serial
import struct

# Edit the path depending on where you stored these
sys.path.append("/usr/local/lib/python3.11/site-packages")


class SentryDriver(Base):
    # Subclass the Viam base component and implement the required functions
    def __init__(self, name: str):
        port = "/dev/ttyUSB0"
        baudrate = 115200
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        self.HEADER = 0xE7
        super().__init__(name)

    def send_data(self, data):
        try:
            self.ser.write(data)
        except Exception as e:
            print(e)

    # BEGIN SERIAL
    # 5: move straight
    def send_move_straight(self, msg):
        # [int distance (mm), int velocity*1000 (m/s)]

        length = 6
        msg_type = 5
        header = struct.pack("BHBBH", self.HEADER, length, 0x00, 0x00, msg_type)
        body = struct.pack("Hh", int(msg[0]), int(msg[1] * 1000))
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        print("move_s ", body)
        self.send_data(msg)

    # 6: spin chasis
    def send_spin_chassis(self, msg):
        # [angle (deg), velocity *1000(deg/s)]
        length = 6
        msg_type = 6
        header = struct.pack("BHBBH", self.HEADER, length, 0x00, 0x00, msg_type)
        body = struct.pack("hh", int(msg[0]), int(msg[1] * 1000))
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.send_data(msg)

    # 7: set power
    def send_set_power(self, msg):
        # [xLinearPower*100, yLinearPower*100, xAnglePower*100, yAnglePower*100], all in range 0-100
        length = 10
        msg_type = 7
        header = struct.pack("BHBBH", self.HEADER, length, 0x00, 0x00, msg_type)
        body = struct.pack(
            "hhhh",
            int(msg[0] * 100),
            int(msg[1] * 100),
            int(msg[2] * 100),
            int(msg[3] * 100),
        )
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        print("set_pow ", body)
        self.send_data(msg)

    # 8: set_velocity
    def send_set_velocity(self, msg):
        # [xVelocity,yVelocity,yawVelocity,pitchVelocity]
        length = 10
        msg_type = 8
        header = struct.pack("BHBBH", self.HEADER, length, 0x00, 0x00, msg_type)
        body = struct.pack(
            "hhhh",
            int(msg[0] * 100),
            int(msg[1] * 100),
            int(msg[2] * 100),
            int(msg[3] * 100),
        )
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        print("set_vel: ", body)
        self.send_data(msg)

    # 9: stop
    def send_stop_chassis(self, msg):
        length = 3
        msg_type = 9
        header = struct.pack("BHBBH", self.HEADER, length, 0x00, 0x00, msg_type)
        body = struct.pack("?", 1)
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.send_data(msg)

    # END SERIAL

    async def move_straight(
        self,
        distance: int,
        velocity: float,
        extra: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        if distance == 0 or velocity == 0:
            return await self.stop()

        # Convert velocity in mm/s to correct number to send in the CMD_MOVE_FORWARD/BACKWARD command.
        conversion_factor = 0.05
        velocity = velocity * conversion_factor

        # if velocity > 0:
        #     self.position += distance
        # else:
        #     self.position -= distance
        self.send_move_straight(msg=[distance, velocity])

        self.is_stopped = False

    async def spin(
        self,
        angle: float,
        velocity: float,
        extra: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        if angle == 0 or velocity == 0:
            return await self.stop()

        spin_time = angle / velocity

        # Convert from velocity in deg/s to what to send to robot.
        conversion_factor = 0.85
        vel = velocity * conversion_factor

        if velocity > 0:
            self.angle += angle
            self.stand()

        else:
            self.angle -= angle
            self.stand()

        self.send_spin_chassis(msg=[angle, velocity])
        time.sleep(spin_time)

        self.is_stopped = False

    async def set_power(
        self,
        linear: Vector3,
        angular: Vector3,
        extra: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        self.linear_pwr = linear
        self.angular_pwr = angular

        # Convert from power percentage (0-1) to what to send to robot.
        # conversion_factor_a = 30
        conversion_factor_l = 25
        # ang_vel = int(angular.z * conversion_factor_a)
        lin_vel = int(linear.y * conversion_factor_l)

        if lin_vel != 0:
            self.send_set_power(msg=[linear.x, linear.y, 0, 0])

    async def set_velocity(
        self,
        linear: Vector3,
        angular: Vector3,
        extra: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        self.linear_vel = linear
        self.angular_vel = angular

        # Convert from velocity in deg/s (angular) and in mm/s (linear) to what to send to robot.
        conversion_factor_a = 0.85
        conversion_factor_l = 0.25
        ang_vel = angular.z * conversion_factor_a
        # Linear components, where y is forward and x is side to side
        y_lin_vel = linear.y * conversion_factor_l
        x_lin_vel = linear.x * conversion_factor_l

        self.send_set_velocity(msg=[linear.x, linear.y, 0, 0])

    async def stop(self, extra: Optional[dict[str, Any]] = None, **kwargs):
        self.is_stopped = True

        self.send_stop_chassis(msg=[1])

    async def is_moving(self):
        # return not self.is_stopped
        return False

    async def do(self, command: dict[str, Any]):
        pass

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> "Base.Properties":
        """
        Get the base width and turning radius

        Returns:
            Properties: The properties of the base
        """
        p = self.Properties(width_meeters=0.277, turning_radius_meters=0.325026)
        return p
