# FAST-LIO2 Navigation Plan (Livox Mid-360) for RoboMaster

This document outlines the step-by-step plan for setting up and running navigation using a Livox Mid-360 and FAST-LIO2, leveraging our existing 3D field model (e.g., Autodesk/CAD exported to `.pcd`).

## 1. Prerequisites and Setup (Docker & ROS 2)
- **Environment:** We will build upon the existing CV Docker container.
- **Dockerfile additions (`dockerfile.nav` or similar):**
  - Install `Livox-SDK2`.
  - Install `livox_ros_driver2` (configured for Mid-360).
  - Install ROS 2 dependencies for `FAST_LIO` and a map-matching localization package (e.g., NDT matching or `FAST_LIO_LOCALIZATION`).
- **Hardware:** Livox Mid-360 securely mounted, connected via Ethernet.

## 2. 3D Model, URDF & Critical Constraints
- **URDF Requirements:** We do *not* need a complex 3D URDF for our robot. A simple, basic URDF (even just linking `base_link` to `livox_frame` using the extrinsics) is sufficient. FAST-LIO2 only requires to know where the LiDAR is relative to the center of the robot.
- **Converting the Field Model:** The existing Autodesk/CAD 3D model of the field must be converted into a PointCloud map (`.pcd`). 
  - This is typically done by sampling points from the CAD mesh to generate a dense point cloud. 
  - **Why?** Since the map is static, we only need a `.pcd` to match the real-time lidar sweeps against. This removes the need for active SLAM (mapping) during runtime. Instead of building the map on the fly, the robot will perform **pure localization**, calculating its position (`map -> odom -> base_link`) by comparing live FAST-LIO2 odometry against the pre-generated field `.pcd`.
- **Critical Constraints (RoboMaster Environment Gotchas):**
  - **The "Lid" Height (Z-Slicing):** The 3D-to-2D conversion or 3D map must account for the physical height of the LiDAR. If elements like overhead bars or low-hanging obstacles exist in the CAD but aren't seen by a LiDAR mounted at 0.5m, your map must reflect the actual 2D slice/view at that specific `0.5m` height to avoid false matches.
  - **Dynamic Obstacles:** The CAD map tells you where the field walls are, but it won't include other enemy/friendly robots. You must still run a **Local Costmap** using the live Mid-360 point cloud to "see" dynamic obstacles immediately in front of you and navigate around them safely.

## 3. Jetson Orin Nano Super Specifics
Because we are deploying on a Jetson Orin Nano Super, consider the following platform specifics:
- **Build from Source:** `livox_ros_driver2` and `FAST_LIO` should be built natively on the Jetson (or via the ARM64 Docker container) to leverage ARM optimizations.
- **Why FAST-LIO2?** FAST-LIO2 is an extremely efficient LiDAR-inertial odometry system. Instead of complex, memory-heavy point matching, it relies on a tightly coupled **Iterated Kalman Filter (IKF)**. This allows it to fuse the Mid-360's high-frequency IMU with its point cloud sweeps to provide ultra-fast, highly accurate local odometry (`odom -> base_link`) with minimal CPU overhead on the Jetson.
- **Power Mode:** Run the existing `./power.sh` script on the Jetson host before starting the Docker container. This script sets the Jetson to `MAXN` mode (`nvpmodel -m 0`), forces max clocks (`jetson_clocks`), and spins up the PWM fan to prevent LiDAR odometry drift due to CPU throttling.
- **Network Interrupts:** FAST-LIO2 is highly sensitive to packet drops. Ensure the ethernet connection (`eth0`) to the Mid-360 is prioritized and MTU is correctly set (typically 1500) within the Jetson's network manager.
- **Memory Management:** Jetson Orin Nano shares memory between the CPU and GPU. Point cloud processing can be memory intensive; configure swap space if the container aggressively kills the `localization` nodes during map loading.

## 4. File Structure & Configurations
- **`livox_mid360.json` / `.yaml`:** Network settings (Robot IP, LiDAR IP) and extrinsics.
- **`fast_lio_mid360.yaml`:** FAST-LIO2 configuration tailored for Mid-360 (IMU characteristics, FOV, etc.).
- **Global Map:** The existing 3D field model converted to a format usable by the localization node (typically `.pcd`).
- **`navigation.launch.py`:** A single ROS 2 launch file that brings up:
  1. `livox_ros_driver2`
  2. `FAST_LIO` (real-time odometry)
  3. Map server / Localization node (matching FAST-LIO odometry to the global `.pcd` map)
  4. The custom Python control loop node (`simple_navigator.py`).
- **Python Nodes (e.g., in `yolov8_ros/scripts/Nav/`):**
  - **`simple_navigator.py`:** The main control node containing the state machine and blind-drive logic (template provided below).
  - **`utils.py` (Optional):** Helper functions for math, quaternion to Euler conversions, or parsing specific game-state elements.

## 3. Run Sequence (Step-by-Step)
1. **Build Container:** Rebuild the CV container with the new dependencies (`docker build ...`).
2. **Launch Stack:** Run `ros2 launch <package_name> navigation.launch.py`.
   - This starts the LiDAR, localization, and the control loop node.

## 4. Simple Control Loop Node (Python - Blind Point-to-Point)
We will implement a Python ROS 2 node containing a simple state-machine to blindly drive between the start position and the middle of the field back-to-back, based on the localized position.

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class SimpleNavigator(Node):
    def __init__(self):
        super().__init__('simple_navigator')
        # TODO: Update topics based on actual localization output
        self.odom_sub = self.create_subscription(Odometry, '/localization/odometry', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Define Coordinates
        self.MIDDLE_X, self.MIDDLE_Y = 5.0, 5.0 # TODO: Adjust to actual middle coords
        self.START_X, self.START_Y = 0.0, 0.0   # TODO: Adjust to actual start coords
        self.TOLERANCE = 0.2
        
        self.states = ["GO_TO_MIDDLE", "WAIT_AT_MIDDLE", "GO_TO_START", "WAIT_AT_START"]
        self.current_state = "GO_TO_MIDDLE"
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        # TODO: Extract yaw from quaternion if angular control is needed for blind drive
        
    def reached(self, target_x, target_y):
        dist = math.sqrt((target_x - self.current_x)**2 + (target_y - self.current_y)**2)
        return dist < self.TOLERANCE

    def send_blind_velocity(self, target_x, target_y):
        # Basic P-controller for blind point-to-point driving
        twist = Twist()
        base_speed = 0.5
        
        # Simple directional drive (assuming holonomic or simple diff-drive)
        # TODO: Tune control logic based on actual chassis kinematics
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        
        # extremely simplified example
        twist.linear.x = base_speed if dx > 0 else -base_speed if dx < 0 else 0.0
        twist.linear.y = base_speed if dy > 0 else -base_speed if dy < 0 else 0.0
        
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def control_loop(self):
        # TODO: Determine state transition based on external criteria (e.g., game state, vision)
        if self.current_state == "GO_TO_MIDDLE":
            self.send_blind_velocity(self.MIDDLE_X, self.MIDDLE_Y)
            if self.reached(self.MIDDLE_X, self.MIDDLE_Y):
                self.stop_robot()
                self.current_state = "WAIT_AT_MIDDLE"
                
        elif self.current_state == "WAIT_AT_MIDDLE":
            # TODO: Implement criteria to trigger return to start
            # if should_return():
            #     self.current_state = "GO_TO_START"
            pass # Placeholder
                
        elif self.current_state == "GO_TO_START":
            self.send_blind_velocity(self.START_X, self.START_Y)
            if self.reached(self.START_X, self.START_Y):
                self.stop_robot()
                self.current_state = "WAIT_AT_START"
                
        elif self.current_state == "WAIT_AT_START":
            # TODO: Implement criteria to trigger move to middle
            # if should_move_to_middle():
            #     self.current_state = "GO_TO_MIDDLE"
            pass # Placeholder

def main(args=None):
    rclpy.init(args=args)
    node = SimpleNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
