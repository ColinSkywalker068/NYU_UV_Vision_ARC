cmake_minimum_required(VERSION 3.8)
project(rm_decision_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AllRobotHP.msg"
  "msg/FriendLocation.msg"
  "msg/GameStatus.msg"
  "msg/RFID.msg"
  "msg/RobotControl.msg"
  "msg/RobotStatus.msg"
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
