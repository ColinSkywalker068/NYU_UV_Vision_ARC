cmake_minimum_required(VERSION 3.8)
project(cv_auto_aim_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Armor.msg"
  "msg/Armors.msg"
  "msg/TrackedArmor.msg"
  "msg/TrackedArmors.msg"
  "msg/Target.msg"
  "msg/SpinInfo.msg"
  "msg/AimSolution.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime std_msgs geometry_msgs)

ament_package()
