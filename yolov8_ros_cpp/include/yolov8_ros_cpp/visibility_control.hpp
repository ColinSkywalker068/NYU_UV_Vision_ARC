#ifndef YOLOV8_ROS_CPP__VISIBILITY_CONTROL_HPP_
#define YOLOV8_ROS_CPP__VISIBILITY_CONTROL_HPP_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define YOLOV8_ROS_CPP_EXPORT __attribute__ ((dllexport))
    #define YOLOV8_ROS_CPP_IMPORT __attribute__ ((dllimport))
  #else
    #define YOLOV8_ROS_CPP_EXPORT __declspec(dllexport)
    #define YOLOV8_ROS_CPP_IMPORT __declspec(dllimport)
  #endif
  #ifdef YOLOV8_ROS_CPP_BUILDING_LIBRARY
    #define YOLOV8_ROS_CPP_PUBLIC YOLOV8_ROS_CPP_EXPORT
  #else
    #define YOLOV8_ROS_CPP_PUBLIC YOLOV8_ROS_CPP_IMPORT
  #endif
  #define YOLOV8_ROS_CPP_PUBLIC_TYPE YOLOV8_ROS_CPP_PUBLIC
  #define YOLOV8_ROS_CPP_LOCAL
#else
  #define YOLOV8_ROS_CPP_EXPORT __attribute__ ((visibility("default")))
  #define YOLOV8_ROS_CPP_IMPORT
  #if __GNUC__ >= 4
    #define YOLOV8_ROS_CPP_PUBLIC __attribute__ ((visibility("default")))
    #define YOLOV8_ROS_CPP_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define YOLOV8_ROS_CPP_PUBLIC
    #define YOLOV8_ROS_CPP_LOCAL
  #endif
  #define YOLOV8_ROS_CPP_PUBLIC_TYPE
#endif

#endif
