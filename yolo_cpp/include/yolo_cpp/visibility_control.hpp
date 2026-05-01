#ifndef YOLO_CPP__VISIBILITY_CONTROL_HPP_
#define YOLO_CPP__VISIBILITY_CONTROL_HPP_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define YOLO_CPP_EXPORT __attribute__ ((dllexport))
    #define YOLO_CPP_IMPORT __attribute__ ((dllimport))
  #else
    #define YOLO_CPP_EXPORT __declspec(dllexport)
    #define YOLO_CPP_IMPORT __declspec(dllimport)
  #endif
  #ifdef YOLO_CPP_BUILDING_LIBRARY
    #define YOLO_CPP_PUBLIC YOLO_CPP_EXPORT
  #else
    #define YOLO_CPP_PUBLIC YOLO_CPP_IMPORT
  #endif
  #define YOLO_CPP_PUBLIC_TYPE YOLO_CPP_PUBLIC
  #define YOLO_CPP_LOCAL
#else
  #define YOLO_CPP_EXPORT __attribute__ ((visibility("default")))
  #define YOLO_CPP_IMPORT
  #if __GNUC__ >= 4
    #define YOLO_CPP_PUBLIC __attribute__ ((visibility("default")))
    #define YOLO_CPP_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define YOLO_CPP_PUBLIC
    #define YOLO_CPP_LOCAL
  #endif
  #define YOLO_CPP_PUBLIC_TYPE
#endif

#endif
