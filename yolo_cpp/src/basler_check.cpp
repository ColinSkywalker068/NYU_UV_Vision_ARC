/// @file basler_check.cpp
/// @brief Standalone Basler/Pylon diagnostic matching the C++ camera backend sequence.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#ifdef HAVE_PYLON
#include <GenApi/GenApi.h>
#include <pylon/PylonIncludes.h>
#endif

namespace
{

struct Config
{
  int device_index{0};
  bool device_index_set{false};
  std::string serial;
  int imgsz_width{640};
  int imgsz_height{640};
  double exposure_us{5000.0};
  double gain_db{5.0};
  int fps{90};
  bool uncapped_fps{true};
  bool flip{false};
  bool use_full_sensor{true};
  bool centered_roi{true};
  bool skip_config{false};
  bool probe_after_step{false};
  int grab_count{3};
  int timeout_ms{1000};
  std::string save_frame_path;
};

void print_status(const char * level, const std::string & label, const std::string & detail = "")
{
  std::cout << '[' << level << "] " << label;
  if (!detail.empty()) {
    std::cout << ": " << detail;
  }
  std::cout << '\n';
}

std::string bool_text(bool value)
{
  return value ? "true" : "false";
}

template<typename T>
std::string to_string_any(const T & value)
{
  std::ostringstream out;
  out << value;
  return out.str();
}

bool arg_is(const std::string & arg, const std::string & name)
{
  return arg == "--" + name;
}

std::optional<std::string> arg_value(const std::string & arg, const std::string & name)
{
  const std::string prefix = "--" + name + "=";
  if (arg.rfind(prefix, 0) == 0) {
    return arg.substr(prefix.size());
  }
  return std::nullopt;
}

std::string require_next(int argc, char ** argv, int & index, const std::string & name)
{
  if (index + 1 >= argc) {
    throw std::runtime_error("Missing value for --" + name);
  }
  ++index;
  return argv[index];
}

std::string read_value(int argc, char ** argv, int & index, const std::string & arg, const std::string & name)
{
  if (const auto value = arg_value(arg, name)) {
    return *value;
  }
  return require_next(argc, argv, index, name);
}

int parse_int(const std::string & text, const std::string & name)
{
  try {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception & exc) {
    throw std::runtime_error("Invalid integer for --" + name + ": " + text + " (" + exc.what() + ")");
  }
}

double parse_double(const std::string & text, const std::string & name)
{
  try {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception & exc) {
    throw std::runtime_error("Invalid number for --" + name + ": " + text + " (" + exc.what() + ")");
  }
}

void print_usage()
{
  std::cout
    << "Usage: ros2 run yolo_cpp basler_check [options]\n\n"
    << "Mirrors the C++ Basler runtime path:\n"
    << "  PylonInitialize -> Enumerate/CreateFirstDevice -> Open -> GenICam config\n"
    << "  -> ImageFormatConverter BGR8 -> StartGrabbing(LatestImageOnly, ProvidedByInstantCamera)\n"
    << "  -> RetrieveResult -> Convert -> optional cv::flip\n\n"
    << "Options:\n"
    << "  --mirror-stm-launch          Use stm.launch-style flip=true with current defaults\n"
    << "  --device-index N             Device index from Pylon enumeration (default 0)\n"
    << "  --serial SERIAL              Select device by serial number\n"
    << "  --fps N                      Basler FPS parameter/default actual_fps (default 90)\n"
    << "  --uncapped-fps               Set AcquisitionFrameRateEnable=false (default)\n"
    << "  --capped-fps                 Set AcquisitionFrameRateEnable=true and AcquisitionFrameRate=fps\n"
    << "  --exposure-us VALUE          ExposureTime value (default 5000.0)\n"
    << "  --gain-db VALUE              Gain value (default 5.0)\n"
    << "  --flip / --no-flip           Apply same 180-degree cv::flip as runtime capture\n"
    << "  --use-full-sensor            Skip Width/Height/Offset ROI writes (default)\n"
    << "  --crop-to-imgsz              Mirror basler_use_full_sensor=false ROI writes\n"
    << "  --imgsz-width N              ROI width when cropping (default 640)\n"
    << "  --imgsz-height N             ROI height when cropping (default 640)\n"
    << "  --centered-roi               Center ROI offsets when cropping (default)\n"
    << "  --no-centered-roi            Do not write ROI offsets\n"
    << "  --skip-config                Open/grab only; no GenICam writes\n"
    << "  --probe-after-step           Grab one frame after each GenICam write\n"
    << "  --grab-count N               Final number of frames to grab (default 3)\n"
    << "  --timeout-ms N               RetrieveResult timeout (default 1000)\n"
    << "  --save-frame PATH            Save first final grabbed frame\n";
}

Config parse_args(int argc, char ** argv)
{
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else if (arg_is(arg, "mirror-stm-launch")) {
      config.flip = true;
    } else if (arg_is(arg, "device-index") || arg_value(arg, "device-index")) {
      config.device_index = parse_int(read_value(argc, argv, i, arg, "device-index"), "device-index");
      config.device_index_set = true;
    } else if (arg_is(arg, "serial") || arg_value(arg, "serial")) {
      config.serial = read_value(argc, argv, i, arg, "serial");
    } else if (arg_is(arg, "fps") || arg_value(arg, "fps")) {
      config.fps = parse_int(read_value(argc, argv, i, arg, "fps"), "fps");
    } else if (arg_is(arg, "uncapped-fps")) {
      config.uncapped_fps = true;
    } else if (arg_is(arg, "capped-fps")) {
      config.uncapped_fps = false;
    } else if (arg_is(arg, "exposure-us") || arg_value(arg, "exposure-us")) {
      config.exposure_us = parse_double(read_value(argc, argv, i, arg, "exposure-us"), "exposure-us");
    } else if (arg_is(arg, "gain-db") || arg_value(arg, "gain-db")) {
      config.gain_db = parse_double(read_value(argc, argv, i, arg, "gain-db"), "gain-db");
    } else if (arg_is(arg, "flip")) {
      config.flip = true;
    } else if (arg_is(arg, "no-flip")) {
      config.flip = false;
    } else if (arg_is(arg, "use-full-sensor")) {
      config.use_full_sensor = true;
    } else if (arg_is(arg, "crop-to-imgsz")) {
      config.use_full_sensor = false;
    } else if (arg_is(arg, "imgsz-width") || arg_value(arg, "imgsz-width")) {
      config.imgsz_width = parse_int(read_value(argc, argv, i, arg, "imgsz-width"), "imgsz-width");
    } else if (arg_is(arg, "imgsz-height") || arg_value(arg, "imgsz-height")) {
      config.imgsz_height = parse_int(read_value(argc, argv, i, arg, "imgsz-height"), "imgsz-height");
    } else if (arg_is(arg, "centered-roi")) {
      config.centered_roi = true;
    } else if (arg_is(arg, "no-centered-roi")) {
      config.centered_roi = false;
    } else if (arg_is(arg, "skip-config")) {
      config.skip_config = true;
    } else if (arg_is(arg, "probe-after-step")) {
      config.probe_after_step = true;
    } else if (arg_is(arg, "grab-count") || arg_value(arg, "grab-count")) {
      config.grab_count = parse_int(read_value(argc, argv, i, arg, "grab-count"), "grab-count");
    } else if (arg_is(arg, "timeout-ms") || arg_value(arg, "timeout-ms")) {
      config.timeout_ms = parse_int(read_value(argc, argv, i, arg, "timeout-ms"), "timeout-ms");
    } else if (arg_is(arg, "save-frame") || arg_value(arg, "save-frame")) {
      config.save_frame_path = read_value(argc, argv, i, arg, "save-frame");
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (config.grab_count <= 0) {
    throw std::runtime_error("--grab-count must be positive");
  }
  if (config.timeout_ms <= 0) {
    throw std::runtime_error("--timeout-ms must be positive");
  }
  return config;
}

#ifdef HAVE_PYLON
struct PylonRuntime
{
  PylonRuntime()
  {
    Pylon::PylonInitialize();
  }

  ~PylonRuntime()
  {
    Pylon::PylonTerminate();
  }
};

std::string pylon_exception_text(const GenICam::GenericException & exc)
{
  std::ostringstream out;
  out << exc.GetDescription();
  if (exc.GetSourceFileName() != nullptr) {
    out << " [" << exc.GetSourceFileName() << ':' << exc.GetSourceLine() << ']';
  }
  return out.str();
}

std::string exception_text()
{
  try {
    throw;
  } catch (const GenICam::GenericException & exc) {
    return std::string("GenICam::GenericException: ") + pylon_exception_text(exc);
  } catch (const std::exception & exc) {
    return std::string(typeid(exc).name()) + ": " + exc.what();
  } catch (...) {
    return "non-std exception";
  }
}

template<typename Fn>
bool run_step(const std::string & label, Fn && fn)
{
  print_status("STEP", label);
  try {
    fn();
    print_status("PASS", label);
    return true;
  } catch (...) {
    print_status("FAIL", label, exception_text());
    return false;
  }
}

std::string device_info_text(const Pylon::CDeviceInfo & info)
{
  std::ostringstream out;
  out << "model=" << info.GetModelName()
      << ", serial=" << info.GetSerialNumber()
      << ", class=" << info.GetDeviceClass()
      << ", friendly=" << info.GetFriendlyName()
      << ", full=" << info.GetFullName();
  return out.str();
}

template<typename NodeT>
std::string node_read_value(GenApi::INodeMap & node_map, const char * name)
{
  try {
    NodeT node(node_map.GetNode(name));
    if (!node) {
      return "(missing)";
    }
    if (!GenApi::IsReadable(node)) {
      return "(not readable)";
    }
    return to_string_any(node->GetValue());
  } catch (...) {
    return std::string("(") + exception_text() + ")";
  }
}

std::string enum_node_read_value(GenApi::INodeMap & node_map, const char * name)
{
  try {
    GenApi::CEnumerationPtr node(node_map.GetNode(name));
    if (!node) {
      return "(missing)";
    }
    if (!GenApi::IsReadable(node)) {
      return "(not readable)";
    }
    GenApi::CEnumEntryPtr entry(node->GetCurrentEntry());
    if (!entry || !GenApi::IsReadable(entry)) {
      return to_string_any(node->GetIntValue());
    }
    std::ostringstream out;
    out << entry->GetSymbolic() << " (" << entry->GetValue() << ")";
    return out.str();
  } catch (...) {
    return std::string("(") + exception_text() + ")";
  }
}

void print_node_snapshot(GenApi::INodeMap & node_map)
{
  print_status("INFO", "Width", node_read_value<GenApi::CIntegerPtr>(node_map, "Width"));
  print_status("INFO", "Height", node_read_value<GenApi::CIntegerPtr>(node_map, "Height"));
  print_status("INFO", "OffsetX", node_read_value<GenApi::CIntegerPtr>(node_map, "OffsetX"));
  print_status("INFO", "OffsetY", node_read_value<GenApi::CIntegerPtr>(node_map, "OffsetY"));
  print_status("INFO", "PixelFormat", enum_node_read_value(node_map, "PixelFormat"));
  print_status("INFO", "ExposureAuto", enum_node_read_value(node_map, "ExposureAuto"));
  print_status("INFO", "ExposureTime", node_read_value<GenApi::CFloatPtr>(node_map, "ExposureTime"));
  print_status("INFO", "Gain", node_read_value<GenApi::CFloatPtr>(node_map, "Gain"));
  print_status(
    "INFO",
    "AcquisitionFrameRateEnable",
    node_read_value<GenApi::CBooleanPtr>(node_map, "AcquisitionFrameRateEnable"));
  print_status(
    "INFO",
    "AcquisitionFrameRate",
    node_read_value<GenApi::CFloatPtr>(node_map, "AcquisitionFrameRate"));
}

template<typename NodeT, typename ValueT>
bool set_if_writable_logged(GenApi::INodeMap & node_map, const char * name, const ValueT & value)
{
  std::ostringstream label;
  label << "set " << name << "=" << value;
  print_status("STEP", label.str());
  try {
    NodeT node(node_map.GetNode(name));
    if (!node) {
      print_status("SKIP", name, "node missing; runtime would skip");
      return true;
    }
    if (!GenApi::IsWritable(node)) {
      print_status("SKIP", name, "node not writable; runtime would skip");
      return true;
    }
    node->SetValue(value);
    print_status("PASS", name, node_read_value<NodeT>(node_map, name));
    return true;
  } catch (...) {
    print_status("FAIL", name, exception_text());
    return false;
  }
}

bool set_enum_if_writable_logged(GenApi::INodeMap & node_map, const char * name, const char * value)
{
  std::ostringstream label;
  label << "set " << name << "=" << value;
  print_status("STEP", label.str());
  try {
    GenApi::CEnumerationPtr node(node_map.GetNode(name));
    if (!node) {
      print_status("SKIP", name, "node missing; runtime would skip");
      return true;
    }
    if (!GenApi::IsWritable(node)) {
      print_status("SKIP", name, "node not writable; runtime would skip");
      return true;
    }
    GenApi::CEnumEntryPtr entry(node->GetEntryByName(value));
    if (!entry || !GenApi::IsReadable(entry)) {
      print_status("SKIP", name, "enum entry missing or not readable; runtime would skip");
      return true;
    }
    node->SetIntValue(entry->GetValue());
    print_status("PASS", name, enum_node_read_value(node_map, name));
    return true;
  } catch (...) {
    print_status("FAIL", name, exception_text());
    return false;
  }
}

bool grab_once(Pylon::CInstantCamera & camera, Pylon::CImageFormatConverter & converter, const Config & config)
{
  try {
    if (camera.IsGrabbing()) {
      camera.StopGrabbing();
    }
    camera.StartGrabbing(
      Pylon::GrabStrategy_LatestImageOnly,
      Pylon::GrabLoop_ProvidedByInstantCamera);

    Pylon::CGrabResultPtr result;
    camera.RetrieveResult(config.timeout_ms, result, Pylon::TimeoutHandling_Return);
    if (!result) {
      print_status("FAIL", "probe grab", "RetrieveResult returned null");
      camera.StopGrabbing();
      return false;
    }
    if (!result->GrabSucceeded()) {
      std::ostringstream detail;
      detail << "code=" << result->GetErrorCode()
             << ", description=" << result->GetErrorDescription();
      print_status("FAIL", "probe grab", detail.str());
      camera.StopGrabbing();
      return false;
    }

    Pylon::CPylonImage image;
    converter.Convert(image, result);
    cv::Mat frame(
      static_cast<int>(result->GetHeight()),
      static_cast<int>(result->GetWidth()),
      CV_8UC3,
      image.GetBuffer());
    if (config.flip) {
      cv::flip(frame, frame, -1);
    }
    std::ostringstream detail;
    detail << frame.cols << 'x' << frame.rows << " step=" << frame.step;
    print_status("PASS", "probe grab", detail.str());
    camera.StopGrabbing();
    return true;
  } catch (...) {
    print_status("FAIL", "probe grab", exception_text());
    try {
      if (camera.IsGrabbing()) {
        camera.StopGrabbing();
      }
    } catch (...) {
    }
    return false;
  }
}

bool maybe_probe_after_step(
  Pylon::CInstantCamera & camera, Pylon::CImageFormatConverter & converter, const Config & config)
{
  if (!config.probe_after_step) {
    return true;
  }
  return grab_once(camera, converter, config);
}

void configure_converter(Pylon::CImageFormatConverter & converter)
{
  converter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
  converter.OutputBitAlignment = Pylon::OutputBitAlignment_MsbAligned;
}

bool configure_camera(
  Pylon::CInstantCamera & camera, Pylon::CImageFormatConverter & converter, const Config & config)
{
  if (config.skip_config) {
    print_status("INFO", "config", "skipping all GenICam writes");
    return true;
  }

  auto & node_map = camera.GetNodeMap();
  bool ok = true;
  const auto apply = [&](bool step_ok) {
    ok = ok && step_ok;
    if (step_ok) {
      ok = maybe_probe_after_step(camera, converter, config) && ok;
    }
  };

  apply(set_if_writable_logged<GenApi::CBooleanPtr>(
    node_map, "AcquisitionFrameRateEnable", !config.uncapped_fps));
  if (!config.uncapped_fps) {
    apply(set_if_writable_logged<GenApi::CFloatPtr>(
      node_map, "AcquisitionFrameRate", static_cast<double>(config.fps)));
  }
  apply(set_enum_if_writable_logged(node_map, "ExposureAuto", "Off"));
  apply(set_if_writable_logged<GenApi::CFloatPtr>(node_map, "ExposureTime", config.exposure_us));
  apply(set_if_writable_logged<GenApi::CFloatPtr>(node_map, "Gain", config.gain_db));

  if (!config.use_full_sensor) {
    GenApi::CIntegerPtr width_node(node_map.GetNode("Width"));
    GenApi::CIntegerPtr height_node(node_map.GetNode("Height"));
    GenApi::CIntegerPtr offset_x(node_map.GetNode("OffsetX"));
    GenApi::CIntegerPtr offset_y(node_map.GetNode("OffsetY"));
    if (width_node) {
      apply(set_if_writable_logged<GenApi::CIntegerPtr>(
        node_map,
        "Width",
        std::min(width_node->GetMax(), static_cast<int64_t>(config.imgsz_width))));
    }
    if (height_node) {
      apply(set_if_writable_logged<GenApi::CIntegerPtr>(
        node_map,
        "Height",
        std::min(height_node->GetMax(), static_cast<int64_t>(config.imgsz_height))));
    }
    if (config.centered_roi && offset_x && offset_y && width_node && height_node) {
      const int64_t centered_x =
        std::max<int64_t>(0, (width_node->GetMax() - width_node->GetValue()) / 2);
      const int64_t centered_y =
        std::max<int64_t>(0, (height_node->GetMax() - height_node->GetValue()) / 2);
      if (GenApi::IsWritable(offset_x)) {
        apply(set_if_writable_logged<GenApi::CIntegerPtr>(node_map, "OffsetX", centered_x));
      }
      if (GenApi::IsWritable(offset_y)) {
        apply(set_if_writable_logged<GenApi::CIntegerPtr>(node_map, "OffsetY", centered_y));
      }
    }
  }

  return ok;
}

bool grab_final_frames(
  Pylon::CInstantCamera & camera, Pylon::CImageFormatConverter & converter, const Config & config)
{
  print_status(
    "STEP",
    "StartGrabbing LatestImageOnly/ProvidedByInstantCamera for final capture");
  try {
    if (camera.IsGrabbing()) {
      camera.StopGrabbing();
    }
    camera.StartGrabbing(
      Pylon::GrabStrategy_LatestImageOnly,
      Pylon::GrabLoop_ProvidedByInstantCamera);
    print_status("PASS", "StartGrabbing");
  } catch (...) {
    print_status("FAIL", "StartGrabbing", exception_text());
    return false;
  }

  bool ok = true;
  bool saved = false;
  for (int frame_index = 0; frame_index < config.grab_count; ++frame_index) {
    try {
      Pylon::CGrabResultPtr result;
      camera.RetrieveResult(config.timeout_ms, result, Pylon::TimeoutHandling_Return);
      if (!result) {
        print_status("FAIL", "frame " + std::to_string(frame_index), "RetrieveResult returned null");
        ok = false;
        continue;
      }
      if (!result->GrabSucceeded()) {
        std::ostringstream detail;
        detail << "code=" << result->GetErrorCode()
               << ", description=" << result->GetErrorDescription();
        print_status("FAIL", "frame " + std::to_string(frame_index), detail.str());
        ok = false;
        continue;
      }
      Pylon::CPylonImage image;
      converter.Convert(image, result);
      cv::Mat frame(
        static_cast<int>(result->GetHeight()),
        static_cast<int>(result->GetWidth()),
        CV_8UC3,
        image.GetBuffer());
      if (config.flip) {
        cv::flip(frame, frame, -1);
      }
      std::ostringstream detail;
      detail << frame.cols << 'x' << frame.rows << " step=" << frame.step;
      print_status("PASS", "frame " + std::to_string(frame_index), detail.str());

      if (!config.save_frame_path.empty() && !saved) {
        if (cv::imwrite(config.save_frame_path, frame)) {
          print_status("PASS", "saved frame", config.save_frame_path);
        } else {
          print_status("FAIL", "saved frame", config.save_frame_path);
          ok = false;
        }
        saved = true;
      }
    } catch (...) {
      print_status("FAIL", "frame " + std::to_string(frame_index), exception_text());
      ok = false;
    }
  }

  try {
    if (camera.IsGrabbing()) {
      camera.StopGrabbing();
    }
    print_status("PASS", "StopGrabbing");
  } catch (...) {
    print_status("FAIL", "StopGrabbing", exception_text());
    ok = false;
  }
  return ok;
}

int run_with_pylon(const Config & config)
{
  print_status("INFO", "basler_check", "C++ runtime-sequence diagnostic");
  print_status(
    "INFO",
    "config",
    "fps=" + std::to_string(config.fps) +
    ", uncapped_fps=" + bool_text(config.uncapped_fps) +
    ", exposure_us=" + to_string_any(config.exposure_us) +
    ", gain_db=" + to_string_any(config.gain_db) +
    ", flip=" + bool_text(config.flip) +
    ", use_full_sensor=" + bool_text(config.use_full_sensor));

  bool ok = true;
  try {
    PylonRuntime pylon_runtime;
    (void)pylon_runtime;
    auto & factory = Pylon::CTlFactory::GetInstance();
    Pylon::DeviceInfoList_t devices;
    factory.EnumerateDevices(devices);
    if (devices.empty()) {
      print_status("FAIL", "Pylon enumeration", "no devices found");
      ok = false;
    } else {
      print_status("PASS", "Pylon enumeration", std::to_string(devices.size()) + " device(s)");
      for (std::size_t i = 0; i < devices.size(); ++i) {
        print_status("INFO", "device [" + std::to_string(i) + "]", device_info_text(devices[i]));
      }
    }

    if (!ok) {
      return 1;
    }

    const bool use_runtime_default_device = config.serial.empty() && !config.device_index_set;
    std::size_t selected_index = 0;
    if (use_runtime_default_device) {
      print_status(
        "INFO",
        "selected device",
        "runtime CreateFirstDevice(); first enumerated device is " + device_info_text(devices[0]));
    } else if (!config.serial.empty()) {
      bool found = false;
      for (std::size_t i = 0; i < devices.size(); ++i) {
        if (config.serial == std::string(devices[i].GetSerialNumber().c_str())) {
          selected_index = i;
          found = true;
          break;
        }
      }
      if (!found) {
        print_status("FAIL", "device selection", "serial not found: " + config.serial);
        return 1;
      }
    } else {
      if (config.device_index < 0 ||
        static_cast<std::size_t>(config.device_index) >= devices.size())
      {
        print_status("FAIL", "device selection", "index out of range");
        return 1;
      }
      selected_index = static_cast<std::size_t>(config.device_index);
    }
    if (!use_runtime_default_device) {
      print_status("INFO", "selected device", device_info_text(devices[selected_index]));
    }

    Pylon::CInstantCamera camera;
    Pylon::CImageFormatConverter converter;

    const std::string attach_label = use_runtime_default_device ?
      "Attach first device via CreateFirstDevice" : "Attach selected device";
    ok = run_step(attach_label, [&]() {
      if (use_runtime_default_device) {
        camera.Attach(factory.CreateFirstDevice());
      } else {
        camera.Attach(factory.CreateDevice(devices[selected_index]));
      }
    }) && ok;
    ok = run_step("Open camera", [&]() {
      camera.Open();
    }) && ok;
    if (!ok) {
      return 1;
    }

    print_node_snapshot(camera.GetNodeMap());
    if (config.probe_after_step) {
      ok = run_step("Configure ImageFormatConverter BGR8 for probes", [&]() {
        configure_converter(converter);
      }) && ok;
    }
    ok = configure_camera(camera, converter, config) && ok;
    if (!config.probe_after_step) {
      ok = run_step("Configure ImageFormatConverter BGR8", [&]() {
        configure_converter(converter);
      }) && ok;
    }
    print_node_snapshot(camera.GetNodeMap());
    ok = grab_final_frames(camera, converter, config) && ok;

    ok = run_step("Close camera", [&]() {
      if (camera.IsOpen()) {
        camera.Close();
      }
    }) && ok;
  } catch (...) {
    print_status("FAIL", "unexpected top-level exception", exception_text());
    ok = false;
  }
  return ok ? 0 : 1;
}
#endif

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const Config config = parse_args(argc, argv);
#ifdef HAVE_PYLON
    return run_with_pylon(config);
#else
    (void)config;
    print_status(
      "FAIL",
      "Pylon support",
      "yolo_cpp was built without HAVE_PYLON; rebuild with /opt/pylon visible");
    return 1;
#endif
  } catch (const std::exception & exc) {
    print_status("FAIL", "argument/setup error", exc.what());
    return 2;
  }
}
