#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <vector>

#include "yolo_cpp/detection_postprocess.hpp"

TEST(DetectionPostprocessTest, AppliesLocalNmsForTwoClassObbExportWithoutEmbeddedNms)
{
  const std::vector<int64_t> shape{1, 7, 2};
  const std::vector<float> tensor_data{
    100.0F,
    102.0F,  // cx
    100.0F,
    100.0F,  // cy
    40.0F,
    40.0F,  // width
    20.0F,
    20.0F,  // height
    0.90F,
    0.80F,  // blue confidence
    0.10F,
    0.20F,  // red confidence
    0.00F,
    0.00F,  // angle
  };

  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(), shape, 640, 640, 2, 0.25F, true, false, false, nullptr);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections.front().class_id, 0);
  EXPECT_NEAR(detections.front().confidence, 0.90F, 1e-6F);
  ASSERT_TRUE(detections.front().obb_angle.has_value());
  EXPECT_NEAR(*detections.front().obb_angle, 0.0F, 1e-6F);
  EXPECT_TRUE(detections.front().has_corners);
  EXPECT_NEAR(detections.front().corners_model[0].x, 80.0F, 1e-6F);
  EXPECT_NEAR(detections.front().corners_model[0].y, 110.0F, 1e-6F);
  EXPECT_NEAR(detections.front().corners_model[2].x, 120.0F, 1e-6F);
  EXPECT_NEAR(detections.front().corners_model[2].y, 90.0F, 1e-6F);
  EXPECT_GT(detections.front().bbox_model.width, 0);
  EXPECT_GT(detections.front().bbox_model.height, 0);
}

TEST(DetectionPostprocessTest, InfersTwoClassObbShapeWhenMetadataIsMissing)
{
  const std::vector<int64_t> shape{1, 7, 1};
  const std::vector<float> tensor_data{
    100.0F,  // cx
    100.0F,  // cy
    40.0F,   // width
    20.0F,   // height
    0.10F,   // blue confidence
    0.90F,   // red confidence
    0.00F,   // angle
  };

  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(), shape, 640, 640, 1, 0.25F, true, false, false, nullptr);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections.front().class_id, 1);
  EXPECT_NEAR(detections.front().confidence, 0.90F, 1e-6F);
  ASSERT_TRUE(detections.front().obb_angle.has_value());
  EXPECT_NEAR(*detections.front().obb_angle, 0.0F, 1e-6F);
}

TEST(DetectionPostprocessTest, ParsesYolo26EndToEndObbRows)
{
  std::vector<float> tensor_data(300 * 7, 0.0F);
  tensor_data[0] = 100.0F;  // cx
  tensor_data[1] = 100.0F;  // cy
  tensor_data[2] = 40.0F;   // width
  tensor_data[3] = 20.0F;   // height
  tensor_data[4] = 0.90F;   // confidence
  tensor_data[5] = 1.0F;    // class id
  tensor_data[6] = 0.25F;   // angle

  const std::vector<int64_t> shape{1, 300, 7};
  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(), shape, 640, 640, 2, 0.25F, true, false, std::nullopt, nullptr);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections.front().class_id, 1);
  EXPECT_NEAR(detections.front().confidence, 0.90F, 1e-6F);
  ASSERT_TRUE(detections.front().obb_angle.has_value());
  EXPECT_NEAR(*detections.front().obb_angle, 0.25F, 1e-6F);
  EXPECT_TRUE(detections.front().has_corners);
  EXPECT_GT(detections.front().bbox_model.width, 0);
  EXPECT_GT(detections.front().bbox_model.height, 0);
}

TEST(DetectionPostprocessTest, ParsesAlternateEndToEndObbRows)
{
  std::vector<float> tensor_data(300 * 7, 0.0F);
  tensor_data[0] = 100.0F;  // cx
  tensor_data[1] = 100.0F;  // cy
  tensor_data[2] = 40.0F;   // width
  tensor_data[3] = 20.0F;   // height
  tensor_data[4] = 0.25F;   // angle
  tensor_data[5] = 0.90F;   // confidence
  tensor_data[6] = 1.0F;    // class id

  const std::vector<int64_t> shape{1, 300, 7};
  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(), shape, 640, 640, 2, 0.25F, true, false, std::nullopt, nullptr);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections.front().class_id, 1);
  EXPECT_NEAR(detections.front().confidence, 0.90F, 1e-6F);
  ASSERT_TRUE(detections.front().obb_angle.has_value());
  EXPECT_NEAR(*detections.front().obb_angle, 0.25F, 1e-6F);
}

TEST(DetectionPostprocessTest, PreservesClassMetadataAndSeparatesBinaryTensorRtPlan)
{
  const auto path = std::filesystem::temp_directory_path()
    / ("yolo_metadata_test_" + std::to_string(
         std::chrono::steady_clock::now().time_since_epoch().count()) + ".engine");
  const std::string metadata_json =
    R"({"task":"obb","end2end":true,"args":{"nms":false},"names":{"0":"blue","1":"red"}})";
  const std::string raw_plan("TRT\0PLAN", 8);
  const auto metadata_size = static_cast<std::uint32_t>(metadata_json.size());
  {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&metadata_size), sizeof(metadata_size));
    stream << metadata_json;
    stream.write(raw_plan.data(), static_cast<std::streamsize>(raw_plan.size()));
  }
  const auto metadata = yolo_cpp::read_engine_metadata_from_file(path.string());
  EXPECT_EQ(metadata.class_names.at(0), "blue");
  EXPECT_EQ(metadata.class_names.at(1), "red");
  EXPECT_EQ(metadata.task.value_or(""), "obb");
  EXPECT_TRUE(metadata.end2end);
  ASSERT_TRUE(metadata.export_includes_nms.has_value());
  EXPECT_FALSE(*metadata.export_includes_nms);
  // The deployed YOLO26 export has end2end=true and args.nms=false together.
  std::vector<float> output(300 * 7, 0.0F);
  const std::vector<float> row{100.0F, 100.0F, 40.0F, 20.0F, 0.9F, 1.0F, 0.25F};
  std::copy(row.begin(), row.end(), output.begin());
  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    output.data(), {1, 300, 7}, 640, 640, 2, 0.25F, true,
    metadata.end2end, metadata.export_includes_nms, nullptr);
  ASSERT_EQ(detections.size(), 1U);
  EXPECT_NEAR(detections.front().confidence, 0.9F, 1e-6F);
  EXPECT_EQ(detections.front().class_id, 1);
  EXPECT_EQ(detections.front().bbox_model, cv::Rect(80, 90, 40, 20));
  EXPECT_EQ(yolo_cpp::read_engine_plan_from_file(path.string()), raw_plan);
  {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(raw_plan.data(), static_cast<std::streamsize>(raw_plan.size()));
  }
  EXPECT_EQ(yolo_cpp::read_engine_plan_from_file(path.string()), raw_plan);
  EXPECT_TRUE(yolo_cpp::read_engine_metadata_from_file(path.string()).class_names.empty());
  std::filesystem::remove(path);
}

TEST(DetectionPostprocessTest, RejectsInvalidEndToEndRows)
{
  const std::vector<float> valid{100.0F, 100.0F, 40.0F, 20.0F, 0.9F, 1.0F, 0.25F};
  for (const auto& invalid : std::vector<std::pair<int, float>>{
         {4, 637.5F}, {0, std::numeric_limits<float>::quiet_NaN()},
         {5, std::numeric_limits<float>::infinity()}, {6, std::numeric_limits<float>::quiet_NaN()},
         {2, 0.0F}, {3, -1.0F}}) {
    auto row = valid;
    row[invalid.first] = invalid.second;
    const auto detections = yolo_cpp::parse_yolo_output_tensor(
      row.data(), {1, 1, 7}, 640, 640, 2, 0.25F, true, true, false, nullptr);
    EXPECT_TRUE(detections.empty()) << "invalid attribute " << invalid.first;
  }
}
