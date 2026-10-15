#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "yolo_cpp/detection_defaults.hpp"
#include "yolo_cpp/detection_postprocess.hpp"

TEST(DetectionPostprocessTest, DefaultConfidenceGateRejectsScoresBelowSixtyFivePercent)
{
  const std::vector<int64_t> shape{1, 7, 2};
  const std::vector<float> tensor_data{
    100.0F,
    300.0F,  // cx
    100.0F,
    300.0F,  // cy
    40.0F,
    40.0F,  // width
    20.0F,
    20.0F,  // height
    0.64F,
    0.65F,  // blue confidence
    0.10F,
    0.10F,  // red confidence
    0.00F,
    0.00F,  // angle
  };

  const auto detections = yolo_cpp::parse_yolo_output_tensor(tensor_data.data(), shape, 640, 640, 2,
    yolo_cpp::kDefaultDetectionConfidence, true, false, false, nullptr);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_NEAR(detections.front().confidence, 0.65F, 1e-6F);
  EXPECT_EQ(detections.front().bbox_model.x, 280);
}

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

TEST(DetectionPostprocessTest, ParsesYolo26RowsEvenWhenExportNmsIsFalse)
{
  std::vector<float> tensor_data(300 * 7, 0.0F);
  tensor_data[0] = 320.0F;
  tensor_data[1] = 240.0F;
  tensor_data[2] = 80.0F;
  tensor_data[3] = 40.0F;
  tensor_data[4] = 0.90F;
  tensor_data[5] = 1.0F;
  tensor_data[6] = 0.25F;

  const std::vector<int64_t> shape{1, 300, 7};
  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(), shape, 640, 640, 2, 0.65F, true, true, false, nullptr);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections.front().class_id, 1);
  EXPECT_NEAR(detections.front().confidence, 0.90F, 1e-6F);
  EXPECT_EQ(detections.front().bbox_model, cv::Rect(280, 220, 80, 40));
}

TEST(DetectionPostprocessTest, MissingMetadataDoesNotTreatClassIdAsConfidence)
{
  std::vector<float> tensor_data(300 * 7, 0.0F);
  tensor_data[0] = 320.0F;
  tensor_data[1] = 240.0F;
  tensor_data[2] = 80.0F;
  tensor_data[3] = 40.0F;
  tensor_data[4] = 0.01F;
  tensor_data[5] = 1.0F;
  tensor_data[6] = 0.0F;

  const std::vector<int64_t> shape{1, 300, 7};
  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(), shape, 640, 640, 1, 0.65F, true, false, std::nullopt, nullptr);

  EXPECT_TRUE(detections.empty());
}

TEST(DetectionPostprocessTest, MissingMetadataKeepsChannelFirstTwoClassObbRaw)
{
  constexpr int kBoxes = 8;
  std::vector<float> tensor_data(7 * kBoxes, 0.0F);
  tensor_data[0 * kBoxes] = 320.0F;
  tensor_data[1 * kBoxes] = 240.0F;
  tensor_data[2 * kBoxes] = 80.0F;
  tensor_data[3 * kBoxes] = 40.0F;
  tensor_data[4 * kBoxes] = 0.10F;
  tensor_data[5 * kBoxes] = 0.90F;
  tensor_data[6 * kBoxes] = 0.25F;

  const std::vector<int64_t> shape{1, 7, kBoxes};
  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(), shape, 640, 640, 1, 0.65F, true, false, std::nullopt, nullptr);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections.front().class_id, 1);
  EXPECT_NEAR(detections.front().confidence, 0.90F, 1e-6F);
}

TEST(DetectionPostprocessTest, RejectsOutOfRangeConfidenceAndInvalidGeometry)
{
  const std::vector<int64_t> shape{1, 300, 7};
  std::vector<float> tensor_data(300 * 7, 0.0F);
  tensor_data[0] = 320.0F;
  tensor_data[1] = 240.0F;
  tensor_data[2] = 80.0F;
  tensor_data[3] = 40.0F;
  tensor_data[4] = 1.50F;
  tensor_data[5] = 0.0F;
  tensor_data[6] = 0.0F;
  tensor_data[7] = 320.0F;
  tensor_data[8] = 240.0F;
  tensor_data[9] = -80.0F;
  tensor_data[10] = 40.0F;
  tensor_data[11] = 0.90F;
  tensor_data[12] = 0.0F;
  tensor_data[13] = 0.0F;

  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(), shape, 640, 640, 2, 0.65F, true, true, false, nullptr);

  EXPECT_TRUE(detections.empty());
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
