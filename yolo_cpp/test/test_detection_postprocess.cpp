#include "yolo_cpp/detection_postprocess.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(DetectionPostprocessTest, AppliesLocalNmsForTwoClassObbExportWithoutEmbeddedNms)
{
  const std::vector<int64_t> shape{1, 7, 2};
  const std::vector<float> tensor_data{
    100.0F, 102.0F,  // cx
    100.0F, 100.0F,  // cy
    40.0F, 40.0F,    // width
    20.0F, 20.0F,    // height
    0.90F, 0.80F,    // blue confidence
    0.10F, 0.20F,    // red confidence
    0.00F, 0.00F,    // angle
  };

  const auto detections = yolo_cpp::parse_yolo_output_tensor(
    tensor_data.data(),
    shape,
    640,
    640,
    2,
    0.25F,
    true,
    false,
    false,
    nullptr);

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
