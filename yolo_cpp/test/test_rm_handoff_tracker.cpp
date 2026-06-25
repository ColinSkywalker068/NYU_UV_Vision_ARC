#include <gtest/gtest.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "yolo_cpp/rm_handoff_tracker.hpp"

namespace {

auto_aim_interfaces::msg::Armor make_armor(const std::string &number,
                                           const std::string &type,
                                           double image_distance, double x,
                                           double y, double z,
                                           double yaw = 0.0) {
  auto_aim_interfaces::msg::Armor armor;
  armor.number = number;
  armor.type = type;
  armor.distance_to_image_center = image_distance;
  armor.pose.position.x = x;
  armor.pose.position.y = y;
  armor.pose.position.z = z;
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  armor.pose.orientation = tf2::toMsg(q);
  return armor;
}

auto_aim_interfaces::msg::Armors
make_armors(std::initializer_list<auto_aim_interfaces::msg::Armor> armors) {
  auto_aim_interfaces::msg::Armors msg;
  msg.armors.assign(armors.begin(), armors.end());
  return msg;
}

yolo_cpp::RmHandoffTracker make_tracker(double max_distance = 1.0,
                                        double max_yaw = 1.0) {
  yolo_cpp::RmHandoffTracker tracker(max_distance, max_yaw);
  tracker.configure_ekf(1.0 / 60.0, 20.0, 100.0, 800.0, 0.05, 0.02);
  tracker.tracking_thres = 0;
  tracker.lost_thres = 2;
  return tracker;
}

} // namespace

TEST(RmHandoffTrackerTest, InitializesClosestToImageCenter) {
  auto tracker = make_tracker();
  tracker.init(make_armors({
      make_armor("1", "small", 40.0, 1.0, 0.0, 2.0),
      make_armor("3", "large", 5.0, 2.0, 0.0, 2.0),
  }));

  EXPECT_EQ(tracker.tracker_state, yolo_cpp::RmHandoffTracker::kDetecting);
  EXPECT_EQ(tracker.tracked_id, "3");
  EXPECT_EQ(tracker.tracked_armor.number, "3");
  EXPECT_EQ(tracker.tracked_armors_num,
            yolo_cpp::RmHandoffArmorsNum::kBalance2);
}

TEST(RmHandoffTrackerTest, MatchesSameIdClosestToPrediction) {
  auto tracker = make_tracker();
  tracker.init(make_armors({make_armor("1", "small", 0.0, 0.0, 0.0, 1.0)}));

  tracker.update(make_armors({
      make_armor("2", "small", 0.0, 0.0, 0.0, 1.0),
      make_armor("1", "small", 0.0, 0.40, 0.0, 1.0),
      make_armor("1", "small", 0.0, 0.02, 0.0, 1.0),
  }));

  EXPECT_EQ(tracker.tracker_state, yolo_cpp::RmHandoffTracker::kTracking);
  EXPECT_EQ(tracker.tracked_armor.number, "1");
  EXPECT_NEAR(tracker.tracked_armor.pose.position.x, 0.02, 1e-6);
}

TEST(RmHandoffTrackerTest, KeepsTrackingTrueStateThroughTempLostEquivalent) {
  auto tracker = make_tracker();
  tracker.init(make_armors({make_armor("1", "small", 0.0, 0.0, 0.0, 1.0)}));
  tracker.update(make_armors({make_armor("1", "small", 0.0, 0.01, 0.0, 1.0)}));
  ASSERT_EQ(tracker.tracker_state, yolo_cpp::RmHandoffTracker::kTracking);

  tracker.update(
      make_armors({make_armor("1", "small", 0.0, 0.01, 0.0, 1.0, 2.0)}));

  EXPECT_EQ(tracker.tracker_state, yolo_cpp::RmHandoffTracker::kTempLost);
  EXPECT_NEAR(tracker.target_state(6), 2.0, 1e-6);
}

TEST(RmHandoffTrackerTest, ClampsRadiusAndUsesOutpostArmorCount) {
  auto tracker = make_tracker();
  tracker.init(
      make_armors({make_armor("outpost", "small", 0.0, 0.0, 0.0, 1.0)}));
  EXPECT_EQ(tracker.tracked_armors_num,
            yolo_cpp::RmHandoffArmorsNum::kOutpost3);

  tracker.target_state(8) = 0.01;
  tracker.ekf.set_state(tracker.target_state);
  tracker.update(make_armors({}));

  EXPECT_NEAR(tracker.target_state(8), 0.12, 1e-9);
}
