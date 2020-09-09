#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>

namespace asl_pepper_collision_check {

class PepperCollisionChecker {
public:
  explicit PepperCollisionChecker(ros::NodeHandle& n) : nh_(n) {
    // Topic names.
    const std::string kLaserScanTopic = "/scan1";
    const std::string kCmdVelTopic = "/cmd_vel";

    // Publishers and subscribers.
    laser_sub_ = nh_.subscribe(kLaserScanTopic, 1000, &PepperCollisionChecker::laserCallback, this);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(kCmdVelTopic, 1);
  }
  ~PepperCollisionChecker() {
  }

protected:
  /// \brief receives joystick messages
  void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    // TODO: check safe

    // Send killMove message
    geometry_msgs::Twist cmd_vel_msg;
    cmd_vel_msg.angular.x = 1337.;
    cmd_vel_pub_.publish(cmd_vel_msg);
    // Raise Flag
    // TODO
  }

private:
  ros::NodeHandle& nh_;
  ros::Subscriber laser_sub_;
  ros::Publisher cmd_vel_pub_;

}; // class PepperCollisionChecker

} // namespace asl_pepper_motion_planning

using namespace asl_pepper_collision_check;

int main(int argc, char **argv) {

  ros::init(argc, argv, "asl_pepper_collision_check");
  ros::NodeHandle n;
  PepperCollisionChecker pepper_collision_checker(n);

  try {
    ros::spin();
  }
  catch (const std::exception& e) {
    ROS_ERROR_STREAM("Exception: " << e.what());
    return 1;
  }
  catch (...) {
    ROS_ERROR_STREAM("Unknown Exception.");
    return 1;
  }

  return 0;
}
