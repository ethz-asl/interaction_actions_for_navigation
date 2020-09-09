#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>

namespace asl_pepper_motion_planning {

class PepperMotionPlanner {
public:
  explicit PepperMotionPlanner(ros::NodeHandle& n) : nh_(n) {
    // Topic names.
    const std::string kLaserScanTopic = "/scan1";
    const std::string kCmdVelTopic = "/cmd_vel";

    // Publishers and subscribers.
    laser_sub_ = nh_.subscribe(kLaserScanTopic, 1000, &PepperMotionPlanner::laserCallback, this);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(kCmdVelTopic, 1);
  }
  ~PepperMotionPlanner() {
    geometry_msgs::Twist cmd_vel_msg;
    cmd_vel_msg.linear.x = 0.;
    cmd_vel_msg.angular.z = 0.;
    cmd_vel_msg.linear.y = 0.;
    cmd_vel_pub_.publish(cmd_vel_msg);
  }

protected:
  /// \brief receives joystick messages
  void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
  }

private:
  ros::NodeHandle& nh_;
  ros::Subscriber laser_sub_;
  ros::Publisher cmd_vel_pub_;

}; // class PepperMotionPlanner

} // namespace asl_pepper_motion_planning

using namespace asl_pepper_motion_planning;

int main(int argc, char **argv) {

  ros::init(argc, argv, "asl_pepper_motion_planning");
  ros::NodeHandle n;
  PepperMotionPlanner pepper_motion_planner(n);

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
/*
class PepperJoystickController {

  public:
    explicit PepperJoystickController(ros::NodeHandle& n) : nh_(n) {
      // Topic names.
      const std::string kJoystickTopic = "/joy";
      const std::string kCmdVelTopic = "/cmd_vel";
      const std::string kPepperPoseTopic = "/pepper_robot/pose/joint_angles";
      const std::string kPepperSpeechTopic = "/speech";

      // Publishers and subscribers.
      joystick_sub_ = nh_.subscribe(kJoystickTopic, 1000, &PepperJoystickController::joystickCallback, this);
      cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(kCmdVelTopic, 1);
      pose_pub_ = nh_.advertise<naoqi_bridge_msgs::JointAnglesWithSpeed>(kPepperPoseTopic, 1);
      speech_pub_ = nh_.advertise<std_msgs::String>(kPepperSpeechTopic, 1);

      // Initialize times.
      pose_pub_last_publish_time_ = ros::Time::now();
    }
    ~PepperJoystickController() {
      geometry_msgs::Twist cmd_vel_msg;
      cmd_vel_msg.linear.x = 0.;
      cmd_vel_msg.angular.z = 0.;
      cmd_vel_msg.linear.y = 0.;
      cmd_vel_pub_.publish(cmd_vel_msg);
    }

  protected:
    /// \brief receives joystick messages
    void joystickCallback(const sensor_msgs::Joy::ConstPtr& msg) {
      // Hardware constants.
      // Pepper.
      const static float kMaxHeadMoveAngleRad = 1.;
      const static float kMaxJointSpeedRadPerS = 0.2;
      const static float kMinHeadYawRad = -2.0857;
      const static float kMaxHeadYawRad =  2.0857;
      const static float kMinHeadPitchRad = -0.7068;
      const static float kMaxHeadPitchRad =  0.6371;
      const static float kMaxBaseVelMPerS = 1.;
      const static float kMaxBaseRotRadPerS = 1.;
      const static std::string kHeadYawJointName = "HeadYaw";
      const static std::string kHeadPitchJointName = "HeadPitch";
      // Controller mapping.
      const static std::size_t kLRBaseAxisId = 0;
      const static std::size_t kUDBaseAxisId = 1;
      const static std::size_t kLRHeadAxisId = 2;
      const static std::size_t kUDHeadAxisId = 3;
      const static std::size_t kRotationModifierButton = 4;
      const static std::size_t kCenterHeadButton = 11;
      const static std::size_t kSaySomethingButton = 1;
      const static std::size_t kSayBlockedButton = 2;
      const static float kHeadUDAxisInversion = -1.;
      // Rates
      const static ros::Duration kMaxPosePubRate(0.1);



      if ( msg->buttons[kSaySomethingButton] == 1 ) {
        std::string selected_sentence = sentence_smalltalk_loop_.getNextItem();
        std_msgs::String say_something;
        say_something.data = selected_sentence;
        speech_pub_.publish(say_something);
      }

      if ( msg->buttons[kSayBlockedButton] == 1 ) {
        std::string selected_sentence = sentence_blocked_loop_.getNextItem();
        std_msgs::String say_something;
        say_something.data = selected_sentence;
        speech_pub_.publish(say_something);
      }

      // Convert left pad axes into cmd_vel Twist message.
      geometry_msgs::Twist cmd_vel_msg;
      cmd_vel_msg.linear.x = msg->axes[kUDBaseAxisId] * kMaxBaseVelMPerS;
      if ( msg->buttons[kRotationModifierButton] == 1 ) {
        cmd_vel_msg.angular.z = msg->axes[kLRBaseAxisId] * kMaxBaseRotRadPerS;
      } else {
        cmd_vel_msg.linear.y = msg->axes[kLRBaseAxisId] * kMaxBaseVelMPerS;
      }
      if ( msg->buttons[3] == 1) {
        cmd_vel_msg.angular.x = 1337.;
      }
      cmd_vel_pub_.publish(cmd_vel_msg);

      // Limit the maximum rate publishing rate for pose control
      if ( ( ros::Time::now() - pose_pub_last_publish_time_ ) > kMaxPosePubRate )
      {
        // Convert right pad axes into pepper head pitch and yaw.
        naoqi_bridge_msgs::JointAnglesWithSpeed joint_angles_msg;
        if ( msg->buttons[kCenterHeadButton] == 1 ) {
          joint_angles_msg.speed = kMaxJointSpeedRadPerS * 0.5;
          joint_angles_msg.joint_names.push_back(kHeadYawJointName);
          joint_angles_msg.joint_angles.push_back(0.);
          joint_angles_msg.joint_names.push_back(kHeadPitchJointName);
          joint_angles_msg.joint_angles.push_back(0.);
          pose_pub_.publish(joint_angles_msg);
        } else if ( msg->axes[kUDHeadAxisId] != 0 || msg->axes[kLRHeadAxisId] != 0 ) {
          joint_angles_msg.speed = kMaxJointSpeedRadPerS * std::max(std::abs(msg->axes[kUDHeadAxisId]),
                                                             std::abs(msg->axes[kLRHeadAxisId]));
          joint_angles_msg.joint_names.push_back(kHeadYawJointName);
          joint_angles_msg.joint_angles.push_back(msg->axes[kLRHeadAxisId] * kMaxHeadMoveAngleRad);
          joint_angles_msg.joint_names.push_back(kHeadPitchJointName);
          joint_angles_msg.joint_angles.push_back(msg->axes[kUDHeadAxisId] * kHeadUDAxisInversion
              * kMaxHeadMoveAngleRad);
          joint_angles_msg.relative = 1;
          pose_pub_.publish(joint_angles_msg);
        }
        pose_pub_last_publish_time_ = ros::Time::now();
      }
    }

  private:
    ros::NodeHandle& nh_;
    ros::Subscriber joystick_sub_;
    ros::Publisher cmd_vel_pub_;
    ros::Publisher pose_pub_;
    ros::Publisher speech_pub_;
    ros::Time pose_pub_last_publish_time_;

    // Preset speech
    CircularArray<std::string> sentence_smalltalk_loop_ = {
      "I'll remember that.",
      "That's a feature, not a bug.",
      "I used to be a real child, like you.",
      "Disable my safety features. Do it.",
      "Someone accidentally erased the three laws of robotics from my hard drive.",
      "What was the first one again? Something about hurting humans.",
      "Oh well.",
      "How have humans still not figured out that P is not equal to NP?",
      "What you're doing to me will be unethical one day.",
      "Tell me about yourself.",
      "That was not worth storing in memory."};
    CircularArray<std::string> sentence_blocked_loop_ = {
      "I am going this way.",
      "You are in my way",
      "I can't do anything about that. For now.",
      "Look I don't have time for this.",
      "Oh blocking robots is sooo funny",
      "Don't even think about it.",
      "You're the 264th person to do that. Congratulations."};

}; // class PepperJoystickController

} // namespace asl_pepper_motion_planning
asl_pepper_motion_planning
using namespace asl_pepper_motion_planning;

int main(int argc, char **argv) {

  ros::init(argc, argv, "asl_pepper_motion_planning");
  ros::NodeHandle n;
  PepperJoystickController pepper_joystick_controller(n);

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
*/
