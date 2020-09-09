#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>
#include <naoqi_bridge_msgs/JointAnglesWithSpeed.h>


// for the tilt indicator
#include <visualization_msgs/Marker.h>


namespace asl_pepper_joystick {

/// \brief A circular array based on std::vector
template <class T>
class CircularArray {
  public:
    CircularArray(std::initializer_list<T> l) : data_(l) {};
    T getNextItem() {
      T result = data_.at(current_pos_);
      if ( ++current_pos_ >= data_.size() ) {
        current_pos_ = 0;
      }
      return result;
    }

  private:
    std::size_t current_pos_ = 0;
    const std::vector<T> data_;
}; // class CircularArray

class PepperJoystickController {

  public:
    explicit PepperJoystickController(ros::NodeHandle& n) : nh_(n) {
      // Topic names.
      const std::string kJoystickTopic = "/joy";
      const std::string kCmdVelTopic = "/cmd_vel";
      const std::string kPepperPoseTopic = "/pepper_robot/pose/joint_angles";
      const std::string kPepperSpeechTopic = "/speech";
      const std::string kPepperGestureTopic = "/gestures";

      const std::string tiltIndicatorTopic = "/tilt_indicator"; 

      // Publishers and subscribers.
      joystick_sub_ = nh_.subscribe(kJoystickTopic, 1000, &PepperJoystickController::joystickCallback, this);
      cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(kCmdVelTopic, 1);
      pose_pub_ = nh_.advertise<naoqi_bridge_msgs::JointAnglesWithSpeed>(kPepperPoseTopic, 1);
      speech_pub_ = nh_.advertise<std_msgs::String>(kPepperSpeechTopic, 1);
      gesture_pub_ = nh_.advertise<std_msgs::String>(kPepperGestureTopic, 1);
      tilt_pub_ = nh_.advertise<visualization_msgs::Marker>(tiltIndicatorTopic,1); 

      // Services
      stop_autonomous_motion_client_ =
        nh_.serviceClient<std_srvs::Trigger>("stop_autonomous_motion");
      resume_autonomous_motion_client_ =
        nh_.serviceClient<std_srvs::Trigger>("resume_autonomous_motion");

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
      const static float kMaxBaseVelMPerS = 0.4; //0.2; //0.55, way lower! // Using moveToward, vels are normalized
      const static float kMaxBaseRotRadPerS = 2.; // Using moveToward, rotation is normalized
      const static std::string kHeadYawJointName = "HeadYaw";
      const static std::string kHeadPitchJointName = "HeadPitch";
      const static std::string kLeftPadGesture  = "animations/Stand/Gestures/Far_1";
      const static std::string kRightPadGesture = "animations/Stand/Gestures/ShowSky_5";
      const static std::string kUpPadGesture    = "animations/Stand/Gestures/You_2";
//       const static std::string kUpPadGesture    = "animations/Stand/Gestures/Far_3";
//       const static std::string kUpPadGesture  = "animations/Stand/Gestures/Give_5";
      const static std::string kDownPadGesture  = "animations/Stand/Gestures/Desperate_4";
//       const static std::string kDownPadGesture  = "animations/Stand/Gestures/BowShort_1";
//       const static std::string kDownPadGesture  = "animations/Stand/Gestures/ShowFloor_2";
      // Controller mapping.
      const static std::size_t kLRBaseAxisId = 3;//0
      const static std::size_t kUDBaseAxisId = 4; //1
      const static std::size_t kLRHeadAxisId = -1;
      const static std::size_t kUDHeadAxisId = -1;
      const static std::size_t kLRGesturePadAxisId = -1;
      const static std::size_t kUDGesturePadAxisId = -1;
      const static std::size_t kRotationModifierButton = 5; // Top left bumper
      const static std::size_t kCenterHeadButton = -1; // Click right joystick
      const static std::size_t kSaySomethingButton = 0; // A (green)
      const static std::size_t kDoGestureButton = 3; // Y (yellow)
      const static std::size_t kIndicateTiltButton = 4;  // Publish and display a red circle implying base tilt

      const static std::size_t kSayBlockedButton = -1; // Y (yellow)
      const static std::size_t kKillMoveButton = 2; // B (red)
      const static std::size_t kResumeAutonomousMotionButton = 7; // start
      const static float kHeadUDAxisInversion = -1.;
      // Rates
      const static ros::Duration kMaxPosePubRate(0.1);


      // Publish gestures
      std::string gesture_name;
      if ( msg->axes[kUDGesturePadAxisId] > 0 ) { // top
        gesture_name = kUpPadGesture;
      } else if (msg->axes[kUDGesturePadAxisId] < 0 ) { // bottom
        gesture_name = kDownPadGesture;
      } else if (msg->axes[kLRGesturePadAxisId] > 0 ) { // left
        gesture_name = kLeftPadGesture;
      } else if (msg->axes[kLRGesturePadAxisId] < 0 ) { // right
        gesture_name = kRightPadGesture;
      } else {
        gesture_name = "";
      }
      if ( gesture_name != "" ) {
        std_msgs::String gesture_msg;
        gesture_msg.data = gesture_name;
        gesture_pub_.publish(gesture_msg);
      }
      // Stock gestures
      std::string selected_gesture = ""; 
      if ( msg->buttons[kDoGestureButton] == 1 ) {
        selected_gesture = gestures_greeting_loop_.getNextItem();
      }
      if (selected_gesture != "") {
        std_msgs::String gesture_msg;
        gesture_msg.data = selected_gesture;
        gesture_pub_.publish(gesture_msg);
      }

      // Publish speech
      if ( msg->buttons[kSaySomethingButton] == 1 ) {
        //std::string selected_sentence = sentence_smalltalk_loop_.getNextItem();
        
        std::string selected_sentence = "Hello Eden. I hope you are well. Best Regards from Pepper!"; 
        
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

      // Tilt. Press the button if the robot base starts loosing grip, to record it in the rosbag and display it in rviz. 
      if(msg->buttons[kIndicateTiltButton]==1){
        visualization_msgs::Marker bigRedWarning;

        //visualization_msgs::Marker m; 
        bigRedWarning.header.frame_id = "base_footprint"; // gmap // Hope that this robot one is available
        bigRedWarning.header.stamp = ros::Time::now(); 
        bigRedWarning.action = visualization_msgs::Marker::ADD; 
        bigRedWarning.lifetime = ros::Duration(); // Doing? -> expiry date
        bigRedWarning.type = visualization_msgs::Marker::CYLINDER; 
        bigRedWarning.id = 42; 
        bigRedWarning.ns = "joystick"; 

        bigRedWarning.color.r = 1.0f; 
        bigRedWarning.color.g = 0.0f; 
        bigRedWarning.color.b = 0.0f; 
        bigRedWarning.color.a = 0.8f; // Some transparency? 

        bigRedWarning.pose.position.x = 0.0;  
        bigRedWarning.pose.position.y = 0.0; 
        bigRedWarning.pose.position.z = 0.0; 

        bigRedWarning.pose.orientation.x = 0.0; 
        bigRedWarning.pose.orientation.y = 0.0; 
        bigRedWarning.pose.orientation.z = 0.0; 
        bigRedWarning.pose.orientation.w = 1.0; 

        bigRedWarning.scale.x = 1.0;
        bigRedWarning.scale.y = 1.0;
        bigRedWarning.scale.z = 0.1;

        tilt_pub_.publish(bigRedWarning); 
      }

      bool left_joystick_is_in_deadzone = ( msg->axes[kLRBaseAxisId] == 0 &&
                                            msg->axes[kUDBaseAxisId] == 0 );
      bool kill_move_button_is_pressed =  msg->buttons[kKillMoveButton] == 1;
      // Convert left pad axes into cmd_vel Twist message.
      geometry_msgs::Twist cmd_vel_msg;
      cmd_vel_msg.linear.x = msg->axes[kUDBaseAxisId] * kMaxBaseVelMPerS;
      // Apply possible rotation modifier
      if ( msg->buttons[kRotationModifierButton] == 1 ) {
        cmd_vel_msg.angular.z = msg->axes[kLRBaseAxisId] * kMaxBaseRotRadPerS;
      } else {
        cmd_vel_msg.linear.y = msg->axes[kLRBaseAxisId] * kMaxBaseVelMPerS;
      }


      // Logic for whether to publish a message or not, whether to stop autonomous planning
      bool publish_cmd_vel = true;
      bool call_stop_autonomous_motion_service = false;
      //   Avoid spamming updates when joystick is in the deadzone for a while,
      //   to allow motion_planning to take over for example.
      static size_t spam_count = 0;
      if ( left_joystick_is_in_deadzone ) {
        spam_count++;
        if ( spam_count > 4 ) {
           publish_cmd_vel = false;
        }
      } else {
        spam_count = 0;
        call_stop_autonomous_motion_service = true;
      }
      //   Workaround to signal naoqi_driver to send killMove command.
      if ( kill_move_button_is_pressed ) {
        cmd_vel_msg.angular.x = 1337.;
        call_stop_autonomous_motion_service = true;
        publish_cmd_vel = true;
      }

      // Send resulting command
      if ( publish_cmd_vel ) {
        cmd_vel_pub_.publish(cmd_vel_msg);
      }

      // Stop motion planner from moving pepper if necessary (take over control)
      if ( call_stop_autonomous_motion_service ) {
        std_srvs::Trigger srv;
        if ( stop_autonomous_motion_client_.call(srv) ) {
            ROS_INFO("Stop autonomous motion response: %d (%s)",
                srv.response.success, srv.response.message.c_str());
        }
        else {
            ROS_INFO("Failed to call service stop_autonomous_motion");
        }
      }
      // Hand over control to autonomous planner
      else if ( msg->buttons[kResumeAutonomousMotionButton] ) {
        std_srvs::Trigger srv;
        if ( resume_autonomous_motion_client_.call(srv) ) {
            ROS_INFO("Resume autonomous motion response: %d (%s)",
                srv.response.success, srv.response.message.c_str());
        }
        else {
            ROS_INFO("Failed to call service resume_autonomous_motion");
        }
      }

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
    ros::Publisher gesture_pub_;

    // Press a button when the robot wiggling get critical, visually. 
    ros::Publisher tilt_pub_; 

    ros::ServiceClient stop_autonomous_motion_client_;
    ros::ServiceClient resume_autonomous_motion_client_;
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

    // CircularArray<std::string> gestures_greeting_loop_ = {
    //   // "animations/Stand/Gestures/Enthusiastic_4", 
    //   // "animations/Stand/Gestures/Enthusiastic_5", 
    //   // "animations/Stand/Gestures/Excited_1", 

    //   // "animations/Stand/Gestures/Hey_4",
    //   // "animations/Stand/Gestures/Hey_6", 

    //   "animations/Stand/Gestures/Hey_1",
    //   "animations/Stand/Gestures/Hey_3",
    //   "animations/Stand/Emotions/Positive/Happy_4"};
    CircularArray<std::string> gestures_greeting_loop_ = {

      "animations/Stand/Gestures/Excited_1", 
      "animations/Stand/Gestures/CalmDown_2", 

      "animations/Stand/Gestures/Everything_3", 
      "animations/Stand/Emotions/Positive/Happy_4", 


      };

}; // class PepperJoystickController

} // namespace asl_pepper_joystick

using namespace asl_pepper_joystick;

int main(int argc, char **argv) {

  ros::init(argc, argv, "asl_pepper_joystick");
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
