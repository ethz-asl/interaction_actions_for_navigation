#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>

#include <glog/logging.h>
#include <tf/transform_listener.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

namespace crop_laser_scan {


class LaserScanCropper {

  public:
    explicit LaserScanCropper(ros::NodeHandle& n) : nh_(n) {
      // Topic names.
      const std::string kScanTopic = "scan";
      const std::string kCroppedScanTopic = "cropped_scan";

      // Publishers and subscribers.
      scan_sub_ = nh_.subscribe(kScanTopic, 1000, &LaserScanCropper::scanCallback, this);
      cropped_scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>(kCroppedScanTopic, 1);

      nh_.getParam("crop_angle_min", kScanCropAngleMin);
      nh_.getParam("crop_angle_max", kScanCropAngleMax);
    }
    ~LaserScanCropper() {}

  protected:
    /// \brief receives scan 1 messages
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
      VLOG(3) << "scancallback";
      VLOG(5) << msg->header.frame_id;

      // Find the index of the first and last points within angle crop
      double first_scan_index_after_anglemin = (kScanCropAngleMin - msg->angle_min) / msg->angle_increment ;
      double last_scan_index_before_anglemax = (kScanCropAngleMax - msg->angle_min) / msg->angle_increment ;
      if ( first_scan_index_after_anglemin < 0 || last_scan_index_before_anglemax < 0 ) {
        LOG(ERROR) << "Angle index should not have negative value:"
          << first_scan_index_after_anglemin << " " << last_scan_index_before_anglemax << std::endl
          << "angle_increment: " << msg->angle_increment << ", angle_min: " << msg->angle_min
          << ", kScanCropAngleMin: " << kScanCropAngleMin
          << ", kScanCropAngleMax: " << kScanCropAngleMax;
      }
      size_t i_first = std::max(0., std::ceil(first_scan_index_after_anglemin));
      size_t i_last = std::min(std::floor(last_scan_index_before_anglemax), msg->ranges.size() - 1.);
      size_t cropped_len = i_last - i_first + 1;
      // Angles
      double angle_first = msg->angle_min + i_first * msg->angle_increment;
      double angle_last = angle_first + (cropped_len - 1.) * msg->angle_increment;

      // Generate the scan to fill in.
      sensor_msgs::LaserScan cropped_scan;
      cropped_scan.header = msg->header;
      cropped_scan.angle_increment = msg->angle_increment;
      cropped_scan.time_increment = msg->time_increment;
      cropped_scan.scan_time = msg->scan_time;
      cropped_scan.range_min = msg->range_min;
      cropped_scan.range_max = msg->range_max;
      cropped_scan.angle_min = angle_first;
      cropped_scan.angle_max = angle_last;
      cropped_scan.ranges.resize(cropped_len);
      cropped_scan.intensities.resize(cropped_len);

      // Fill in ranges.
      for ( size_t j = 0; j < cropped_len; j++ ) {
        size_t i = j + i_first;
        cropped_scan.ranges.at(j) = msg->ranges.at(i);
      }

      // Fill in intensities. if no intensities, spoof intensities
      if ( msg->intensities.size() == 0 ) {
        for ( size_t j = 0; j < cropped_len; j++ ) {
          size_t i = j + i_first;
          cropped_scan.intensities.at(j) = 1.0;
        }
      } else {
        for ( size_t j = 0; j < cropped_len; j++ ) {
          size_t i = j + i_first;
          cropped_scan.intensities.at(j) = msg->intensities.at(i);
        }
      }


      VLOG(3) << "";
      // publish result.
      cropped_scan_pub_.publish(cropped_scan);
     }


  private:
    // ROS
    ros::NodeHandle& nh_;
    ros::Subscriber scan_sub_;
    ros::Publisher cropped_scan_pub_;
    // Params
    double kScanCropAngleMin; // = -1.82;
    double kScanCropAngleMax; // =  1.87;

}; // class LaserScanCropper

} // namespace crop_laser_scan

using namespace crop_laser_scan;

int main(int argc, char **argv) {

  ros::init(argc, argv, "crop_laser_scan");
  ros::NodeHandle n("~"); // private node handle (~ gets replaced with node name)
  LaserScanCropper laser_scan_cropper(n);

  try {
//     ros::MultiThreadedSpinner spinner(2); // Necessary to allow concurrent callbacks.
//     spinner.spin();
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

