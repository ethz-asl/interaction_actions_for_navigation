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
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl_ros/transforms.h>
#include <laser_geometry/laser_geometry.h>


namespace combine_laser_scans {

template <class T>
class Setable {
  public:
    Setable() : is_set_(false) {};
    ~Setable() {};
    T get() const {
      if ( !is_set_ ) {
        throw std::runtime_error("Attempted to access value which is not set.");
      }
      return data_;
    }
    void set(const T& data) { data_ = data; }
    bool is_set() const { return is_set_; }
  private:
    T data_;
    bool is_set_;
};

// \brief A thread-safe buffer for storing a value 
template <class T>
class ProtectedBuffer {
 public:
  ProtectedBuffer() {};
  ~ProtectedBuffer() {};
  // \brief empties the buffered value, and copies it to value_out.
  // Returns false if the buffer was empty, true otherwise.
  bool flushValue(T &value_out) {
    mutex_.lock();
    bool value_is_set = value_is_set_;
    if ( value_is_set ) {
      value_out = protected_value_;
    }
    value_is_set_ = false;
    mutex_.unlock();
    if ( value_is_set ) {
      return true;
    }
    return false;
  }
  // \brief empties the buffered value.
  // Returns false if the buffer was empty, true otherwise.
  bool flushValue() {
    mutex_.lock();
    bool value_is_set = value_is_set_;
    value_is_set_ = false;
    mutex_.unlock();
    return value_is_set;
  }
  void setValue(const T &value) {
    mutex_.lock();
    protected_value_ = value;
    value_is_set_ = true;
    mutex_.unlock();
    return;
  }
  // Wait until a value is set and then flush.
  // if timeout_ms is 0, waits forever.
  bool waitUntilSetAndFlush(T &value_out, const size_t timeout_ms = 0) {
    size_t kDT_ms = 1;
    size_t total_time_waited_ms = 0;
    while ( true ) {
      if ( flushValue(value_out) ) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kDT_ms));
      total_time_waited_ms += kDT_ms;
      if ( timeout_ms != 0 && total_time_waited_ms >= timeout_ms ) {
        return false;
      }
    }
  }
 private:
  std::mutex mutex_;
  T protected_value_;
  bool value_is_set_ = false;
}; // class ProtectedBuffer

class LaserScanCombiner {

  public:
    explicit LaserScanCombiner(ros::NodeHandle& n) : nh_(n) {
      // Topic names.
      std::vector<std::string> kScanTopics;
      kScanTopics.push_back("scan1");
      kScanTopics.push_back("scan2");
      const std::string kScan1Topic = "scan1";
      const std::string kScan2Topic = "scan2";
      const std::string kCombinedScanTopic = "combined_scan";

      // State
      tf_is_known_.resize(kScanTopics.size(), false);
      kTargetScanFrame = "base_footprint";
      tf_lidar_i_in_target_.resize(kScanTopics.size());
      last_scan_in_target_frame_.resize(kScanTopics.size());
      last_scan_is_unpublished_.resize(kScanTopics.size(), false); // not available yet, therefore not unpublished
      is_waiting_for_first_scan_.resize(kScanTopics.size(), true);

      // Publishers and subscribers.
      scan_1_sub_ = nh_.subscribe(kScan1Topic, 1, &LaserScanCombiner::scan1Callback, this);
      scan_2_sub_ = nh_.subscribe(kScan2Topic, 1, &LaserScanCombiner::scan2Callback, this);
      combined_scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>(kCombinedScanTopic, 1);
    }
    ~LaserScanCombiner() {}

  protected:
    /// \brief Does nothing is Tf is known, otherwise fetches tf
    void setTFOfLidarInTarget(const sensor_msgs::LaserScan& msg, const size_t& this_lidar_id) {
      constexpr bool kAssumeStaticTransform = true;
      constexpr size_t kTFTimeout_ms = 1000;
      constexpr size_t kMsToNs = 1000000;

      if ( !kAssumeStaticTransform ) {
        LOG(ERROR) << "Dynamic transforms are not implemented.";
        ros::shutdown;
        return;
      }
      // Find the transform between 1 and 2 if it is not known yet.
      if ( !tf_is_known_.at(this_lidar_id) ) {
      LOG(INFO) << "Waiting for transform between scan and target frame: " <<
        msg.header.frame_id << " " << this_lidar_id << " " << kTargetScanFrame;
        try {
          ros::Duration kTFWait = ros::Duration(0, 200*kMsToNs);
          tf_listener_.waitForTransform(msg.header.frame_id, kTargetScanFrame,
                                        msg.header.stamp + kTFWait,
                                        ros::Duration(0, kTFTimeout_ms * kMsToNs));
          tf_listener_.lookupTransform(msg.header.frame_id, kTargetScanFrame,
                                       msg.header.stamp + kTFWait,
                                       tf_lidar_i_in_target_.at(this_lidar_id));
        }
        catch ( tf::TransformException &ex ) {
          LOG(ERROR) << "Error while looking up transform between scan and target : " <<
            ex.what();
          return;
        }
        tf_is_known_.at(this_lidar_id) = true;
        LOG(INFO) << "Transform found.";
      }
    }

    void mergeScanIntoScanOfSameFrame(
        const sensor_msgs::LaserScan& scan1, 
        sensor_msgs::LaserScan& scan2) const {
      const std::string kMergeStrategy = "keep_closest";
      const ros::Duration kDesiredSyncThreshold(0.1);
      // Assumes the scans are in the same format! (angular resolution, frame, etc.. identical)
      // if (scan1.header.frame_id ... )
      if ( scan1.ranges.size() != scan2.ranges.size() ) {
        LOG(ERROR) << "Scans to merge must be identical";
        ros::shutdown;
      }
      if ( ( scan1.header.stamp - scan2.header.stamp ) > kDesiredSyncThreshold ||
           ( scan2.header.stamp - scan1.header.stamp ) > kDesiredSyncThreshold) { // now API for absolute value of a Duration? ok then...
        LOG(ERROR) << "Sync tolerance exceeded.";
      }
      for ( size_t i = 0; i < scan1.ranges.size(); i++ ) {
        if ( scan1.ranges.at(i) == 0 ) { continue; }
        if ( scan1.ranges.at(i) < scan2.ranges.at(i) || scan2.ranges.at(i) == 0 ) {
          scan2.ranges.at(i) = scan1.ranges.at(i);
          scan2.intensities.at(i) = scan1.intensities.at(i);
        }
      }
    }

    // "Because the stamp of a sensor_msgs/LaserScan is the time of the first measurement..." - wiki.ros.org/laser_geometry
    sensor_msgs::LaserScan generateEmptyOutputScan(const sensor_msgs::LaserScan msg) const {
      // Create a new scan
      sensor_msgs::LaserScan scan;
      scan.header.frame_id = kTargetScanFrame;
      scan.header.stamp = msg.header.stamp;
      scan.angle_increment = 0.00581718236208; // approximately increment of SICK Tim571
      scan.time_increment = msg.time_increment;
      scan.scan_time = msg.scan_time;
      scan.range_min = msg.range_min;
      scan.range_max = msg.range_max;
//       scan.angle_min = reference_scan1_.angle_min; // start with the same angle
      scan.angle_min = 0.;
      scan.ranges.resize(2*M_PI / scan.angle_increment, 0.);
      scan.intensities.resize(scan.ranges.size(), 0.);
      scan.angle_max =
        scan.angle_increment * ( scan.ranges.size() - 1 ); // sh. be full circle
      return scan;
    }

    sensor_msgs::LaserScan transformScanToTargetFrame(
        const sensor_msgs::LaserScan& msg, const size_t& this_lidar_id) const {
      sensor_msgs::LaserScan target_scan = generateEmptyOutputScan(msg);
      tf::StampedTransform t_T_a = tf_lidar_i_in_target_.at(this_lidar_id);
      
      // Find radius from target to lidar, within which points should be ignored (within robot)
      tf::Vector3 zeros(0., 0., 0.);
      tf::Vector3 lidarxyz_in_target = t_T_a.inverse() * zeros;
      float kEstRobotRadius = sqrt( lidarxyz_in_target.x() * lidarxyz_in_target.x() + 
                                    lidarxyz_in_target.y() * lidarxyz_in_target.y() );

      // a_x_i, a_y_i = a_r_i cos a_th_i, a_r_i sin a_th_i
      sensor_msgs::PointCloud cloud;
      static laser_geometry::LaserProjection projector_;
      // TODO do this manually to preserve intensities?
      projector_.projectLaser(msg, cloud);
      // Find intensity values if desired
      bool found_intensities = false;
      std::vector<float> intensities;
      try {
        VLOG(3) << "";
        for ( size_t i = 0; i < cloud.channels.size(); i++ ) {
          if ( cloud.channels.at(i).name == "intensity" ) {
            intensities = cloud.channels.at(i).values;
            if ( intensities.size() == cloud.points.size() ) {
              found_intensities = true;
              break;
            }
            LOG(ERROR) << "In scan, mismatch between intensity and points array sizes.";
          }
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "Could not find intensities for scan: " << e.what();
      }

      // Transform each point into the target scan
      for ( size_t i = 0; i < msg.ranges.size(); i++ ) {
        // if the point is a no-return (range = 0), no point in projecting
        if ( msg.ranges.at(i) < msg.range_min ) { continue; }
        if ((cloud.points[i].x * cloud.points[i].x +
             cloud.points[i].y * cloud.points[i].y +
             cloud.points[i].z * cloud.points[i].z ) < msg.range_min * msg.range_min ) {
//           LOG(ERROR) << msg.ranges.at(i); // this shouldn't be happening
          continue; }
        // Extract point
        tf::Vector3 a_p_i(
            cloud.points[i].x,
            cloud.points[i].y,
            cloud.points[i].z);
        // convert this point to a point in the target frame
        // (t_x_i, t_y_i) = t_T_a * (a_x_i, t_y_i)
        tf::Vector3 t_p_i = t_T_a.inverse() * a_p_i;
        // t_r_i, t_th_i = norm(t_x_i, t_y_i), atan2(t_x_i, t_y_i)
        float angle = atan2(t_p_i.y(), t_p_i.x());
        float range = sqrt(t_p_i.y() * t_p_i.y() + t_p_i.x() * t_p_i.x());
        if ( angle < 0 ) { angle += 2 * M_PI; } // atan2 [-pi, pi] -> now [0, 2pi]
        // reject points within robot.
        if ( range < kEstRobotRadius ) {
          continue;
        }
        // fill in range in corresponding angle bracket
        CHECK( ( angle / target_scan.angle_increment )  >= 0 );
        size_t index = round(angle / target_scan.angle_increment);
        if ( index == target_scan.ranges.size() ) {
          index = 0;
        }
        VLOG(2) << "angle: " << angle;
        VLOG(2) << "min angle: " << target_scan.angle_min;
        VLOG(2) << "max angle: " << target_scan.angle_max;
        VLOG(2) << "angle inc: " << target_scan.angle_increment;
        VLOG(2) << "rel angle: " << angle;
        VLOG(2) << "index: " << index;
        target_scan.ranges.at(index) = range;
        if ( found_intensities ) {
          target_scan.intensities.at(index) = intensities.at(i);
        }
      }
      return target_scan;
    }

    /// \brief receives scan 1 messages
    void scan1Callback(const sensor_msgs::LaserScan::ConstPtr& msg) {
      scanCallback(msg, 0);
    }

    void scan2Callback(const sensor_msgs::LaserScan::ConstPtr& msg) {
      scanCallback(msg, 1);
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg, const size_t& this_lidar_id) {

      // Set the reference static transform between lasers.
      setTFOfLidarInTarget(*msg, this_lidar_id);
      // Unless the tf has been found, the rest of the callback can not be executed
      if ( !tf_is_known_.at(this_lidar_id) ) {
        return;
      }

      // Transform this scan to target frame
      sensor_msgs::LaserScan scan_in_target_frame = transformScanToTargetFrame(*msg, this_lidar_id);

      // On first run, only set the last scan.
      if ( is_waiting_for_first_scan_.at(this_lidar_id) ) {
        last_scan_in_target_frame_.at(this_lidar_id) = scan_in_target_frame;
        is_waiting_for_first_scan_.at(this_lidar_id) = false;
        last_scan_is_unpublished_.at(this_lidar_id) = true;
        LOG(INFO) << "First scan received for sensor " << this_lidar_id;
        return;
      }

      // Possible actions: 
      // store scan to merge later with next scan of other lidar
      // publish scan alone
      // merge with stored scan from other lidar and publish
      // Check edge cases.
      if ( last_scan_is_unpublished_.at(this_lidar_id) ) {
        VLOG(1) << "the last scan of this lidar went unpublished for a whole cycle. " << 
          "This could indicate that the prediction heuristic is failing or assumptions are unmet.";
      }
      mutex_.lock(); // ensure thread-safety when dealing with buffers of other lidars
      // If an unpublished scan is available, merge and publish. makes sense for 2 lidars
      sensor_msgs::LaserScan merged_scan = scan_in_target_frame;
      bool unpublished_scan_is_available = false;
      for ( size_t other_lidar_id = 0; other_lidar_id < last_scan_in_target_frame_.size();
          other_lidar_id++) {
        if ( this_lidar_id == other_lidar_id ) { continue; }
        if ( last_scan_is_unpublished_.at(other_lidar_id) ) {
          unpublished_scan_is_available = true;
          // get unpublished scan
          sensor_msgs::LaserScan last_scan_in_target_frame_of_other_lidar = 
            last_scan_in_target_frame_.at(other_lidar_id);
          // merge
          mergeScanIntoScanOfSameFrame(last_scan_in_target_frame_of_other_lidar, merged_scan);
          VLOG(2) << "Merging " << other_lidar_id << " into " << this_lidar_id;
          last_scan_is_unpublished_.at(other_lidar_id) = false;
        }
      }
      if ( unpublished_scan_is_available ) {
        combined_scan_pub_.publish(merged_scan);
        last_scan_is_unpublished_.at(this_lidar_id) = false;
      } else {
        last_scan_is_unpublished_.at(this_lidar_id) = true;
        // Otherwise try to predict if we should switch the grouping to improve sync
        // The following strategy only makes sense for 2 unsynced lidars:
        //
        // time ---->            |now
        //                   |midpoint   |hypothesized next scan if constant rate
        //  first lidar: 1   x   1       ?
        // second lidar:       2       ?
        // published   : |_____|
        // Heuristic: if last published scan of other lidar (2) is closer to now than half the time since
        //            the last scan of this lidar (1), our current grouping is inefficient.
        //            solution -> Publish this scan alone to switch staggering order.
        //            Assuming constant publishing rate for both lidars this guarantees
        //            as-close-as-possible lidar merging.
        sensor_msgs::LaserScan last_scan_in_target_frame_of_this_lidar =
          last_scan_in_target_frame_.at(this_lidar_id);
        ros::Duration time_since_last_scan_of_this_lidar = 
          msg->header.stamp - last_scan_in_target_frame_of_this_lidar.header.stamp;
        ros::Time midpoint = last_scan_in_target_frame_of_this_lidar.header.stamp + 
          ros::Duration(time_since_last_scan_of_this_lidar.sec / 2.,
                        time_since_last_scan_of_this_lidar.nsec / 2.);

        for ( size_t other_lidar_id = 0; other_lidar_id < last_scan_in_target_frame_.size();
            other_lidar_id++) {
          if ( this_lidar_id == other_lidar_id ) { continue; }
          sensor_msgs::LaserScan last_scan_in_target_frame_of_other_lidar = 
            last_scan_in_target_frame_.at(other_lidar_id);
          if ( last_scan_in_target_frame_of_other_lidar.header.stamp > midpoint ) {
            // Publish this scan alone in the hope of improving group sync
            VLOG(1) << "Publishing scan " << this_lidar_id << " alone to switch order";
            combined_scan_pub_.publish(scan_in_target_frame);
            last_scan_is_unpublished_.at(this_lidar_id) = false;
          }
        }
      }
      mutex_.unlock();
      last_scan_in_target_frame_.at(this_lidar_id) = scan_in_target_frame;
    }




  private:
    // ROS
    ros::NodeHandle& nh_;
    ros::Subscriber scan_1_sub_;
    ros::Subscriber scan_2_sub_;
    ros::Publisher combined_scan_pub_;
    tf::TransformListener tf_listener_;
    // State
    std::vector<bool> tf_is_known_;
    std::string kTargetScanFrame;
    std::vector<tf::StampedTransform> tf_lidar_i_in_target_;
    std::vector<sensor_msgs::LaserScan> last_scan_in_target_frame_;
    std::vector<bool> last_scan_is_unpublished_;
    std::vector<bool> is_waiting_for_first_scan_;
    std::mutex mutex_;

}; // class LaserScanCombiner

} // namespace combine_laser_scans

using namespace combine_laser_scans;

int main(int argc, char **argv) {

  ros::init(argc, argv, "combine_laser_scans");
  ros::NodeHandle n;
  LaserScanCombiner laser_scan_combiner(n);

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

