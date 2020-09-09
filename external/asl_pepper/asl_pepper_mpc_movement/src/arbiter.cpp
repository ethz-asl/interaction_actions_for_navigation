//#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include "std_msgs/Float64.h"
#include "math.h"
#include <eigen3/Eigen/Dense> // works with eigen3/ added. 
//#include <eigen3/Eigen/Core> // works with eigen3/ added. 

//#include "local_interpolation.h"
#include "data_manager.h"

// Custom Message: 
#include "asl_pepper_mpc_movement/RawMpcData.h"

// read out values to file
#include <iostream>
#include <fstream>
#include <ctime>
#include <stdio.h>
//using namespace std;

#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// Alternative planner stuff: 
#include <tf/transform_listener.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <base_local_planner/trajectory_planner_ros.h>

// Idea: Send goal commands from here, for testing, and check for arrival or other metrics. 

void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg){


}


int main(int argc, char **argv)

{
  
  ros::init(argc, argv, "arbiter");
  ros::NodeHandle node("~");

  // Wait with sending the first goal, until the controller is active == /cmd_vel starts sending. 
  ros::Subscriber sub_cmd_vel = node.subscribe("/cmd_vel", 10, &cmdVelCallback); 

  int rate = 20;

  ros::Rate loop_rate(rate); // how often ros excecutes the while loop below

  // Printing the iteration block
  int countIterations = 0;// counts iterations


  while (ros::ok()){
  
    std::cout << "Arbiter in action. i = " << countIterations << std::endl;
    countIterations++; 

    // Publishing a goal setpoint
    if(countIterations == 50){
      //if(!butler.receivedInitialCallbacks){

      // Goal message example: 
      // header: 
      //   seq: 0
      //   stamp: 
      //     secs: 4
      //     nsecs:  99999999
      //   frame_id: "base_footprint"
      // pose: 
      //   position: 
      //     x: -1.66830646992
      //     y: -0.175897836685
      //     z: 0.0
      //   orientation: 
      //     x: 0.0
      //     y: 0.0
      //     z: 0.999386680552
      //     w: 0.0350180344421

      geometry_msgs::PoseStamped someGoal; 
      someGoal.header.frame_id = "base_footprint"; 

      someGoal.pose.position.x = -1.7; 
      someGoal.pose.position.y = -0.2; 
      someGoal.pose.position.z = 0.0; 

      someGoal.pose.orientation.x = 0.0; 
      someGoal.pose.orientation.y = 0.0; 
      someGoal.pose.orientation.z = 0.999386680552; 
      someGoal.pose.orientation.w = 0.0350180344421; 

      //pub_goal.publish(someGoal); 

    }

    if(countIterations>150){
      ros::shutdown(); // Sufficient to stop other stuff, too? Need some way to 'rosnode kill -a'? 

    }

    ros::spinOnce();
    loop_rate.sleep();
    
  }
  
}

