
#include "data_manager.h"

Data_Manager::Data_Manager(){
  // std::cout << "Constructor called." << std::endl; 
  turmoil_timestamp_=ros::Time::now().toSec(); 
}

void Data_Manager::storeParameters(double dt, double v_ref, double max_speed, int n_obst, int n_agent, double robot_radius, 
  double lookahead_distance, double goal_proximity, 
  double turnTime, double turnLimit){

  desired_speed_ = v_ref; 
  max_speed_ = max_speed; 
  acado_dt_ = dt; 

  n_obst_ = n_obst; 

  // Consider storing the dimensions as params somewhere to avoid hard-coded stuff? 
  // More flexible to changes on the Matlab end? 
  obst_data_.resize(n_obst_, 3); // 3, not 4. Getting rid of the online weights. Do 'ignored' ones by just setting them far away. 
  obst_data_.setZero(); 

  x_turmoil_.resize(4); 
  x_turmoil_.setZero(); 

  n_agent_ = n_agent; 
  agent_posPlus_.resize(n_agent_, 3); // x, y, radius
  agent_posPlus_.setZero(); 
  agent_velo_.resize(n_agent_,2); // Horizon length and mpc states
  agent_velo_.setZero(); 

  robot_radius_ = robot_radius; 
  lookahead_distance_ = lookahead_distance; 
  goal_proximity_ = goal_proximity; 

  turnTime_ = turnTime; // Completing the turn towards the right direction
  turnLimit_ = turnLimit; // Jerk of the turn

  mpc_full_state_.resize(ACADO_N+1, ACADO_NX); 
  mpc_full_state_.setZero(); 

  if(n_agent_> 0){
    x_agent_.resize(1, n_agent); 
  }
  x_agent_.setZero(); 

  storedFirstParams_ = true; 

}

void Data_Manager::storeTurmoilSystemParameters(double k_turmoil, double c_turmoil, double robot_mass, std::vector<double> A, std::vector<double> B){

  k_turmoil_ = k_turmoil; // Turmoil spring constant
  c_turmoil_ = k_turmoil; // Turmoil damping
  robot_mass_ = robot_mass; // Robot mass

  // Turmoil system dynamics
  A_turmoil_.setZero(); 
  B_turmoil_.setZero(); 

  // Eigen::Matrix<double, 4, 4> A_turmoil;
  // Eigen::Matrix<double, 4, 2> B_turmoil; 

  // Hmmm, bricolage. Works, though! 
  double* A_ptr = &A[0]; 
  Eigen::Map<Eigen::MatrixXd> A_mapper(A_ptr, 4, 4); // Transpose afterwards (fills colums first)
  A_turmoil_ = A_mapper.transpose();

  double* B_ptr = &B[0]; 
  Eigen::Map<Eigen::MatrixXd> B_mapper(B_ptr, 2, 4);  
  B_turmoil_ = B_mapper.transpose(); 

  std::cout << "The stored A: " << A_turmoil_ << std::endl; 
  std::cout << "The stored B: " << B_turmoil_ << std::endl; 
}

void Data_Manager::storeWeights(double weight_pos, double weight_velo, double weight_turmoil, double weight_turmoil_E, 
    double weight_obst, double weight_agent, double weight_input){

  // std::cout << "Storing weights..." << std::endl; 

  if(storedFirstParams_){

    // Not row. They are columns somewhy. 
    Eigen::VectorXd turm(4);
    turm.setOnes(); 

    Eigen::VectorXd obst(n_obst_);
    obst.setOnes(); 

    Eigen::VectorXd agents(n_agent_);
    agents.setOnes(); 

    // Somehow needed for initialisation. 
    weights_.setZero(4+4+2+n_obst_+n_agent_+2); 

    weights_ << weight_pos, weight_pos, weight_velo, weight_velo, // x and y each
      turm*weight_turmoil, 
      weight_turmoil_E, weight_turmoil_E, 
      obst*(weight_obst/(n_obst_*1.0)), // Nasty int trap
      agents*(weight_agent/(n_agent_*1.0)), 
      weight_input, weight_input; 

    // Removed arrival weight switch
    // // More weight on position. 
    // // Eigen: Tail() method. 
    // arrival_weights_ = weights_; 
    // arrival_weights_(0) = 10*weight_pos; 
    // arrival_weights_(1) = 10*weight_pos; 

    // std::cout << "Weights (running): " << weights_ << std::endl; 
    // std::cout << "Weights (arrival): " << arrival_weights_ << std::endl; 

  }else{
    std::cout << "WARNING: Skipping weight storage since n_obst and n_agent haven't been set yet. " << std::endl; 

  }
}

// fancy panel with weight adaptation and stuff. 
void Data_Manager::panelCallback(asl_pepper_mpc_movement::panelConfig &config, uint32_t level){

  // std::cout << "Received some params. Some string: " << config.SomeWeirdStringStuff << std::endl; 

  storeWeights(config.weight_pos, config.weight_velo, config.weight_turmoil, config.weight_turmoil_E, 
    config.weight_obst, config.weight_agent, config.weight_input); 

  // No direct route...
  // agentDetection_ = config.agent_detection; 
  //   enum agentDetection {disable = 0, rwth_tracker = 1, dr_spaam = 2};
  // agentDetection agentDetection_= disable; 
  switch(config.agent_detection){
    case agentDetection::disable : agentDetection_=disable;   break;
    case agentDetection::rwth_tracker : agentDetection_=rwth_tracker;   break;
    case agentDetection::dr_spaam : agentDetection_=dr_spaam;   break;
  }

  setBigRedButtonStatus(config.big_red_button); 
}

bool Data_Manager::stopAutonomousMotionServiceCall(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res){
  STOP_ = true; 
  return true; 
} 
bool Data_Manager::resumeAutonomousMotionServiceCall(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res){
  STOP_ = false; 
  return true; 
}

void Data_Manager::setBigRedButtonStatus(bool STOP){
  STOP_ = STOP; 
}

bool Data_Manager::isStopped(){
  return STOP_;
}

void Data_Manager::vanishAgentsIfNecessary(){
  if(agentDetection_ == disable){

    firstAgentReceived_ = true; 
   
    for(int setting = 0; setting < n_agent_; setting++){

      agent_posPlus_(setting, 0) = x_position_+100.0; 
      agent_posPlus_(setting, 1) = y_position_+100.0; 
      agent_posPlus_(setting, 2) = 0.5; 

      agent_velo_(setting, 0) = 0.0; 
      agent_velo_(setting, 1) = 0.0; 

    }
  }
}

bool Data_Manager::receivedInitialCallbacks(){
  // std::cout << "Flags: firstOdomReceived=" << firstOdomReceived_ << "; firstObstReceived=" << 
  // firstObstReceived_ << "; firstAgentReceived=" << firstAgentReceived_  << std::endl; 

  return (firstOdomReceived_ && firstTfReceived_ && firstObstReceived_ && firstAgentReceived_); 
}

// Position is not accurate enough, drift! 
// void Data_Manager::robotOdometryCallback(const nav_msgs::Odometry::ConstPtr& msg){

//   firstOdomReceived_ = true; 
//   x_position_ = msg->pose.pose.position.x;
//   y_position_ = msg->pose.pose.position.y;
//   z_position_ = msg->pose.pose.position.y;
//   //std::cout << "Odometry callback. Position: " << x_position_ << "/" << y_position_ << std::endl; 

//   // Quaternion block needed? 
//   // x_quaternion_ = msg->pose.pose.orientation.x;
//   // y_quaternion_ = msg->pose.pose.orientation.y;
//   // z_quaternion_ = msg->pose.pose.orientation.z;
//   // w_quaternion_ = msg->pose.pose.orientation.w;

//   // Need Yaw only
//   tf2::Quaternion quat(
//     msg->pose.pose.orientation.x,
//     msg->pose.pose.orientation.y,
//     msg->pose.pose.orientation.z,
//     msg->pose.pose.orientation.w);
//   tf2::Matrix3x3 m(quat);

//   double roll, pitch, yaw;
//   m.getRPY(roll, pitch, yaw);

//   //std::cout << "The yaw computed in robotOdometryCallback: " << yaw << std::endl; 
//   yaw_ =  yaw; 
//   //std::cout << "Odometry callback. Calculated the following yaw: " << yaw_ << std::endl; 

//   // These are in robot coordinates? Unlike the position ones? Weird. It's all specified in the definition! 
//   // # This represents an estimate of a position and velocity in free space.  
//   // # The pose in this message should be specified in the coordinate frame given by header.frame_id.
//   // # The twist in this message should be specified in the coordinate frame given by the child_frame_id
//   // Header header
//   // string child_frame_id
//   // geometry_msgs/PoseWithCovariance pose
//   // geometry_msgs/TwistWithCovariance twist

//   // Proper translation this time, instead of hand-crafted 2d rotation?  Seems to be tricky. 

//   double x_velo_robo = msg->twist.twist.linear.x; 
//   double y_velo_robo = msg->twist.twist.linear.y; 
//   //std::cout << "Odometry callback. Received values(x/y): " << x_velo_robo << "/" << y_velo_robo << std::endl; 

//   // Careful: Need positive direction here. 
//   x_velocity_ = std::cos(yaw)*x_velo_robo-std::sin(yaw)*y_velo_robo; 
//   y_velocity_ = std::sin(yaw)*x_velo_robo+std::cos(yaw)*y_velo_robo; 
 
//   // x_velocity_ = msg->twist.twist.linear.x;
//   // y_velocity_ = msg->twist.twist.linear.y;
//   //std::cout << "Odometry callback. Storing these velocity values (x/y): " << x_velocity_ << "/" << y_velocity_ << std::endl; 

//   z_velocity_ = msg->twist.twist.linear.z;


//   x_angular_velocity_ = msg->twist.twist.angular.x;
//   y_angular_velocity_ = msg->twist.twist.angular.y;
//   z_angular_velocity_ = msg->twist.twist.angular.z;


//   // To prevent the robot from going to the default zero when no goal has arrived yet
//   if (!storedFirstWaypoint_){
//     x_goal_ = x_position_;
//     y_goal_ = y_position_; 
//     yaw_goal_ = yaw_; 

//     storedFirstWaypoint_ = true; 
//   }
//   // Even save the entire pose? 
//   //current_pose_.pose = msg->pose.pose;


//   //std::cout << "robotOdometryCallback called: x/y = " << x_position_ << "/" << y_position_ << std::endl;
// }


void Data_Manager::robotOdometryCallback(const nav_msgs::Odometry::ConstPtr& msg){

  firstOdomReceived_ = true; 

  double x_velo_baseFoot = msg->twist.twist.linear.x; 
  double y_velo_baseFoot = msg->twist.twist.linear.y; 

  // x_velocity_ = msg->twist.twist.linear.x;
  // y_velocity_ = msg->twist.twist.linear.y;
  // Careful: Need positive rotation direction here. 
  x_velocity_ = std::cos(yaw_baseFoot_refMap_)*x_velo_baseFoot-std::sin(yaw_baseFoot_refMap_)*y_velo_baseFoot; 
  y_velocity_ = std::sin(yaw_baseFoot_refMap_)*x_velo_baseFoot+std::cos(yaw_baseFoot_refMap_)*y_velo_baseFoot; 
 
  // std::cout << "Odometry callback. Storing these velocity values (x/y): " << x_velocity_ << "/" << y_velocity_ << std::endl; 

  z_velocity_ = msg->twist.twist.linear.z;

  x_angular_velocity_ = msg->twist.twist.angular.x;
  y_angular_velocity_ = msg->twist.twist.angular.y;
  z_angular_velocity_ = msg->twist.twist.angular.z;

  // Even save the entire pose? 
  // current_pose_.pose = msg->pose.pose;

  //*******************************************//
  double local_dt = ros::Time::now().toSec() - turmoil_timestamp_;

  double u_forwards = 0.0; 
  double u_sideways = 0.0; 
  if(local_dt>0.01){// Avoid division by 0
      u_forwards = (x_velo_baseFoot-previous_velo_x_robot_)/local_dt; 
      u_sideways = (y_velo_baseFoot-previous_velo_y_robot_)/local_dt; 
  }

  // Consider picking a separate dt here, more precise than the acado_dt
  int iterations = std::round(local_dt/acado_dt_); 
  
  // std::cout << "Turmoil System: Number of steps:" << iterations << std::endl; 

  if(iterations>0){
    // Re-distribute the input over the picked interval
    u_forwards = u_forwards*local_dt/(iterations*acado_dt_); 
    u_sideways = u_sideways*local_dt/(iterations*acado_dt_); 

    Eigen::Vector2d u_const; 
    u_const << u_forwards, u_sideways; 

    // std::cout << "Turmoil System: The calculated u:" << u_const << std::endl; 

    // Assuming constant input (Might be unrealistic...)
    for(int it = 0; it<iterations; it++){
      // Simple Matrix multiplication
      x_turmoil_ = ( A_turmoil_*x_turmoil_.transpose() +  B_turmoil_*u_const).transpose(); 
    }
    // std::cout << "Turmoil System: The calculated turmoil:" << x_turmoil_ << std::endl; 

    // Next iteration
    turmoil_timestamp_ = ros::Time::now().toSec(); 
    previous_velo_x_robot_ = x_velo_baseFoot; 
    previous_velo_y_robot_ = y_velo_baseFoot; 

  }else{
    std::cout << "Turmoil System: Not running, Timestep is too short. " << std::endl; 
  }

  // Update the turmoil state by feeding it to the discrete system
  // Turmoil states update, one Input for ACADO: 

  // Manual system propagation, one dt step: 
  // Simple Matrix multiplication? 
  // system_matrix= 

  //   % Dynamics: 
  // % f.add(dot(x_turmoil) == x_turmoil_dot); 
  // % f.add(dot(y_turmoil) == y_turmoil_dot); 
  // % f.add(dot(x_turmoil_dot) == -(k/m)*x_turmoil -(c/m)*x_turmoil_dot + u_x_speed_dot); 
  // % f.add(dot(y_turmoil_dot) == -(k/m)*y_turmoil -(c/m)*y_turmoil_dot + u_y_speed_dot); 
  //std::cout << "Previous velocity values: " << previous_velo_x << "/" << previous_velo_y << std::endl; 

  // double turmoil_c = 0.2; // TODO: Move to param
  // double local_dt = ros::Time::now().toSec() - turmoil_timestamp_; 

  // // Cover zero case? Yep, happens at the start. 
  // // It's not stable in the first place, and therefore not useful at all. 
  // if(local_dt>=0.03){// Race conditions...
  //   double x5_new = x_turmoil_(0) + local_dt*x_turmoil_(2); 
  //   double x6_new = x_turmoil_(1) + local_dt*x_turmoil_(3);
  //   double x7_new = x_turmoil_(2) + local_dt*( -(k_turmoil_/robot_mass_)*x_turmoil_(0) -(turmoil_c/robot_mass_)*x_turmoil_(2)) + (x_velocity_-previous_velo_x_robot_); // dt/dt cancels out.
  //   double x8_new = x_turmoil_(3) + local_dt*( -(k_turmoil_/robot_mass_)*x_turmoil_(1) -(turmoil_c/robot_mass_)*x_turmoil_(3)) + (y_velocity_-previous_velo_y_robot_); 

  //   x_turmoil_ << x5_new, x6_new, x7_new, x8_new; 

  //   std::cout << "Turmoil reset with dt = " << local_dt << std::endl; 
  //   std::cout << "... x_velo/previous  = " << x_velocity_ << "/" << previous_velo_x_robot_ << std::endl; 
  //   std::cout << "... y_velo/previous  = " << y_velocity_ << "/" << previous_velo_y_robot_ << std::endl; 

  // }

  // turmoil_timestamp_ = ros::Time::now().toSec(); 
  // previous_velo_x_robot_ = x_velocity_; 
  // previous_velo_y_robot_ = y_velocity_; 

}


void Data_Manager::robotPositionUpdate(geometry_msgs::TransformStamped& shift){

  firstTfReceived_ = true; 

  // std::cout << "Updating the robot position. " << std::endl; 

  // Accepting the reference map as the global origin...
  x_position_ = shift.transform.translation.x; 
  y_position_ = shift.transform.translation.y;
  z_position_ = shift.transform.translation.z;

  // std::cout << "TF stuff. Position: " << x_position_ << "/" << y_position_ << std::endl; 

  // Quaternion block needed? 
  // x_quaternion_ = msg->pose.pose.orientation.x;
  // y_quaternion_ = msg->pose.pose.orientation.y;
  // z_quaternion_ = msg->pose.pose.orientation.z;
  // w_quaternion_ = msg->pose.pose.orientation.w;

  // Need Yaw: 
  tf2::Quaternion quat(
    shift.transform.rotation.x,
    shift.transform.rotation.y,
    shift.transform.rotation.z,
    shift.transform.rotation.w);
  tf2::Matrix3x3 m(quat);

  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  yaw_baseFoot_refMap_ = yaw; 

  // std::cout << "TF stuff. Calculated the following yaw: " << yaw_baseFoot_refMap_ << std::endl; 

  // These are in robot coordinates? Unlike the position ones? Weird. It's all specified in the definition! 
  // # This represents an estimate of a position and velocity in free space.  
  // # The pose in this message should be specified in the coordinate frame given by header.frame_id.
  // # The twist in this message should be specified in the coordinate frame given by the child_frame_id
  // Header header
  // string child_frame_id
  // geometry_msgs/PoseWithCovariance pose
  // geometry_msgs/TwistWithCovariance twist

  // Proper translation this time, instead of hand-crafted 2d rotation?  Seems to be tricky. 

  // To prevent the robot from going to the default zero when no goal has arrived yet
  if (!storedFirstWaypoint_){
    x_goal_ = x_position_;
    y_goal_ = y_position_; 
    yaw_goal_ = yaw_baseFoot_refMap_; 

    storedFirstWaypoint_ = true; 
  }
  // Even save the entire pose? 
  //current_pose_.pose = msg->pose.pose;

  //std::cout << "robotOdometryCallback called: x/y = " << x_position_ << "/" << y_position_ << std::endl;
}

void Data_Manager::plannerCallback (const nav_msgs::Path::ConstPtr& msg){

  // Varies a lot, from 1 to 5-7 up to 20
  //std::cout << "We're listening. Path length: " << msg->poses.size() << std::endl; 

  // No need to store it.
  // std::Vector<flaot> accumulatedDistance; 
  float accumulatedDistance = 0.0f; 
  float bit = 0.0f; 
  int pickedPoint = 0; 
  bool foundIntermediateTarget = false; 

  for (int it = 0; it<msg->poses.size(); it++){
      // Euclidian Distance
      if (it == 0){
          bit = sqrt( pow(msg->poses[it].pose.position.x - x_position_,2) + pow(msg->poses[it].pose.position.y - y_position_,2)); 
      }else{
          bit = sqrt( pow(msg->poses[it].pose.position.x - msg->poses[it-1].pose.position.x,2) + 
                      pow(msg->poses[it].pose.position.y - msg->poses[it-1].pose.position.y,2)); 
      }

      accumulatedDistance += bit; 
      //std::cout << "Toggler bool: " << foundTarget << " Goal: " << lookahead_distance_ << " Gathered: " << accumulatedDistance << std::endl; 

      if (!foundIntermediateTarget && (accumulatedDistance > lookahead_distance_)){
          pickedPoint = it; 
          foundIntermediateTarget = true; 
          if(pickedPoint == msg->poses.size()-1){
            waypointIsGoal_ = true; 
          }else{
            waypointIsGoal_ = false; 
          }
          // No break, need the total distance
      }
      //std::cout << "**  A Waypoint (x/y): " << msg->poses[it].pose.position.x << "/" << msg->poses[it].pose.position.y << std::endl; 
  }

  // Just pick the goal in this case. 
  if(!foundIntermediateTarget){
    pickedPoint = msg->poses.size()-1; // Array indexing stuff.  
    waypointIsGoal_ = true; 
  }
  
  //std::cout << "Accumulated distance between path waypoints: " << accumulatedDistance << std::endl; 

  // Path seems to be in local robot coordinates
  x_goal_ = msg->poses[pickedPoint].pose.position.x; 
  y_goal_ = msg->poses[pickedPoint].pose.position.y; 

  // Calculate speed setpoints based on the general direction: 
  if (accumulatedDistance > goal_proximity_){
    double ant_x = x_goal_-x_position_; 
    double ant_y = y_goal_-y_position_; 

    // // Huh! Sum's not constant with that...
    // double phi = atan2(ant_y, ant_x); 
    // x_speed_goal_ = cos(phi)*desired_speed_; 
    // y_speed_goal_ = sin(phi)*desired_speed_; 

    double m = abs(ant_x/ant_y); 
    double x_speed_share = desired_speed_*(m/(1+m)); 
    double y_speed_share = desired_speed_-x_speed_share; 
    if(ant_x>=0.0){
      x_speed_goal_ = x_speed_share; 
    }else{
      x_speed_goal_ = -x_speed_share; 
    }
    if(ant_y>=0.0){
      y_speed_goal_ = y_speed_share; 
    }else{
      y_speed_goal_ = -y_speed_share; 
    }
    // Consider taking out of if statement. Good aspect element: Doesn't keep changing direction when circling a goal. 

    yaw_goal_ = atan2(ant_y, ant_x); 
    //std::cout << "Planner Callback. Calculated this target yaw: " << yaw_goal_ << std::endl; 

    // Temporarily, to avoid coordinate/model issues! 
    // yaw_goal_ = 0.0; 
    // yaw_goal_ = M_PI/2.0; 

    // Remove, relict from weight switch? 
    setAcadoOptimisationWeights(weights_); 

  }else{

    // That's too radical? Better do a ramp? 
    x_speed_goal_ = 0.0; 
    y_speed_goal_ = 0.0; 

    // Rebalance weights for more positional accuracy (As the ACADO guys did) (and to ensure that it acually does go to the desired position? )
    // Messed up...
    // setAcadoOptimisationWeights(arrival_weights_);
  }

  // std::cout << "The picked setpoint (path entry " << pickedPoint << "):" << std::endl;  
  // std::cout << "--- x_position=" << x_position_ << "; y_position=" << y_position_ << std::endl; 
  // std::cout << "--- x_goal=" << x_goal_ <<"; y_goal=" << y_goal_ << std::endl; 
  // std::cout << "--- x_speed_goal=" << x_speed_goal_ << "; y_speed_goal=" << y_speed_goal_ << std::endl; 
}

void Data_Manager::obstCallback (const boost::shared_ptr<sensor_msgs::LaserScan_<std::allocator<void> > const>& msg){

  firstObstReceived_ = true; 
  int count = msg->ranges.size(); 

  //std::cout << "Obstacle callback. Grabbing info: " << msg->range_min <<"//"<< msg->range_max <<"//"<< msg->ranges.size() << std::endl; 
  
  // sort ranges: 
  // initialize original index locations

  // Trouble: Ranges are 0 for the chose points. Comment in the message: 
  //"float32 range_min        # minimum range value [m]\n"
  //"float32 range_max        # maximum range value [m]\n"
  //"\n"
  //"float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)\n"

  std::vector<float> dist = msg->ranges;

  double min = msg->range_min; 
  double max = msg->range_max; 
  if(min>robot_radius_){
    std::cout << "WARNING: Obst callback: Minimum scan range is shorter than the robot radius." << std::endl; 
  }
  // std::cout << "Scan properties: min=" << min << " max=" << max << std::endl; 

  // Just set low entries to high ones! 
  // Better library functions somewhere? Vector comparison stuff in c++20? 
  for (int goer = 0; goer<dist.size(); goer++){
    if(dist[goer] < min){
      dist[goer] = max; 
    }
  }

  // More filtering: Reduce the overlap of the resulting circles
  int slice = 0; 
  double tempRange = 0.0;
  int localSlices = 0; 

  while(slice<dist.size()){
    // Calculate the angle corresponding to that certain range setpoint. 
    tempRange = dist[slice]; 
    localSlices = floor(atan2(robot_radius_, tempRange)/msg->angle_increment); // No factor two, allow for some margim. 
    
    // Finally using do-while! 
    do{
      slice++; 
      localSlices--; 
      // incremented for the first already, good. 
      if(abs(dist[slice]-tempRange) < 0.5*robot_radius_){
        //std::cout << "Resetting..." << std::endl; 
        dist[slice] = max; 
      }else{
        localSlices = 0; // soft break
      }
    }while(localSlices>0); 
  }

  // Push back the ones too close. 
  // Attempt to recover/prevent situations where the robot gets stuck due to constraint violation. Nasty. 
  for(int reco = 0; reco<dist.size(); reco++){
    if(dist[reco]<robot_radius_){
      dist[reco] = robot_radius_+0.1; 
    }
  }

  /* Need idx...
  for( auto &element : dist){
    if (element)
  }*/

  std::vector<int> pos(count);
  std::iota(pos.begin(), pos.end(), 0); // Fills vector with increasing values
  // sort position based on dist, great. 
  std::stable_sort(pos.begin(), pos.end(), [&dist](size_t i1, size_t i2) {return dist[i1] < dist[i2];});

  obst_data_.setZero(); 
  double section = msg->angle_increment; 
  for(int grabbing = 0; grabbing < n_obst_; grabbing++){

    // Translate to global coordinates + Store locally
    // Def: Angle is around the z axis, clockwise, zero = forward = x axis. Unusual -> switch sin and cos. 

    //std::cout << "Conversion test: index=" << pos[grabbing] << " angle=" << pos[grabbing]*section << " range="<< dist[pos[grabbing]] << std::endl; 
    
    if(grabbing < count){
      obst_data_(grabbing, 0) = x_position_ + dist[pos[grabbing]]*cos(pos[grabbing]*section+yaw_baseFoot_refMap_); // Global x direction
      obst_data_(grabbing, 1) = y_position_ + dist[pos[grabbing]]*sin(pos[grabbing]*section+yaw_baseFoot_refMap_); 
      obst_data_(grabbing, 2) = robot_radius_; 
    }else{// Far away from the robot, just to be sure. 
      obst_data_(grabbing, 0) = x_position_ + 100.0; 
      obst_data_(grabbing, 1) = y_position_ + 100.0; 
      obst_data_(grabbing, 2) = robot_radius_; 
    }

  }

  //std::cout << "Here is the final obstacle matrix:\n" << obst_data_ << std::endl;

}

// Store Agent positions + Velocities
//void agentCallback (const spencer_tracking_msgs::TrackedPersons_<std::allocator<int> agentMsgItter>::ConstPtr& msg){
void Data_Manager::agentCallback (const boost::shared_ptr<frame_msgs::TrackedPersons_<std::allocator<void> > const>& msg){

  // Ony use one source at a time, even if both detectors are running. 
  if(agentDetection_ == rwth_tracker){

    firstAgentReceived_ = true; 
    int n = msg->tracks.size(); 

    // std::cout << "Received agent callback, n = " << n << std::endl; 

    // build 2d vector with disdances (no need for root)
    std::vector<float> dist;
    double dd = 0.0; 

    for(int coronaRubbish = 0; coronaRubbish < n; coronaRubbish++){
      dd = pow(msg->tracks[coronaRubbish].pose.pose.position.x - x_position_,2) + pow(msg->tracks[coronaRubbish].pose.pose.position.y - y_position_,2); 
      dist.push_back(dd); 
    }

    // sort indices based on dist, great. 
    std::vector<int> number(n);
    std::iota(number.begin(), number.end(), 0); // Fills vector with increasing values
    std::stable_sort(number.begin(), number.end(), [&dist](size_t i1, size_t i2) {return dist[i1] < dist[i2];});

    // Storing Data
    agent_posPlus_.setZero(); 
    agent_velo_.setZero(); 
    for(int setting = 0; setting < n_agent_; setting++){
      if(setting < n){ // Need to set to bogus values to avoid derivative-of-sqrt(0)-issue. 
        // Idea: Radius scaled with covariance (average in x and y)? 
        agent_posPlus_(setting, 0) = msg->tracks[number[setting]].pose.pose.position.x; 
        agent_posPlus_(setting, 1) = msg->tracks[number[setting]].pose.pose.position.y; 
        agent_posPlus_(setting, 2) = 0.5; // Consider getting that from the detector, obese people? 

        agent_velo_(setting, 0) = msg->tracks[number[setting]].twist.twist.linear.x; 
        agent_velo_(setting, 1) = msg->tracks[number[setting]].twist.twist.linear.y; 
      }else{
        // Far away, being certain. 
        agent_posPlus_(setting, 0) = x_position_+100.0; 
        agent_posPlus_(setting, 1) = y_position_+100.0; 
        agent_posPlus_(setting, 2) = 0.5; 

        agent_velo_(setting, 0) = 0.0; 
        agent_velo_(setting, 1) = 0.0; 
      }
      
    }
    //std::cout << "The resulting agent posPlus vector:\n"  << agent_posPlus_ << std::endl; 
    //std::cout << "And the agent velovector:\n"  << agent_velo_ << std::endl; 
  }
}

void Data_Manager::agentCallbackDrSpaam(const boost::shared_ptr<geometry_msgs::PoseArray_<std::allocator<void> > const>& msg){

  if(agentDetection_ == dr_spaam){

    firstAgentReceived_ = true; 
    int n = msg->poses.size(); 

    std::cout << "Received agent callback (DR-SPAAM version!), n = " << n << std::endl; 

    // build 2d vector with disdances (no need for root)
    std::vector<float> dist;
    double dd = 0.0; 

    // Assuming that it's in the laser scan frame
    for(int coronaRubbish = 0; coronaRubbish < n; coronaRubbish++){
      dd = pow(msg->poses[coronaRubbish].position.x - x_position_,2) + pow(msg->poses[coronaRubbish].position.y - y_position_,2); 
      dist.push_back(dd); 
    }

    // sort indices based on dist, great. 
    std::vector<int> number(n);
    std::iota(number.begin(), number.end(), 0); // Fills vector with increasing values
    std::stable_sort(number.begin(), number.end(), [&dist](size_t i1, size_t i2) {return dist[i1] < dist[i2];});

    // Storing Data
    agent_posPlus_.setZero(); 
    agent_velo_.setZero(); 

    double frontLaserOffset_x = 0.2520; 
    double frontLaserOffset_y = 0.0; 

    for(int setting = 0; setting < n_agent_; setting++){
      if(setting < n){ // Need to set to bogus values to avoid derivative-of-sqrt(0)-issue. 

        double agent_x_baseFoot = msg->poses[number[setting]].position.x + frontLaserOffset_x; 
        double agent_y_baseFoot = msg->poses[number[setting]].position.y + frontLaserOffset_y; 

        // There's an offset if it's in the front_scan tf... Need to consider this! 
        // Assuming constant ones: 0.2520/0/0.2340

        agent_posPlus_(setting, 0) = x_position_ + std::cos(yaw_baseFoot_refMap_)*agent_x_baseFoot-std::sin(yaw_baseFoot_refMap_)*agent_y_baseFoot; 
        agent_posPlus_(setting, 1) = y_position_ + std::sin(yaw_baseFoot_refMap_)*agent_x_baseFoot+std::cos(yaw_baseFoot_refMap_)*agent_y_baseFoot; 
        agent_posPlus_(setting, 2) = 0.5; // TODO: Think about that! 

        // Hmmm, There's no velocity info in this topic. Estimate somehow? (Past positions etc.)
        // agent_velo_(setting, 0) = msg->tracks[number[setting]].twist.twist.linear.x; 
        // agent_velo_(setting, 1) = msg->tracks[number[setting]].twist.twist.linear.y; 
        agent_velo_(setting, 0) = 0.0; 
        agent_velo_(setting, 1) = 0.0; 

      }else{
        // Far away, we're safe. 
        agent_posPlus_(setting, 0) = x_position_+100.0; 
        agent_posPlus_(setting, 1) = y_position_+100.0; 
        agent_posPlus_(setting, 2) = 0.5; 

        agent_velo_(setting, 0) = 0.0; 
        agent_velo_(setting, 1) = 0.0; 
      }
      
    }
    std::cout << "The resulting agent posPlus vector:\n"  << agent_posPlus_ << std::endl; 
    std::cout << "And the agent velovector:\n"  << agent_velo_ << std::endl; 
  }
}

void Data_Manager::setAcadoOptimisationWeights (Eigen::VectorXd weight_vec){
  W_ = weight_vec.asDiagonal();
  WN_ = weight_vec.head(ACADO_NYN).asDiagonal(); // Tricky: Probably leaving out the output ones. 

  // Update acado variables with new weighting matrices
  Eigen::Map<Eigen::Matrix<double, ACADO_NY, ACADO_NY>>(const_cast<double*>
      (acadoVariables.W)) = W_.transpose();
  Eigen::Map<Eigen::Matrix<double, ACADO_NYN, ACADO_NYN>>(const_cast<double*>
      (acadoVariables.WN)) = WN_.transpose();
}

void Data_Manager::setAcadoInput(){

  Eigen::RowVectorXd x_0(ACADO_NX);

  x_0.setZero(); 
  x_0 << x_position_, y_position_, x_velocity_, y_velocity_, // basic states
     x_turmoil_, // Turmoil states. Reset after a collision? % Should be based on 'Input applied actually?'
     agent_posPlus_.col(0).transpose(), agent_posPlus_.col(1).transpose(), agent_posPlus_.col(2).transpose(); //Agent states (position + radius)
  
  // std::cout << "x0: " << x_0 << std::endl;
  // std::cout << "Basic states: " << x_0.head(4) << std::endl;

  if(acadoAdditionalFlush_) {
    setAcadoAdditionalFlush(x_0);
    acadoAdditionalFlush_ = false;
  }

  // Map to h
  // Eigen::Matrix<double, ACADO_NY, ACADO_NY> W_;
  Eigen::RowVectorXd x_ref_obst(n_obst_); 
  x_ref_obst.setZero(); 
  Eigen::RowVectorXd x_ref_agent(n_agent_); 
  x_ref_agent.setZero(); 

  Eigen::RowVectorXd x_ref(ACADO_NY); // Acado_ny = 12 atm. Energy states? Must be 4x basic, 4x turmoil, 2x Energy, 2x Input. 
  x_ref.setZero(); 
  x_ref << x_goal_, y_goal_, x_speed_goal_, y_speed_goal_,  //x, y, x_speed, y_speed
       0.0, 0.0, 0.0, 0.0,  // Turmoil states
       0.0, 0.0, // Turmoil energy
       x_ref_obst, // Obstacles 
       x_ref_agent, // agents
       0.0, 0.0;// Inputs

  // std::cout << "x_ref: " << x_ref << std::endl;
  //std::cout << "Setpoint: " << x_ref.head(4) << std::endl;

  // Online Data
  Eigen::RowVectorXd od_tot(ACADO_NOD); 
  od_tot.setZero(); 

  od_tot << x_ref.head(4), 
      obst_data_.col(0).transpose(), obst_data_.col(1).transpose(), obst_data_.col(2).transpose(), // Obstacle position, radius
      agent_velo_.col(0).transpose(), agent_velo_.col(1).transpose(), // Current agent speed
      yaw_baseFoot_refMap_; // Angle
  
  // std::cout << "Online data row: " << od_tot << std::endl; 

  for(int i = 0; i < ACADO_N; ++i) {
    // Eigen: 'at i/0 of size 1/Const'
    x_reference_.block(i, 0, 1, ACADO_NY) = x_ref; 

    // They: Lots of isnan() tests

    acado_online_data_.block(i, 0, 1, ACADO_NOD) = od_tot;//Eigen::RowVectorXd::Zero(ACADO_NOD);
  }

  // End state
  // Could get this with head()... 
  x_referenceN_.setZero(); // Just to be safe
  x_referenceN_ << x_goal_, y_goal_, x_speed_goal_, y_speed_goal_,  // Keep speed up, not zero for velocities. 
      0.0, 0.0, 0.0, 0.0,  // Turmoil states
      0.0, 0.0, // Turmoil energy, no inputs
      x_ref_obst, 
      x_ref_agent; // Agents

  // Last entry of online data. Is it different? Just due to the loop design, assumingly. 
  acado_online_data_.block(ACADO_N, 0, 1, ACADO_NOD) = od_tot; //x_ref.head(4);

  // std::cout << "reference_: " << reference_ << std::endl;
  // std::cout << "referenceN_: " << referenceN_ << std::endl;

  // pass updated system params to acado
  // Hope that the transpose is correct
  Eigen::Map<Eigen::Matrix<double, ACADO_NX, 1>>(const_cast<double*>(acadoVariables.x0)) = x_0;
  Eigen::Map<Eigen::Matrix<double, ACADO_NY, ACADO_N>>(const_cast<double*>(acadoVariables.y)) = x_reference_.transpose();
  Eigen::Map<Eigen::Matrix<double, ACADO_NYN, 1>>(const_cast<double*>(acadoVariables.yN)) = x_referenceN_.transpose();
  Eigen::Map<Eigen::Matrix<double, ACADO_NOD, ACADO_N + 1>>(const_cast<double*>(acadoVariables.od)) = acado_online_data_.transpose();

}

void Data_Manager::flushStatesNextTime(){
  acadoAdditionalFlush_ = true; 
}

void Data_Manager::setAcadoAdditionalFlush(Eigen::VectorXd x0){
  for (int i = 0; i < ACADO_N + 1; i++) {
    state_.block(i, 0, 1, ACADO_NX) << x0.transpose();
  }
  // Fills the state matrix with the initial state
  Eigen::Map<Eigen::Matrix<double, ACADO_NX, ACADO_N + 1>>(const_cast<double*>(acadoVariables.x)) = state_.transpose();
  input_.setZero();
  Eigen::Map<Eigen::Matrix<double, ACADO_NU, ACADO_N>>(const_cast<double*>(acadoVariables.u)) = input_.transpose();
}


void Data_Manager::resetLocalMPCParameters(){

  // All relevant ones gotten? 

  // set all matrices to zero
  W_.setZero();
  WN_.setZero();
  input_.setZero();
  state_.setZero();
  x_reference_.setZero();
  x_referenceN_.setZero();
  acado_online_data_.setZero();

  x_speed_output_robot_ = 0.0; 
  y_speed_output_robot_ = 0.0; 
  //set output vector to zero  
  // output_.clear();
  // output_.push_back(0.0);
  // output_.push_back(0.0);
  x_turmoil_.setZero(); 
  obst_data_.setZero(); 

  agent_posPlus_.setZero(); 
  agent_velo_.setZero(); 

  // Doesn't make much sense here? 
  // default weighting matrix, should be parametrised in future
  Eigen::VectorXd w_diag(ACADO_NY); 
  w_diag = Eigen::VectorXd::Zero(ACADO_NY); // Or ::Ones()

  // Eigen::VectorXd w_diag(ACADO_NY);
  // W_diag << 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.;

  setAcadoOptimisationWeights(w_diag);
} 

bool Data_Manager::calculateRobotPublisherOutput() {

    double velo_x_global = x_velocity_ + acado_dt_ * acadoVariables.u[0]; 
    double velo_y_global = y_velocity_ + acado_dt_ * acadoVariables.u[1]; 

    if (std::isnan(velo_x_global) || std::isnan(velo_y_global)) {
      return false;
    }

    // Rotation: 
    // // x2=cosβx1−sinβy1
    // // y2=sinβx1+cosβy1

    // output.push_back(std::cos(yaw_)*x_speed_output_-std::sin(yaw_)*y_speed_output_); 
    // output.push_back(std::sin(yaw_)*x_speed_output_+std::cos(yaw_)*y_speed_output_);   

    // Simple: Just turn towards the goal, P-controller, basically. 
    double angle_change = (yaw_goal_-yaw_baseFoot_refMap_); 
    // Spillover
    if(angle_change>M_PI){
      angle_change = angle_change-2*M_PI; 
    }else if(angle_change<-M_PI){
      angle_change = angle_change+2*M_PI; 
    }

    double angle_velo = 0.0; 
    if(turnTime_>0.0){
      angle_velo = angle_change/turnTime_; 
    }

    // capping
    // w_output_ = std::clamp( (yaw_goal_-yaw_)/turnTime_, -turnLimit_, turnLimit_); // Not working ...
    angle_velo = (angle_velo < -turnLimit_) ? -turnLimit_ : (turnLimit_ < angle_velo) ? turnLimit_ : angle_velo; 

    w_output_ = angle_velo; 

    

    // Correction due to the ongoing robot rotation, as in the RVO controller?: 
    // TODO: Verify correction sign + factor 0.5 once a proper environment is available. 
    double correction = 0.0; //acado_dt_*angle_velo*0.5; // Should compensate: Too much at first, too little later. 

    // Careful: Need to rotate in the 'negative' direction here. 
    x_speed_output_robot_ = std::cos(-yaw_baseFoot_refMap_ + correction)*velo_x_global-std::sin(-yaw_baseFoot_refMap_ + correction)*velo_y_global; 
    y_speed_output_robot_ = std::sin(-yaw_baseFoot_refMap_ + correction)*velo_x_global+std::cos(-yaw_baseFoot_refMap_ + correction)*velo_y_global; 

    return true;

}

// Redundant now
void Data_Manager::storeCleanTurmoilStates(){

  bool nanFoundTest = isnan(acadoVariables.x[4+ACADO_NX]) || isnan(acadoVariables.x[5+ACADO_NX]) || isnan(acadoVariables.x[6+ACADO_NX]) || isnan(acadoVariables.x[7+ACADO_NX]); 
  x_turmoil_.setZero(); 
  if(!nanFoundTest){// Might be redundant now since it only gets called in the non-NAN-case. 
    // Next element to avoid the integration of the system. 
    x_turmoil_ << acadoVariables.x[4+ACADO_NX], acadoVariables.x[5+ACADO_NX], acadoVariables.x[6+ACADO_NX], acadoVariables.x[7+ACADO_NX]; 
  }
  // std::cout << "Turmoil state storage: " << x_turmoil_ << std::endl; 
}

// Redundant now
void Data_Manager::storeDeadbeatTurmoilStates(double previous_velo_x, double previous_velo_y){
  
  // Manual system propagation, one dt step: 
  // Simple Matrix multiplication? 
  // system_matrix= 

  //   % Dynamics: 
  // % f.add(dot(x_turmoil) == x_turmoil_dot); 
  // % f.add(dot(y_turmoil) == y_turmoil_dot); 
  // % f.add(dot(x_turmoil_dot) == -(k/m)*x_turmoil -(c/m)*x_turmoil_dot + u_x_speed_dot); 
  // % f.add(dot(y_turmoil_dot) == -(k/m)*y_turmoil -(c/m)*y_turmoil_dot + u_y_speed_dot); 
  std::cout << "Previous velocity values: " << previous_velo_x << "/" << previous_velo_y << std::endl; 

  double turmoil_c = 0.2; // TODO: Move to param

  double x5_new = x_turmoil_(0) + acado_dt_*x_turmoil_(2); 
  double x6_new = x_turmoil_(1) + acado_dt_*x_turmoil_(3);

  // Velocity is zero, not the input! The Input is rather negative... How to compute? 
  // Previus setpoint, get difference, divide by dt? 
  // Should be in the dmc_vel field, still. 
  // Could grab it from the odom topic, too? Would be more 'realistic', but less fitting for the reconstruction. 
  double x7_new = x_turmoil_(2) + acado_dt_*( -(k_turmoil_/robot_mass_)*x_turmoil_(0) -(turmoil_c/robot_mass_)*x_turmoil_(2) + (0.0-previous_velo_x)/acado_dt_); 
  double x8_new = x_turmoil_(3) + acado_dt_*( -(k_turmoil_/robot_mass_)*x_turmoil_(1) -(turmoil_c/robot_mass_)*x_turmoil_(3) + (0.0-previous_velo_y)/acado_dt_); 

  x_turmoil_ << x5_new, x6_new, x7_new, x8_new; 
  //x_turmoil_ << 100.0, 100.0, 100.0, 100.0;
}

// Redundant now
void Data_Manager::updateTurmoilStates(){

  // Turmoil states update, one Input for ACADO: 

  // Manual system propagation, one dt step: 
  // Simple Matrix multiplication? 
  // system_matrix= 

  //   % Dynamics: 
  // % f.add(dot(x_turmoil) == x_turmoil_dot); 
  // % f.add(dot(y_turmoil) == y_turmoil_dot); 
  // % f.add(dot(x_turmoil_dot) == -(k/m)*x_turmoil -(c/m)*x_turmoil_dot + u_x_speed_dot); 
  // % f.add(dot(y_turmoil_dot) == -(k/m)*y_turmoil -(c/m)*y_turmoil_dot + u_y_speed_dot); 
  //std::cout << "Previous velocity values: " << previous_velo_x << "/" << previous_velo_y << std::endl; 

  double turmoil_c = 0.2; // TODO: Move to param
  double local_dt = ros::Time::now().toSec() - turmoil_timestamp_; 

  // Cover zero case? Yep, happens at the start. 
  // It's not stable in the first place, and therefore not useful at all. 
  if(local_dt>=0.03){// Race conditions...
    double x5_new = x_turmoil_(0) + local_dt*x_turmoil_(2); 
    double x6_new = x_turmoil_(1) + local_dt*x_turmoil_(3);
    double x7_new = x_turmoil_(2) + local_dt*( -(k_turmoil_/robot_mass_)*x_turmoil_(0) -(turmoil_c/robot_mass_)*x_turmoil_(2)) + (x_velocity_-previous_velo_x_robot_); // dt/dt cancels out.
    double x8_new = x_turmoil_(3) + local_dt*( -(k_turmoil_/robot_mass_)*x_turmoil_(1) -(turmoil_c/robot_mass_)*x_turmoil_(3)) + (y_velocity_-previous_velo_y_robot_); 

    x_turmoil_ << x5_new, x6_new, x7_new, x8_new; 

    std::cout << "Turmoil reset with dt = " << local_dt << std::endl; 
    std::cout << "... x_velo/previous  = " << x_velocity_ << "/" << previous_velo_x_robot_ << std::endl; 
    std::cout << "... y_velo/previous  = " << y_velocity_ << "/" << previous_velo_y_robot_ << std::endl; 

  }

  turmoil_timestamp_ = ros::Time::now().toSec(); 
  previous_velo_x_robot_ = x_velocity_; 
  previous_velo_y_robot_ = y_velocity_; 

}

void Data_Manager::storeFullMpcStates(){

  mpc_full_state_.setZero(); // needed? 
  mpc_full_state_ = Eigen::Map<Eigen::Matrix<double,ACADO_N+1,ACADO_NX,Eigen::RowMajor> >(acadoVariables.x); 

  // std::cout << "The entire state matrix:\n" << Eigen::Map<Eigen::Matrix<double,ACADO_N+1,ACADO_NX,Eigen::RowMajor> >(acadoVariables.x) << std::endl;

}

void Data_Manager::getRobotPublisherOutput(std::vector<double>& output) {

    // push_back() adds at the end, so should be good. 
    output.clear();
    output.push_back(x_speed_output_robot_); 
    output.push_back(y_speed_output_robot_);   
    output.push_back(w_output_);   
}

void Data_Manager::getOptimisationWeights(Eigen::VectorXd& weights) {
    // ZERO DAY: Don't ignore the arrival weights! 
    weights = weights_; 
}


void Data_Manager::prepareObstMessage(visualization_msgs::MarkerArray& obst_markers){
    obst_markers.markers.resize(n_obst_); 

    for (int alle = 0; alle<n_obst_; alle++){
      //visualization_msgs::Marker m; 
      obst_markers.markers[alle].header.frame_id = "reference_map"; // gmap
      obst_markers.markers[alle].header.stamp = ros::Time::now(); 
      obst_markers.markers[alle].action = visualization_msgs::Marker::ADD; 
      obst_markers.markers[alle].lifetime = ros::Duration(); // Doing? 
      obst_markers.markers[alle].type = visualization_msgs::Marker::CYLINDER; 
      obst_markers.markers[alle].id = alle+10; 
      obst_markers.markers[alle].ns = "mpc_pepper"; 

      obst_markers.markers[alle].color.r = 0.0f; 
      obst_markers.markers[alle].color.g = 1.0f; 
      obst_markers.markers[alle].color.b = 0.0f; 
      obst_markers.markers[alle].color.a = 1.0f;

      obst_markers.markers[alle].pose.position.x = obst_data_(alle,0);  
      obst_markers.markers[alle].pose.position.y = obst_data_(alle,1); 
      obst_markers.markers[alle].pose.position.z = 0.0; 

      obst_markers.markers[alle].pose.orientation.x = 0.0; 
      obst_markers.markers[alle].pose.orientation.y = 0.0; 
      obst_markers.markers[alle].pose.orientation.z = 0.0; 
      obst_markers.markers[alle].pose.orientation.w = 1.0; 

      // Cylinder -> Possibly an ellypse. 
      obst_markers.markers[alle].scale.x = obst_data_(alle,2);
      obst_markers.markers[alle].scale.y = obst_data_(alle,2);
      obst_markers.markers[alle].scale.z = 0.05;
    }
}

void Data_Manager::prepareTurmoilIndicator(visualization_msgs::MarkerArray& turmoil_indicator){

    double x_turmoil = x_turmoil_(0); 
    double x_turmoil_dot = x_turmoil_(1); 
    double y_turmoil = x_turmoil_(2); 
    double y_turmoil_dot = x_turmoil_(3); 

    /*
    % 'Energy' of these turmoil states
    E_excitation_x = 0.5*(k/m)*x(:,5).^2; 
    E_movement_x = 0.5*m*x(:,7).^2; 
    E_excitation_y = 0.5*(k/m)*x(:,6).^2; 
    E_movement_y = 0.5*m*x(:,8).^2; 

    % Energy. Gets squared later. 
    x_turmoil_energy = sqrt(0.5*(k/m)*x_turmoil.^2 + 0.5*m*x_turmoil_dot.^2 +1); % Add 1 to avoid the root derivative issue around 0! 
    y_turmoil_energy = sqrt(0.5*(k/m)*y_turmoil.^2 + 0.5*m*y_turmoil_dot.^2 +1); % Add 1 to avoid the root derivative issue around 0!
    */

    std::vector<double> energy; 
    energy.push_back(0.5*(k_turmoil_/robot_mass_)*x_turmoil*x_turmoil); 
    energy.push_back(0.5*robot_mass_*x_turmoil_dot*x_turmoil_dot); 
    energy.push_back(0.5*(k_turmoil_/robot_mass_)*y_turmoil*y_turmoil); 
    energy.push_back(0.5*robot_mass_*y_turmoil_dot*y_turmoil_dot); 

    //std::cout << "Energy values: " << energy[0] << " " << energy[1] << " " << energy[2] << " " << energy[3] << std::endl; 
    
    double rescaleFactor = 20;
    double adder = 0.0; 

    turmoil_indicator.markers.resize(4); 

    for (int alle = 0; alle<4; alle++){
      //visualization_msgs::Marker m; 
      turmoil_indicator.markers[alle].header.frame_id = "base_footprint"; // To be above the robot? 
      turmoil_indicator.markers[alle].header.stamp = ros::Time::now(); 
      turmoil_indicator.markers[alle].action = visualization_msgs::Marker::ADD; 
      turmoil_indicator.markers[alle].lifetime = ros::Duration(); // Doing? Doing? -> expiry date
      turmoil_indicator.markers[alle].type = visualization_msgs::Marker::CYLINDER; 
      turmoil_indicator.markers[alle].id = alle+100; // Avoid intefering with other markers! 
      turmoil_indicator.markers[alle].ns = "mpc_pepper"; 

      // Alternate green and blue
      turmoil_indicator.markers[alle].color.r = 0.0f; 
      if(alle % 2 == 0){
        turmoil_indicator.markers[alle].color.g = 1.0f; 
        turmoil_indicator.markers[alle].color.b = 0.0f; 
      }else{
        turmoil_indicator.markers[alle].color.g = 0.0f; 
        turmoil_indicator.markers[alle].color.b = 1.0f; 
      }
      turmoil_indicator.markers[alle].color.a = 1.0f;

      // On top of pepper
      // This point seems to be in the middle of the cylinder. 
      turmoil_indicator.markers[alle].pose.position.x = 0.0;  
      turmoil_indicator.markers[alle].pose.position.y = 0.0; 
      turmoil_indicator.markers[alle].pose.position.z = 2.0+adder+rescaleFactor*0.5*energy[alle]; //Starts in the middle of it. 
      adder += rescaleFactor*energy[alle]; 

      turmoil_indicator.markers[alle].pose.orientation.x = 0.0; 
      turmoil_indicator.markers[alle].pose.orientation.y = 0.0; 
      turmoil_indicator.markers[alle].pose.orientation.z = 0.0; 
      turmoil_indicator.markers[alle].pose.orientation.w = 1.0; 

      // Set the scale of the marker -- 1x1x1 here means 1m on a side
      // Behaviour with the cylinder? -> Possibly an ellypse. 
      // Goes both ways -> factor 0.5 for cylinder height? No, It's the total height. 
      turmoil_indicator.markers[alle].scale.x = 0.2;
      turmoil_indicator.markers[alle].scale.y = 0.2;
      turmoil_indicator.markers[alle].scale.z = rescaleFactor*energy[alle]; // Change depending on the energy. 

      //std::cout << "Adding stuff, i = " << alle << std::endl; 
    }

}

void Data_Manager::prepareMpcPredictionIndicator(visualization_msgs::MarkerArray& pred_markers, int thingsPerAgent){

  //std::cout << "MpcPredInd: Entering the function. " << std::endl; 

  thingsPerAgent = std::min(thingsPerAgent, ACADO_N+1); 
  int markerCount = thingsPerAgent*(1+n_agent_); // Main guy + once for each agent.

  pred_markers.markers.resize(markerCount); 

  std::vector<int> pickedFutures; 
  double prober = (ACADO_N+1)-1; // Array accessing -- -1 
  double step = (ACADO_N+1)/(thingsPerAgent*1.0); // Int division issues...

  // Last, for sure. 
  while(prober>0.0){
    pickedFutures.push_back(std::round(prober)); 

    prober -= step; 
  }

  int firstAgentValueWithinState = 8; // 4x basic states, 4x turmoil states, start at 0 already compared to ML, but the next one. 
  int markerIndex = 0; 

  for(int futurist : pickedFutures){
    for(int entity = 0; entity<(n_agent_+1); entity++){

      //visualization_msgs::Marker m; 
      pred_markers.markers[markerIndex].header.frame_id = "reference_map"; // MPC works in the global frame. -- not gmap? 
      pred_markers.markers[markerIndex].header.stamp = ros::Time::now(); 
      pred_markers.markers[markerIndex].action = visualization_msgs::Marker::ADD; 
      pred_markers.markers[markerIndex].lifetime = ros::Duration(); // Doing?  -> expiry date
      pred_markers.markers[markerIndex].type = visualization_msgs::Marker::CYLINDER; // consider using line strips instead
      pred_markers.markers[markerIndex].id = 1000+markerIndex; // Avoid intefering with other markers! 
      pred_markers.markers[markerIndex].ns = "mpc_pepper"; 

      // Light blue: 67.8% red... Really that much in red and green? 
      pred_markers.markers[markerIndex].color.r = 0.678f; 
      pred_markers.markers[markerIndex].color.g = 0.847f; 
      pred_markers.markers[markerIndex].color.b = 0.902f; 
      pred_markers.markers[markerIndex].color.a = 0.5f; // Some transparency? 

      // This point seems to be in the middle of the cylinder. 
      if(entity == 0){// Main robot case
        pred_markers.markers[markerIndex].pose.position.x = mpc_full_state_(futurist, 0); // x value
        pred_markers.markers[markerIndex].pose.position.y = mpc_full_state_(futurist, 1); // y value
        // "Radius"
        pred_markers.markers[markerIndex].scale.x = robot_radius_;
        pred_markers.markers[markerIndex].scale.y = robot_radius_;
      }else{// Agent case, offset! 
        pred_markers.markers[markerIndex].pose.position.x = mpc_full_state_(futurist, firstAgentValueWithinState+0*n_agent_+(entity-1)); // x value
        pred_markers.markers[markerIndex].pose.position.y = mpc_full_state_(futurist, firstAgentValueWithinState+1*n_agent_+(entity-1)); // y value
        // "Radius"
        pred_markers.markers[markerIndex].scale.x = mpc_full_state_(futurist, firstAgentValueWithinState+2*n_agent_+(entity-1));
        pred_markers.markers[markerIndex].scale.y = mpc_full_state_(futurist, firstAgentValueWithinState+2*n_agent_+(entity-1));
      }

      pred_markers.markers[markerIndex].pose.position.z = 0.0; //Starts in the middle of it. 

      pred_markers.markers[markerIndex].pose.orientation.x = 0.0; 
      pred_markers.markers[markerIndex].pose.orientation.y = 0.0; 
      pred_markers.markers[markerIndex].pose.orientation.z = 0.0; 
      pred_markers.markers[markerIndex].pose.orientation.w = 1.0; 

      pred_markers.markers[markerIndex].scale.z = 0.05; // Just some height
      
      markerIndex++; 
    }
  }

}

