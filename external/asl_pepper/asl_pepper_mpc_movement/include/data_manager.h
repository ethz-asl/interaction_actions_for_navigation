
#include "math.h"
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Sorting for selecting nearest entries
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>

// #include <geometry_msgs/Twist.h>
// #include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// Persons. Two different sources. (sim that contains all and detection which might fail?)
// #include <spencer_tracking_msgs/TrackedPerson.h>
// #include <spencer_tracking_msgs/TrackedPersons.h>
#include <frame_msgs/TrackedPerson.h>
#include <frame_msgs/TrackedPersons.h>

#include <sensor_msgs/LaserScan.h>
#include <std_srvs/Trigger.h>
#include <std_srvs/TriggerResponse.h>
#include <std_srvs/Empty.h>

// from sensor_msgs.msg import LaserScan
// from std_srvs.srv import Trigger, TriggerResponse
#include "asl_pepper_mpc_movement/RawMpcData.h"

// TF stuff
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

//#include <Eigen/Dense>
#include <eigen3/Eigen/Dense>

// Parameter panel
#include <dynamic_reconfigure/server.h>
#include <asl_pepper_mpc_movement/panelConfig.h>

// Acado, variables and functions
#include "acado_common.h" // Contains lots of constants etc. 
#include "acado_auxiliary_functions.h"
// Use void acado_printHeader( ); in main to display the properties nicely at the start? 
ACADOvariables acadoVariables;
ACADOworkspace acadoWorkspace;

class Data_Manager
{
private:

  bool acadoAdditionalFlush_ = true; // Once at start, may have to be re-done in NAN case
  bool storedFirstWaypoint_ = false; 
  bool storedFirstParams_ = false; // Just to make sure that n_obst and n_agent have been set. 
  bool waypointIsGoal_ = false; 

  bool firstOdomReceived_ = false; 
  bool firstObstReceived_ = false; 
  bool firstAgentReceived_ = false; 
  bool firstTfReceived_ = false; 

  // They: Interpolation stuff

  double x_position_ = 0.0;
  double y_position_ = 0.0;
  double z_position_ = 0.0;
  // double x_quaternion_ = 0.0;
  // double y_quaternion_ = 0.0;
  // double z_quaternion_ = 0.0;
  // double w_quaternion_ = 0.0;

  //double yaw_ = 0.0; 
  double yaw_baseFoot_refMap_  = 0.0; 

  double x_velocity_ = 0.0;
  double y_velocity_ = 0.0;
  double z_velocity_ = 0.0;
  double x_angular_velocity_ = 0.0;
  double y_angular_velocity_ = 0.0;
  double z_angular_velocity_ = 0.0;

  // States
  Eigen::RowVectorXd x_turmoil_;
  Eigen::RowVectorXd x_agent_;

  // Setpoint 
  double x_goal_ = 0.0;
  double y_goal_ = 0.0; 
  double x_speed_goal_ = 0.0; 
  double y_speed_goal_ = 0.0; 
  double yaw_goal_ = 0.0; 

  // Gets set when loading the parametres
  double acado_dt_ = 0.0; 
  double desired_speed_ = 0.0; 
  double max_speed_ = 0.0; 

  int n_obst_ = 0; 
  int n_agent_ = 0; 

  // To keep track of Turmoil states (turmoil propagation rigmarole) 
  double turmoil_timestamp_ = 0.0; 
  double previous_velo_x_robot_ = 0.0; 
  double previous_velo_y_robot_ = 0.0; 

  // Obstacle data. Columns of x, y, radius. 'length' gets reset from 3 to n_obst when loading this parameter. 
  // Eigen::MatrixXd obst_data_(3,3); 
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> obst_data_;

  // Agents. x, y, radius. And velocity separately (since they're needed in different places)
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> agent_posPlus_; 
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> agent_velo_; 

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mpc_full_state_; 

  double robot_radius_ = 0.0; 
  double lookahead_distance_ = 0.0; 
  double goal_proximity_ = 0.0; 

  double turnTime_ = 0.0; // Completing the turn to head towards the right direction
  double turnLimit_ = 0.0; // Jerk of the turn

  double k_turmoil_ = 0.0; // Turmoil energy spring constant
  double c_turmoil_ = 0.0; // Turmoil system damping
  double robot_mass_ = 0.0; // Robot mass

  // Discrete Turmoil System (Tustin approximation, generated in Matlab)
  Eigen::Matrix<double, 4, 4> A_turmoil_;
  Eigen::Matrix<double, 4, 2> B_turmoil_; 


  Eigen::VectorXd weights_; 
  // Eigen::VectorXd arrival_weights_; 

  // Stuff needed by ACADO
  // Define matrices and vectors that are passed to acado at each timestep
  // (for initialization of the current state, online data, and reference):
  Eigen::Matrix<double, ACADO_NY, ACADO_NY> W_;
  Eigen::Matrix<double, ACADO_NYN, ACADO_NYN> WN_;
  Eigen::Matrix<double, ACADO_N + 1, ACADO_NX> state_;
  Eigen::Matrix<double, ACADO_N, ACADO_NU> input_;
  Eigen::Matrix<double, ACADO_N, ACADO_NY> x_reference_;
  Eigen::Matrix<double, 1, ACADO_NYN> x_referenceN_;
  Eigen::Matrix<double, ACADO_N + 1, ACADO_NOD> acado_online_data_;

  // output
  double x_speed_output_robot_ = 0.0; 
  double y_speed_output_robot_ = 0.0; 
  double w_output_ = 0.0; 

  bool STOP_ = true; // false; //Safer default value! 

  enum agentDetection {disable = 0, rwth_tracker = 1, dr_spaam = 2};
  agentDetection agentDetection_= disable; 

public:

  Data_Manager(); 

  // Parameter loading callbacks
  void storeParameters(double dt, double v_ref, double max_speed, int n_obstacle, int n_agent, double robot_radius, 
    double lookahead_distance, double goal_proximity, 
    double turnTime, double turnLimit); 

  void storeTurmoilSystemParameters(double k_turmoil, double c_turmoil, double robot_mass, 
    std::vector<double> A, std::vector<double> B); 

  void storeWeights (double weight_pos, double weight_velo, double weight_turmoil, double weight_turmoil_E, 
    double weight_obst, double weight_agent, double weight_input);

  // fancy panel with weight adaptation and stuff. 
  void panelCallback(asl_pepper_mpc_movement::panelConfig &config, uint32_t level);

  bool receivedInitialCallbacks(); 

  // They: Frenet frame path interpolation methods

  // Various ROS callbacks
  // -- Store the position locally. 
  void robotOdometryCallback (const nav_msgs::Odometry::ConstPtr& msg);
  void robotPositionUpdate(geometry_msgs::TransformStamped& shift); 
  
  // -- Calculate the next setpoint from the entire path
  void plannerCallback (const nav_msgs::Path::ConstPtr& msg); 

  // -- Store Agent positions + velocities. One topic needs frame_msgs the other spencer_tracking_msgs! 
  void agentCallback (const boost::shared_ptr<frame_msgs::TrackedPersons_<std::allocator<void> > const>& msg); 

  // -- DR Spaam aggent version
  void agentCallbackDrSpaam(const boost::shared_ptr<geometry_msgs::PoseArray_<std::allocator<void> > const>& msg); 

  // -- Obstacles
  void obstCallback (const boost::shared_ptr<sensor_msgs::LaserScan_<std::allocator<void> > const>& msg); 
  // void obstCallback (const sensor_msgs::LaserScan_& msg); 

  // Changing W and Wn in the Acado space
  void setAcadoOptimisationWeights(Eigen::VectorXd weight_vec);

  // prepares acado variables for the next iteration
  void setAcadoInput();

  void flushStatesNextTime(); 

  // Needed? Only called during the first iteration. Consider deleting this. Same in Matlab, though. Might have to do with space allocation or smth. 
  // NAN handling, too. 
  void setAcadoAdditionalFlush(Eigen::VectorXd x0);

  // Sets everything to 0
  void resetLocalMPCParameters();

  // Returns the result which should be applied in the next time step
  bool calculateRobotPublisherOutput();
  void getRobotPublisherOutput(std::vector<double>& output);

  // Deprecated turmoil handling
  // NO, do that in the odometry callback: From what gets applied, not the desired inputs. 
  // Could just calculate it locally in both cases? Coordinate issue, though. 
  void storeCleanTurmoilStates(); // Predicted ones within MPC
  // Used when MPC failed, keep track of the (deadbeat) input, still. Needs previous speedto compute the corresponding input. 
  void storeDeadbeatTurmoilStates(double previous_velo_x, double previous_velo_y); 
  // Not working due to the unstable discrete system
  void updateTurmoilStates(); 

  void storeFullMpcStates(); 

  void getOptimisationWeights(Eigen::VectorXd& weights);

  // Publishers for visualisations
  void prepareObstMessage(visualization_msgs::MarkerArray& obst_markers); 
  void prepareTurmoilIndicator(visualization_msgs::MarkerArray& turmoil_markers); 
  void prepareMpcPredictionIndicator(visualization_msgs::MarkerArray& pred_markers, int thingsPerAgent); // Need to skip some to avoid lag.  

  // void disableAgentStuff(); 
  void vanishAgentsIfNecessary(); 

  // Emergency stop button
  bool stopAutonomousMotionServiceCall(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res); 
  bool resumeAutonomousMotionServiceCall(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res); 
  void setBigRedButtonStatus(bool STOP); 
  bool isStopped(); 

  // Theirs: 
/*
  bool goalReached();

  void transformationGItoGR();

  void setIRQuaternion(geometry_msgs::TransformStamped transform);

  void setIRTransBool(bool setter);
*/

};
