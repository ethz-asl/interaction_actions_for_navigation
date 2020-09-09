
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include "std_msgs/Float64.h"
#include "math.h"
#include <eigen3/Eigen/Dense> // works with eigen3/ added, not just eigen/

// Contains subscriber callbacks, data storage and more utility. 
#include "data_manager.h"

// Custom Data Message: 
#include "asl_pepper_mpc_movement/RawMpcData.h"

// read out values to file
// #include <iostream>
// #include <fstream>
// #include <ctime>
// #include <stdio.h>
#include <chrono>

#include <geometry_msgs/Twist.h>

// Translating ACADO error codes
#include "MessageHandling.hpp"

int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "path_planning_mpc");
  ros::NodeHandle node("~");

  Data_Manager butler; // Handles callbacks and many other functions needed wrap the acado code. 

  // Subscribers. As many as 13 method versions! http://docs.ros.org/melodic/api/roscpp/html/classros_1_1NodeHandle.html
  // -- Robot position data
  ros::Subscriber sub_odom = node.subscribe("/pepper_robot/odom", 10, &Data_Manager::robotOdometryCallback, &butler); 

  // -- TF needed for proper localisation
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  // -- Setpoint selection
  ros::Subscriber sub_planner = node.subscribe("/global_planner/global_path", 10, &Data_Manager::plannerCallback, &butler); 

  // -- Agents
  ros::Subscriber sub_agent = node.subscribe("/rwth_tracker/tracked_persons", 10, &Data_Manager::agentCallback, &butler); 

  // -- Agents, version for the live detection, some deep learning tracker
  ros::Subscriber sub_agent_dr_spaam = node.subscribe("/dr_spaam_detections", 10, &Data_Manager::agentCallbackDrSpaam, &butler); 

  // -- Obstacles
  ros::Subscriber sub_obst = node.subscribe("/combined_scan" , 10, &Data_Manager::obstCallback, &butler); 

  // Publishers. 
  // -- Velocity commands
  ros::Publisher pub_vel_cmd = node.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

  // -- Picked laserscan obstacles (Just to visualise)
  ros::Publisher pub_obst = node.advertise<visualization_msgs::MarkerArray>("obstacles_picked", 10); 

  // -- Turmoil states (overhead column as a visual indicator)
  ros::Publisher pub_turmoil = node.advertise<visualization_msgs::MarkerArray>("turmoil_states", 10); 

  // -- Predicted states for the robot + Agents 
  ros::Publisher pub_prediction = node.advertise<visualization_msgs::MarkerArray>("mpc_predictions", 10); 

  // -- Raw MPC data ("online data", state matrix etc, to add to a rosbag)
  ros::Publisher pub_rawMpcData = node.advertise<asl_pepper_mpc_movement::RawMpcData>("raw_mpc_data", 10); 

  // Button for emergency stop, resuming
  ros::ServiceServer serv_fullStop = node.advertiseService("/stop_autonomous_motion", &Data_Manager::stopAutonomousMotionServiceCall, &butler);
  ros::ServiceServer serv_resume = node.advertiseService("/resume_autonomous_motion", &Data_Manager::resumeAutonomousMotionServiceCall, &butler);

  // Get parameters from param server. Just used for loading, see data_manager
  int rate;
  bool stop_storage; // Not used directly, passed over to butler.STOP_
  std_srvs::Trigger serviceTrigger; 

  double acado_dt; 
  double desired_speed; 
  double max_speed; 

  double n_obst; 
  double n_agent; 

  double robot_radius;
  double lookahead_distance;
  double goal_proximity;

  double turnTime; // Completing the turn in the right direction
  double turnLimit; // Jerk of the turn

  double k_turmoil; // Turmoil spring constant
  double c_turmoil; // Turmoil system damping
  double robot_mass; // Robot mass
  std::vector<double> A_turmoil;
  std::vector<double> B_turmoil;

  // std::vector<double> weights_vector;
  // std::vector<double> goal_weights_vector;
  double weight_pos; 
  double weight_velo; 
  double weight_turmoil; 
  double weight_turmoil_E; 
  double weight_obst; // All together
  double weight_agent; // Sum
  double weight_input; 

  if(
      !node.getParam("rate", rate)
      || !node.getParam("big_red_button", stop_storage)

      || !node.getParam("acado_dt", acado_dt)
      || !node.getParam("desired_speed", desired_speed)
      || !node.getParam("max_speed", max_speed)
      || !node.getParam("n_obst", n_obst)
      || !node.getParam("n_agent", n_agent)

      || !node.getParam("robot_radius", robot_radius)
      || !node.getParam("lookahead_distance", lookahead_distance)
      || !node.getParam("goal_proximity", goal_proximity)

      || !node.getParam("turnTime", turnTime)
      || !node.getParam("turnLimit", turnLimit)

      || !node.getParam("k_turmoil", k_turmoil)
      || !node.getParam("c_turmoil", c_turmoil)
      || !node.getParam("robot_mass", robot_mass)
      || !node.getParam("a_turmoil", A_turmoil)
      || !node.getParam("b_turmoil", B_turmoil)

      // || !node.getParam("weights", weights_vector)
      // || !node.getParam("arrival_weights", arrival_weights_vector)
      || !node.getParam("weight_pos", weight_pos)
      || !node.getParam("weight_velo", weight_velo)
      || !node.getParam("weight_turmoil", weight_turmoil)
      || !node.getParam("weight_turmoil_E", weight_turmoil_E)
      || !node.getParam("weight_obst", weight_obst)
      || !node.getParam("weight_agent", weight_agent)
      || !node.getParam("weight_input", weight_input)

      ){
    ROS_ERROR("Unable to load parameter");
    return EXIT_FAILURE;
  };
  //std::cout << weights_vector.at(1) << std::endl; 

  // Store everything within the data manager: 
  butler.setBigRedButtonStatus(stop_storage); 
  butler.storeParameters(acado_dt, desired_speed, max_speed, n_obst, n_agent, robot_radius, 
    lookahead_distance, goal_proximity, 
    turnTime, turnLimit); 
  butler.storeTurmoilSystemParameters(k_turmoil, c_turmoil, robot_mass, A_turmoil, B_turmoil); 
  // butler.storeWeights(weights_vector, arrival_weights_vector); 
  butler.storeWeights(weight_pos, weight_velo, weight_turmoil, weight_turmoil_E, weight_obst, weight_agent, weight_input); 

  // Acado error type translation
  MessageHandling messageHandler; 
  // MPC output to cmd_vel input preparation
  std::vector<double> output;
  geometry_msgs::Twist cmd_vel; // Publishing the results in this thing. 

  ros::Rate loop_rate(rate); // Waiting during the while-loop below, before the next iteration

  // Acado related
  butler.resetLocalMPCParameters();
  acado_initializeSolver();

  // Fancy interactive panel with sliders for weights
  dynamic_reconfigure::Server<asl_pepper_mpc_movement::panelConfig> fancyPanelServer;
  dynamic_reconfigure::Server<asl_pepper_mpc_movement::panelConfig>::CallbackType f;
  
  f = boost::bind(&Data_Manager::panelCallback, &butler, _1, _2); // What are the last two arguments? 
  fancyPanelServer.setCallback(f); 

  // Time consumption measurements. hrc seems more precise/useful than ros timing
  ros::Time timer_ros; 
  auto timer_hrc = std::chrono::high_resolution_clock::now();
  double duration_hrc; 

  // Printing the terminal iteration info block
  int info = 0;// counts iterations
  int info_size = 0; // number of info digits
  std::string filler = "i =       "; 

  int acado_status = 0; // Flag returned by the ACADO code. 
  bool acado_success; 
  //bool arrivedAtGoal; // Could check that and send out a deadbeat signal. Entering Hover state, otherwise.  

  // Position info which is more accurate than the one from odom
  geometry_msgs::TransformStamped shift; 

  while (ros::ok()){
  
    info++; 
    info_size = std::to_string(info).size(); // inefficient? 
    std::cout << "**********************************" << std::endl;
    std::cout << "Starting new MPC iteration. N = " << ACADO_N << std::endl;
    std::cout << "**********  " << filler.replace(9-info_size, info_size, std::to_string(info)) << "  **********" << std::endl;

    // std::cout  << "RVL test: " << std::endl; 
    // for(int all =0; all<100; all++){
    //   std::cout << all <<": "<< messageHandler.getErrorString(all) << std::endl; 
    // }
    // messageHandler.listAllMessages(); // goes to file

    // Sends agents far away if no info on them is available (panel option 'disable agents')
    butler.vanishAgentsIfNecessary(); 

    try{
      shift = tfBuffer.lookupTransform("reference_map", "base_footprint", ros::Time(0)); 
      butler.robotPositionUpdate(shift); 
    }catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ROS_WARN("Can't look up the TF translation at the moment! "); 
      ros::Duration(1.0).sleep(); // Hmmm...
      continue;
    }

    timer_ros = ros::Time::now(); 
    timer_hrc = std::chrono::high_resolution_clock::now();

    // They: Lots of path/interpolation stuff

    acado_success = false; // Pessimistic view
    if(butler.receivedInitialCallbacks()){
    
      // update inputs for Acado
      butler.setAcadoInput();
      acado_preparationStep();
      acado_status = acado_feedbackStep();

      // std::cout << "Acado status after the feedback step: " << acado_status << std::endl;

      // TODO:Could add a control command buffer for the failure and NAN case
      if(acado_status == 0){

        // std::cout << "The entire state matrix:\n" << Eigen::Map<Eigen::Matrix<double,ACADO_N+1,ACADO_NX,Eigen::RowMajor> >(acadoVariables.x) << std::endl;
        // std::cout << "The entire output matrix:\n" << Eigen::Map<Eigen::Matrix<double,ACADO_N+1,ACADO_NU,Eigen::RowMajor> >(acadoVariables.u) << std::endl;
        
        // Solved differently now (all in the odom callback)
        // butler.updateTurmoilStates(); 

        // Just for analysis, doesn't get propagated. May contain NAN! 
        butler.storeFullMpcStates(); 

        if(butler.calculateRobotPublisherOutput()){
          acado_success = true; 
        }else{
          ROS_WARN_STREAM("MPC: Solver did not fail, but there's an NAN state somewhere. ");
          ROS_WARN_STREAM("MPC: Flushing states next time. ");
          butler.flushStatesNextTime(); 
        }

      }else{

        ROS_WARN_STREAM("MPC: Solver failed with status " << acado_status << " (" << messageHandler.getErrorString(acado_status) << ")");
        ROS_WARN_STREAM("MPC: Re-initialising...");

        // Attempt to recover by flushing everyting
        butler.resetLocalMPCParameters();
        acado_initializeSolver();
      }
    }

    // Moved Turmoil propagation to the odom topic subscriber. Consider doing form TF? 
    // // Turmoil propagation: Anyways, depends on success flag? Keeps old values, otherwise? 
    // if(acado_success){
    //   butler.storeCleanTurmoilStates(); // Only store these when it was successful
    // }else{
    //   // Keeps old values at the moment. 
    // }

    // Theirs: Goal reached case. Need for structure below? 

    // Preparing/possibly sending a command
    cmd_vel.linear.z = 0.0;
    cmd_vel.angular.x = 0.0; 
    cmd_vel.angular.y = 0.0;
    
    if(butler.isStopped()){
      ROS_WARN_STREAM("Stopped! Publishing All-Zero command."); 
      // Publish deadbeat
      cmd_vel.linear.x = 0.0;
      cmd_vel.linear.y = 0.0;
      cmd_vel.angular.z = 0.0;
      pub_vel_cmd.publish(cmd_vel);

    }else{
      if(acado_success){

        butler.getRobotPublisherOutput(output); 
        // cmd_vel.linear.x = output[0];  
        // cmd_vel.linear.y = output[1];

        // Enforce a constraint on this layer to avoid the corresponding constraint in ACADO. 
        if(output[0]>max_speed){
          ROS_WARN_STREAM("MPC produced a reckless forward speed. Capping! ");
          cmd_vel.linear.x = max_speed;  
        }else{
          cmd_vel.linear.x = output[0];  
        }

        if(output[1]>max_speed){
          ROS_WARN_STREAM("MPC produced a reckless sideways speed. Capping!");
          cmd_vel.linear.y = max_speed;  
        }else{
          cmd_vel.linear.y = output[1];  
        }

        cmd_vel.angular.z = output[2];//0.5;// 1.0; 
        std::cout << "Publishing velocity commands: x_velo = " << cmd_vel.linear.x << "; y_velo = " << cmd_vel.linear.y << "; w = " << cmd_vel.angular.z << std::endl;  
        pub_vel_cmd.publish(cmd_vel);
      }else{
        // Don't send anything. Could do a stop, too, or depend behaviour on arrival. 
        // Send 'Deadbeat' if arrived at the goal? Design choice. 
      }
    }

    // Gaining time /reducing the computational load when turning that stuff off when it's not needed? 
    // Publishers which are needed to evaluate the results via rosbags and display what's going on in RVIZ
    // -- Obstacles
    visualization_msgs::MarkerArray obst_markers; 
    butler.prepareObstMessage(obst_markers); 
    pub_obst.publish(obst_markers); 

    // -- Turmoil states
    visualization_msgs::MarkerArray turmoil_indicator; 
    butler.prepareTurmoilIndicator(turmoil_indicator); 
    pub_turmoil.publish(turmoil_indicator); 

    // -- Stuff from the MPC prediction
    if(butler.receivedInitialCallbacks()){
      visualization_msgs::MarkerArray pred_indicator; 
      butler.prepareMpcPredictionIndicator(pred_indicator, 5); 
      pub_prediction.publish(pred_indicator); 
    }

    // -- All MPC data (Wrap to avoid zeros/errors)
    if(butler.receivedInitialCallbacks()){
      asl_pepper_mpc_movement::RawMpcData mpc_shipment; 
      // Not as above since the data is available here. 

      // Smarter vector copying? 
      // mpc_shipment.OnlineData = acadoVariables.od; 
      for(int aa = 0; aa<ACADO_NOD; aa++){
        mpc_shipment.OnlineData.push_back(acadoVariables.od[aa]); 
      }

      // Outputs: 
      for(int cc = 0; cc<ACADO_QP_NV; cc++){ // QP_NV: Acado optimisation variables = NxNU = 40*2 (default). Need all input cost for the proper cost plot. 
        mpc_shipment.Outputs.push_back(acadoVariables.u[cc]); 
      }

      // mpc_shipment.onlineData = acadoVariables.od; 
      // That's constant, wouldn't need to add it every time. 
      mpc_shipment.AcadoNx = ACADO_NX; 
      mpc_shipment.AcadoNod = ACADO_NOD; 
      mpc_shipment.AcadoNplus = ACADO_N+1; 
      mpc_shipment.AcadoNu = ACADO_NU; 

      // Weights are diagonal matrices in ACADO -- save some by grabbing raw values instead. 
      // ZERO DAY: That doesn't send the correct arrival weights if the arrival weight distinction is used
      Eigen::VectorXd weights;
      butler.getOptimisationWeights(weights); 
      for(int bb = 0; bb<ACADO_NY; bb++){ // NY: correct correspodence
        mpc_shipment.Weights.push_back(weights[bb]); 
      }

      //  // Get the KKT tolerance of the current iterate.
      // real_t acado_getKKT(  );

      // // Calculate the objective value.
      // real_t acado_getObjective(  );

      mpc_shipment.AcadoCost = acado_getObjective(); 
      mpc_shipment.AcadoSuccessFlag = acado_status; 

      for(int bb = 0; bb<(ACADO_NX*(ACADO_N+1)); bb++){
        mpc_shipment.States.push_back(acadoVariables.x[bb]);
      }
      //mpc_shipment.states = acadoVariables.x; 
      duration_hrc = (std::chrono::high_resolution_clock::now()-timer_hrc).count()*1e-9; 
      mpc_shipment.MpcNodeComputationTime = duration_hrc; 

      // std::cout << "Option ros, fallowing timer result: " << (ros::Time::now()-timer_ros).toSec() << std::endl; 
      // std::cout << "Option hrc, fallowing timer result: " << (std::chrono::high_resolution_clock::now()-timer_hrc).count()*1e-9 << std::endl; // nanoseconds

      if(duration_hrc>(1.0/rate)){
        ROS_WARN_STREAM("MPC node: Probably took more time than allowed for the computations: " << duration_hrc << " s.");
      }

      pub_rawMpcData.publish(mpc_shipment); 
    }

    ros::spinOnce();
    loop_rate.sleep();
    
  }

}
