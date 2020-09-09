# MPC Movement
This module contains an MPC controller, which relies on C-code created in advance via Acado in Matlab.

## Startup
For installation, follow the [asl_pepper instructions]( https://github.com/ethz-asl/asl_pepper/blob/mpc_movement/README.md) with adapted branch choice. The extra modules [RWTH pedestrian tracker](https://drive.google.com/open?id=1sytyNYNqgXDDBekClZnX5h0-fnpElAov) and 
[DR-SPAAM detector](https://github.com/VisualComputingInstitute/DR-SPAAM-Detector) are required, but could be disabled in the code. 
Building with ```catkin make asl_pepper_mpc_movement``` comes next (try the additional ```--force-cmake``` flag in case of ROS message issues). Don't forget sourcing throughout (define an alias): 
```
source ~/peppervenv/bin/activate
source ~/pepper_ws/devel/setup.bash
```
ROS simulations are started with: 
```
roslaunch asl_pepper_mpc_movement controller_triage.launch script_args:="--no-stop" create_rosbag:=false  pass_nogoal_option:="" controller_choice:=1  pass_pso_IfMoveBase:="" mapname:=asl_office_j scenario:=officegroups 
```
Or use, for more overview, ```roscore``` in one terminal, combined with ```mon launch``` in another: 
```
mon launch asl_pepper_mpc_movement controller_triage.launch script_args:="--no-stop" create_rosbag:=false  pass_nogoal_option:="" controller_choice:=1  pass_pso_IfMoveBase:="" mapname:=asl_office_j scenario:=officegroups 
```
In the live setting, follow the Pepper Manual, which involves starting roscore and the Ethernet connection. In a third terminal: 
```
mon launch asl_pepper_mpc_movement live_mpc_controller_pure.launch mapname:=asl_office_k
```
## Tipps
- Toggling the emergency stop button via terminal: 
```
rosservice call /stop_autonomous_motion
rosservice call /resume_autonomous_motion
```
- If the map matcher has trouble because the environment has changed, create a new map by walking the robot around and using ```rosrun map_server map_saver -f asl_office_x /map:=/gmap``` to create a new map. 
- Run ```rosclean check``` and subsequent ```rosclean purge``` regularly to free space
- Scenario editing within the ```interaction_actions``` module:
```
python gui_world_editor.py --map-folder ~/maps/ --map-name asl_office_j --scenario-folder ../scenarios/ --scenario irosasl_office_j2
python draw_scenes.py --scenario-folder ../scenarios/ --map-name asl_office_j --map-folder ~/maps/ 
```
- TF overview with ```rosrun tf view_frames``` 
- Check the joystick configuration with ```jstest-gtk```
- The dynamic reconfiguration panel has flaws: Default values from the YAML file parameter loading overwrite the ones specified for the panel, but they get applied once the panel is used. This could cause the stop button to be toggled after having released the robot by other means (joystick or manual service call). 

## Possible Future Features
- Caching MPC commands for more than one prediction step to have a backup in case of solver failure. 
- Stuckness recovery by switching into a state where only obstacle weights apply. This would avoid tinkering with the intermediate goal. 
- Choose behaviour after arrival: Lingering around the destination and keep dodging pedestrians or do a full stop? (Might depend on the global planner behaviour for that case.) 
- Smaller sampling for turmoil propagation, independent of the other setup. (Update A and B with the tustin discretisation matlab script.)

<!-- Readme Manual Syntax: -->
<!-- Titles: More hashtags = lower: ## Pepper Simulator-->
<!-- Image: 
![Resources for Pepper Robot](https://github.com/ethz-asl/asl_pepper/raw/master/asl_pepper.jpg "Pepper Resources") -->

<!-- Fancy writing: ```asl_pepper_joystick``` -->
<!-- Link: (see [RL Readme](https://github.com/ethz-asl/asl_pepper/blob/master/wiki/RL_README.md) ) -->

<!-- Dotting with minus: - open a new terminal, run -->
