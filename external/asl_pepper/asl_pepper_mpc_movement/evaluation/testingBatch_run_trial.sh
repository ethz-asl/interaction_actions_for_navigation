
# Adapted version of the one in asl_pepper_evaluation
# Used for early RVO, Move base and MPC runs

# check args
if [ -z $1 ]
then
  echo "argument (output dir) required" && exit 1
fi

# check if ROS is sourced
[ ! -z $ROS_DISTRO ] || { echo -e '\033[0;31mError: ROS is not sourced.\033[0m' && exit 1 ; }

# Adapted version of the one in asl_pepper_evaluation
# Deprecated
# Added MPC and RVO controller options to extend the set that was generated for IROSASL

RUN_DIR=$1/$(date +"%Y_%m_%d_%H_%M")

for TRIAL_NUMBER in {1..2}
do
  for SCENE_NUMBER in {5..5}
  do
    for MAP_NUMBER in {3..3}
    do
      for PLANNER_NUMBER in {7..7}
      do
        # check if roscore is already running
        rostopic list >/dev/null 2> /dev/null && echo "ROS core is already running! aborting." && exit 1

        # start roscore
        roscore &
        sleep 1

        # start recording rosbag
        ROSBAG_DIR="$RUN_DIR/planner_$PLANNER_NUMBER/map_$MAP_NUMBER/scene_$SCENE_NUMBER"
        mkdir -p $ROSBAG_DIR
        ROSBAG="$ROSBAG_DIR/trial_$TRIAL_NUMBER.bag"
        rosbag record -a -O $ROSBAG __name:=iros_bag_recorder &
        sleep 1

        if [[ $MAP_NUMBER == 1 ]]
        then
          N_DS=1  # amount of downsampling to apply for ia planning
          MAP_NAME=asl
          MAX_RUNTIME=300
        fi
        if [[ $MAP_NUMBER == 2 ]]
        then
          N_DS=2
          MAP_NAME=unity_scene_map
          MAX_RUNTIME=300
        fi
        if [[ $MAP_NUMBER == 3 ]]
        then
          N_DS=3
          MAP_NAME=asl_office_j
          MAX_RUNTIME=600
        fi

        SCENARIO="iros${MAP_NAME}${SCENE_NUMBER}"

        # start my experiment
        if [[ $PLANNER_NUMBER == 1 ]]
        then
          timeout 1200 roslaunch ia_ros auto_ros_ia_node.launch \
            scenario:=$SCENARIO mapname:=$MAP_NAME ia_downsampling_passes:=$N_DS \
            max_runtime:=$MAX_RUNTIME
        fi
        if [[ $PLANNER_NUMBER == 2 ]]
        then
          timeout 1200 roslaunch asl_pepper_move_base auto_move_base.launch \
            scenario:=$SCENARIO mapname:=$MAP_NAME ia_downsampling_passes:=$N_DS \
            max_runtime:=$MAX_RUNTIME
        fi
        if [[ $PLANNER_NUMBER == 3 ]]
        then
          timeout 1200 roslaunch ia_ros auto_ros_ia_node.launch script_args:="--only-intend" \
            scenario:=$SCENARIO mapname:=$MAP_NAME ia_downsampling_passes:=$N_DS \
            max_runtime:=$MAX_RUNTIME
        fi
        if [[ $PLANNER_NUMBER == 4 ]]
        then
          timeout 1200 roslaunch ia_ros auto_ros_ia_node.launch script_args:="--only-say" \
            scenario:=$SCENARIO mapname:=$MAP_NAME ia_downsampling_passes:=$N_DS \
            max_runtime:=$MAX_RUNTIME
        fi
        if [[ $PLANNER_NUMBER == 5 ]]
        then
          timeout 1200 roslaunch ia_ros auto_ros_ia_node.launch script_args:="--only-nudge" \
            scenario:=$SCENARIO mapname:=$MAP_NAME ia_downsampling_passes:=$N_DS \
            max_runtime:=$MAX_RUNTIME
        fi
        if [[ $PLANNER_NUMBER == 7 ]] # MPC planner
        then
          timeout 1200 roslaunch asl_pepper_mpc_movement mpc_controller_pure.launch \
            scenario:=$SCENARIO mapname:=$MAP_NAME ia_downsampling_passes:=$N_DS \
            max_runtime:=$MAX_RUNTIME
        fi
        if [[ $PLANNER_NUMBER == 8 ]] # RVO planner
        then
          #Clean RVO launch file instead of that? 
          # timeout 1200 roslaunch asl_pepper_motion_planning rvo_planner.launch \
          #   scenario:=$SCENARIO mapname:=$MAP_NAME ia_downsampling_passes:=$N_DS \
          #   max_runtime:=$MAX_RUNTIME
          timeout 1200 roslaunch asl_pepper_mpc_movement controller_triage.launch \
            scenario:=$SCENARIO mapname:=$MAP_NAME ia_downsampling_passes:=$N_DS \
            max_runtime:=$MAX_RUNTIME controller_choice:=2 create_rosbag:=false \
            rviz:=false script_args:="--no-stop" dr_spaam_start:=false

        fi

        # stop rosbag record gracefully
        rosnode kill iros_bag_recorder
        sleep 2  # give rosbag time to finish

        # kill roscore
        JOBS="$(jobs -p)"
        if [ -n "${JOBS}" ]
        then
          kill ${JOBS}
        fi

        wait

      done
    done
  done
done
