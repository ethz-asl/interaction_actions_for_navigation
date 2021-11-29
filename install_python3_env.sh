# pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag tf tf2_ros message_filters std_srvs tf2_geometry_msgs cv_bridge visualization_msgs
cd ~/Code/pepper_ws/src/responsive
pip install .
cd ~/Code/pepper_ws/src/frame_soft/yolo_to_3d
pip install .
cp ~/Code/pepper_ws/src/navrep/external/frame_msgs ~/pepper3venv/lib/python3.6/site-packages/ -r
cp ~/Code/pepper_ws/src/interaction_actions_for_navigation/external/rwth_perception_people_msgs ~/pepper3venv/lib/python3.6/site-packages/ -r
cp ~/Code/pepper_ws/src/interaction_actions_for_navigation/external/darknet_ros_msgs ~/pepper3venv/lib/python3.6/site-packages/ -r

