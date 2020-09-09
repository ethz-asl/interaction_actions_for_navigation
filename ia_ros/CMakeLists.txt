cmake_minimum_required(VERSION 2.8.3)
project(ia_ros)
find_package(catkin_simple REQUIRED)
catkin_simple()
cs_install()
cs_install_scripts(
  scripts/ros_ia_node
  scripts/ros_sim_node
  scripts/ia_dashboard
)
