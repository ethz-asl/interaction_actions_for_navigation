cmake_minimum_required(VERSION 2.8.3)
project(asl_pepper_motion_planning)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin_simple REQUIRED)
# find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs sensor_msgs geometry_msgs naoqi_bridge_msgs)

catkin_simple()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
# catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES asl_pepper_motion_planning
#  CATKIN_DEPENDS joy
#  DEPENDS system_lib
# )

include_directories(include ${catkin_INCLUDE_DIRS})

add_compile_options(-std=c++11)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)

## Declare a C++ library
# add_library(asl_pepper_motion_planning
#   src/${PROJECT_NAME}/asl_pepper_motion_planning.cpp
# )

## Declare a C++ executable
cs_add_executable(asl_pepper_motion_planning_node src/asl_pepper_motion_planning_node.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(asl_pepper_motion_planning_node ${catkin_LIBRARIES})

cs_install()
cs_install_scripts(scripts/global_planner scripts/pepper_2d_simulator scripts/test_rl scripts/motion_planner scripts/rvo_planner)
