cmake_minimum_required(VERSION 2.8.3)
project(frame_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	message_generation
	std_msgs
	geometry_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
	FILES
	DetectedPerson.msg
	DetectedPersons.msg
	TrackedPerson.msg
	TrackedPersons.msg
	TrackedPerson2d.msg
	TrackedPersons2d.msg
	PersonTrajectoryEntry.msg
	PersonTrajectory.msg
	PersonTrajectories.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
	DEPENDENCIES
	std_msgs
	geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
catkin_package(
   CATKIN_DEPENDS message_runtime
)


###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)
