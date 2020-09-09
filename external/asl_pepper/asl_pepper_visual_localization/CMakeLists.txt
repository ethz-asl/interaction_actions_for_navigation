cmake_minimum_required(VERSION 2.8.3)
project(asl_pepper_visual_localization)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS 
roscpp std_msgs opencv3_catkin cv_bridge image_transport
)


#generate_messages( DEPENDENCIES std_msgs)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES asl_pepper_visual_localization
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)





###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
 ${catkin_INCLUDE_DIRS}
 ${opencv3_catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/asl_pepper_visual_localization.cpp
# )

## Declare a C++ executable
#add_executable(talker src/talker.cpp)
#add_executable(listener src/listener.cpp)

## Specify libraries to link a library or executable target against
#target_link_libraries(talker ${catkin_LIBRARIES})
#target_link_libraries(listener ${catkin_LIBRARIES})

#add_executable(talker src/talker.cpp)
#target_link_libraries(talker ${catkin_LIBRARIES})
#add_dependencies(talker asl_pepper_visual_localization_generate_messages_cpp)

#add_executable(listener src/listener.cpp)
#target_link_libraries(listener ${catkin_LIBRARIES})
#add_dependencies(listener asl_pepper_visual_localization_generate_messages_cpp)

#add_executable(camera_subscriber_ben src/camera_subscriber_ben.cpp)
#target_link_libraries(camera_subscriber_ben ${catkin_LIBRARIES} ${opencv3_catkin_LIBRARIES})
#add_dependencies(camera_subscriber_ben asl_pepper_visualisation_generate_messages_cpp)

#add_executable(sub1 src/sub1.cpp)
#target_link_libraries(sub1 ${catkin_LIBRARIES} ${opencv3_catkin_LIBRARIES})

#add_executable(subtest src/subtest.cpp)
#target_link_libraries(subtest ${catkin_LIBRARIES} ${opencv3_catkin_LIBRARIES})


#add_executable(save_pict src/save_pict.cpp)
#target_link_libraries(save_pict ${catkin_LIBRARIES} ${opencv3_catkin_LIBRARIES})

#add_executable(testsift src/testsift.cpp)
#target_link_libraries(testsift ${catkin_LIBRARIES} ${opencv3_catkin_LIBRARIES})

#add_executable(my_cpp_tests src/my_cpp_tests.cpp)
#target_link_libraries(my_cpp_tests ${catkin_LIBRARIES} ${opencv3_catkin_LIBRARIES})


