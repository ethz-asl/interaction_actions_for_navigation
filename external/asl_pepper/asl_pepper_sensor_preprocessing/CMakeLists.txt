cmake_minimum_required(VERSION 2.8.3)
project(asl_pepper_sensor_preprocessing)

add_definitions(-std=c++11)

find_package(catkin_simple 0.1.0 REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_executable(combine_laser_scans src/combine_laser_scans.cpp)
cs_add_executable(crop_laser_scan src/crop_laser_scan.cpp)

#add_doxygen(REQUIRED)

cs_install()
cs_export()
