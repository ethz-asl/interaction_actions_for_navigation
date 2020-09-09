cmake_minimum_required(VERSION 2.8.3)
project(asl_pepper_gmapping)

add_definitions(-std=c++11)

find_package(catkin_simple 0.1.0 REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#add_doxygen(REQUIRED)

cs_install()
cs_export()
