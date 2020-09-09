launches orb_slam_2_ros with the Pepper configuration. Requires a funtional orb_slam_2_ros package, using the branch feature/rgbd_simple

Recommended install is to unpack orbslam_and_dependencies.7z in your catkin_ws/src/ folder.

Make sure to install the following debian packages first:

libnlopt-dev
libeigen3-dev
libopencv-dev
autoconf

Also ensure to put an ORBvoc.txt file in vocabulary/ 
An example ORBvoc.txt can be found in the ORBSLAM2 repository.

TODO: pre-process camera images to register depth and front images.
