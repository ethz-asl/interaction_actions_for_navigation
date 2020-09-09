# asl_pepper
Code and resources for working with the Pepper robot.

![Resources for Pepper Robot](https://github.com/ethz-asl/asl_pepper/raw/master/asl_pepper.jpg "Pepper Resources")

This repository contains several ROS packages necessary to work with Pepper at ASL.
The main functionalities are:
- ROS bringup. Publishes all sensor data and subscribes to command topics - In ```asl_pepper_basic_functions```
- Joystick control - In ```asl_pepper_joystick```
- Mapping (using gmapping), localization (using map_matcher), and basic motion planning - In ```asl_pepper_motion_planning```
- RL based navigation - In ```asl_pepper_motion_planning/scripts``` (see [RL Readme](https://github.com/ethz-asl/asl_pepper/blob/master/wiki/RL_README.md) )

## Install

TODO daniel: crowdbot sim dependencies
catkin_crowdbotsim --branch ETH
remove file_server ?
catkin build crowdbotsim


```
# Makes sure basic dependencies are installed
sudo echo "I shouldn't copy-paste code I don't understand!"
sudo apt -y install git
sudo apt -y install python-catkin-tools python-wstool
sudo apt -y install python-pip virtualenv
sudo apt -y install libssh2-1-dev
```

```
# Makes sure ROS is sourced
[ ! -z $ROS_DISTRO ] || source /opt/ros/melodic/setup.bash || source /opt/ros/kinetic/setup.bash
[ ! -z $ROS_DISTRO ] || { echo -e '\033[0;31mError: ROS is not sourced.\033[0m' && exit 1 ; }
# Install to a new ros workspace:
mkdir -p ~/Code/pepper_ws/src
cd ~/Code/pepper_ws
catkin config --merge-devel
catkin config --extend /opt/ros/$ROS_DISTRO
catkin config -DCMAKE_BUILD_TYPE=Release
cd src
git clone git@github.com:ethz-asl/asl_pepper.git --branch devel
```

```
# apt-get Dependencies
# (Ensure dependencies listed in dependencies.rosinstall get cloned correctly!
# Some require SSH authentification.)
sudo apt -y install autoconf build-essential libtool
sudo apt -y install ros-$ROS_DISTRO-pepper-* ros-$ROS_DISTRO-naoqi-driver
sudo apt -y install ros-$ROS_DISTRO-joy
sudo apt -y install ros-$ROS_DISTRO-costmap-converter
sudo apt -y install ros-$ROS_DISTRO-move-base ros-$ROS_DISTRO-teb-local-planner
sudo apt -y install ros-$ROS_DISTRO-map-server
cd ~/Code/pepper_ws/src
wstool init
wstool merge asl_pepper/dependencies.rosinstall
wstool update
```

```
# Create and source a virtualenv
cd ~
virtualenv peppervenv --system-site-packages --python=python2.7
source ~/peppervenv/bin/activate
pip install numpy matplotlib Cython rospkg pyyaml
# (latest numba has build error on python 2)
pip install numba==0.44 llvmlite==0.30
```

```
# Python dependencies
source ~/peppervenv/bin/activate
cd ~/Code/pepper_ws/src/asl_pepper/asl_pepper_2d_simulator/python
pip install -e .
cd ~/Code/pepper_ws/src
{ python -c "import pyniel" && cd ~/Documents/pyniel && echo "Existing pyniel found." ; } || \
{ git clone git@github.com:danieldugas/pyniel.git && echo "Cloning pyniel." && cd pyniel ; }
pip install -e .
cd ~/Code/pepper_ws/src
git clone git@github.com:danieldugas/range_libc.git --branch comparisons
cd range_libc/pywrapper
python setup.py install
cd ~/Code/pepper_ws/src
git clone git@github.com:danieldugas/pymap2d.git
cd pymap2d
pip install .
cd ~/Code/pepper_ws/src
git clone git@github.com:danieldugas/pylidar2d.git
cd pylidar2d
pip install .
cd ~/Code/pepper_ws/src
git clone git@github.com:ethz-asl/interaction_actions --branch devel
cd interaction_actions/python/cIA
pip install .
cd ..
pip install -e .
cd ~/Code/pepper_ws/src
git clone git@github.com:ethz-asl/pepper_local_planning.git responsive --branch asldemo
cd responsive/lib_dwa
pip install .
cd ../lib_clustering
pip install .
# External python dependencies
cd ~/Code/pepper_ws/src
git clone git@github.com:danieldugas/Python-RVO2.git
cd Python-RVO2
pip install .
```

```
# Pedsim and subdependencies
cd ~/Code/pepper_ws/src
git clone https://github.com/srl-freiburg/pedsim_ros.git
cd pedsim_ros
git submodule update --init --recursive
cd ~/Code/pepper_ws/src
```

```
# (Optional) Realsense2 - If you intend to run the real Pepper robot with RGB-D camera.
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository -y "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u
sudo add-apt-repository -y "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
sudo apt update
sudo apt install -y librealsense2 librealsense2-dkms librealsense2-utils librealsense2-dev ros-$ROS_DISTRO-ddynamic-reconfigure
cd ~/Code/pepper_ws/src
git clone git@github.com:IntelRealSense/realsense-ros.git --branch 2.2.12
cd ~/Code/pepper_ws
catkin build realsense2_camera
```


```
# (Optional) Reinforcement Learning Dependencies
source ~/peppervenv/bin/activate
pip install tensorflow
# install baselines fork, with some functions modified to work with python 2
cd ~/peppervenv && mkdir -p src && cd src
git clone https://github.com/danieldugas/baselines.git --branch python2_ros
cd baselines
pip install -e .
```

```
# (Optional) RWTH Tracker Dependencies
sudo apt install -y qt4-default
source ~/peppervenv/bin/activate
pip install torch==1.1.0 torchvision==0.3.0
pip install tensorflow==1.8
cd ~/Code/pepper_ws/src
# if this didn't work, download the frame_soft repo from the google drive link below (see section frame_soft in this README)
# and place the frame_soft folder in ~/Code/pepper_ws/src/
git clone git@git.rwth-aachen.de:sabarinath.mahadevan/frame_soft.git --branch crowdbot_mobilenet
git clone git@github.com:danieldugas/darknet_ros.git --recursive --branch crowdbot
git clone git@github.com:danieldugas/tensorflow_object_detector.git --branch crowdbot
cd tensorflow_object_detector/data/models
wget -r -np -nH --cut-dirs=3 -R index.html* http://robotics.ethz.ch/~asl-datasets/pepper/tensorflow_models/ssd_mobilenet_v2_coco_2018_03_29/
cd ~/Code/pepper_ws/src
git clone https://github.com/Kukanani/vision_msgs.git --branch kinetic-devel
cd ~/Code/pepper_ws
catkin build tensorflow_object_detector rwth_crowdbot_launch framesoft_tracking_rviz_plugin
```

```
# (Optional) EPFL RDS Reactive Planner
sudo apt install -y libsdl2-dev
sudo apt install -y ros-$ROS_DISTRO-velodyne-msgs ros-$ROS_DISTRO-velodyne-pointcloud
cd ~/Code/pepper_ws/src
git clone git@github.com:epfl-lasa/rds.git --branch pepper_crowdbot
cd rds/rds
make demo_rds
catkin build rds_ros rds_gui_ros
```

```
# Build
cd ~/Code/pepper_ws
catkin build pedsim_visualizer pedsim_simulator spencer_tracking_rviz_plugin
catkin build gmapping map_matcher
catkin build pylidar2d_ros ia_ros responsive
catkin build asl_pepper
```

```
# vim: %s:Code/pepper_ws:Documents/Code/pepper_ws:g ; %s:peppervenv:rospyenv:g
```

### Test the executables

see the Pepper Simulator tutorial



# Tutorials

## Waypoint Navigation

- Follow the [installation instructions](https://github.com/ethz-asl/asl_pepper#install)
- open a new terminal, run
```
source ~/peppervenv/bin/activate
source ~/Code/pepper_ws/devel/setup.bash
roslaunch asl_pepper_motion_planning rvo_planner_sim.launch
```

## Pepper Simulator

- Follow the [installation instructions](https://github.com/ethz-asl/asl_pepper#install)
- open a new terminal, run
```
source ~/peppervenv/bin/activate
source ~/Code/pepper_ws/devel/setup.bash
roslaunch asl_pepper_2d_simulator pepper_2d_simulator.launch
```
- run rviz, load [this example visualization configuration.](http://robotics.ethz.ch/~asl-datasets/pepper/pepper_sensor_demo.rviz)

By now the simulator is running, and the robot can be controlled by publishing messages to the ```/cmd_vel``` topic
This can be done for example by running rqt robot_steering.

Otherwise, an example motion planner can be used to control the robot by setting goals:
- open another terminal, run
```
source ~/peppervenv/bin/activate
source ~/Code/pepper_ws/devel/setup.bash
roslaunch asl_pepper_motion_planning motion_planner.launch script_args:="--hot-start"
```
- Wait for a position estimate to be found (~10 seconds)
- Add a goal by pressing the 'g' keyboard key while in rviz, then clicking at the desired goal position.

## Visualize Pepper Sensors from a Recording

- Follow the [installation instructions](https://github.com/ethz-asl/asl_pepper#install)
- Download [this demo2.bag file](http://robotics.ethz.ch/~asl-datasets/pepper/demo2.bag) to your home directory: ```~/rosbags/demo2.bag```
- open a new terminal, run 
```
source ~/Code/pepper_ws/devel/setup.bash
roslaunch asl_pepper_rosbags demo2.bag
```
- run rviz, load [this example visualization configuration.](http://robotics.ethz.ch/~asl-datasets/pepper/pepper_sensor_demo.rviz)

## Reinforcement Learning for Pepper

see the [reinforcement learning README page](https://github.com/ethz-asl/asl_pepper/blob/master/wiki/RL_README.md#rl-environment-for-pepper-navigation)

# Batteries Not Included

## Maps

Some nodes, for example motion_planning can be given a reference map to work against. Though a single map, office_full is included in the github repository for convenience, more maps are available in the ASL google drive -> "Pepper" shared drive -> maps folder. For compatibility, it is recommended to place them in a ~/maps folder in your home directory.

```
ln -s -i ~/Code/pepper_ws/src/asl_pepper/maps ~/maps
```

## frame_soft

```
https://drive.google.com/open?id=1sytyNYNqgXDDBekClZnX5h0-fnpElAov
```

## Rosbags

rosbags are available in the ASL google drive -> "Pepper" shared drive -> "rosbags" folder. 
For compatibility, place them in a ~/rosbag folder in your home directory according to the following structure:
```
rosbags
├── CLA_rosbags
│   ├── 2019-06-14-10-04-06.bag
│   └── 2019-06-14-10-13-03.bag
├── demo1.bag
├── demo2.bag
├── full_sensors2.bag
├── full_sensors.bag
├── HG_rosbags
│   ├── 2019-04-05-11-56-07.bag
│   ├── 2019-04-05-11-58-01.bag
│   ├── 2019-04-05-12-01-42.bag
│   ├── 2019-04-05-12-04-58.bag
│   ├── 2019-04-05-13-01-00.bag
│   ├── 2019-04-05-13-03-25.bag
│   ├── 2019-04-05-13-08-23.bag
│   ├── 2019-04-05-13-12-11.bag
│   ├── hg_map.bag
│   └── scared_girl.bag
├── merged_demo2.bag
├── realsense_crowd.bag
├── realsense_test.bag
└── temp.bag
```



# Other

## Known Issues

- Maplab dependency model results in duplicate packages -> remove duplicates
- asl_pepper_cartographer (not installed by default) requires Cartographer, which requires Protobuf 3.4.1. using something like https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf to upgrade system protobuf breaks gazebo and maybe other things.
- Gmapping had to be modified in order to avoid crashing with hi-res laser


## Uninstall

```
$ cd ~
$ rm -rf ~/peppervenv
$ rm -rf ~/Code/pepper_ws
```

