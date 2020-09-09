# interaction_actions_for_navigation

## Install

install [ROS](http://wiki.ros.org/melodic/Installation/Ubuntu)

```
set -e
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
[ ! -z $ROS_DISTRO ] || { echo -e '\033[0;31mError: ROS is not sourced.\033[0m' &&
                          echo "to install ROS, visit: http://wiki.ros.org/melodic/Installation/Ubuntu" &&
                          sleep 3 && exit 1 ; }
# Install to a new ros workspace:
mkdir -p ~/IAN/ian_ws/src
cd ~/IAN/ian_ws
catkin config --merge-devel
catkin config --extend /opt/ros/$ROS_DISTRO
catkin config -DCMAKE_BUILD_TYPE=Release
cd src
git clone https://github.com/ethz-asl/interaction_actions_for_navigation.git --branch devel
```

```
# apt-get Dependencies
# (Ensure dependencies listed in dependencies.rosinstall get cloned correctly)
sudo apt -y install autoconf build-essential libtool
sudo apt -y install ros-$ROS_DISTRO-pepper-* ros-$ROS_DISTRO-naoqi-driver
sudo apt -y install ros-$ROS_DISTRO-joy
sudo apt -y install ros-$ROS_DISTRO-costmap-converter
sudo apt -y install ros-$ROS_DISTRO-move-base ros-$ROS_DISTRO-teb-local-planner
sudo apt -y install ros-$ROS_DISTRO-map-server
cd ~/IAN/ian_ws/src
wstool init
wstool merge interaction_actions_for_navigation/dependencies.rosinstall
wstool update
```

```
# Create and source a virtualenv
cd ~/IAN
virtualenv ianvenv --system-site-packages --python=python2.7
source ~/IAN/ianvenv/bin/activate
pip install numpy matplotlib Cython rospkg pyyaml
# (latest numba has build error on python 2)
pip install numba==0.44 llvmlite==0.30
```

```
# Python dependencies
source ~/IAN/ianvenv/bin/activate
cd ~/IAN/ian_ws/src/interaction_actions_for_navigation/external/asl_pepper/asl_pepper_2d_simulator/python
pip install -e .
cd ~/IAN/ian_ws/src
{ python -c "import pyniel" && cd ~/Documents/pyniel && echo "Existing pyniel found." ; } || \
{ git clone https://github.com/danieldugas/pyniel.git && echo "Cloning pyniel." && cd pyniel ; }
pip install -e .
cd ~/IAN/ian_ws/src
git clone https://github.com/danieldugas/range_libc.git --branch comparisons
cd range_libc/pywrapper
python setup.py install
cd ~/IAN/ian_ws/src
git clone https://github.com/danieldugas/pymap2d.git
cd pymap2d
pip install .
cd ~/IAN/ian_ws/src
git clone https://github.com/danieldugas/pylidar2d.git
cd pylidar2d
pip install .
cd ~/IAN/ian_ws/src/interaction_actions_for_navigation/python/cIA
pip install .
cd ..
pip install -e .
cd ~/IAN/ian_ws/src
git clone https://github.com/ethz-asl/pepper_local_planning.git responsive --branch asldemo
cd responsive/lib_dwa
pip install .
cd ../lib_clustering
pip install .
# External python dependencies
cd ~/IAN/ian_ws/src
git clone https://github.com/danieldugas/Python-RVO2.git
cd Python-RVO2
pip install .
```

```
# Pedsim and subdependencies
cd ~/IAN/ian_ws/src
git clone https://github.com/srl-freiburg/pedsim_ros.git
cd pedsim_ros
git submodule update --init --recursive
cd ~/IAN/ian_ws/src
```

```
# Build
cd ~/IAN/ian_ws
catkin build pedsim_visualizer pedsim_simulator spencer_tracking_rviz_plugin
catkin build gmapping map_matcher
catkin build asl_pepper_gmapping asl_pepper_2d_simulator
catkin build pylidar2d_ros ia_ros responsive
```
