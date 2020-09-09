#!/bin/bash
# Ensure naoqi_driver is already running
rosrun camera_calibration cameracalibrator.py --size 7x6 --square 0.07 image:=/pepper_robot/camera/ir/image_raw camera:=/pepper_robot/camera/ir --no-service-check &
DIR=`dirname $0`
PCKG_DIR="$DIR/../"
sleep 1
echo "######"
echo "Starting calibration frame generation. Ensure the robot and target are static when capturing frames."
python "$DIR/generate_calibration_frames.py"
cd "$PCKG_DIR/calibration_data"
rm calibration.bag
echo "######"
echo "Starting Kalibr calibration script"
rosrun kalibr kalibr_bagcreater --folder "$PCKG_DIR/calibration_data" --output-bag "$PCKG_DIR/calibration_data/calibration.bag"
rosrun kalibr kalibr_calibrate_cameras --bag "$PCKG_DIR/calibration_data/calibration.bag" --topics /cam0/image_raw /cam1/image_raw --models pinhole-radtan pinhole-radtan --target "$PCKG_DIR/config/calibration_grid.yaml"
