#!/bin/bash

# Udpate apt
sudo apt update && sudo apt upgrade -y
# Install RViz plugin
sudo apt install ros-$ROS_DISTRO-rviz-imu-plugin

# Install sensor packages
cd ../ # .../src
git clone -b dev_4.0.6 --recursive https://github.com/stereolabs/zed-ros2-wrapper.git
git clone -b ros2 https://github.com/ros-drivers/nmea_navsat_driver.git
git clone https://github.com/CLOBOT-Co-Ltd/myahrs_ros2_driver.git
git clone https://github.com/mint-lab/mint_cart_ros.git
git clone -b foxy-devel https://github.com/dongwookheo/ros2_humble_ublox_f9r.git

# Fix myahrs_ros2_driver.cpp
cd ./myahrs_ros2_driver/myahrs_ros2_driver/src
sed -i 's/this->declare_parameter(\("[a-z_]*"\))/this->declare_parameter(\1, 0.0)/g' ./myahrs_ros2_driver.cpp

# Apply revised launch files
cd ../../../mint_cart_ros/changed_files
cp myahrs_ros2_driver.launch.py ../../myahrs_ros2_driver/myahrs_ros2_driver/launch/myahrs_ros2_driver_noviewer.launch.py
cp ublox_serial_driver.yaml ../../nmea_navsat_driver/config/ublox_serial_driver.yaml
cp ublox_serial.launch.py ../../nmea_navsat_driver/launch/ublox_serial.launch.py

# Install ROS Packages Dependencies
cd ../../../../ # ros woskspace
source /root/.bashrc
rosdep update
rosdep install --from-path src --ignore-src -r -y

# Alert finished script
echo "Finished install packages!"
echo "Go to ROS workspace && build with 'colcon build'"
echo "And then 'sourcing bash files'"
