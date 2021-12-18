#!/bin/bash

# UPDATING THE VIRTUAL MACHINE
# Open a terminal window (Ctrl + Alt + T), and enter the following:
sudo apt-get update
sudo apt-get install ros-noetic-husky-desktop
sudo apt-get install ros-noetic-husky-simulator

# RUNNING A VIRTUAL HUSKY
# Open a terminal window, and enter:
# roslaunch husky_gazebo husky_empty_world.launch

# Open another terminal window, and enter:
# roslaunch husky_viz view_robot.launch

# rostopic pub /cmd_vel geometry_msgs/Twist "linear:
#         x: 0.5
#         y: 0.0
#         z: 0.0
# angular:
#         x: 0.0
#         y: 0.0
#         z: 0.0" -r 10
# UPDATING THE VIRTUAL MACHINE
# Open a terminal window (Ctrl + Alt + T), and enter the following:

sudo apt-get update
sudo apt-get install ros-noetic-warthog-simulator
sudo apt-get install ros-noetic-warthog-desktop
sudo apt-get install ros-noetic-warthog-navigation

# RUNNING A VIRTUAL WARTHOG
# Open a terminal window, and enter:
# roslaunch warthog_gazebo empty_world.launch

# Open another terminal window, and enter:
# roslaunch warthog_viz view_robot.launch

# Before we begin, install Git to pull packages from GitHub, and pygame, to provide us with the tools to map out Husky’s movements:
sudo apt-get install git
sudo apt-get install python-pygame

# The first step is to install rqt! We will also be installing some common plugins to create our dashboard
sudo apt-get install ros-noetic-rqt ros-noetic-rqt-common-plugins ros-noetic-rqt-robot-plugins

# We’ll begin by installing Clearpath’s Jackal simulation, desktop, and navigation packages.
sudo apt-get install ros-noetic-jackal-simulator ros-noetic-jackal-desktop