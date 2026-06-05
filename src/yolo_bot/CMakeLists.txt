cmake_minimum_required(VERSION 3.0.2)
project(yolo_bot)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
