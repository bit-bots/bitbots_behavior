cmake_minimum_required(VERSION 2.8.3)
project(bitbots_body_behavior)

find_package(catkin REQUIRED COMPONENTS rospy)

find_package(catkin COMPONENTS bitbots_docs)
enable_bitbots_docs()

catkin_python_setup()

catkin_package()
