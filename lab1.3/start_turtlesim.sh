#! /bin/bash

colcon build && source install/setup.sh && ros2 run turtlesim turtlesim_node
