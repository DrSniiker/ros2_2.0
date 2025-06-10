# to start run:

first souce the terminals using ´soure install/setup.bash´

then run:

´ros2 run navigator navigator_node.py´

´ros2 run navigator a_star_node.py´

in seperate terminals

then run:

´ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=\<your_map_file\>.yaml´

## in rviz

use "2D Pose Estimate" to place the robot

then use "Publish Point" to choose a goal point

when the preview binary map appears press any button to close it, as you do the a* algorithm will be displayed, if a path is constructed closing the map will start the navigation loop
