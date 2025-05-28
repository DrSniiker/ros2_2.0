# Obstacle Detection

## Instructions

### Setup and Running the Lab

1. **Terminal 1: Launch the Main Simulation**
   ```
   cd ~/workspace
   source install/setup.bash
   ros2 launch bringup main.launch.py
   ```
   
   Optional: To disable RViZ, add the `enable_rviz:=False` parameter:
   ```
   ros2 launch bringup main.launch.py enable_rviz:=False
   ```

2. **Terminal 2: Launch the Obstacle Detection**
   ```
   cd ~/workspace
   source install/setup.bash
   ros2 launch obstacle_detection obstacle_detection.launch.py
   ```

### Important Note

All instructions and details about how to complete this lab are contained within the `obstacle_detection.py` file located at:
```
lab2/src/obstacle_detection/obstacle_detection/obstacle_detection.py
```
# Discoveries

## Topic /odom
### nav_msgs/msg/Odometry Message
#### geometry_msgs/msg/PoseWithCovariance pose
#####  geometry_msgs/msg/Pose pose
#### geometry_msgs/msg/TwistWithCovariance twist
##### geometry_msgs/msg/Twist twist

För att röra på sig skicka Twist medelande till /cmd_vel noden
```
twist = self.tele_twist
self.cmd_vel_pub.publish(twist)
```
