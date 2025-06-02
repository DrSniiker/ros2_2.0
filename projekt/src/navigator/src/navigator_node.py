#!/usr/bin/env python3
#
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Jeonggeun Lim, Ryan Shim, Gilbert

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from std_msgs.msg import UInt8MultiArray, MultiArrayDimension
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from time import sleep
import numpy
import cv2
import csv


class Turtlebot3Navigator(Node):

    def __init__(self):
        super().__init__('turtlebot3_navigator_node')
        print('TurtleBot3 Obstacle Detection - Auto Move Enabled')
        print('----------------------------------------------')
        print('stop angle: -90 ~ 90 deg')
        print('stop distance: 0.5 m')
        print('----------------------------------------------')

        self.scan_ranges = []
        self.has_scan_received = False
        self.has_map_received = True

        self.stop_distance = 0.2
        self.tele_twist = Twist()
        self.tele_twist.linear.x = 0.0
        self.tele_twist.angular.z = 0.0

        self.robot_pose = Point()
        self.goal_pose = Point()
        self.origin_offset = Point()
        self.map_resolution = 0.0

        qos = QoSProfile(depth=10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        self.obstacle_detected = self.create_publisher(Bool, 'obs_det', qos)

        self.a_star_map_pub = self.create_publisher(UInt8MultiArray, 'a_star_map', qos)

        self.start_goal_coords_pub = self.create_publisher(UInt8MultiArray, 'start_goal_coords', qos)

        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)

        self.cmd_vel_raw_sub = self.create_subscription(
            Twist,
            'cmd_vel_raw',
            self.cmd_vel_raw_callback,
            qos_profile=qos_profile_sensor_data)
        
        self.cost_map_sub = self.create_subscription(
            OccupancyGrid,
            'global_costmap/costmap',
            self.cost_map_callback,
            qos_profile=qos_profile_sensor_data)

        self.cost_map_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.amcl_pose_callback,
            qos_profile=qos_profile_sensor_data)

        self.cost_map_sub = self.create_subscription(
            PointStamped,
            'clicked_point',
            self.clicked_point_callback,
            qos_profile=qos_profile_sensor_data)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        self.has_scan_received = True

    def amcl_pose_callback(self, msg):
        self.robot_pose = msg.pose.pose.position

    def clicked_point_callback(self, msg):
        self.goal_pose = msg.point
        self.has_map_received = True

    def cost_map_callback(self, msg):
        occupancy_grid_data = msg.data
        map_width = msg.info.width
        map_height = msg.info.height
        self.origin_offset = msg.info.origin.position
        self.map_resolution = msg.info.resolution
        if self.has_map_received:
            self.has_map_received = False
            self.create_map(occupancy_grid_data, map_height, map_width)

    def cmd_vel_raw_callback(self, msg):
        self.tele_twist = msg

    def timer_callback(self):
        if self.has_scan_received and self.has_map_received:
            self.has_scan_received = False

    def global_to_discrete(self, globalX, globalY):
        discreteX = (globalX - self.origin_offset.x)/self.map_resolution
        discreteY = (globalY - self.origin_offset.y)/self.map_resolution
        return (int(discreteX), int(discreteY))

    def discrete_to_global(self, coords):
        coords = numpy.array(coords, dtype=numpy.int32)
        coords *= self.map_resolution
        coords[:, 0] = coords[:, 0] + self.origin_offset.x
        coords[:, 1] = coords[:, 1] + self.origin_offset.y
        return coords
    
    # construct multiarray message
    def multi_array_constructor(self, array):
        msg = UInt8MultiArray()

        # Flatten the 2D array for the 'data' field
        msg.data = [int(item) for sublist in array for item in sublist]

        # Define the layout dimensions
        dim_row = MultiArrayDimension()
        dim_row.label = "rows"
        dim_row.size = len(array)
        dim_row.stride = len(array) * len(array[0])

        dim_col = MultiArrayDimension()
        dim_col.label = "cols"
        dim_col.size = len(array[0])
        dim_col.stride = len(array[0])

        msg.layout.dim = [dim_row, dim_col]
        msg.layout.data_offset = 0
        
        return msg

    def multi_array_deconstructor(self, msg):
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size

        # Reconstruct the 2D array from flat data list
        data = msg.data
        array_2d = [data[i * cols:(i + 1) * cols] for i in range(rows)]
        
        return array_2d


    def create_map(self, data, height, width):
        print('create map')
        maze2D = numpy.array(data, dtype=numpy.int8).reshape((height, -width))

        start = (2, 2)
        goal = (2, 18)
        maze = self.maze_from_csv('maze.csv')
        maze = numpy.array(maze, dtype=numpy.int8)

        print('############### BEFORE FUNCTION CALL ')
        print(f'{self.robot_pose.x=}')
        # robot_pose_relative = self.global_to_discrete(self.robot_pose.x, self.robot_pose.y)
        # goal_pose_relative = self.global_to_discrete(self.goal_pose.x, self.goal_pose.y)

        robot_pose_relative = self.global_to_discrete(start[0], start[1])
        goal_pose_relative = self.global_to_discrete(goal[0], goal[1])

        print('map')
        map_msg = UInt8MultiArray()
        map_msg = self.multi_array_constructor(maze)
        self.a_star_map_pub.publish(map_msg)

        print('points')
        points = [robot_pose_relative, goal_pose_relative]
        points = numpy.array(points, dtype=numpy.int8)
        start_goal_msg = UInt8MultiArray()
        start_goal_msg= self.multi_array_constructor(points)
        self.start_goal_coords_pub.publish(start_goal_msg)

        # A*

        print(robot_pose_relative)
        print(goal_pose_relative)

        self.print_map_cv2(maze, robot_pose_relative, goal_pose_relative) #TODO add path

    def print_map_cv2(self, map2D, robot_pose, goal_pose, threshold=50):
        """
        Visualize a 2D map with robot and goal positions using OpenCV.

        Parameters:
        - map2D: 2D list or numpy array representing the map (0 = free space, 1 = obstacle)
        - robot_pose: tuple (x, y) for robot position
        - goal_pose: tuple (x, y) for goal position
        """
        # Convert map to numpy array if it isn't already
        map2D = numpy.array(map2D, dtype=numpy.uint8)

        # Apply threshold: pixels above threshold are obstacles (1), below are free (0)
        binary_map = (map2D >= threshold).astype(numpy.uint8)

        # Normalize map to 0-255 for visualization (0 = white, 1 = black obstacle)
        # Invert so free space is white, obstacles are black
        img = 255 * (1 - binary_map)

        # Convert single channel to BGR for colored markings
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw origin as a red circle
        cv2.circle(img_color, (0,0), radius=3, color=(0, 0, 255), thickness=-1)

        # Draw robot position as a blue circle
        cv2.circle(img_color, robot_pose, radius=5, color=(255, 0, 0), thickness=-1)

        # Draw goal position as a green circle
        cv2.circle(img_color, goal_pose, radius=5, color=(0, 255, 0), thickness=-1)

        # Optionally resize for better visibility
        scale = 4
        img_resized = cv2.resize(img_color, (img_color.shape[1]*scale, img_color.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

        # Show the image
        cv2.imshow('Map Visualization', img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_obstacle(self):
        while True:
            sleep(1)
        
        left_range = int(len(self.scan_ranges) / 4)
        right_range = int(len(self.scan_ranges) * 3 / 4)

        obstacle_distance = min(
            min(self.scan_ranges[0:left_range]),
            min(self.scan_ranges[right_range:360])
        )

        twist = Twist()
        print(f'{obstacle_distance=:.2f}')
        # print(f'{self.has_scan_received=}')
        if obstacle_distance < self.stop_distance:
            twist.linear.x = 0.0
            twist.angular.z = self.tele_twist.angular.z
            self.get_logger().info('Obstacle detected! Stopping.', throttle_duration_sec=2)
            self.cmd_vel_pub.publish(twist)
            sleep(2)
            twist.linear.x = -0.1
            self.cmd_vel_pub.publish(twist)
            sleep(2)
        else:
            twist = self.tele_twist
            self.cmd_vel_pub.publish(twist)

    def maze_from_csv(self, filename):
        maze = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Filtrera bort tomma celler i varje rad
                filtered_row = [cell for cell in row if cell.strip() != '']
                if not filtered_row:
                    # Hoppa över tomma rader
                    continue
                maze.append([int(cell) for cell in filtered_row])
        return maze


def main(args=None):
    rclpy.init(args=args)
    turtlebot3_navigator_node = Turtlebot3Navigator()
    rclpy.spin(turtlebot3_navigator_node)

    turtlebot3_navigator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
