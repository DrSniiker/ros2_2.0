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
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from time import sleep
import numpy


class Turtlebot3ObstacleDetection(Node):

    def __init__(self):
        super().__init__('turtlebot3_obstacle_detection')
        print('TurtleBot3 Obstacle Detection - Auto Move Enabled')
        print('----------------------------------------------')
        print('stop angle: -90 ~ 90 deg')
        print('stop distance: 0.5 m')
        print('----------------------------------------------')

        self.scan_ranges = []
        self.has_scan_received = False
        self.has_map_received = False

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
        print(f'{self.has_scan_received=}')
        if self.has_scan_received and self.has_map_received:
            self.detect_obstacle()
            self.has_scan_received = False
    
    def create_map(self, data, height, width):
        maze2D = numpy.array(data, dtype=numpy.int8).reshape((height, width))
        # Kart hantering
        robot_pose_relative = Point()
        robot_pose_relative.x = (self.robot_pose.x - self.origin_offset.x)/self.map_resolution
        robot_pose_relative.y = (self.robot_pose.y - self.origin_offset.y)/self.map_resolution

        goal_pose_relative = Point()
        goal_pose_relative.x = (self.goal_pose.x - self.origin_offset.x)/self.map_resolution
        goal_pose_relative.y = (self.goal_pose.y - self.origin_offset.y)/self.map_resolution

        # A*

        print(robot_pose_relative.x, robot_pose_relative.y)
        print(goal_pose_relative.x, goal_pose_relative.y)
        self.print_maze_with_path(maze2D, (int(robot_pose_relative.x), int(robot_pose_relative.y)), (int(goal_pose_relative.x), int(goal_pose_relative.y)), 2)

    def print_maze_with_path(self, maze, start, goal, path):
        for i, row in enumerate(maze):
            line = ''
            for j, cell in enumerate(row):
                if (i, j) == start:
                    line += 'O'  # Startpunkt
                elif (i, j) == goal:
                    line += 'X'  # Målpunkt
                # elif path and (i, j) in path:
                #     line += '*'  # Del av vägen
                else:
                    line += '█' if cell >= 100 else ' '
            print(line)

    def detect_obstacle(self):
        while True:
            print('len: ', len(self.occupancy_grid_data))
            print('w*h: ', self.map_width*self.map_height)
            print(self.map_height, self.map_width)
        
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


def main(args=None):
    rclpy.init(args=args)
    turtlebot3_obstacle_detection = Turtlebot3ObstacleDetection()
    rclpy.spin(turtlebot3_obstacle_detection)

    turtlebot3_obstacle_detection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
