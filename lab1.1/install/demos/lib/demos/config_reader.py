#! /usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class ConfigReaderNode(Node):
    def __init__(self):
        super().__init__("config_reader")
        
        # Declare parameters with default values
        self.declare_parameter("message", "Default message")
        self.declare_parameter("timer_period", 1.0)
        
        # Get parameter values
        self.message = self.get_parameter("message").value
        self.timer_period = self.get_parameter("timer_period").value
        
        # Log the parameters
        self.get_logger().info(f'Using message: "{self.message}"')
        self.get_logger().info(f'Using timer period: {self.timer_period} seconds')
        
        # Set up publisher and timer
        self.i = 0
        self.pub = self.create_publisher(String, "config_topic", 10)
        self.tmr = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        # Always get the latest parameter values
        current_message = self.get_parameter("message").value
        
        msg = String()
        msg.data = f"{current_message}: {self.i}"
        self.i += 1
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.pub.publish(msg)
        
        # Check if timer_period has changed
        current_period = self.get_parameter("timer_period").value
        if current_period != self.timer_period:
            self.get_logger().info(f"Timer period changed from {self.timer_period} to {current_period}")
            self.timer_period = current_period
            self.destroy_timer(self.tmr)
            self.tmr = self.create_timer(self.timer_period, self.timer_callback)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = ConfigReaderNode()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            node.destroy_node()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == "__main__":
    main() 