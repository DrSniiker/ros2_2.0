#! /usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class ParamTalker(Node):
    def __init__(self):
        super().__init__("param_talker")

        # Declare parameters with default values
        self.declare_parameter("message", "Hello World")
        self.declare_parameter("timer_period", 1.0)

        # Get parameter values
        self.message = self.get_parameter("message").value
        self.timer_period = self.get_parameter("timer_period").value

        # Log the parameters
        self.get_logger().info(f'Using message: "{self.message}"')
        self.get_logger().info(f"Using timer period: {self.timer_period} seconds")

        self.i = 0
        self.pub = self.create_publisher(String, "chatter", 10)
        self.tmr = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = f"{self.message}: {self.i}"
        self.i += 1
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.pub.publish(msg)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = ParamTalker()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            # Clean up the node
            node.destroy_node()
    finally:
        # Shutdown only once, and only if we initialized
        try:
            rclpy.shutdown()
        except:
            # Ignore any shutdown errors
            pass


if __name__ == "__main__":
    main()
