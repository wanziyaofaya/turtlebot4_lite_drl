import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import gymnasium as gym

class SlamRLNode(Node):
    def __init__(self):
        super().__init__('slam_rl_node')
        self.subscription_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.publisher_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.env = gym.make(None) # TODO: Add custom environment  
        self.state = None  
        self.agent_initialized = False

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        self.state = self.process_state(ranges)

    def odom_callback(self, msg):
        pass  # Add odometry handling if needed

    def process_state(self, scan_data):
        # normalize and return np arr
        return np.clip(scan_data, 0, 10)

    def step(self):
        if self.state is not None and not self.agent_initialized:
            self.agent_initialized = True
            observation, _ = self.env.reset()
        
        if self.agent_initialized:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = SlamRLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
