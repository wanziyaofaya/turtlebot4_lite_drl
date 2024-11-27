import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class TurtleBotNavEnv(gym.Env):
    def __init__(self):
        super(TurtleBotNavEnv, self).__init__()
        
        # Init ros2 node
        rclpy.init()
        self.node = rclpy.create_node('turtlebot_nav_env')
        
        # Define action and observation spaces
        # 4 Discrete Actions (forward, backwards, left, right)
        self.action_space = gym.spaces.Discrete(4)

        # Continuous observation (LiDAR scans)
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(360,), dtype=np.float32
        )
        
        # Pub/Sub
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # State
        self.state = None
        self.goal_position = np.array([5.0, 5.0]) # TODO set goal position based on gazebo world
        self.current_position = np.array([0.0, 0.0]) # TODO set initial position based on gazebo world
        self.last_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        self.done = False
        self.collision = False

    def scan_callback(self, msg):
        pass

    def odom_callback(self, msg):
        pass

    def reset(self):
        self.done = False
        self.collision = False
        self.last_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        self._send_stop_command()
        return self._get_state()
    
    def step(self, action):
        pass

    def _take_action(self, action):
        pass

    def _send_stop_command(self):
        pass
    
    def _get_state(self):
        if self.state is None:
            return np.zeros(self.observation_space.shape)
        return self.state

    def _calculate_reward(self):
        pass

    def _is_done(self):
        if self._is_collision():
            self.done = True
        elif np.linalg.norm(self.goal_position - self.current_position) < 0.5:
            self.done = True
        return self.done

    def _is_collision(self):
        pass
    
    def close(self):
        self._send_stop_command()
        self.node.destroy_node()
        rclpy.shutdown()
