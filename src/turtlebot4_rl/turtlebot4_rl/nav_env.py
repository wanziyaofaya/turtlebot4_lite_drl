import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from gz.transport14 import Node
from gz.msgs11.pose_pb2 import Pose
from gz.msgs11.boolean_pb2 import Boolean
import time
import math
import tf_transformations

# Constants
GOAL_REACH_THRESHOLD = 0.5  # 目标到达阈值（米）

class TurtleBotNavEnv(gym.Env):
    def __init__(self, start_position, goal_position, max_wait_for_observation=5.0):
        super().__init__()

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node('turtlebot_nav_env')

        # Define action spaces
        # Bounds for moving [linear, angular]
        self.action_space = gym.spaces.Box(low=np.array([-3.0, -1.5]), high=np.array([3.0, 1.5]), dtype=np.float32)

        # Continuous observation (LiDAR scans + robot state)
        # LiDAR: 640 values (0.0-10.0m) + robot state: 4 values
        # Robot state: [distance_to_goal, angle_to_goal, prev_linear_vel, prev_angular_vel]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.zeros(640), np.array([0.0, -np.pi, -3.0, -1.5])]),
            high=np.concatenate([np.full(640, 12.0), np.array([20.0, np.pi, 3.0, 1.5])]),
            dtype=np.float32
        )

        # Pub/Sub
        self.cmd_vel_pub = self.node.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # State
        self.lidar_data = None
        self.start_position = np.array(start_position, dtype=np.float32)
        self.goal_position = np.array(goal_position, dtype=np.float32)
        self.current_position = np.copy(self.start_position)
        self.current_yaw = 0.0
        self.done = False
        self.max_wait_for_observation = max_wait_for_observation

        # Odometry calibration
        self.odom_position_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.odom_calibrated = False
        self.yaw_offset = 0.0

        # Previous velocities for robot state
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0

        self._reset_robot_position()
        self._print_and_log("TurtleBotNavEnv initialized.")

    def scan_callback(self, msg):
        """Updates state with current scan data."""
        self.lidar_data = np.array(msg.ranges, dtype=np.float32)
        # Replace inf values with the maximum LiDAR range (12.0m)
        self.lidar_data[np.isinf(self.lidar_data)] = 12.0

    def odom_callback(self, msg):
        """Updates current position and orientation, applying odometry offsets if calibrated."""
        # Extract position
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y

        # Extract orientation (yaw)
        odom_q = msg.pose.pose.orientation
        odom_euler = tf_transformations.euler_from_quaternion([
            odom_q.x,
            odom_q.y,
            odom_q.z,
            odom_q.w
        ])
        odom_yaw = odom_euler[2]  # Yaw angle in radians

        if not self.odom_calibrated:
            # Calibrate odometry offsets
            self.odom_position_offset = np.array([
                odom_x - self.start_position[0],
                odom_y - self.start_position[1]
            ], dtype=np.float32)

            # Set yaw offset based on desired yaw (facing downwards)
            desired_yaw = -math.pi / 2  # Facing downwards (270 degrees)
            self.yaw_offset = desired_yaw - odom_yaw

            self.odom_calibrated = True
            # self._print_and_log(f"Odometry calibrated. Position offset: {self.odom_position_offset}, Orientation offset: {self.yaw_offset:.2f} radians.")

            # Reset current position and yaw to start position and desired yaw (facing downwards)
            self.current_position = np.copy(self.start_position)
            self.current_yaw = desired_yaw
            return

        # Apply position offset
        adjusted_x = odom_x - self.odom_position_offset[0]
        adjusted_y = odom_y - self.odom_position_offset[1]
        self.current_position = np.array([adjusted_x, adjusted_y], dtype=np.float32)

        # Apply orientation offset
        adjusted_yaw = odom_yaw + self.yaw_offset
        # Normalize yaw to [-pi, pi]
        adjusted_yaw = (adjusted_yaw + math.pi) % (2 * math.pi) - math.pi
        self.current_yaw = adjusted_yaw

    def seed(self, seed=0):
        """Set the random seed for reproducibility."""
        super().seed(seed)
        np.random.seed(seed)

    def reset(self, *, seed=None, options=None, start_position=None, goal_position=None):
        """Reset the environment. Optionally set new start and goal positions."""
        if start_position is not None:
            self.start_position = np.array(start_position, dtype=np.float32)
        if goal_position is not None:
            self.goal_position = np.array(goal_position, dtype=np.float32)

        # Print start and goal positions for this episode
        self._print_and_log(f"Episode starting - Start position: [{self.start_position[0]:.2f}, {self.start_position[1]:.2f}], Goal position: [{self.goal_position[0]:.2f}, {self.goal_position[1]:.2f}]")

        super().reset(seed=seed)

        self._send_stop_command()
        self.done = False

        # Reset position in Gazebo
        self._reset_robot_position()

        # Reset state variables
        self.lidar_data = None

        # Wait for initial observations
        if not self._wait_for_new_state():
            raise RuntimeError("No LiDAR data received after reset timeout.")

        return self._get_state(), {}

    def step(self, action):
        """Execute one step in the environment."""
        # Take the action and save it
        self._take_action(action)
        self.last_action = action

        # Wait for new sensor data
        if not self._wait_for_new_state():
            raise RuntimeError("No LiDAR data received after step timeout.")

        # Check termination conditions
        done, collision, min_lidar = self._is_collision()
        distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        target = distance_to_goal < GOAL_REACH_THRESHOLD
        
        if target:
            done = True
            self._print_and_log("Goal reached!")

        # Calculate reward
        reward = self._calculate_reward(target, collision, min_lidar)

        # 构造info字典，标记成功或碰撞
        info = {}
        if target:
            info['is_success'] = True
        elif collision:
            info['is_collision'] = True

        return self._get_state(), reward, done, False, info

    def _take_action(self, action):
        """Send velocity command to the robot."""
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # self._print_and_log(f"Action received: {action}")

        linear, angular = action
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)

        # Store current velocities as previous velocities for next step
        self.prev_linear_vel = msg.twist.linear.x
        self.prev_angular_vel = msg.twist.angular.z

        self.cmd_vel_pub.publish(msg)

    def _send_stop_command(self):
        """Send zero velocity to the robot."""
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.cmd_vel_pub.publish(msg)

    def _get_state(self):
        """Return the current state (LiDAR readings + robot state)."""
        # LiDAR data
        if self.lidar_data is None:
            # If no state available, return zeros for LiDAR data
            lidar_data = np.zeros(640, dtype=np.float32)
        else:
            lidar_data = self.lidar_data.copy()
        
        # Robot state: [distance_to_goal, angle_to_goal, prev_linear_vel, prev_angular_vel]
        distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        
        # Calculate angle to goal relative to robot's current orientation
        goal_vector = self.goal_position - self.current_position
        angle_to_goal_global = np.arctan2(goal_vector[1], goal_vector[0])
        angle_to_goal = angle_to_goal_global - self.current_yaw
        # Normalize angle to [-pi, pi]
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        robot_state = np.array([
            distance_to_goal,
            angle_to_goal,
            self.prev_linear_vel,
            self.prev_angular_vel
        ], dtype=np.float32)
        
        # Combine LiDAR data with robot state
        combined_state = np.concatenate([lidar_data, robot_state])
        return combined_state

    def _calculate_reward(self, target, collision, min_laser):
        if target:
            return 1000.0  # 到达目标的高奖励
        elif collision:
            return -1000.0  
        else:
            # 每步惩罚
            step_penalty = -0.01

            # 距离目标的奖励（越接近目标奖励越高）
            distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
            if distance_to_goal < 2.0:
                distance_reward = max(0, 1 - distance_to_goal / 2.0) * 2  # 归一化到 [0,2]
            else:
                distance_reward = 0.0

            # 激励机器人保持线速度并减少角速度
            linear_vel = self.last_action[0] if hasattr(self, 'last_action') else 0.0
            angular_vel = self.last_action[1] if hasattr(self, 'last_action') else 0.0
            velocity_reward = linear_vel - abs(angular_vel) * 0.5

            # 激励机器人远离障碍物
            obstacle_penalty = max(0, 1 - min_laser * 2.0) * 0.5

            # 综合奖励
            reward = step_penalty + distance_reward  + velocity_reward  - obstacle_penalty 
            return reward

    def _is_collision(self):
        """Check if a collision has occurred based on LiDAR data."""
        collision_threshold = 0.3
        min_lidar = np.min(self.lidar_data) if self.lidar_data is not None else float('inf')
        collision = min_lidar < collision_threshold
        if collision:
            self._print_and_log("Collision detected!")
        return collision, collision, min_lidar

    def _wait_for_new_state(self):
        """
        Spin until a new LiDAR scan is received or timeout.
        Return True if new state is received, False otherwise.
        """
        start_time = time.time()
        initial_state = self.lidar_data
        while (self.lidar_data is initial_state) and (time.time() - start_time < self.max_wait_for_observation):
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.lidar_data is not initial_state

    def _reset_robot_position(self):
        """
        Reset the robot's position and synchronize odometry.
        """
        node = Node()
        pose_msg = Pose()
        pose_msg.name = "turtlebot4"

        pose_msg.position.x = float(self.start_position[0])
        pose_msg.position.y = float(self.start_position[1])
        pose_msg.position.z = 0.0

        yaw = -math.pi / 2  # Desired yaw in radians (facing downwards)
        pose_msg.orientation.w = math.cos(yaw / 2.0)
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = math.sin(yaw / 2.0)

        service_name = "/world/maze/set_pose"
        timeout_ms = 1000

        try:
            result, response = node.request(service_name, pose_msg, Pose, Boolean, timeout_ms)
            if not response.data:
                raise RuntimeError("Failed to reset the robot position.")
        except Exception as e:
            raise RuntimeError(f"Service call failed: {e}")

        time.sleep(0.1)

        self._calibrate_odom()

    def _calibrate_odom(self):
        """Spinlock until odometry callback received to determine correct offsets to use."""
        self.odom_calibrated = False
        self.odom_position_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.yaw_offset = 0.0

        # self._print_and_log("Calibrating odometry offsets...")

        start_time = time.time()
        timeout = 5.0  # seconds

        while not self.odom_calibrated and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if not self.odom_calibrated:
            raise RuntimeError("Odometry calibration timed out.")

    def _print_and_log(self, message):
        self.node.get_logger().info(message)

    def close(self):
        self._send_stop_command()
        self.node.destroy_node()
        rclpy.shutdown()
