import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from gz.transport14 import Node as GzNode
from gz.msgs11.pose_pb2 import Pose as GzPose
from gz.msgs11.boolean_pb2 import Boolean
from gz.msgs11.pose_v_pb2 import Pose_V
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener
import time
import math
import tf_transformations

# Constants
GOAL_REACH_THRESHOLD = 0.3  # 目标到达阈值（米）

class TurtleBotNavEnv(gym.Env):
    def __init__(self, start_position, goal_position, max_wait_for_observation=50.0):
        super().__init__()

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node('turtlebot_nav_env')

        # Velocity limits (use constants so clipping is consistent)
        self.MAX_LINEAR_VEL = 3.0
        self.MIN_LINEAR_VEL = -3.0
        self.MAX_ANGULAR_VEL = 1.5
        self.MIN_ANGULAR_VEL = -1.5

        # Define action spaces
        self.action_space = gym.spaces.Box(
            low=np.array([self.MIN_LINEAR_VEL, self.MIN_ANGULAR_VEL], dtype=np.float32),
            high=np.array([self.MAX_LINEAR_VEL, self.MAX_ANGULAR_VEL], dtype=np.float32),
            dtype=np.float32
        )

        # Continuous observation (LiDAR scans + robot state)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.zeros(640), np.array([0.0, -np.pi, -3.0, -1.5])]),
            high=np.concatenate([np.full(640, 12.0), np.array([20.0, np.pi, 3.0, 1.5])]),
            dtype=np.float32
        )

        # Pub/Sub
        self.cmd_vel_pub = self.node.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Gazebo Transport Node for getting model pose directly
        self.gz_node = GzNode()
        self.robot_model_name = 'turtlebot4'
        
        # Subscribe to model pose topic  
        self.model_pose_topic = f"/model/{self.robot_model_name}/pose"
        success = self.gz_node.subscribe(Pose_V, self.model_pose_topic, self._gz_pose_callback)
        if not success:
            self._print_and_log(f"Failed to subscribe to {self.model_pose_topic}")
        
        # TF2 Buffer and Listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        
        # Environment coordinate frame (you can customize this)
        self.env_frame_id = 'env_frame'
        self.gazebo_world_frame_id = 'world'

        # State
        self.lidar_data = None
        self.start_position = np.array(start_position, dtype=np.float32)
        self.goal_position = np.array(goal_position, dtype=np.float32)
        self.current_position = np.copy(self.start_position)
        self.current_yaw = 0.0
        self.last_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        self.best_distance_to_goal = np.linalg.norm(self.goal_position - self.start_position)
        self.max_steps_without_improvement = 10000
        self.distance_degradation_limit = 50
        self.done = False
        self.max_wait_for_observation = max_wait_for_observation

        self.previous_position = np.copy(self.start_position)
        self.stationary_steps = 0
        self.stationary_threshold = 0.01  # Threshold to consider the robot as stationary
        self.max_stationary_steps = 100  # Maximum allowed stationary steps before penalty

        # Gazebo model state tracking
        self.gazebo_position = None
        self.gazebo_orientation = None
        self.model_state_received = False

        # Previous velocities for robot state
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0

        self._reset_robot_position()
        self._wait_for_model_state()
        self._print_and_log(f"TurtleBotNavEnv initialized with Gazebo model state tracking")

    def scan_callback(self, msg):
        """Updates state with current scan data."""
        self.lidar_data = np.array(msg.ranges, dtype=np.float32)
        # Replace inf values with the maximum LiDAR range (12.0m)
        self.lidar_data[np.isinf(self.lidar_data)] = 12.0

    def _gz_pose_callback(self, msg):
        """Callback for Gazebo pose topic - receives Pose_V message"""
        try:
            # Find the main robot pose (not link poses)
            robot_pose = None
            for pose in msg.pose:
                # Look for the main robot entity pose
                if pose.name == self.robot_model_name:
                    robot_pose = pose
                    break
            
            if robot_pose is not None:
                # Store Gazebo position and orientation
                self.gazebo_position = np.array([
                    robot_pose.position.x,
                    robot_pose.position.y,
                    robot_pose.position.z
                ], dtype=np.float32)
                
                self.gazebo_orientation = [
                    robot_pose.orientation.x,
                    robot_pose.orientation.y,
                    robot_pose.orientation.z,
                    robot_pose.orientation.w
                ]
                
                # Convert to environment coordinates
                self._update_env_position()
                self.model_state_received = True
                
        except Exception as e:
            pass

    def _get_robot_pose_from_gazebo(self):
        """Get robot pose directly from Gazebo using Gazebo Transport"""
        try:
            # Request pose from Gazebo
            entity_name = self.robot_model_name
            service_name = "/world/maze/pose/info"
            
            # Create request message
            req = GzPose()
            req.name = entity_name
            
            # Make synchronous service call with timeout
            timeout_ms = 1000
            result, response = self.gz_node.request(service_name, req, GzPose, GzPose, timeout_ms)
            
            if result and response:
                # Store Gazebo position and orientation
                self.gazebo_position = np.array([
                    response.position.x,
                    response.position.y,
                    response.position.z
                ], dtype=np.float32)
                
                self.gazebo_orientation = [
                    response.orientation.x,
                    response.orientation.y,
                    response.orientation.z,
                    response.orientation.w
                ]
                
                # Convert to environment coordinates
                self._update_env_position()
                self.model_state_received = True
                return True
            else:
                self._print_and_log(f"Failed to get pose for {entity_name}")
                return False
                
        except Exception as e:
            self._print_and_log(f"Error getting robot pose from Gazebo: {e}")
            return False

    def _update_env_position(self):
        """Convert Gazebo coordinates to environment coordinates using TF transformation"""
        if self.gazebo_position is None or self.gazebo_orientation is None:
            return
            
        self.current_position = self.gazebo_position[:2].copy()  # Use only x, y
            
        # Calculate yaw from quaternion
        _, _, self.current_yaw = tf_transformations.euler_from_quaternion(self.gazebo_orientation)

    def _rotate_2d(self, point, theta):
        """Rotate a 2D point by theta (radians)."""
        c, s = math.cos(theta), math.sin(theta)
        x, y = point
        return np.array([c*x - s*y, s*x + c*y], dtype=np.float32)

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

        super().reset(seed=seed)

        self._send_stop_command()
        self.done = False

        # Reset position in Gazebo
        self._reset_robot_position()
        
        # Wait for model state to be received
        self._wait_for_model_state()
        
        # Reset state variables
        self.lidar_data = None
        self.last_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        self.direction_history = []
        self.previous_position = np.copy(self.start_position)

        # Wait for initial observations
        if not self._wait_for_new_state():
            raise RuntimeError("No LiDAR data received after reset timeout.")

        return self._get_state(), {}

    def step(self, action):
        """Execute one step in the environment."""
        # 保存执行动作前的距离作为"上次距离"
        self.last_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        
        # Execute action once
        self._take_action(action)
        self.last_action = action

        # Wait for both model state and LiDAR to update
        old_position = self.current_position.copy()
        old_lidar_id = id(self.lidar_data)
        start_time = time.time()
        model_updated = False
        lidar_updated = False
        
        while (time.time() - start_time < self.max_wait_for_observation):
            rclpy.spin_once(self.node, timeout_sec=0.05)
            
            # Check if model state has been updated via Gazebo topic callback
            if not model_updated and not np.allclose(self.current_position, old_position, atol=1e-5):
                model_updated = True
            
            # Check if LiDAR has been updated
            if not lidar_updated and id(self.lidar_data) != old_lidar_id:
                lidar_updated = True
            
            # Break if both are updated
            if model_updated and lidar_updated:
                break
        
        if not lidar_updated:
            raise RuntimeError("No LiDAR data received.")
        if not model_updated:
            self._print_and_log("Warning: Model state may not have been updated after action.")

        # Get current state
        done, collision, min_lidar = self._is_collision()
        distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        target = distance_to_goal < GOAL_REACH_THRESHOLD
        if target:
            done = True
            self._print_and_log("Goal reached!")

        reward = self._calculate_reward(target, collision, min_lidar)
        info = {'is_success': False, 'is_collision': False}
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
        # Clip linear and angular velocities to safety limits (ensure robot never exceeds limits)
        linear = float(np.clip(linear, self.MIN_LINEAR_VEL, self.MAX_LINEAR_VEL))
        angular = float(np.clip(angular, self.MIN_ANGULAR_VEL, self.MAX_ANGULAR_VEL))
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)

        # Store current velocities as previous velocities for next step
        self.prev_linear_vel = msg.twist.linear.x
        self.prev_angular_vel = msg.twist.angular.z
        # self._print_and_log(f"Velocities -> linear: {self.prev_linear_vel:.3f} m/s, angular: {self.prev_angular_vel:.3f} rad/s")

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

        # 添加详细调试信息 - 所有坐标均为环境坐标系
        self._print_and_log(
            f"🔍状态: "
            f"起始=[{self.start_position[0]:.3f}, {self.start_position[1]:.3f}] | "
            f"目标=[{self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}] | "
            f"当前=[{self.current_position[0]:.3f}, {self.current_position[1]:.3f}] | "
            f"距离目标={distance_to_goal:.3f}m | "
        )
        
        # Calculate angle to goal relative to robot's current orientatilobal = np.arctan2(goal_vector[1], goal_vector[0])
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
            target_reward = 120.0
            self._print_and_log(f"🎯 REWARD: Target reached! reward={target_reward:.3f}")
            return target_reward
        elif collision:
            collision_reward = -120.0
            self._print_and_log(f"💥 REWARD: Collision! reward={collision_reward:.3f}")
            return collision_reward
        else:
            distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
            if distance_to_goal <= 0.5:
                goal_reward = 0.25 * (1.0 - np.tanh(2.0 * (distance_to_goal - 0.2)))
            else:
                goal_reward = 0.0

            distance_improvement = self.last_distance_to_goal - distance_to_goal
            self._print_and_log(f"distance_improvement = {distance_improvement:.4f}")

            # 奖励参数
            alpha = 40.0  # 增加正向奖励，让靠近目标更有吸引力
            beta = 40.0   # 适度惩罚远离目标的行为
            step_penalty_coef = 0.06

            # === 距离改进奖励/惩罚 ===
            distance_reward = 0.0
            if distance_improvement > 0:
                distance_reward = alpha * distance_improvement
                # Update progress time
                if hasattr(self, 'last_progress_time'):
                    self.last_progress_time = time.time()
            else:
                distance_reward = beta * distance_improvement  # distance_improvement is negative

            # === 步数惩罚 ===
            step_penalty = step_penalty_coef

            # === 障碍物距离惩罚 ===
            obstacle_penalty = max(0, 1 - min_laser * 2.0) * 0.1

            linear_vel = self.last_action[0] if hasattr(self, 'last_action') else 0.0
            angular_vel = abs(self.last_action[1]) if hasattr(self, 'last_action') else 0.0

            velocity_reward = max(0, linear_vel) * 0.07 - angular_vel * 0.05

            # === 计算总奖励 ===
            total_reward = (distance_reward - step_penalty - obstacle_penalty + velocity_reward + goal_reward)

            # 打印详细的奖励分解
            self._print_and_log(
                f"📊 REWARD: "
                f"goal={goal_reward:+.3f} | "
                f"distance={distance_reward:+.3f} | "
                f"step=-{step_penalty:.3f} | "
                f"obstacle=-{obstacle_penalty:.3f} | "
                # f"orientation={orientation_reward:.3f} | "
                f"velocity={velocity_reward:+.3f} | "
                f"TOTAL={total_reward:+.2f}"
            )
            # === 更新状态 ===
            self.previous_position = np.copy(self.current_position)
            return total_reward

    def _count_oscillations(self):
        """
        Count the number of direction changes in the recent movement history.
        Oscillation is detected when the movement direction alternates frequently.
        """
        oscillations = 0
        for i in range(1, len(self.direction_history)):
            if self.direction_history[i] != 0 and self.direction_history[i] != self.direction_history[i-1]:
                oscillations += 1
        return oscillations
     
    def _update_direction_history(self, movement_direction):
        """
        Update the movement direction history with the latest movement.
        """
        self.direction_history.append(movement_direction)
        if len(self.direction_history) > self.max_history:
            self.direction_history.pop(0)

    def _angle_difference(self, current, target):
        """
        Compute the smallest difference between two angles.
        """
        diff = target - current
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    def _is_collision(self):
        """Check if a collision has occurred based on LiDAR data."""
        collision_threshold = 0.25
        min_lidar = np.min(self.lidar_data) if self.lidar_data is not None else float('inf')
        collision = min_lidar < collision_threshold
        if collision:
            self._print_and_log(f"Collision detected! min_lidar={min_lidar:.4f}")
        return collision, collision, min_lidar

    def _wait_for_new_state(self):
        """
        Spin until a new LiDAR scan is received or timeout.
        Return True if new state is received, False otherwise.
        """
        start_time = time.time()
        initial_id = id(self.lidar_data)
        while (id(self.lidar_data) == initial_id) and (time.time() - start_time < self.max_wait_for_observation):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if id(self.lidar_data) == initial_id:
            self._print_and_log("LiDAR data did not update in time.")
            return False
        else:
            return True

    def _reset_robot_position(self):
        """
        Reset the robot's position using Gazebo transport.
        """
        pose_msg = GzPose()
        pose_msg.name = "turtlebot4"
        pose_msg.position.x = float(self.start_position[0])
        pose_msg.position.y = float(self.start_position[1])
        pose_msg.position.z = 0.0

        yaw = -math.pi / 2  # Desired yaw in radians (facing downwards, -y direction in Gazebo)
        pose_msg.orientation.w = math.cos(yaw / 2.0)
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = math.sin(yaw / 2.0)

        service_name = "/world/maze/set_pose"
        timeout_ms = 1000

        self._print_and_log(f"Resetting robot to position: x={self.start_position[0]}, y={self.start_position[1]}")

        try:
            result, response = self.gz_node.request(service_name, pose_msg, GzPose, Boolean, timeout_ms)
            if result and response and response.data:
                self._print_and_log("Robot position reset successfully")
            else:
                self._print_and_log(f"Position reset failed: result={result}, response.data={response.data if response else 'None'}")
                # Don't raise exception, just warn - position tracking will still work
                self._print_and_log("Continuing with Gazebo model state tracking...")
        except Exception as e:
            self._print_and_log(f"Service call failed: {e}")
            self._print_and_log("Continuing with Gazebo model state tracking...")

        time.sleep(0.5)

    def _wait_for_model_state(self):
        """Wait for initial model state from Gazebo topic."""
        self.model_state_received = False
        
        self._print_and_log("Waiting for initial model state from Gazebo topic...")

        start_time = time.time()
        timeout = 5.0  # seconds

        while not self.model_state_received and (time.time() - start_time) < timeout:
            # Allow Gazebo transport to process messages
            time.sleep(0.1)
            
        if not self.model_state_received:
            raise RuntimeError("Model state reception timed out.")

    def _print_and_log(self, message):
        self.node.get_logger().info(message)

    def close(self):
        self._send_stop_command()
        # Unsubscribe from Gazebo topic if needed
        try:
            if hasattr(self, 'gz_node'):
                # Note: gz.transport doesn't have explicit unsubscribe, 
                # the node cleanup handles it
                pass
        except:
            pass
        self.node.destroy_node()
        rclpy.shutdown()
