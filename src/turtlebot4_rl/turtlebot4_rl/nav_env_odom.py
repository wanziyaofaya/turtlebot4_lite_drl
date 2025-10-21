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
GOAL_REACH_THRESHOLD = 0.3  # 目标到达阈值（米）

class TurtleBotNavEnv(gym.Env):
    def __init__(self, start_position, goal_position, max_wait_for_observation=5.0):
        super().__init__()

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node('turtlebot_nav_env')

        # Velocity limits (use constants so clipping is consistent)
        self.MAX_LINEAR_VEL = 0.3
        self.MIN_LINEAR_VEL = -0.3
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
            low=np.concatenate([np.zeros(640), np.array([0.0, -np.pi, -0.3, -1.5])]),
            high=np.concatenate([np.full(640, 12.0), np.array([20.0, np.pi, 0.3, 1.5])]),
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

        # Odometry calibration
        self.odom_position_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.odom_calibrated = False
        self.yaw_offset = 0.0

        # Previous velocities for robot state
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0

        self._reset_robot_position()
        self._print_and_log(f"TurtleBotNavEnv initialized ")

    def scan_callback(self, msg):
        """Updates state with current scan data."""
        self.lidar_data = np.array(msg.ranges, dtype=np.float32)
        # Replace inf values with the maximum LiDAR range (12.0m)
        self.lidar_data[np.isinf(self.lidar_data)] = 12.0

    def odom_callback(self, msg):
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y

        odom_q = msg.pose.pose.orientation
        _, _, odom_yaw = tf_transformations.euler_from_quaternion([
            odom_q.x, odom_q.y, odom_q.z, odom_q.w
        ])

        if not self.odom_calibrated:
            # 记录初始的 odom 姿态
            self.initial_odom = np.array([odom_x, odom_y], dtype=np.float32)
            self.initial_odom_yaw = odom_yaw

            # Gazebo 设定的起点和朝向
            self.start_yaw = -math.pi / 2  # Facing -y
            self.start_position = np.array(self.start_position, dtype=np.float32)

            # 计算旋转和平移偏移
            self.yaw_offset = self.start_yaw - odom_yaw
            self.translation_offset = self.start_position - self._rotate_2d(
                np.array([odom_x, odom_y], dtype=np.float32),
                self.yaw_offset
            )

            self.odom_calibrated = True
            self.current_position = np.copy(self.start_position)
            self.current_yaw = self.start_yaw
            return

        rotated = self._rotate_2d(
            np.array([odom_x, odom_y], dtype=np.float32),
            self.yaw_offset
        )
        adjusted = rotated + self.translation_offset
        self.current_position = adjusted

        self.current_yaw = (odom_yaw + self.yaw_offset + math.pi) % (2 * math.pi) - math.pi

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
        self._calibrate_odom()
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

        # Wait for both odom and LiDAR to update
        old_position = self.current_position.copy()
        old_lidar_id = id(self.lidar_data)
        start_time = time.time()
        odom_updated = False
        lidar_updated = False
        
        while (time.time() - start_time < self.max_wait_for_observation):
            rclpy.spin_once(self.node, timeout_sec=0.05)
            
            # Check if odometry has been updated
            if not odom_updated and not np.allclose(self.current_position, old_position, atol=1e-5):
                odom_updated = True
            
            # Check if LiDAR has been updated
            if not lidar_updated and id(self.lidar_data) != old_lidar_id:
                lidar_updated = True
            
            # Break if both are updated
            if odom_updated and lidar_updated:
                break
        
        if not lidar_updated:
            raise RuntimeError("No LiDAR data received after step timeout.")
        if not odom_updated:
            self._print_and_log("Warning: Odometry data may not have been updated after action.")

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
            f"🔍 状态: "
            f"起始=[{self.start_position[0]:.3f}, {self.start_position[1]:.3f}] | "
            f"目标=[{self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}] | "
            f"当前=[{self.current_position[0]:.3f}, {self.current_position[1]:.3f}] | "
            f"距离目标={distance_to_goal:.3f}m | "
            f"改进={self.last_distance_to_goal - distance_to_goal:+.3f}m | "
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
            target_reward = 20.0
            self._print_and_log(f"🎯 REWARD: Target reached! reward={target_reward:.3f}")
            return target_reward
        elif collision:
            collision_reward = -10.0
            self._print_and_log(f"💥 REWARD: Collision! reward={collision_reward:.3f}")
            return collision_reward
        else:
            distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
            distance_improvement = self.last_distance_to_goal - distance_to_goal
            
            # 奖励参数 - 调整后的版本
            alpha = 200.0  # 增加正向奖励，让靠近目标更有吸引力
            beta = 150.0   # 适度惩罚远离目标的行为
            step_penalty_coef = 0.06
            orientation_scale = 0.2   

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
            obstacle_penalty = max(0, 1 - min_laser * 2.0) * 0.5

            # === 朝向目标角度 ===
            # desired_yaw = math.atan2(
            #     self.goal_position[1] - self.current_position[1],
            #     self.goal_position[0] - self.current_position[0]
            # )
            # yaw_diff = self._angle_difference(self.current_yaw, desired_yaw)
            # orientation_reward = math.cos(yaw_diff) * orientation_scale

            # 改进的速度奖励：更合理的速度激励
            linear_vel = self.last_action[0] if hasattr(self, 'last_action') else 0.0
            angular_vel = abs(self.last_action[1]) if hasattr(self, 'last_action') else 0.0
            velocity_reward = max(0,linear_vel) * 0.3 - abs(angular_vel) * 0.08 # 鼓励前进，适度惩罚旋转
            
            # === 计算总奖励 ===
            total_reward = (distance_reward - step_penalty - obstacle_penalty + velocity_reward)

            # 打印详细的奖励分解
            # self._print_and_log(
            #     f"📊 REWARD BREAKDOWN: "
            #     f"distance={distance_reward:+.3f} | "
            #     f"step=-{step_penalty:.3f} | "
            #     f"obstacle=-{obstacle_penalty:.3f} | "
            #     f"orientation={orientation_reward:.3f} | "
            #     f"velocity={velocity_reward:+.3f} | "
            #     f"TOTAL={total_reward:+.2f}"
            # )
            
            # 打印状态信息
            # self._print_and_log(
            #     f"📍 STATE INFO: "
            #     f"dist_to_goal={distance_to_goal:.3f}m | "
            #     f"improvement={distance_improvement:+.4f}m | "
            #     f"min_laser={min_laser:.3f}m | "
            #     f"stationary_steps={self.stationary_steps} | "
            #     f"yaw_diff={abs(yaw_diff)*180/math.pi:.1f}° | "
            #     f"vel=[{linear_vel:.2f}, {angular_vel:.2f}]"
            # )

            # === 更新状态 ===
            self.previous_position = np.copy(self.current_position)
            # 注意：last_distance_to_goal 现在在 step() 方法开始时更新
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
        Reset the robot's position and synchronize odometry.
        """
        node = Node()
        pose_msg = Pose()
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

        try:
            result, response = node.request(service_name, pose_msg, Pose, Boolean, timeout_ms)
            if not response.data:
                raise RuntimeError("Failed to reset the robot position.")
        except Exception as e:
            raise RuntimeError(f"Service call failed: {e}")

        time.sleep(0.5)

        self._calibrate_odom()

    def _calibrate_odom(self):
        """Spinlock until odometry callback received to determine correct offsets to use."""
        self.odom_calibrated = False
        self.odom_position_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.yaw_offset = 0.0

        # self._print_and_log("Calibrating odometry offsets...")

        start_time = time.time()
        timeout = 3.0  # seconds

        while not self.odom_calibrated and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if not self.odom_calibrated:
            raise RuntimeError("Odometry calibration timed out.")

    def _print_and_log(self, message):
        self.node.get_logger().info(message)

    def close(self):
        self._send_stop_command()
        self.node.destroy_node()
        rclpy.shutdown()
