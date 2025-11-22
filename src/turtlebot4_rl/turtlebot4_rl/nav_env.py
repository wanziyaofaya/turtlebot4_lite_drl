import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from gz.transport14 import Node as GzNode
from gz.msgs11.pose_pb2 import Pose as GzPose
from gz.msgs11.boolean_pb2 import Boolean
from gz.msgs11.pose_v_pb2 import Pose_V
import time
import math
import tf_transformations

# Constants
GOAL_REACH_THRESHOLD = 0.1  # 目标到达阈值（米）

class TurtleBotNavEnv(gym.Env):
    def __init__(self, max_wait_for_observation=50.0, map_bounds=None, min_distance=2, positions_file=None):
        super().__init__()

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node('turtlebot_nav_env')
        
        # 地图边界和位置生成配置
        self.map_bounds = map_bounds if map_bounds is not None else {
            'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2
        }
        self.min_distance = min_distance  # 起点和目标之间的最小距离
        
        # Placeholder values - will be set by reset() before first use
        self.start_position = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_position = np.array([2.0, 2.0], dtype=np.float32)

        # 加载预定义起终点对
        self.positions = None
        self.position_index = 0
        if positions_file is None:
            positions_file = '/home/turtlebot4/turtlebot4_lite_drl/positions_6000.json'
        try:
            import json
            with open(positions_file, 'r') as f:
                self.positions = json.load(f)
            if not isinstance(self.positions, list) or len(self.positions) == 0:
                self.positions = None
                self._print_and_log("positions_6000.json 加载失败或为空，仍将使用随机起终点！")
            else:
                self._print_and_log(f"已加载{len(self.positions)}对起终点，将依次使用。")
        except Exception as e:
            self.positions = None
            self._print_and_log(f"未能加载positions_6000.json: {e}，仍将使用随机起终点！")

        # Velocity limits (use constants so clipping is consistent)
        self.MAX_LINEAR_VEL = 3.0
        self.MIN_LINEAR_VEL = 0.0
        self.MAX_ANGULAR_VEL = 1.9
        self.MIN_ANGULAR_VEL = -1.9

        # Define action spaces
        self.action_space = gym.spaces.Box(
            low=np.array([self.MIN_LINEAR_VEL, self.MIN_ANGULAR_VEL], dtype=np.float32),
            high=np.array([self.MAX_LINEAR_VEL, self.MAX_ANGULAR_VEL], dtype=np.float32),
            dtype=np.float32
        )

        # Continuous observation (64维LiDAR最小值 + robot state)
        # Robot state: [distance_to_goal, angle_to_goal, distance_change, angle_change, prev_linear_vel, prev_angular_vel]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.zeros(64), np.array([0.0, -np.pi, -20.0, -2*np.pi, 0.0, -1.9])]),
            high=np.concatenate([np.full(64, 12.0), np.array([20.0, np.pi, 20.0, 2*np.pi, 3.0, 1.9])]),
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

        # State - will be properly initialized by reset()
        self.lidar_data = None
        self.current_position = None
        self.current_yaw = 0.0
        self.prev_distance_to_goal = 0.0
        self.prev_angle_to_goal = 0.0
        self.done = False
        self.max_wait_for_observation = max_wait_for_observation

        # Gazebo model state tracking
        self.gazebo_position = None
        self.gazebo_orientation = None
        self.model_state_received = False

        # Previous velocities for robot state
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
        
        # Last action for reward calculation
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)

        self._print_and_log(f"TurtleBotNavEnv initialized. Call reset() before first use.")

    def scan_callback(self, msg):
        """Updates state with current scan data."""
        raw_data = np.array(msg.ranges, dtype=np.float32)
        raw_data[np.isinf(raw_data)] = 12.0
        # 分成64段，每段10个数据，取最小值
        if raw_data.shape[0] >= 640:
            processed = [np.min(raw_data[i*10:(i+1)*10]) for i in range(64)]
        else:
            # 如果数据不足640，补齐为64维
            padded = np.pad(raw_data, (0, 640-raw_data.shape[0]), constant_values=12.0)
            processed = [np.min(padded[i*10:(i+1)*10]) for i in range(64)]
        # 转成numpy数组并截断：将所有>=1的测距设为1，保留小于1的值
        processed = np.clip(processed, 0.0, 1)
        self.lidar_data = processed

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

    def _update_env_position(self):
        """Convert Gazebo coordinates to environment coordinates using TF transformation"""
        if self.gazebo_position is None or self.gazebo_orientation is None:
            return
            
        self.current_position = self.gazebo_position[:2].copy()  # Use only x, y
            
        # Calculate yaw from quaternion
        _, _, self.current_yaw = tf_transformations.euler_from_quaternion(self.gazebo_orientation)

    def _generate_random_positions(self):
        """生成随机起点和目标位置，确保不在障碍物内且满足最小距离要求"""
        from turtlebot4_rl.collision import is_spawn_position_valid
        
        max_attempts = 3000
        for _ in range(max_attempts):
            start_x = round(np.random.uniform(self.map_bounds['x_min'], self.map_bounds['x_max']), 2)
            start_y = round(np.random.uniform(self.map_bounds['y_min'], self.map_bounds['y_max']), 2)
            if not is_spawn_position_valid(start_x, start_y, bounds=self.map_bounds):
                continue
            
            goal_x = round(np.random.uniform(self.map_bounds['x_min'], self.map_bounds['x_max']), 2)
            goal_y = round(np.random.uniform(self.map_bounds['y_min'], self.map_bounds['y_max']), 2)
            if not is_spawn_position_valid(goal_x, goal_y, bounds=self.map_bounds):
                continue
            
            distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if distance >= self.min_distance:
                start_pos = np.array([start_x, start_y], dtype=np.float32)
                goal_pos = np.array([goal_x, goal_y], dtype=np.float32)
                return start_pos, goal_pos
        
        # 如果无法生成有效位置，使用默认值
        self._print_and_log("Warning: Could not generate valid positions, using fallback")
        return np.array([0.0, 0.0], dtype=np.float32), np.array([2.0, 2.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Reset the environment with new start and goal positions from positions_6000.json（如有），否则随机。"""
        # Call parent reset first to handle seeding
        super().reset(seed=seed)

        # 使用预定义起终点对
        if self.positions is not None and len(self.positions) > 0:
            pair = self.positions[self.position_index % len(self.positions)]
            self.position_index += 1
            try:
                start = np.array(pair['start'], dtype=np.float32)
                goal = np.array(pair['goal'], dtype=np.float32)
                self.start_position = start
                self.goal_position = goal
            except Exception as e:
                self._print_and_log(f"positions_6000.json 格式错误，使用随机起终点: {e}")
                self.start_position, self.goal_position = self._generate_random_positions()
        else:
            self.start_position, self.goal_position = self._generate_random_positions()

        # Send stop command
        self._send_stop_command()
        self.done = False

        # Reset position in Gazebo
        self._reset_robot_position()
        self._print_and_log(f"Resetting robot to start: x={self.start_position[0]:.2f}, y={self.start_position[1]:.2f} | goal: x={self.goal_position[0]:.2f}, y={self.goal_position[1]:.2f}")
        
        # Wait for model state to be received (updates self.current_position and self.current_yaw)
        self._wait_for_model_state()
        
        # Reset all state variables after we have current_position
        self.lidar_data = None
        self.prev_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        
        # Calculate initial angle to goal
        goal_vector = self.goal_position - self.current_position
        angle_to_goal_global = np.arctan2(goal_vector[1], goal_vector[0])
        self.prev_angle_to_goal = angle_to_goal_global - self.current_yaw
        # Normalize to [-pi, pi]
        self.prev_angle_to_goal = (self.prev_angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        # Reset velocities
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)

        # Wait for initial observations
        if not self._wait_for_new_state():
            raise RuntimeError("No LiDAR data received after reset timeout.")

        return self._get_state(), {}

    def step(self, action):
        """Execute one step in the environment."""
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
            # If no state available, return zeros for LiDAR数据
            lidar_data = np.zeros(64, dtype=np.float32)
        else:
            lidar_data = self.lidar_data.copy()
        
        # Calculate current distance and angle to goal
        distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)

        # 添加详细调试信息 - 所有坐标均为环境坐标系
        # self._print_and_log(
        #     f"🔍状态: "
        #     f"起始=[{self.start_position[0]:.3f}, {self.start_position[1]:.3f}] | "
        #     f"目标=[{self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}] | "
        #     f"当前=[{self.current_position[0]:.3f}, {self.current_position[1]:.3f}] | "
        #     f"距离目标={distance_to_goal:.3f}m | "
        # )
        
        # Calculate angle to goal relative to robot's current orientation
        goal_vector = self.goal_position - self.current_position
        angle_to_goal_global = np.arctan2(goal_vector[1], goal_vector[0])
        angle_to_goal = angle_to_goal_global - self.current_yaw
        # Normalize angle to [-pi, pi]
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate changes from previous step
        distance_change = distance_to_goal - self.prev_distance_to_goal
        angle_change = angle_to_goal - self.prev_angle_to_goal
        # Normalize angle change to [-pi, pi]
        angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi
        
        # Robot state: [distance_to_goal, angle_to_goal, distance_change, angle_change, prev_linear_vel, prev_angular_vel]
        robot_state = np.array([
            distance_to_goal,
            angle_to_goal,
            distance_change,
            angle_change,
            self.prev_linear_vel,
            self.prev_angular_vel
        ], dtype=np.float32)
        
        # Update history for next step
        self.prev_distance_to_goal = distance_to_goal
        self.prev_angle_to_goal = angle_to_goal
        
        # Combine LiDAR data with robot state
        combined_state = np.concatenate([lidar_data, robot_state])
        return combined_state

    def _calculate_reward(self, target, collision, min_laser):
        if target:
            target_reward = 100
            self._print_and_log(f"🎯 REWARD: Target reached! reward={target_reward:.3f}")
            return target_reward
        elif collision:
            collision_reward = -100
            self._print_and_log(f"💥 REWARD: Collision! reward={collision_reward:.3f}")
            return collision_reward
        else:
            distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
            distance_improvement = self.prev_distance_to_goal - distance_to_goal
            alpha = 20.0  
            beta = 20.0   
            step_penalty = 0.05
            distance_reward = 0.0
            if distance_improvement > 0:
                distance_reward = alpha * distance_improvement
                if hasattr(self, 'last_progress_time'):
                    self.last_progress_time = time.time()
            else:
                distance_reward = beta * distance_improvement

            linear_vel = self.last_action[0]
            angular_vel = abs(self.last_action[1])
            velocity_reward = linear_vel * 0.02 - angular_vel * 0.01

            obstacle_penalty = 0.0
            if min_laser < 0.3:
                obstacle_penalty = min_laser - 0.3

            total_reward = distance_reward - step_penalty
            # total_reward = distance_reward - step_penalty + velocity_reward
            # self._print_and_log(
            #     f"➖ REWARD: distance_reward={distance_reward:.3f}, "
            #     f"step_penalty={-step_penalty:.3f}, "
            #     f"velocity_reward={velocity_reward:.3f} | "
            #     f"total_reward={total_reward:.3f}"
            # )
            return total_reward
    
    def _is_collision(self):
        """Check if a collision has occurred based on LiDAR data."""
        collision_threshold = 0.25
        min_lidar = np.min(self.lidar_data) if self.lidar_data is not None else float('inf')
        min_lidar = min_lidar
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

        # Random initial yaw for better generalization
        yaw = np.random.uniform(-math.pi, math.pi)
        pose_msg.orientation.w = math.cos(yaw / 2.0)
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = math.sin(yaw / 2.0)

        service_name = "/world/maze/set_pose"
        timeout_ms = 1000

        # self._print_and_log(f"Resetting robot to position: x={self.start_position[0]}, y={self.start_position[1]}")

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
        
        # self._print_and_log("Waiting for initial model state from Gazebo topic...")

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
