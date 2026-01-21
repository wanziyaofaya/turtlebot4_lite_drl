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
from gz.msgs11.entity_factory_pb2 import EntityFactory  # 新增导入
import time
import math
import tf_transformations
import os
import torch
import tabm
import rtdl_num_embeddings
import sklearn.preprocessing
from typing import NamedTuple

class RegressionLabelStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray

# Constants
GOAL_REACH_THRESHOLD = 0.1  # 全局目标到达阈值（米）
SUBGOAL_SWITCH_THRESHOLD = 0.2  # 子目标切换阈值（米）- 接近子目标时静默切换
SUBGOAL_STOP_GENERATION_DISTANCE = 1.5  # 当机器人距全局终点小于该值时，不再生成新的子目标，直接追踪全局终点
SUBGOAL_CLOSE_TO_GOAL_THRESHOLD = 0.5  # 若预测子目标距全局终点小于该值，则直接使用全局终点

class TurtleBotNavEnv(gym.Env):
    def __init__(self, max_wait_for_observation=5.0, map_bounds=None, min_distance=2.0, positions_file=None, subgoal_model_path=None):
        super().__init__()

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node('turtlebot_nav_env')
        
        # 地图边界和位置生成配置
        # self.map_bounds = map_bounds if map_bounds is not None else {
        #     'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2
        # }
        self.map_bounds = map_bounds if map_bounds is not None else {
             'x_min': -3, 'x_max': 3, 'y_min': -3, 'y_max': 3
          }
        # self.map_bounds = map_bounds if map_bounds is not None else {
        #     'x_min': -5, 'x_max': 5, 'y_min': -5, 'y_max': 5
        # }
        self.min_distance = min_distance  # 起点和目标之间的最小距离
        
        # Placeholder values - will be set by reset() before first use
        self.start_position = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_position = np.array([2.0, 2.0], dtype=np.float32)
        # 加载预定义起终点对
        self.positions = None
        self.position_index = 0
        if positions_file is None:
            # 优先尝试当前工作目录下的文件
            if os.path.exists('positions_6.json'):
                positions_file = os.path.abspath('positions_6.json')
            else:
                positions_file = '/home/wanzi/turtlebot4_lite_drl/positions_6.json'
        
        try:
            import json
            with open(positions_file, 'r') as f:
                self.positions = json.load(f)
            if not isinstance(self.positions, list) or len(self.positions) == 0:
                self.positions = None
                self._print_and_log(f"{positions_file} 加载失败或为空，仍将使用随机起终点！")
            else:
                self._print_and_log(f"已从 {positions_file} 加载{len(self.positions)}对起终点，将依次使用。")
        except Exception as e:
            self.positions = None
            self._print_and_log(f"未能加载 {positions_file}: {e}，仍将使用随机起终点！")
        # Velocity limits (use constants so clipping is consistent)
        self.MAX_LINEAR_VEL = 3.0  # 降低速度，防止因速度过快刹不住车
        self.MAX_ANGULAR_VEL = 1.9
        
        # LiDAR & Goal configuration
        self.LIDAR_MAX_RANGE = 9.0  # Maximum LiDAR range in meters
        self.MAX_GOAL_DIST = 9.0  # Maximum distance for goal normalization

        # Define action spaces in normalized range [-1, 1]
        # action[0]: normalized linear command, action[1]: normalized angular command
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Continuous observation (64维LiDAR最小值 + robot state)
        # Robot state: [distance_to_goal, angle_to_goal, distance_change, angle_change, prev_linear_vel, prev_angular_vel]
        # Normalized observation space: All values roughly in [-1, 1] or [0, 1]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.zeros(32), np.array([0.0, -1.0, -1.0, -1.0, 0.0, -1.0])]),
            high=np.concatenate([np.ones(32), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]),
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
        self.raw_data = None
        self.lidar_seq = 0
        self.min_lidar = None
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
        self.model_state_seq = 0
        self.last_model_state_time = 0.0
        self.last_reset_yaw = 0.0

        # Previous velocities for robot state
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0

        # Cached goal metrics (reused within a step to reduce duplicate math)
        self._cached_model_seq = None
        self._cached_distance_to_goal = None
        self._cached_angle_to_goal = None
        
        # Last action for reward calculation
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)

        # Visualization State
        self.markers_initialized = False
        self.start_marker_name = "start_marker_visual"
        self.goal_marker_name = "goal_marker_visual"
        self.subgoal_marker_name = "subgoal_marker_visual"
        self.subgoal_marker_spawned = False

        # Subgoal Model Setup
        self.subgoal_model = None
        self.subgoal_preprocessing = None
        self.subgoal_label_stats = None
        self.device = torch.device('cpu')
        self.global_goal_position = None  # 真正的全局终点
        self.current_goal = None  # RL 智能体看到的"当前目标"（可能是子目标或全局终点）
        self.using_subgoal = False  # 当前是否在追踪子目标
        self.lidar_data_64 = None
        
        if subgoal_model_path:
             self._load_subgoal_model(subgoal_model_path)

        self._print_and_log(f"TurtleBotNavEnv initialized. Call reset() before first use.")

    def scan_callback(self, msg):
        """Updates state with current scan data."""
        raw_data = np.asarray(msg.ranges, dtype=np.float32)
        raw_data[np.isinf(raw_data)] = self.LIDAR_MAX_RANGE

        # Ensure a fixed length (640 = 32 * 20)
        if raw_data.size < 640:
            raw_data = np.pad(raw_data, (0, 640 - raw_data.size), constant_values=self.LIDAR_MAX_RANGE)
        elif raw_data.size > 640:
            raw_data = raw_data[:640]

        # Vectorized min-pooling into 32 beams
        processed = raw_data.reshape(32, 20).min(axis=1)
        
        # Vectorized min-pooling into 64 beams for Subgoal Model
        processed_64 = raw_data.reshape(64, 10).min(axis=1)

        self.raw_data = raw_data
        self.lidar_data = processed  # np.ndarray(float32), avoids per-step list->array conversion
        self.lidar_data_64 = processed_64
        self.min_lidar = float(processed.min())
        self.lidar_seq += 1

    def _gz_pose_callback(self, msg):
        """Callback for Gazebo pose topic - receives Pose_V message"""
        try:
            # /model/<name>/pose sometimes includes multiple poses (model + links).
            # Prefer the model pose; if name matching is inconsistent, fall back safely.
            robot_pose = None

            if hasattr(msg, 'pose') and msg.pose:
                # 1) Exact match
                for pose in msg.pose:
                    if pose.name == self.robot_model_name:
                        robot_pose = pose
                        break

                # 2) Best-effort match: choose the shortest name containing the model name
                if robot_pose is None:
                    candidates = [p for p in msg.pose if self.robot_model_name in (p.name or "")]
                    if candidates:
                        robot_pose = min(candidates, key=lambda p: len(p.name or ""))

                # 3) Fallback: take the first pose
                if robot_pose is None:
                    robot_pose = msg.pose[0]

            if robot_pose is not None:
                self.gazebo_position = np.array(
                    [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z],
                    dtype=np.float32,
                )

                self.gazebo_orientation = [
                    robot_pose.orientation.x,
                    robot_pose.orientation.y,
                    robot_pose.orientation.z,
                    robot_pose.orientation.w,
                ]

                self._update_env_position()
                self.model_state_received = True
                self.model_state_seq += 1
                self.last_model_state_time = time.time()
                
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
        
        max_attempts = 1000
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

    def _is_subgoal_valid(self, subgoal: np.ndarray) -> bool:
        """使用 is_position_valid 判断 current_goal(子目标) 是否合理（与障碍物/边界保持安全距离）。"""
        if subgoal is None:
            return False
        try:
            x = float(subgoal[0])
            y = float(subgoal[1])
        except Exception:
            return False
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        from turtlebot4_rl.collision import is_position_valid
        return bool(is_position_valid(x, y, bounds=self.map_bounds))

    def reset(self, *, seed=None, options=None):
        """Reset the environment with new random start and goal positions."""
        
        # Call parent reset first to handle seeding
        super().reset(seed=seed)
        
        # Generate new random positions (will use the seed set by super().reset())
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
                self._print_and_log(f"positions_6.json 格式错误，使用随机起终点: {e}")
                self.start_position, self.goal_position = self._generate_random_positions()
        else:
            self.start_position, self.goal_position = self._generate_random_positions()

        self._print_and_log(f"Episode Reset: Start Position: {self.start_position}, Goal Position: {self.goal_position}")

        # Send stop command
        self._send_stop_command()
        self.done = False

        # Update visuals for start and goal in Gazebo
        self._update_marker_visuals()

        # Reset position in Gazebo
        self.reset_success = self._reset_robot_position()
        self._print_and_log(f"Resetting robot to start: x={self.start_position[0]:.2f}, y={self.start_position[1]:.2f} | goal: x={self.goal_position[0]:.2f}, y={self.goal_position[1]:.2f}")
        
        # Wait for model state to be received (updates self.current_position and self.current_yaw)
        # Wait for 5 updates to ensure we clear any stale data from before reset
        self._wait_for_model_state(num_updates=5)
        
        # Reset all state variables after we have current_position
        self.lidar_data = None
        self.min_lidar = None
        self._cached_model_seq = None
        self._cached_distance_to_goal = None
        self._cached_angle_to_goal = None
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

        # Wait for initial observations (wait for 5 updates to ensure fresh LiDAR data)
        if not self._wait_for_new_state(num_updates=5):
            raise RuntimeError("No LiDAR data received after reset timeout.")

        # --- Subgoal Logic Start (Goal Masquerading) ---
        # 保存全局终点，设置 RL 看到的 current_goal
        self.global_goal_position = self.goal_position.copy()
        self.using_subgoal = False

        # 如果离全局终点已经很近，则不启用子目标，直接追踪全局终点
        dist_to_global_goal = float(np.linalg.norm(self.global_goal_position - self.current_position))
        if dist_to_global_goal < SUBGOAL_STOP_GENERATION_DISTANCE:
            self._print_and_log(
                f"[Internal] Close to global goal (dist={dist_to_global_goal:.3f}m < {SUBGOAL_STOP_GENERATION_DISTANCE}), using global goal directly."
            )
            self.current_goal = self.global_goal_position.copy()
            self.using_subgoal = False
        elif self.subgoal_model is not None:
            subgoal = self._predict_subgoal()
            if subgoal is not None:
                dist_subgoal_to_current = float(np.linalg.norm(subgoal - self.current_position))
                dist_subgoal_to_global = float(np.linalg.norm(subgoal - self.global_goal_position))

                # 子目标若落在障碍物/边界安全距离内，直接回退到全局目标
                if not self._is_subgoal_valid(subgoal):
                    self._print_and_log("[Internal] Subgoal invalid by is_position_valid; using global goal instead.")
                    self.current_goal = self.global_goal_position.copy()
                    self.using_subgoal = False

                # 子目标若已非常接近全局终点：直接追踪全局终点（等价于把终点当作子目标）
                elif dist_subgoal_to_global < SUBGOAL_CLOSE_TO_GOAL_THRESHOLD:
                    self._print_and_log(
                        f"[Internal] Subgoal close to global goal (dist={dist_subgoal_to_global:.3f}m < {SUBGOAL_CLOSE_TO_GOAL_THRESHOLD}); using global goal instead."
                    )
                    self.current_goal = self.global_goal_position.copy()
                    self.using_subgoal = False

                # 子目标太近当前位置 -> 认为无效，退化为全局终点（避免刚开始就触发切换）
                elif dist_subgoal_to_current < SUBGOAL_SWITCH_THRESHOLD:
                    self._print_and_log(
                        f"[Internal] Subgoal too close to current position ({dist_subgoal_to_current:.3f}m), using global goal instead."
                    )
                    self.current_goal = self.global_goal_position.copy()
                    self.using_subgoal = False
                else:
                    # 静默设置子目标为 current_goal，RL 不知道这是子目标
                    self.current_goal = subgoal
                    self.using_subgoal = True

                self._print_and_log(f"[Internal] Using current_goal: {self.current_goal} (using_subgoal={self.using_subgoal})")
                if self.subgoal_marker_spawned:
                    self._move_marker(self.subgoal_marker_name, self.current_goal)
                else:
                    self._spawn_marker(self.subgoal_marker_name, self.current_goal, color="0 0 1 1")
                    self.subgoal_marker_spawned = True
            else:
                self._print_and_log("Subgoal prediction failed, using global goal.")
                self.current_goal = self.global_goal_position.copy()
                self.using_subgoal = False
        else:
            self.current_goal = self.global_goal_position.copy()
            self.using_subgoal = False
        # --- Subgoal Logic End ---
        
        # goal_position 现在指向 RL 看到的目标
        self.goal_position = self.current_goal
        
        # Recalculate metrics with current_goal
        self.prev_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        goal_vector = self.goal_position - self.current_position
        angle_to_goal_global = np.arctan2(goal_vector[1], goal_vector[0])
        self.prev_angle_to_goal = angle_to_goal_global - self.current_yaw
        self.prev_angle_to_goal = (self.prev_angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        return self._get_state(), {}

    def step(self, action):
        """Execute one step in the environment."""
        # Execute action once
        self._take_action(action)
        self.last_action = action

        # Wait for both model state and LiDAR to update
        initial_model_seq = self.model_state_seq
        initial_lidar_seq = self.lidar_seq
        deadline = time.monotonic() + float(self.max_wait_for_observation)
        model_updated = False
        lidar_updated = False
        
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0.0, remaining)))
            
            # Check if model state has been updated via Gazebo topic callback
            if not model_updated and self.model_state_seq != initial_model_seq:
                model_updated = True
            
            # Check if LiDAR has been updated
            if not lidar_updated and self.lidar_seq != initial_lidar_seq:
                lidar_updated = True
            
            # Break if both are updated
            if model_updated and lidar_updated:
                break
        
        if not lidar_updated:
            raise RuntimeError("No LiDAR data received.")

        # Compute goal metrics once and cache for reuse in reward/state
        distance_to_goal, angle_to_goal = self._compute_goal_metrics()
        self._cached_model_seq = self.model_state_seq
        self._cached_distance_to_goal = distance_to_goal
        self._cached_angle_to_goal = angle_to_goal

        # Get current state
        done, collision, min_lidar = self._is_collision()
        
        # --- Goal Masquerading: 静默切换目标 ---
        dist_to_current_goal = distance_to_goal
        dist_to_global_goal = np.linalg.norm(self.global_goal_position - self.current_position)
        
        # 检查是否到达全局终点
        target = dist_to_global_goal < GOAL_REACH_THRESHOLD
        
        # 如果正在追踪子目标，检查是否到达子目标：
        # - 若离全局目标足够近 (<1.5m) -> 切回全局终点
        # - 否则 -> 继续预测新的子目标，作为新的 current_goal
        if self.using_subgoal and not target and not collision:
            if dist_to_current_goal < SUBGOAL_SWITCH_THRESHOLD:
                if dist_to_global_goal < SUBGOAL_STOP_GENERATION_DISTANCE:
                    # 接近全局终点，不再生成子目标，静默切回全局终点
                    self._print_and_log(
                        f"[Internal] Close to global goal (dist={dist_to_global_goal:.3f}m < {SUBGOAL_STOP_GENERATION_DISTANCE}), switching to global goal."
                    )
                    self.current_goal = self.global_goal_position.copy()
                    self.using_subgoal = False
                else:
                    # 未接近全局终点：到达一个子目标后，继续预测新的子目标
                    self._print_and_log(
                        f"[Internal] Subgoal reached (dist={dist_to_current_goal:.3f}). Predicting next subgoal..."
                    )
                    next_subgoal = self._predict_subgoal()
                    if next_subgoal is not None:
                        dist_next_to_current = float(np.linalg.norm(next_subgoal - self.current_position))
                        dist_next_to_global = float(np.linalg.norm(next_subgoal - self.global_goal_position))
                        if not self._is_subgoal_valid(next_subgoal):
                            self._print_and_log("[Internal] Next subgoal invalid by is_position_valid; using global goal as fallback.")
                            self.current_goal = self.global_goal_position.copy()
                            self.using_subgoal = False
                        elif dist_next_to_global < SUBGOAL_CLOSE_TO_GOAL_THRESHOLD:
                            self._print_and_log(
                                f"[Internal] Next subgoal close to global goal (dist={dist_next_to_global:.3f}m < {SUBGOAL_CLOSE_TO_GOAL_THRESHOLD}); switching to global goal."
                            )
                            self.current_goal = self.global_goal_position.copy()
                            self.using_subgoal = False
                        elif dist_next_to_current < SUBGOAL_SWITCH_THRESHOLD:
                            self._print_and_log(
                                f"[Internal] Next subgoal too close to current position ({dist_next_to_current:.3f}m); using global goal as fallback."
                            )
                            self.current_goal = self.global_goal_position.copy()
                            self.using_subgoal = False
                        else:
                            self.current_goal = next_subgoal
                            self.using_subgoal = True
                    else:
                        self._print_and_log("[Internal] Subgoal prediction failed; using global goal as fallback.")
                        self.current_goal = self.global_goal_position.copy()
                        self.using_subgoal = False

                # goal_position 始终指向 RL 看到的 current_goal
                self.goal_position = self.current_goal.copy()

                # 更新可视化（子目标 marker 用来显示 current_goal）
                if self.subgoal_marker_spawned:
                    self._move_marker(self.subgoal_marker_name, self.goal_position)

                # 重新计算 metrics（平滑过渡，不产生奖励跳变）
                self.prev_distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
                goal_vector = self.goal_position - self.current_position
                angle_to_goal_global = np.arctan2(goal_vector[1], goal_vector[0])
                self.prev_angle_to_goal = angle_to_goal_global - self.current_yaw
                self.prev_angle_to_goal = (self.prev_angle_to_goal + np.pi) % (2 * np.pi) - np.pi

                # 更新 cached metrics
                distance_to_goal, angle_to_goal = self._compute_goal_metrics()
                self._cached_distance_to_goal = distance_to_goal
                self._cached_angle_to_goal = angle_to_goal

        if target:
            done = True
            self._print_and_log("Goal reached!")
        elif not model_updated:
            done = True
            self._print_and_log("Warning: Model state may not have been updated after action.")

        # 奖励计算：完全基于 current_goal，无子目标特殊奖励
        reward = self._calculate_reward(
            target,
            collision,
            model_updated,
            distance_to_goal=distance_to_goal,
            angle_to_goal=angle_to_goal
        )
        info = {'is_success': False, 'is_collision': False}
        if target:
            info['is_success'] = True
        elif collision:
            info['is_collision'] = True
        return self._get_state(), reward, done, False, info

    def _take_action(self, action):
        """Send velocity command to the robot.

        The RL policy outputs normalized actions in [-1, 1].
        Here we map them to real robot linear/angular velocities.
        """
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # Ensure action is a numpy array and clipped to [-1, 1]
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        norm_linear, norm_angular = action

        # Map normalized linear action [-1, 1] -> [0, MAX_LINEAR_VEL]
        linear = (norm_linear + 1.0) / 2.0 * self.MAX_LINEAR_VEL

        # Map normalized angular action [-1, 1] -> [-MAX_ANGULAR_VEL, MAX_ANGULAR_VEL]
        angular = norm_angular * self.MAX_ANGULAR_VEL

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
            raise RuntimeError("LiDAR data is not available.")
        else:
            # Work on a copy so the original sensor history remains untouched
            lidar_array = np.asarray(self.lidar_data, dtype=np.float32)
            # Clip to max range and normalize to [0, 1]
            # lidar_data = np.clip(lidar_array, 0.0, self.LIDAR_MAX_RANGE) / self.LIDAR_MAX_RANGE
            lidar_data = np.clip(lidar_array, 0.0, 1.0)
            
        
        # Calculate current distance and angle to goal (reuse cached metrics when valid)
        if (
            getattr(self, '_cached_model_seq', None) == self.model_state_seq
            and self._cached_distance_to_goal is not None
            and self._cached_angle_to_goal is not None
        ):
            distance_to_goal = float(self._cached_distance_to_goal)
            angle_to_goal = float(self._cached_angle_to_goal)
        else:
            distance_to_goal, angle_to_goal = self._compute_goal_metrics()
        
        # Calculate changes from previous step
        distance_change = distance_to_goal - self.prev_distance_to_goal
        angle_change = angle_to_goal - self.prev_angle_to_goal
        # Normalize angle change to [-pi, pi]
        angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi
        
        # Normalize robot state components
        norm_distance = np.clip(distance_to_goal / self.MAX_GOAL_DIST, 0.0, 1.0) # Assuming max distance approx 6m
        norm_angle = angle_to_goal / np.pi # [-1, 1]
        norm_linear_vel = self.prev_linear_vel / self.MAX_LINEAR_VEL # [0, 1]
        norm_angular_vel = self.prev_angular_vel / self.MAX_ANGULAR_VEL # [-1, 1] (approx)

        # Robot state: [distance_to_goal, angle_to_goal, distance_change, angle_change, prev_linear_vel, prev_angular_vel]
        robot_state = np.array([
            norm_distance,
            norm_angle,
            distance_change,
            angle_change,
            norm_linear_vel,
            norm_angular_vel
        ], dtype=np.float32)
        
        # Update history for next step
        self.prev_distance_to_goal = distance_to_goal
        self.prev_angle_to_goal = angle_to_goal
        
        # Combine LiDAR data with robot state
        combined_state = np.concatenate([lidar_data, robot_state])
        return combined_state

    def _compute_goal_metrics(self):
        """Compute distance to goal and angle-to-goal in robot frame; angle normalized to [-pi, pi]."""
        distance_to_goal = float(np.linalg.norm(self.goal_position - self.current_position))
        goal_vector = self.goal_position - self.current_position
        angle_to_goal_global = float(np.arctan2(goal_vector[1], goal_vector[0]))
        angle_to_goal = angle_to_goal_global - float(self.current_yaw)
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        return distance_to_goal, float(angle_to_goal)

    def _calculate_reward(self, target, collision, model_updated, distance_to_goal=None, angle_to_goal=None):
        """
        统一的奖励函数 - Goal Masquerading 策略
        RL 智能体只知道它在追踪一个"当前目标"，奖励完全基于距离改进
        没有子目标特殊奖励，保持奖励函数连续平滑
        """
        if target:
            target_reward = 100.0
            self._print_and_log(f"🎯 REWARD: Target reached! reward={target_reward:.3f}")
            return target_reward
        elif collision:
            collision_reward = -100.0
            self._print_and_log(f"💥 REWARD: Collision! reward={collision_reward:.3f}")
            return collision_reward
        elif not model_updated:
            no_update_penalty = -100.0
            self._print_and_log(f"⚠️ REWARD: No model update! reward={no_update_penalty:.3f}")
            return no_update_penalty
        else:
            if distance_to_goal is None or angle_to_goal is None:
                distance_to_goal, angle_to_goal = self._compute_goal_metrics()
            else:
                distance_to_goal = float(distance_to_goal)
                angle_to_goal = float(angle_to_goal)

            distance_improvement = self.prev_distance_to_goal - distance_to_goal

            # Convert angle to a [0, 1] alignment factor.
            # 0 rad -> 1.0 (fully aligned), pi rad -> 0.0 (opposite direction)
            heading_scale = float(np.clip((np.cos(angle_to_goal) + 1.0) / 2.0, 0.0, 1.0))

            # === 距离改进奖励/惩罚 ===
            if distance_improvement > 0:
                # 前进时：对齐越好，奖励越大
                distance_reward = distance_improvement * heading_scale
                if hasattr(self, 'last_progress_time'):
                    self.last_progress_time = time.time()
            else:
                # 后退时：固定为 1，不根据朝向缩放惩罚
                distance_reward = distance_improvement

            # === 步数惩罚 ===
            step_penalty = 0.25

            # === 计算总奖励 ===
            total_reward = 100 * distance_reward - step_penalty

            return total_reward
    
    def _is_collision(self):
        """Check if a collision has occurred based on LiDAR data."""
        collision_threshold = 0.25
        if self.lidar_data is None:
            min_lidar = float('inf')
        elif self.min_lidar is not None:
            min_lidar = float(self.min_lidar)
        else:
            min_lidar = float(np.min(self.lidar_data))
        collision = min_lidar < collision_threshold
        if collision:
            self._print_and_log(f"Collision detected! min_lidar={min_lidar:.4f}")
        return collision, collision, min_lidar

    def _wait_for_new_state(self, num_updates=1):
        """
        Spin until new LiDAR scans are received or timeout.
        Return True if target number of updates are received, False otherwise.
        """
        start_time = time.time()
        target_seq = self.lidar_seq + num_updates
        while (self.lidar_seq < target_seq) and (time.time() - start_time < self.max_wait_for_observation):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.lidar_seq < target_seq:
            self._print_and_log(f"LiDAR data did not update {num_updates} times in time.")
            return False
        else:
            return True

    def _reset_robot_position(self):
        """
        Reset the robot's position using Gazebo transport.
        Returns True if successful, False otherwise.
        """
        pose_msg = GzPose()
        pose_msg.name = "turtlebot4"
        pose_msg.position.x = float(self.start_position[0])
        pose_msg.position.y = float(self.start_position[1])
        pose_msg.position.z = 0.0

        # Random initial yaw for better generalization
        yaw = np.random.uniform(-math.pi, math.pi)
        # yaw = -math.pi / 2
        self.last_reset_yaw = float(yaw)
        pose_msg.orientation.w = math.cos(yaw / 2.0)
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = math.sin(yaw / 2.0)

        service_name = "/world/maze/set_pose"
        timeout_ms = 1000

        try:
            result, response = self.gz_node.request(service_name, pose_msg, GzPose, Boolean, timeout_ms)
            if result and response and response.data:
                self._print_and_log("Robot position reset successfully")
                return True
            else:
                self._print_and_log(f"Position reset failed: result={result}, response.data={response.data if response else 'None'}")
                return False
        except Exception as e:
            self._print_and_log(f"Service call failed: {e}")
            return False

    def _wait_for_model_state(self, num_updates=1):
        """Wait for initial model state from Gazebo topic."""
        self.model_state_received = False
        target_seq = self.model_state_seq + num_updates
        
        # self._print_and_log("Waiting for initial model state from Gazebo topic...")

        start_time = time.time()
        timeout = 5.0  # seconds

        while (self.model_state_seq < target_seq) and (time.time() - start_time) < timeout:
            # Allow Gazebo transport to process messages
            time.sleep(0.01)
            
        if self.model_state_seq < target_seq:
            self._print_and_log(f"Warning: Model state reception timed out (received {self.model_state_seq - (target_seq - num_updates)}/{num_updates}). Using fallback pose for reset.")
            if self.current_position is None:
                self.current_position = self.start_position.copy()
            if not np.isfinite(self.current_yaw):
                self.current_yaw = 0.0
            # If we just requested a reset with random yaw, use that as a reasonable fallback
            self.current_yaw = float(getattr(self, 'last_reset_yaw', 0.0))

    def _print_and_log(self, message):
        self.node.get_logger().info(message)

    def _load_subgoal_model(self, model_path):
        if not os.path.exists(model_path):
            self._print_and_log(f"Subgoal model not found at {model_path}")
            return

        try:
            self._print_and_log(f"Loading subgoal model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            model_params = checkpoint['model_params']
            self.subgoal_preprocessing = checkpoint['preprocessing']
            self.subgoal_label_stats = checkpoint['regression_label_stats']
            
            n_num_features = model_params['n_num_features']
            n_outputs = model_params['n_outputs']
            cat_cardinalities = model_params['cat_cardinalities']
            bins = model_params.get('bins')
            
            if bins is None:
                self._print_and_log("Warning: No bins found in checkpoint, using dummy bins.")
                # Create dummy bins for initialization (will be overwritten by load_state_dict)
                bins = [torch.tensor(np.linspace(0, 1, 129), dtype=torch.float32) for _ in range(n_num_features)]
            
            num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins=bins,
                d_embedding=32,  # 与训练时保持一致
                activation=False,
                version='B',
            )
            
            self.subgoal_model = tabm.TabM.make(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                d_out=n_outputs,
                num_embeddings=num_embeddings,
                n_blocks=3,
                d_block=640,
                dropout=0.0,
                k=8,
            ).to(self.device)
            
            self.subgoal_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.subgoal_model.eval()
            self._print_and_log("Subgoal model loaded successfully.")
            
        except Exception as e:
            self._print_and_log(f"Failed to load subgoal model: {e}")
            self.subgoal_model = None

    def _predict_subgoal(self):
        if self.subgoal_model is None:
            return None
            
        if self.lidar_data_64 is None:
            self._print_and_log("No 64-beam LiDAR data available for prediction.")
            return None

        try:
            current_x, current_y = self.current_position
            goal_x, goal_y = self.global_goal_position
            yaw = self.current_yaw
            lidar = self.lidar_data_64

            # Concatenate: [start_x, start_y, goal_x, goal_y, yaw, lidar...]
            input_features = np.concatenate([
                np.array([current_x, current_y, goal_x, goal_y, yaw], dtype=np.float32),
                lidar
            ])
            
            input_features = input_features.reshape(1, -1)
            input_features = self.subgoal_preprocessing.transform(input_features)
            input_features = np.nan_to_num(input_features, nan=0.0)
            
            input_tensor = torch.as_tensor(input_features, device=self.device).float()
            
            with torch.no_grad():
                output = self.subgoal_model(input_tensor, None)
                # Mean over ensemble dimension (dim 1)
                output = output.mean(dim=1) 
                prediction = output.cpu().numpy()[0]
                
            if self.subgoal_label_stats:
                mean = self.subgoal_label_stats.mean
                std = self.subgoal_label_stats.std
                prediction = prediction * std + mean
            
            self._print_and_log(f"🤖 Neural Network Predicted Subgoal: x={prediction[0]:.4f}, y={prediction[1]:.4f}")
            
            # 如果子目标点距离当前位置过近，认为该预测无效（由上层决定回退策略）
            dist_to_current = float(np.linalg.norm(prediction - self.current_position))
            if dist_to_current < 0.3:
                self._print_and_log(
                    f"⚡ Subgoal too close to current position ({dist_to_current:.4f} < 0.3); ignoring this prediction."
                )
                return None
                
            return prediction.astype(np.float32)
            
        except Exception as e:
            self._print_and_log(f"Error during subgoal prediction: {e}")
            return None

    # ============================================================
    # === Gazebo Visualization Helper Methods ===
    # ============================================================

    def _update_marker_visuals(self):
        """Spawns or moves visual markers for start and goal in Gazebo."""
        if not self.markers_initialized:
            # First run: Spawn models
            # Green for start
            self._spawn_marker(self.start_marker_name, self.start_position, color="0 1 0 1") 
            # Red for goal
            self._spawn_marker(self.goal_marker_name, self.goal_position, color="1 0 0 1")   
            self.markers_initialized = True
        else:
            # Subsequent runs: Move models (faster than respawning)
            self._move_marker(self.start_marker_name, self.start_position)
            self._move_marker(self.goal_marker_name, self.goal_position)

    def _spawn_marker(self, name, position, color="1 0 0 1"):
        """Spawns a static visual-only cylinder using EntityFactory."""
        # SDF for a flat cylinder (marker), static, no collision
        sdf_string = f"""
        <?xml version="1.0" ?>
        <sdf version="1.6">
            <model name="{name}">
                <static>true</static>
                <link name="link">
                    <visual name="visual">
                        <geometry>
                            <cylinder>
                                <radius>0.2</radius>
                                <length>0.01</length>
                            </cylinder>
                        </geometry>
                        <material>
                            <ambient>{color}</ambient>
                            <diffuse>{color}</diffuse>
                            <specular>0 0 0 1</specular>
                        </material>
                    </visual>
                </link>
            </model>
        </sdf>
        """
        
        req = EntityFactory()
        req.sdf = sdf_string
        req.pose.position.x = float(position[0])
        req.pose.position.y = float(position[1])
        req.pose.position.z = 0.01  # Slightly raised to avoid Z-fighting
        
        # Ensure this service name matches your world name (usually 'maze' based on your code)
        service_name = "/world/maze/create" 
        
        try:
            # Using Boolean as response type, largely just need to trigger the service
            self.gz_node.request(service_name, req, EntityFactory, Boolean, 1000)
        except Exception as e:
            self._print_and_log(f"Failed to spawn marker {name}: {e}")

    def _move_marker(self, name, position):
        """Moves an existing marker to a new position."""
        pose_msg = GzPose()
        pose_msg.name = name
        pose_msg.position.x = float(position[0])
        pose_msg.position.y = float(position[1])
        pose_msg.position.z = 0.01
        pose_msg.orientation.w = 1.0 
        
        service_name = "/world/maze/set_pose"
        
        try:
            self.gz_node.request(service_name, pose_msg, GzPose, Boolean, 300)
        except Exception as e:
            self._print_and_log(f"Failed to move marker {name}: {e}")

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