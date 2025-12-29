import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gz.transport14 import Node as GzNode
from gz.msgs11.pose_pb2 import Pose as GzPose
from gz.msgs11.boolean_pb2 import Boolean
from gz.msgs11.entity_factory_pb2 import EntityFactory
import time
import math
import tf_transformations
import os

# Constants
GOAL_REACH_THRESHOLD = 0.1  # 目标到达阈值（米）


class TurtleBotNavEnv(gym.Env):
    """Navigation environment using odometry for pose and LiDAR for perception."""

    def __init__(self, max_wait_for_observation=5.0, map_bounds=None, min_distance=2.0, positions_file=None):
        super().__init__()

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node('turtlebot_nav_env_odom')

        # 地图边界与起终点配置
        self.map_bounds = map_bounds if map_bounds is not None else {
            'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2
        }
        self.min_distance = min_distance

        # 预定义起终点文件
        self.positions = None
        self.position_index = 0
        if positions_file is None:
            if os.path.exists('positions_6000.json'):
                positions_file = os.path.abspath('positions_6000.json')
            else:
                positions_file = '/home/wanzi/turtlebot4_lite_drl/positions_6000.json'
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

        # 速度与范围配置
        self.MAX_LINEAR_VEL = 3.0
        self.MAX_ANGULAR_VEL = 1.9
        self.LIDAR_MAX_RANGE = 6.0
        self.MAX_GOAL_DIST = 6.0

        # 动作空间（归一化到 [-1, 1]）
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 观测空间：32 维 LiDAR + 6 维机器人状态
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.zeros(32), np.array([0.0, -1.0, -1.0, -1.0, 0.0, -1.0])]),
            high=np.concatenate([np.ones(32), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]),
            dtype=np.float32,
        )

        # Pub/Sub
        self.cmd_vel_pub = self.node.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Gazebo 节点用于重置
        self.gz_node = GzNode()

        # 状态变量
        self.lidar_data = None
        self.lidar_seq = 0
        self.min_lidar = None
        self.odom_seq = 0
        self.current_position = None
        self.current_yaw = 0.0
        self.prev_distance_to_goal = 0.0
        self.prev_angle_to_goal = 0.0
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.done = False
        self.max_wait_for_observation = max_wait_for_observation

        # 起点与终点占位
        self.start_position = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_position = np.array([2.0, 2.0], dtype=np.float32)

        # 缓存的目标指标
        self._cached_model_seq = None
        self._cached_distance_to_goal = None
        self._cached_angle_to_goal = None

        # 重置时的随机航向保存
        self.last_reset_yaw = 0.0

        # 可视化标记
        self.markers_initialized = False
        self.start_marker_name = "start_marker_visual"
        self.goal_marker_name = "goal_marker_visual"

        self._print_and_log("TurtleBotNavEnv (odom) initialized. Call reset() before first use.")

    # ============================================================
    # 回调
    # ============================================================
    def scan_callback(self, msg):
        raw_data = np.asarray(msg.ranges, dtype=np.float32)
        raw_data[np.isinf(raw_data)] = self.LIDAR_MAX_RANGE
        if raw_data.size < 640:
            raw_data = np.pad(raw_data, (0, 640 - raw_data.size), constant_values=self.LIDAR_MAX_RANGE)
        elif raw_data.size > 640:
            raw_data = raw_data[:640]
        processed = raw_data.reshape(32, 20).min(axis=1)
        self.lidar_data = processed
        self.min_lidar = float(processed.min())
        self.lidar_seq += 1

    def odom_callback(self, msg):
        try:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            self.current_position = np.array([pos.x, pos.y], dtype=np.float32)
            _, _, yaw = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            self.current_yaw = float(yaw)
            self.odom_seq += 1
        except Exception:
            pass

    # ============================================================
    # 核心流程
    # ============================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 生成或加载起终点
        if self.positions is not None and len(self.positions) > 0:
            pair = self.positions[self.position_index % len(self.positions)]
            self.position_index += 1
            try:
                self.start_position = np.array(pair['start'], dtype=np.float32)
                self.goal_position = np.array(pair['goal'], dtype=np.float32)
            except Exception as e:
                self._print_and_log(f"positions_6000.json 格式错误，使用随机起终点: {e}")
                self.start_position, self.goal_position = self._generate_random_positions()
        else:
            self.start_position, self.goal_position = self._generate_random_positions()

        self._print_and_log(f"Episode Reset: Start={self.start_position}, Goal={self.goal_position}")

        self._send_stop_command()
        self.done = False

        # 在仿真中标记起点与终点
        self._update_marker_visuals()

        # 重置机器人位姿
        self._reset_robot_position()

        # 等待里程计稳定（刷掉陈旧数据）
        self._wait_for_odom_updates(num_updates=5)

        # 重置缓存与历史
        self.lidar_data = None
        self.min_lidar = None
        self._cached_model_seq = None
        self._cached_distance_to_goal = None
        self._cached_angle_to_goal = None

        if self.current_position is None:
            self.current_position = self.start_position.copy()
        if not np.isfinite(self.current_yaw):
            self.current_yaw = float(getattr(self, 'last_reset_yaw', 0.0))

        # 初始化目标角度与距离历史
        self.prev_distance_to_goal = float(np.linalg.norm(self.goal_position - self.current_position))
        goal_vector = self.goal_position - self.current_position
        angle_to_goal_global = float(np.arctan2(goal_vector[1], goal_vector[0]))
        self.prev_angle_to_goal = (angle_to_goal_global - self.current_yaw + np.pi) % (2 * np.pi) - np.pi

        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)

        # 等待新的 LiDAR 数据
        if not self._wait_for_new_state(num_updates=5):
            raise RuntimeError("No LiDAR data received after reset timeout.")

        return self._get_state(), {}

    def step(self, action):
        # 发送动作
        self._take_action(action)
        self.last_action = action

        initial_odom_seq = self.odom_seq
        initial_lidar_seq = self.lidar_seq
        deadline = time.monotonic() + float(self.max_wait_for_observation)
        odom_updated = False
        lidar_updated = False

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0.0, remaining)))
            if not odom_updated and self.odom_seq != initial_odom_seq:
                odom_updated = True
            if not lidar_updated and self.lidar_seq != initial_lidar_seq:
                lidar_updated = True
            if odom_updated and lidar_updated:
                break

        if not lidar_updated:
            raise RuntimeError("No LiDAR data received.")
        if not odom_updated:
            self._print_and_log("Warning: Odom may not have updated after action.")

        # 缓存目标指标
        distance_to_goal, angle_to_goal = self._compute_goal_metrics()
        self._cached_model_seq = self.odom_seq
        self._cached_distance_to_goal = distance_to_goal
        self._cached_angle_to_goal = angle_to_goal

        # 终止与奖励
        done, collision, min_lidar = self._is_collision()
        target = distance_to_goal < GOAL_REACH_THRESHOLD
        if target:
            done = True
            self._print_and_log("Goal reached!")
        elif not odom_updated:
            done = True

        reward = self._calculate_reward(
            target,
            collision,
            odom_updated,
            distance_to_goal=distance_to_goal,
            angle_to_goal=angle_to_goal,
        )

        info = {'is_success': bool(target), 'is_collision': bool(collision)}
        return self._get_state(), reward, done, False, info

    # ============================================================
    # 动作与观测
    # ============================================================
    def _take_action(self, action):
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        norm_linear, norm_angular = action

        linear = (norm_linear + 1.0) / 2.0 * self.MAX_LINEAR_VEL
        angular = norm_angular * self.MAX_ANGULAR_VEL

        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)

        self.prev_linear_vel = msg.twist.linear.x
        self.prev_angular_vel = msg.twist.angular.z

        self.cmd_vel_pub.publish(msg)

    def _send_stop_command(self):
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.cmd_vel_pub.publish(msg)

    def _get_state(self):
        if self.lidar_data is None:
            raise RuntimeError("LiDAR data is not available.")
        lidar_array = np.asarray(self.lidar_data, dtype=np.float32)
        lidar_data = np.clip(lidar_array, 0.0, 1.0)

        if (
            getattr(self, '_cached_model_seq', None) == self.odom_seq
            and self._cached_distance_to_goal is not None
            and self._cached_angle_to_goal is not None
        ):
            distance_to_goal = float(self._cached_distance_to_goal)
            angle_to_goal = float(self._cached_angle_to_goal)
        else:
            distance_to_goal, angle_to_goal = self._compute_goal_metrics()

        distance_change = distance_to_goal - self.prev_distance_to_goal
        angle_change = angle_to_goal - self.prev_angle_to_goal
        angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi

        norm_distance = np.clip(distance_to_goal / self.MAX_GOAL_DIST, 0.0, 1.0)
        norm_angle = angle_to_goal / np.pi
        norm_linear_vel = self.prev_linear_vel / self.MAX_LINEAR_VEL
        norm_angular_vel = self.prev_angular_vel / self.MAX_ANGULAR_VEL

        robot_state = np.array([
            norm_distance,
            norm_angle,
            distance_change,
            angle_change,
            norm_linear_vel,
            norm_angular_vel,
        ], dtype=np.float32)

        self.prev_distance_to_goal = distance_to_goal
        self.prev_angle_to_goal = angle_to_goal

        combined_state = np.concatenate([lidar_data, robot_state])
        return combined_state

    # ============================================================
    # 计算与奖励
    # ============================================================
    def _compute_goal_metrics(self):
        distance_to_goal = float(np.linalg.norm(self.goal_position - self.current_position))
        goal_vector = self.goal_position - self.current_position
        angle_to_goal_global = float(np.arctan2(goal_vector[1], goal_vector[0]))
        angle_to_goal = angle_to_goal_global - float(self.current_yaw)
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        return distance_to_goal, float(angle_to_goal)

    def _calculate_reward(self, target, collision, odom_updated, distance_to_goal=None, angle_to_goal=None):
        if target:
            target_reward = 100.0
            self._print_and_log(f"🎯 REWARD: Target reached! reward={target_reward:.3f}")
            return target_reward
        elif collision:
            collision_reward = -100.0
            self._print_and_log(f"💥 REWARD: Collision! reward={collision_reward:.3f}")
            return collision_reward
        elif not odom_updated:
            no_update_penalty = -100.0
            self._print_and_log(f"⚠️ REWARD: No odom update! reward={no_update_penalty:.3f}")
            return no_update_penalty
        else:
            if distance_to_goal is None or angle_to_goal is None:
                distance_to_goal, angle_to_goal = self._compute_goal_metrics()
            distance_improvement = self.prev_distance_to_goal - distance_to_goal
            heading_scale = float(np.clip((np.cos(angle_to_goal) + 1.0) / 2.0, 0.0, 1.0))
            if distance_improvement > 0:
                distance_reward = distance_improvement * heading_scale
                if hasattr(self, 'last_progress_time'):
                    self.last_progress_time = time.time()
            else:
                distance_reward = distance_improvement
            step_penalty = 0.5
            total_reward = 100 * distance_reward - step_penalty
            return total_reward

    def _is_collision(self):
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

    # ============================================================
    # 等待与重置
    # ============================================================
    def _wait_for_new_state(self, num_updates=1):
        start_time = time.time()
        target_seq = self.lidar_seq + num_updates
        while (self.lidar_seq < target_seq) and (time.time() - start_time < self.max_wait_for_observation):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.lidar_seq < target_seq:
            self._print_and_log(f"LiDAR data did not update {num_updates} times in time.")
            return False
        return True

    def _wait_for_odom_updates(self, num_updates=1):
        start_time = time.time()
        target_seq = self.odom_seq + num_updates
        while (self.odom_seq < target_seq) and (time.time() - start_time) < self.max_wait_for_observation:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.odom_seq < target_seq:
            self._print_and_log(f"Warning: Odom reception timed out (received {self.odom_seq - (target_seq - num_updates)}/{num_updates}). Using fallback pose.")
            if self.current_position is None:
                self.current_position = self.start_position.copy()
            if not np.isfinite(self.current_yaw):
                self.current_yaw = 0.0
            self.current_yaw = float(getattr(self, 'last_reset_yaw', 0.0))

    def _reset_robot_position(self):
        pose_msg = GzPose()
        pose_msg.name = "turtlebot4"
        pose_msg.position.x = float(self.start_position[0])
        pose_msg.position.y = float(self.start_position[1])
        pose_msg.position.z = 0.0

        yaw = np.random.uniform(-math.pi, math.pi)
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
            else:
                self._print_and_log(f"Position reset failed: result={result}, response.data={response.data if response else 'None'}")
                self._print_and_log("Continuing with odom tracking...")
        except Exception as e:
            self._print_and_log(f"Service call failed: {e}")
            self._print_and_log("Continuing with odom tracking...")

    # ============================================================
    # 实用工具
    # ============================================================
    def _generate_random_positions(self):
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
            distance = np.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2)
            if distance >= self.min_distance:
                return np.array([start_x, start_y], dtype=np.float32), np.array([goal_x, goal_y], dtype=np.float32)
        self._print_and_log("Warning: Could not generate valid positions, using fallback")
        return np.array([0.0, 0.0], dtype=np.float32), np.array([2.0, 2.0], dtype=np.float32)

    def _print_and_log(self, message):
        self.node.get_logger().info(message)

    # ============================================================
    # 可视化标记
    # ============================================================

    def _update_marker_visuals(self):
        """Spawn or move visual markers for start/goal in Gazebo."""
        if not self.markers_initialized:
            self._spawn_marker(self.start_marker_name, self.start_position, color="0 1 0 1")
            self._spawn_marker(self.goal_marker_name, self.goal_position, color="1 0 0 1")
            self.markers_initialized = True
        else:
            self._move_marker(self.start_marker_name, self.start_position)
            self._move_marker(self.goal_marker_name, self.goal_position)

    def _spawn_marker(self, name, position, color="1 0 0 1"):
        """Spawn a flat cylinder marker via Gazebo entity factory."""
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
        req.pose.position.z = 0.01

        service_name = "/world/maze/create"
        try:
            self.gz_node.request(service_name, req, EntityFactory, Boolean, 1000)
        except Exception as e:
            self._print_and_log(f"Failed to spawn marker {name}: {e}")

    def _move_marker(self, name, position):
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
        try:
            self.node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
