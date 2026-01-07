import numpy as np
import rclpy
from math import *

class RuleBasedSubgoalGenerator:
    def __init__(self, lidar_size=640, max_range=12.0):
        self.lidar_size = lidar_size
        self.max_range = max_range

    def calculate_heuristic_score(x, y, dist_o, dist_g, dist_l1, dist_l2, kernel_size, resolution):
        from turtlebot4_rl.collision import is_position_valid
        d1 = np.tanh(np.exp((dist_o / dist_l1) ** 2) / exp((dist_l2 / dist_l1) ** 2)) * dist_l2
        d2 = dist_g

        point_information = 0
        count = 0
        half_size = kernel_size / 2
        x_min = max(-1.5, x - half_size)
        x_max = min(1.5, x + half_size)
        y_min = max(-1.5, y - half_size)
        y_max = min(1.5, y + half_size)
        xi = x_min
        while xi < x_max:
            yj = y_min
            while yj < y_max:
                if not is_position_valid(xi, yj):
                    point_information += 5
                else:
                    point_information += 1
                count += 1
                yj += resolution
            xi += resolution
        if count == 0:
            I = 0
        else:
            I = min(50, exp(point_information / count))
        return d1 + d2 + I
    
    def get_subgoal(self, lidar_data, odomX, odomY, angle, dist_s, dist_g):
        """
        输入: lidar_data (np.ndarray), 机器人位置和朝向
        输出: subgoal (np.ndarray) - 生成的子目标点坐标 (x, y)
        """
        if lidar_data is None or len(lidar_data) != self.lidar_size:
            return np.array([0.0, 0.0], dtype=np.float32)

        nodes = []
        # --- new_nodes 规则 ---
        for i in range(1, len(lidar_data)):
            if lidar_data[i] < 7.5:
                dist = lidar_data[i]
                angl = i / 4 - 90
                qx = dist * np.cos(np.radians(angl + angle))
                qy = dist * np.sin(np.radians(angl + angle))
                nodes.append([qx + odomX, qy + odomY])
            if abs(lidar_data[i - 1] - lidar_data[i]) > 1.5 and lidar_data[i - 1] < 8.5 and lidar_data[i] < 8.5:
                dist = (lidar_data[i - 1] + lidar_data[i]) / 2
                angl = i / (2 * 2) - 90
                qx = dist * np.cos(np.radians(angl + angle))
                qy = dist * np.sin(np.radians(angl + angle))
                nodes.append([qx + odomX, qy + odomY])

        # --- free_space_nodes 规则 ---
        count5 = 0
        for i in range(1, len(lidar_data)):
            if 4.5 < lidar_data[i] < 9.9:
                count5 += 1
                continue
            if count5 > 35:
                dist = 4
                angl = (i - count5 / 2) / (2 * 2) - 90
                qx = dist * np.cos(np.radians(angl + angle))
                qy = dist * np.sin(np.radians(angl + angle))
                nodes.append([qx + odomX, qy + odomY])
                count5 = 0
            else:
                count5 = 0

        # --- infinite_nodes 规则 ---
        tmp_i = 0
        save_i = 0
        for i in range(1, len(lidar_data)):
            if lidar_data[i] < 6.9:
                if i - tmp_i > 50 and tmp_i > 0:
                    dist = min(lidar_data[save_i], lidar_data[i])
                    angl = (i - (i - save_i) / 2) / (2 * 2) - 90
                    qx = dist * np.cos(np.radians(angl + angle))
                    qy = dist * np.sin(np.radians(angl + angle))
                    nodes.append([qx + odomX, qy + odomY])
                tmp_i = i

        # 计算每个点的启发式分数并选择分数最小的点作为子目标点
        if not nodes:
            return np.array([0.0, 0.0], dtype=np.float32)

        nodes = np.array(nodes)
        heuristic_scores = []

        dist_l1 = 5  # Inner distance limit for local heuristics discount
        dist_l2 = 10  # Outer distance limit for local heuristics discount
        kernel_size = 4 # Kernel size in pixels for map information heuristic calculation
        for node in nodes:
            x, y = node
            score = self.calculate_heuristic_score(x, y, dist_s, dist_g, dist_l1, dist_l2, kernel_size,resolution=0.01)
            heuristic_scores.append(score)

        idx = np.argmin(heuristic_scores)
        return nodes[idx]

if __name__ == "__main__":
    from turtlebot4_rl.nav_env_hrl import TurtleBotNavEnv

    # 随机生成起点和终点，类似rl_node.py
    def generate_random_positions(min_distance=2, map_bounds=None):
        import random
        import numpy as np
        if map_bounds is None:
            map_bounds = {'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2}
            # map_bounds = {'x_min': -3, 'x_max': 3, 'y_min': -3, 'y_max': 3}
        max_attempts = 3000
        for _ in range(max_attempts):
            start_x = round(random.uniform(map_bounds['x_min'], map_bounds['x_max']), 2)
            start_y = round(random.uniform(map_bounds['y_min'], map_bounds['y_max']), 2)
            goal_x = round(random.uniform(map_bounds['x_min'], map_bounds['x_max']), 2)
            goal_y = round(random.uniform(map_bounds['y_min'], map_bounds['y_max']), 2)
            distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if distance >= min_distance:
                start_pos = np.array([start_x, start_y], dtype=np.float32)
                goal_pos = np.array([goal_x, goal_y], dtype=np.float32)
                return start_pos, goal_pos
        # fallback
        return np.array([0.0, 0.0], dtype=np.float32), np.array([2.0, 2.0], dtype=np.float32)

    if __name__ == "__main__":
        from turtlebot4_rl.nav_env_hrl import TurtleBotNavEnv
        # 随机生成起点和终点
        start, goal = generate_random_positions()
        env = TurtleBotNavEnv(start, goal)
        generator = RuleBasedSubgoalGenerator()
        # 等待激光数据更新
        import time
        timeout = 5.0
        start_time = time.time()
        while env.lidar_data is None and (time.time() - start_time < timeout):
            rclpy.spin_once(env.node, timeout_sec=0.05)
        lidar_data = env.lidar_data
        # 获取机器人当前位姿
        odomX = env.current_position[0]
        odomY = env.current_position[1]
        angle = env.current_yaw
        dist_s = np.linalg.norm(env.current_position - start)
        dist_g = np.linalg.norm(env.current_position - goal)
        subgoal = generator.get_subgoal(lidar_data, odomX, odomY, angle, dist_s, dist_g)
        print(f"起点: {start}, 终点: {goal}, 机器人位置: ({odomX:.2f}, {odomY:.2f}), 朝向: {angle:.2f}, 生成的子目标点: {subgoal}")
