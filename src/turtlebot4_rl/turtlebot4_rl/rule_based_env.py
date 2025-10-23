import numpy as np
import rclpy
class RuleBasedSubgoalGenerator:
    """
    基于规则的子目标点生成器。
    只使用激光信息（LiDAR）来生成子目标点。
    """
    def __init__(self, lidar_size=640, max_range=12.0):
        self.lidar_size = lidar_size
        self.max_range = max_range

    def get_subgoal(self, lidar_data):
        """
        输入: lidar_data (np.ndarray) - 机器人的激光信息
        输出: subgoal (np.ndarray) - 生成的子目标点坐标 (x, y)
        """
        # 简单规则：选择距离最大的激光束方向作为子目标点
        if lidar_data is None or len(lidar_data) != self.lidar_size:
            # 激光数据无效，返回原点
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # 找到距离最大的激光束
        max_idx = np.argmax(lidar_data)
        max_dist = lidar_data[max_idx]
        angle = (max_idx / self.lidar_size) * 2 * np.pi - np.pi  # 假设激光覆盖360度
        
        # 以机器人为原点，计算子目标点坐标
        x = max_dist * np.cos(angle)
        y = max_dist * np.sin(angle)
        return np.array([x, y], dtype=np.float32)

if __name__ == "__main__":
    from turtlebot4_rl.nav_env import TurtleBotNavEnv

    # 随机生成起点和终点，类似rl_node.py
    def generate_random_positions(min_distance=2.0, map_bounds=None):
        import random
        import numpy as np
        if map_bounds is None:
            map_bounds = {'x_min': -1.5, 'x_max': 1.5, 'y_min': -1.5, 'y_max': 1.5}
        max_attempts = 1000
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
        from turtlebot4_rl.nav_env import TurtleBotNavEnv
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
        subgoal = generator.get_subgoal(lidar_data)
        print(f"起点: {start}, 终点: {goal}, 生成的子目标点: {subgoal}")
