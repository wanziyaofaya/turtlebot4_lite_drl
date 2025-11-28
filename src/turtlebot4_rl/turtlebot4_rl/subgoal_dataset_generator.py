# subgoal_dataset_generator.py

import os
import numpy as np
import random
from turtlebot4_rl.collision import is_spawn_position_valid
from datetime import datetime

def generate_random_positions(map_bounds, min_distance=2):
    """生成不在障碍物内且距离足够的随机起点和终点"""
    max_attempts = 3000
    for _ in range(max_attempts):
        start_x = round(random.uniform(map_bounds['x_min'], map_bounds['x_max']), 2)
        start_y = round(random.uniform(map_bounds['y_min'], map_bounds['y_max']), 2)
        if not is_spawn_position_valid(start_x, start_y, bounds=map_bounds):
            continue
        goal_x = round(random.uniform(map_bounds['x_min'], map_bounds['x_max']), 2)
        goal_y = round(random.uniform(map_bounds['y_min'], map_bounds['y_max']), 2)
        if not is_spawn_position_valid(goal_x, goal_y, bounds=map_bounds):
            continue
        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        if distance >= min_distance:
            start_pos = np.array([start_x, start_y], dtype=np.float32)
            goal_pos = np.array([goal_x, goal_y], dtype=np.float32)
            return start_pos, goal_pos
    print("[WARN] Could not generate valid random positions, using fallback positions")
    return np.array([0.0, 0.0], dtype=np.float32), np.array([2.0, 2.0], dtype=np.float32)


def generate_subgoal_dataset(env, model_dir, num_samples=5000, output_file='improved_astar_subgoal_dataset.txt', min_distance=2):
    """
    生成子目标点数据集，每条数据包括：起点、终点、子目标点、激光信息。
    """
    from turtlebot4_rl.improved_astar import astar
    map_bounds = {'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2}
    dataset_path = os.path.join(model_dir, output_file)

    # 检查文件是否存在
    file_exists = os.path.isfile(dataset_path)

    with open(dataset_path, 'a') as f:  # 使用追加模式
        if not file_exists:
            # 如果文件不存在，写入表头
            f.write('start_x,start_y,goal_x,goal_y,subgoal_x,subgoal_y,lidar_0,...,lidar_63\n')
        
        for i in range(num_samples):
            start, goal = generate_random_positions(map_bounds, min_distance)
            env.start_position = start
            env.goal_position = goal
            obs, _ = env.reset()
            # 提取64维激光信息
            lidar = obs[:64]
            if getattr(env, 'lidar_data', None) is None:
                env.lidar_data = lidar
            path = astar(start, goal, resolution=0.01, env=env)
            if path is None or len(path) < 2:
                print(f"[WARN] Astar failed or path too short for start={start}, goal={goal}")
                continue
            subgoal = path[1]
            lidar_str = ','.join([f"{v:.4f}" for v in lidar])
            f.write(f"{start[0]:.4f},{start[1]:.4f},{goal[0]:.4f},{goal[1]:.4f},{subgoal[0]:.4f},{subgoal[1]:.4f},{lidar_str}\n")
            print(f"Sample {i+1}: start={start}, goal={goal}, subgoal={subgoal}")
    print(f"Subgoal dataset generated: {dataset_path}")


# 可执行入口
if __name__ == "__main__":
    import argparse
    from turtlebot4_rl.nav_env import TurtleBotNavEnv

    parser = argparse.ArgumentParser(description="生成TurtleBot子目标点数据集")
    parser.add_argument('--model_dir', type=str, default='models', help='数据集保存目录')
    parser.add_argument('--num_samples', type=int, default=5000, help='生成样本数量')
    parser.add_argument('--output_file', type=str, default='improved_astar_subgoal_dataset.txt', help='输出文件名')
    parser.add_argument('--min_distance', type=float, default=2, help='起点与终点最小距离')
    parser.add_argument('--start_x', type=float, default=0.0, help='起点x坐标')
    parser.add_argument('--start_y', type=float, default=0.0, help='起点y坐标')
    parser.add_argument('--goal_x', type=float, default=5.0, help='终点x坐标')
    parser.add_argument('--goal_y', type=float, default=5.0, help='终点y坐标')
    args = parser.parse_args()

    # 初始化环境（只用默认参数，起点终点后续设置）
    env = TurtleBotNavEnv(
        max_wait_for_observation=50.0,
        map_bounds={'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2},
        min_distance=args.min_distance
    )

    # 生成数据集
    generate_subgoal_dataset(
        env,
        args.model_dir,
        num_samples=args.num_samples,
        output_file=args.output_file,
        min_distance=args.min_distance
    )

# python3 src/turtlebot4_rl/turtlebot4_rl/subgoal_dataset_generator.py