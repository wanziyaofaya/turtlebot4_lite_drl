import os
def generate_subgoal_dataset(env, model_dir, num_samples=500000, output_file='subgoal_dataset.txt', min_distance=2):
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
            # 写入64维雷达信息的表头
            lidar_headers = ",".join([f"lidar_{k}" for k in range(64)])
            f.write(f'start_x,start_y,goal_x,goal_y,subgoal_x,subgoal_y,{lidar_headers}\n')
            # f.write(f'start_x,start_y,goal_x,goal_y,yaw,subgoal_x,subgoal_y,{lidar_headers}\n')
        
        for i in range(num_samples):
            env.reset()

            # 使用实际位置和姿态，而不是请求的复位位置
            start = env.current_position if env.current_position is not None else env.start_position
            goal = env.goal_position
            
            # 直接使用环境类中已经处理好的 64 维雷达数据
            lidar = env.lidar_data_64
            if lidar is None:
                print(f"[WARN] LiDAR data is None for start={start}, goal={goal}")
                continue
            
            path = astar(start, goal, resolution=0.01, env=env)
            # print(f"Astar path from {start} to {goal}: {path}")
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
    parser.add_argument('--num_samples', type=int, default=500000, help='生成样本数量')
    parser.add_argument('--output_file', type=str, default='subgoal_dataset.txt', help='输出文件名')
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