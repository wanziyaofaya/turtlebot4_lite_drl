import os
import time
import rclpy

def generate_subgoal_dataset(env, model_dir, num_samples=250000, output_file='subgoal_dataset_0.35.txt', min_distance=2):
    """
    生成子目标点数据集，每条数据包括：起点、终点、yaw角度、子目标点、激光信息。
    确保 start、goal、yaw、lidar_data 和 A* 规划都在同一时刻、同一位置采集。
    """
    from turtlebot4_rl.improved_astar import astar
    map_bounds = {'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2}
    dataset_path = os.path.join(model_dir, output_file)

    # 检查文件是否存在
    file_exists = os.path.isfile(dataset_path)

    with open(dataset_path, 'a') as f:  # 使用追加模式
        if not file_exists:
            # 写入64维雷达信息的表头，新增yaw列
            lidar_headers = ",".join([f"lidar_{k}" for k in range(64)])
            f.write(f'start_x,start_y,goal_x,goal_y,yaw,subgoal_x,subgoal_y,{lidar_headers}\n')
        
        for i in range(num_samples):
            env.reset()

            # 检查复位是否成功，失败则跳过
            if not getattr(env, 'reset_success', True):
                print(f"[WARN] Position reset failed, skipping sample {i+1}")
                continue

            # 使用实际位置和姿态，而不是请求的复位位置
            start = env.current_position if env.current_position is not None else env.start_position
            goal = env.goal_position
            yaw = env.current_yaw  # 获取当前yaw角度

            # 直接使用环境类中已经处理好的 64 维雷达数据
            lidar = env.lidar_data_64
            
            # # 等待机器人稳定并获取最新的传感器数据
            # # 多次 spin 确保位置和 LiDAR 数据同步更新
            # stable_wait_iterations = 20
            # for _ in range(stable_wait_iterations):
            #     rclpy.spin_once(env.node, timeout_sec=0.05)
            #     time.sleep(0.02)
            
            # # 记录当前 LiDAR 序列号
            # lidar_seq_before = env.lidar_seq
            # model_seq_before = env.model_state_seq
            
            # # 等待新的一帧数据，确保位置和 LiDAR 是同一时刻的
            # timeout = 0.5
            # start_time = time.time()
            # while time.time() - start_time < timeout:
            #     rclpy.spin_once(env.node, timeout_sec=0.05)
            #     # 等待 LiDAR 和位置都更新
            #     if env.lidar_seq > lidar_seq_before and env.model_state_seq > model_seq_before:
            #         break
            
            # # 使用请求的复位位置（整数/简单小数），而不是实际位置（带误差的四位小数）
            # start = env.start_position.copy()  # 请求的起点位置
            # goal = env.goal_position.copy()    # 请求的终点位置
            # lidar = env.lidar_data_64.copy() if env.lidar_data_64 is not None else None
            
            if lidar is None:
                print(f"[WARN] LiDAR data is None for start={start}, goal={goal}")
                continue

            path = astar(start, goal, resolution=0.01, env=env)
            # print(f"Astar path from {start} to {goal}: {path}")
            
            # # 使用同步后的位置进行 A* 规划
            # path = astar(tuple(start), tuple(goal), resolution=0.01, env=env)
            
            if path is None or len(path) < 2:
                print(f"[WARN] Astar failed or path too short for start={start}, goal={goal}")
                continue
            
            subgoal = path[1]
            lidar_str = ','.join([f"{v:.4f}" for v in lidar])
            f.write(f"{start[0]:.4f},{start[1]:.4f},{goal[0]:.4f},{goal[1]:.4f},{yaw:.4f},{subgoal[0]:.4f},{subgoal[1]:.4f},{lidar_str}\n")
            # 起点终点四舍五入到两位小数（与positions_6000.json一致），子目标保留4位小数
            # f.write(f"{start[0]:.2f},{start[1]:.2f},{goal[0]:.2f},{goal[1]:.2f},{yaw:.4f},{subgoal[0]:.4f},{subgoal[1]:.4f},{lidar_str}\n")
            print(f"Sample {i+1}: start={start}, goal={goal}, yaw={yaw:.4f}, subgoal={subgoal}")
    print(f"Subgoal dataset generated: {dataset_path}")


# 可执行入口
if __name__ == "__main__":
    import argparse
    from turtlebot4_rl.nav_env_hrl import TurtleBotNavEnv

    parser = argparse.ArgumentParser(description="生成TurtleBot子目标点数据集")
    parser.add_argument('--model_dir', type=str, default='models', help='数据集保存目录')
    parser.add_argument('--num_samples', type=int, default=250000, help='生成样本数量')
    parser.add_argument('--output_file', type=str, default='subgoal_dataset_0.35.txt', help='输出文件名')
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
        # map_bounds={'x_min': -3, 'x_max': 3, 'y_min': -3, 'y_max': 3},
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