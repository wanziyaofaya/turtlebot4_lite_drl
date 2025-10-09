import os
import numpy as np
from stable_baselines3 import PPO, DQN, SAC
from turtlebot4_rl.nav_env import TurtleBotNavEnv
from turtlebot4_rl.collision import is_spawn_position_valid

def test_model(model_path, algorithm='PPO', episodes=10, min_distance=1.0):
    """
    测试训练好的模型。
    :param model_path: 模型文件路径
    :param algorithm: 使用的算法 (PPO, DQN, SAC)
    :param episodes: 测试的回合数
    :param min_distance: 起点和目标点之间的最小距离
    """
    # 检查模型文件是否存在
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")

    # 定义地图边界
    map_bounds = {'x_min': -9.5, 'x_max': 9.5, 'y_min': -9.5, 'y_max': 9.5}

    # 初始化环境
    env = TurtleBotNavEnv(np.array([0.0, 0.0], dtype=np.float32), np.array([5.0, 5.0], dtype=np.float32))

    # 加载模型
    algorithms = {'PPO': PPO, 'DQN': DQN, 'SAC': SAC}
    if algorithm not in algorithms:
        raise ValueError(f"不支持的算法: {algorithm}")

    model = algorithms[algorithm].load(model_path, env=env)

    # 开始测试
    for episode in range(1, episodes + 1):
        # 随机生成起点和目标点
        for _ in range(1000):
            start_x = np.random.uniform(map_bounds['x_min'], map_bounds['x_max'])
            start_y = np.random.uniform(map_bounds['y_min'], map_bounds['y_max'])
            if not is_spawn_position_valid(start_x, start_y, bounds=map_bounds):
                continue

            goal_x = np.random.uniform(map_bounds['x_min'], map_bounds['x_max'])
            goal_y = np.random.uniform(map_bounds['y_min'], map_bounds['y_max'])
            if not is_spawn_position_valid(goal_x, goal_y, bounds=map_bounds):
                continue

            distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if distance >= min_distance:
                start_pos = np.array([start_x, start_y], dtype=np.float32)
                goal_pos = np.array([goal_x, goal_y], dtype=np.float32)
                break
        else:
            print("无法生成有效的起点和目标点，使用默认值。")
            start_pos = np.array([0.0, 0.0], dtype=np.float32)
            goal_pos = np.array([5.0, 5.0], dtype=np.float32)

        obs, _ = env.reset(start_position=start_pos, goal_position=goal_pos)
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"回合 {episode}: 总奖励: {total_reward}")

    env.close()

if __name__ == '__main__':
    # 示例测试代码
    test_model(model_path='models/PPO/model_20250929_161427.zip', algorithm='PPO', episodes=10, min_distance=1.0)
    # python src/turtlebot4_rl/turtlebot4_rl/test_model.py