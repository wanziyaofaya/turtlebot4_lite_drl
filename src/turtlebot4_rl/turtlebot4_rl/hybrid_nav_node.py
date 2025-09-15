import rclpy
from rclpy.node import Node
import numpy as np
from turtlebot4_rl.nav_env import TurtleBotNavEnv
from stable_baselines3 import DQN
from astar import astar, remove_redundant_nodes

class HybridNavNode(Node):
    def __init__(self, start, goal, model_path=None, threshold=0.2):
        super().__init__('hybrid_nav_node')
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.threshold = threshold
        self.env = TurtleBotNavEnv(self.start, self.goal, is_discrete=True)
        if model_path:
            self.model = DQN.load(model_path, env=self.env)
            self.get_logger().info(f"Loaded DQN model from {model_path}")
        else:
            self.model = DQN('MlpPolicy', self.env, verbose=1)
            self.get_logger().info("Initialized new DQN model.")

    def run(self):
        # 1. 用A*规划全局路径
        path = astar(tuple(self.start), tuple(self.goal))
        if not path:
            self.get_logger().error("A*未找到可行路径！")
            return
        simple_path = remove_redundant_nodes(path)
        self.get_logger().info(f"A*规划路径点数: {len(simple_path)}")

        # 2. 依次将simple_path的点作为DQN的子目标点
        for idx, sub_goal in enumerate(simple_path):
            self.get_logger().info(f"导航到子目标点 {idx+1}/{len(simple_path)}: {sub_goal}")
            obs, _ = self.env.reset(start_position=self.env.current_position, goal_position=sub_goal)
            done = False
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                # 判断是否到达子目标点
                if np.linalg.norm(self.env.current_position - np.array(sub_goal)) < self.threshold:
                    self.get_logger().info(f"已到达子目标点 {idx+1}")
                    break
                if done or truncated:
                    self.get_logger().warning("本回合终止，未到达子目标点。重置环境。")
                    break
        self.get_logger().info("已到达全局终点！")
        self.env.close()


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_x', type=float, default=-9.0)
    parser.add_argument('--start_y', type=float, default=3.0)
    parser.add_argument('--goal_x', type=float, default=-2.0)
    parser.add_argument('--goal_y', type=float, default=-1.0)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args(args)

    rclpy.init(args=args)
    node = HybridNavNode(
        start=[args.start_x, args.start_y],
        goal=[args.goal_x, args.goal_y],
        model_path=args.model_path,
        threshold=args.threshold
    )
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
