import rclpy
from rclpy.node import Node
from turtlebot4_rl.nav_env import TurtleBotNavEnv
from stable_baselines3 import PPO, DQN, SAC
import numpy as np
import argparse

class TurtleBotRLNode(Node):
    def __init__(self, start_x=0.0, start_y=0.0, goal_x=10.0, goal_y=10.0, algorithm='DQN'):
        super().__init__('turtlebot_rl_node')

        self.declare_parameter('start_position', [start_x, start_y])
        self.declare_parameter('goal_position', [goal_x, goal_y])
        self.declare_parameter('algorithm', algorithm)

        self.start_position = np.array(self.get_parameter('start_position').value)
        self.goal_position = np.array(self.get_parameter('goal_position').value)
        self.algorithm = self.get_parameter('algorithm').value.upper()

        self.env = TurtleBotNavEnv(self.start_position, self.goal_position, self.algorithm == 'DQN')
        self.env.goal_position = self.goal_position

        self.model = self._load_algorithm(self.algorithm)

        self.get_logger().info(
            f"Start: {self.start_position}, Goal: {self.goal_position}, Algorithm: {self.algorithm}"
        )

    def _load_algorithm(self, algorithm_name):
        # Map algorithm names to RL models
        algorithms = {
            'PPO': PPO,
            'DQN': DQN,
            'SAC': SAC,
        }
        if algorithm_name not in algorithms:
            self.get_logger().error(f"Algorithm {algorithm_name} is not supported!")
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")
        return algorithms[algorithm_name]("MlpPolicy", self.env, verbose=1)

    def train(self, timesteps=1000):
        self.get_logger().info(f"Training {self.algorithm} for {timesteps} timesteps.")
        self.model.learn(total_timesteps=timesteps)
        self.model.save(f"{self.algorithm}_turtlebot_model")
        self.get_logger().info(f"Training completed. Model saved as {self.algorithm}_turtlebot_model.")
    
    def evaluate(self, episodes=5):
        self.get_logger().info(f"Evaluating {self.algorithm} for {episodes} episodes.")
        for episode in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                total_reward += reward
            self.get_logger().info(f"Episode {episode + 1}: Total Reward: {total_reward}")

def main(args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--start_x', type=float, default=0.0)
    arg_parser.add_argument('--start_y', type=float, default=0.0)
    arg_parser.add_argument('--goal_x', type=float, default=10.0)
    arg_parser.add_argument('--goal_y', type=float, default=10.0)
    arg_parser.add_argument('--algorithm', type=str, default='DQN')

    parsed = arg_parser.parse_args(args=args)

    rclpy.init(args=args)
    node = TurtleBotRLNode(parsed.start_x, parsed.start_y, parsed.goal_x, parsed.goal_y, parsed.algorithm)
    node.train()
    node.evaluate()
    rclpy.spin(node)
    rclpy.shutdown()