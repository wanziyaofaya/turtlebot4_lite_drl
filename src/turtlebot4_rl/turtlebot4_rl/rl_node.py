# turtlebot_rl_node.py

import rclpy
from rclpy.node import Node
from turtlebot4_rl.nav_env import TurtleBotNavEnv
from stable_baselines3 import PPO, DQN, SAC
import numpy as np
import argparse
import os
import csv
from datetime import datetime
import torch

class TurtleBotRLNode(Node):
    def __init__(self, algorithm='PPO', timesteps=10000, episodes=10, positions_file=None, model_path=None):
        super().__init__('turtlebot_rl_node')
        
        self.algorithm = algorithm.upper()
        self.timesteps = timesteps
        self.episodes = episodes
        self.positions_file = positions_file
        self.model_path = model_path

        self.model_dir = os.path.join('models', self.algorithm)
        os.makedirs(self.model_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(self.model_dir, f"metrics_{timestamp}.txt")
        with open(self.metrics_file, 'w') as f:
            f.write("Task_Start_X,Task_Start_Y,Task_Goal_X,Task_Goal_Y,Model_Path,Episode,Total_Reward\n")

        # Load positions if file provided
        if self.positions_file:
            self.position_pairs = self._load_positions(self.positions_file)
        else:
            # Default single task
            self.position_pairs = [(np.array([0.0, 0.0], dtype=np.float32), 
                                   np.array([5.0, 5.0], dtype=np.float32))]

        # Initialize environment with first position pair
        start_pos, goal_pos = self.position_pairs[0]
        self.env = TurtleBotNavEnv(start_pos, goal_pos)

        self.model = self._load_algorithm(self.algorithm, self.model_path)

        self.get_logger().info(
            f"Algorithm: {self.algorithm}, Timesteps: {self.timesteps}, Episodes: {self.episodes}, "
            f"Tasks: {len(self.position_pairs)}, Model Path: {self.model_path}"
        )

    def _load_positions(self, file_path):
        """Load start and goal positions from a file."""
        position_pairs = []
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for idx, row in enumerate(reader):
                    try:
                        start_x, start_y, goal_x, goal_y = map(float, row)
                        position_pairs.append((np.array([start_x, start_y], dtype=np.float32),
                                               np.array([goal_x, goal_y], dtype=np.float32)))
                    except ValueError:
                        self.get_logger().warning(f"Skipping line {idx + 1} due to conversion error: {row}")
        except FileNotFoundError:
            self.get_logger().error(f"Positions file {file_path} not found.")
        except Exception as e:
            self.get_logger().error(f"Error reading positions file {file_path}: {e}")
        return position_pairs

    def _load_algorithm(self, algorithm_name, model_path):
        """Load or initialize the RL model based on the specified algorithm."""
        algorithms = {
            'PPO': PPO,
            'DQN': DQN,
            'SAC': SAC,
        }
        if algorithm_name not in algorithms:
            self.get_logger().error(f"Algorithm {algorithm_name} is not supported!")
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        if model_path and os.path.isfile(model_path):
            self.get_logger().info(f"Loading pre-trained model from {model_path}")
            model = algorithms[algorithm_name].load(model_path, env=self.env)
        else:
            if model_path:
                self.get_logger().warning(f"Model path {model_path} not found. Initializing a new model.")
            
            # 为PPO添加更稳定的超参数
            if algorithm_name == 'PPO':
                model = algorithms[algorithm_name](
                    "MlpPolicy", 
                    self.env, 
                    verbose=1,
                    learning_rate=3e-4,  # 降低学习率
                    n_steps=512,  # 减少步数
                    batch_size=64,  # 减少批次大小
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    max_grad_norm=0.5,  # 添加梯度裁剪
                    policy_kwargs=dict(
                        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        activation_fn=torch.nn.Tanh
                    )
                )
            else:
                model = algorithms[algorithm_name]("MlpPolicy", self.env, verbose=1, device='cpu')
        return model

    def train_and_evaluate(self):
        """Train the model on each position pair and evaluate."""
        for idx, (start, goal) in enumerate(self.position_pairs):

            # Reset the environment with new start and goal positions
            obs, _ = self.env.reset(start_position=start, goal_position=goal)

            # Train the model
            self.get_logger().info(f"Training on Task {idx + 1} for {self.timesteps} timesteps.")
            self.model.learn(total_timesteps=self.timesteps)
            model_path = os.path.join(self.model_dir, f"model_task_{idx + 1}.zip")
            self.model.save(model_path)
            self.get_logger().info(f"Model saved to {model_path}.")

            # Evaluate the model
            self.get_logger().info(f"Evaluating on Task {idx + 1} for {self.episodes} episodes.")
            for episode in range(1, self.episodes + 1):
                obs, _ = self.env.reset(start_position=start, goal_position=goal)
                done = False
                total_reward = 0.0
                while not done:
                    action, _states = self.model.predict(obs)
                    obs, reward, done, truncated, info = self.env.step(action)
                    total_reward += reward
                self.get_logger().info(f"Task {idx + 1} - Episode {episode}: Total Reward: {total_reward}")

                # Log the metrics
                with open(self.metrics_file, 'a') as f:
                    f.write(f"{start[0]},{start[1]},{goal[0]},{goal[1]},{model_path},{episode},{total_reward}\n")

    def close(self):
        self.env.close()
        self.get_logger().info("Environment closed.")

def main(args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--algorithm', type=str, default='PPO', help='RL Algorithm to use (PPO, DQN, SAC)')
    arg_parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps to train per task')
    arg_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate per task')
    arg_parser.add_argument('--positions_file', type=str, default=None, help='Path to positions file for curriculum learning')
    arg_parser.add_argument('--model_path', type=str, default=None, help='Path to a pre-trained model zip file to load and build upon')

    parsed = arg_parser.parse_args(args=args)

    rclpy.init(args=args)
    node = TurtleBotRLNode(
        algorithm=parsed.algorithm,
        timesteps=parsed.timesteps,
        episodes=parsed.episodes,
        positions_file=parsed.positions_file,
        model_path=parsed.model_path
    )
    node.train_and_evaluate()
    node.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
