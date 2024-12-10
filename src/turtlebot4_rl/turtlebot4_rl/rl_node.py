# turtlebot_rl_node.py

import rclpy
from rclpy.node import Node
from turtlebot4_rl.nav_env import TurtleBotNavEnv
from stable_baselines3 import PPO, DQN, SAC
import numpy as np
import argparse
import os
import sys
import csv
from datetime import datetime

class TurtleBotRLNode(Node):
    def __init__(self, start_x=0.0, start_y=0.0, goal_x=10.0, goal_y=10.0, algorithm='DQN', timesteps=10000, episodes=10, positions_file=None):
        super().__init__('turtlebot_rl_node')

        self.declare_parameter('start_position', [start_x, start_y])
        self.declare_parameter('goal_position', [goal_x, goal_y])
        self.declare_parameter('algorithm', algorithm)
        self.declare_parameter('timesteps', timesteps)
        self.declare_parameter('episodes', episodes)
        self.declare_parameter('positions_file', positions_file if positions_file else "")

        self.start_position = np.array(self.get_parameter('start_position').value)
        self.goal_position = np.array(self.get_parameter('goal_position').value)
        self.algorithm = self.get_parameter('algorithm').value.upper()
        self.timesteps = self.get_parameter('timesteps').value
        self.episodes = self.get_parameter('episodes').value
        self.positions_file = self.get_parameter('positions_file').value

        # Load position pairs for curriculum learning
        self.position_pairs = []
        if self.positions_file:
            self.position_pairs = self._load_positions(self.positions_file)
            if not self.position_pairs:
                sys.exit(1)
        else:
            # Single start and goal position
            self.position_pairs = [(self.start_position, self.goal_position)]

        self.model_dir = os.path.join('models', self.algorithm)
        os.makedirs(self.model_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(self.model_dir, f"metrics_{timestamp}.txt")
        with open(self.metrics_file, 'w') as f:
            f.write("Task_Start_X,Task_Start_Y,Task_Goal_X,Task_Goal_Y,Model_Path,Episode,Total_Reward\n")

        initial_start, initial_goal = self.position_pairs[0]
        self.env = TurtleBotNavEnv(initial_start, initial_goal, self.algorithm == 'DQN')

        self.model = self._load_algorithm(self.algorithm)

        self.get_logger().info(
            f"Start: {initial_start}, Goal: {initial_goal}, Algorithm: {self.algorithm}, Timesteps: {self.timesteps}, Episodes: {self.episodes}"
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

    def train_and_evaluate(self):
        """Train the model on each position pair and evaluate."""
        for idx, (start, goal) in enumerate(self.position_pairs):
            self.get_logger().info(f"Starting Task {idx + 1}: Start={start}, Goal={goal}")

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
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.env.step(action)
                    done = done or truncated
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
    arg_parser.add_argument('--start_x', type=float, default=0.0, help='Starting X position')
    arg_parser.add_argument('--start_y', type=float, default=0.0, help='Starting Y position')
    arg_parser.add_argument('--goal_x', type=float, default=10.0, help='Goal X position')
    arg_parser.add_argument('--goal_y', type=float, default=10.0, help='Goal Y position')
    arg_parser.add_argument('--algorithm', type=str, default='DQN', help='RL Algorithm to use (PPO, DQN, SAC)')
    arg_parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps to train per task')
    arg_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate per task')
    arg_parser.add_argument('--positions_file', type=str, default=None, help='Path to positions file for curriculum learning')

    parsed = arg_parser.parse_args(args=args)

    rclpy.init(args=args)
    try:
        node = TurtleBotRLNode(
            start_x=parsed.start_x,
            start_y=parsed.start_y,
            goal_x=parsed.goal_x,
            goal_y=parsed.goal_y,
            algorithm=parsed.algorithm,
            timesteps=parsed.timesteps,
            episodes=parsed.episodes,
            positions_file=parsed.positions_file
        )
        node.train_and_evaluate()
        node.close()
    except Exception as e:
        node.get_logger().error(f"An error occurred: {e}")
    finally:
        rclpy.shutdown()