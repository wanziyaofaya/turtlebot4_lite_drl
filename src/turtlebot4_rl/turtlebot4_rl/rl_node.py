# turtlebot_rl_node.py

import rclpy
from rclpy.node import Node
from turtlebot4_rl.nav_env import TurtleBotNavEnv
from turtlebot4_rl.collision import point_in_obstacle
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np
import argparse
import os
import csv
from datetime import datetime
import torch
import random
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(BaseCallback):
    """Custom callback for logging all episode metrics to Tensorboard."""

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_rewards = 0
        self.total_lengths = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals['dones'][0]:
            # Track episode statistics
            current_episode_reward = sum(self.locals['rewards'])
            current_episode_length = len(self.locals['rewards'])

            self.episode_rewards.append(current_episode_reward)
            self.episode_lengths.append(current_episode_length)

            # Update global statistics
            self.total_rewards += current_episode_reward
            self.total_lengths += current_episode_length
            self.episode_count += 1

            # Calculate and log global averages
            avg_reward_all = self.total_rewards / self.episode_count
            avg_length_all = self.total_lengths / self.episode_count
            self.logger.record('episode/reward_avg_all', avg_reward_all)
            self.logger.record('episode/length_avg_all', avg_length_all)

        return True


    def __init__(self, algorithm='PPO', timesteps=10000, episodes=10, model_path=None, min_distance=2.0, eval_episodes=10):
        super().__init__('turtlebot_rl_node')

        self.algorithm = algorithm.upper()
        self.timesteps = timesteps
        self.episodes = episodes
        self.model_path = model_path
        self.min_distance = min_distance
        self.eval_episodes = eval_episodes

        # Map boundaries (based on the warehouse map)
        self.map_bounds = {'x_min': -9.5, 'x_max': 9.5, 'y_min': -9.5, 'y_max': 9.5}

        self.model_dir = os.path.join('models', self.algorithm)
        os.makedirs(self.model_dir, exist_ok=True)

        # Setup Tensorboard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log = os.path.join('tensorboard_logs', self.algorithm, timestamp)
        os.makedirs(self.tensorboard_log, exist_ok=True)
        self.tb_writer = SummaryWriter(self.tensorboard_log)

        self.metrics_file = os.path.join(self.model_dir, f"metrics_{timestamp}.txt")
        with open(self.metrics_file, 'w') as f:
            f.write("Start_X,Start_Y,Goal_X,Goal_Y,Model_Path,Episode,Total_Reward\n")

        # Initialize environment with random positions
        start_pos, goal_pos = self._generate_random_positions()
        self.env = TurtleBotNavEnv(start_pos, goal_pos)

        self.model = self._load_algorithm(self.algorithm, self.model_path)

        self.get_logger().info(
            f"Algorithm: {self.algorithm}, Timesteps: {self.timesteps}, Episodes: {self.episodes}, Model Path: {self.model_path}"
        )
        self.get_logger().info(f"Tensorboard logs will be saved to: {os.path.abspath(self.tensorboard_log)}")
        self.get_logger().info("To view training progress, run: tensorboard --logdir tensorboard_logs")

    def _generate_random_positions(self):
        """Generate random start and goal positions that are not in obstacles and meet distance requirements. 坐标保留两位小数"""
        max_attempts = 1000
        for _ in range(max_attempts):
            start_x = round(random.uniform(self.map_bounds['x_min'], self.map_bounds['x_max']), 2)
            start_y = round(random.uniform(self.map_bounds['y_min'], self.map_bounds['y_max']), 2)
            if point_in_obstacle(start_x, start_y):
                continue
            goal_x = round(random.uniform(self.map_bounds['x_min'], self.map_bounds['x_max']), 2)
            goal_y = round(random.uniform(self.map_bounds['y_min'], self.map_bounds['y_max']), 2)
            if point_in_obstacle(goal_x, goal_y):
                continue
            distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if distance >= self.min_distance:
                start_pos = np.array([start_x, start_y], dtype=np.float32)
                goal_pos = np.array([goal_x, goal_y], dtype=np.float32)
                return start_pos, goal_pos
        self.get_logger().warning("Could not generate valid random positions, using fallback positions")
        return np.array([0.0, 0.0], dtype=np.float32), np.array([5.0, 5.0], dtype=np.float32)

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
            model = algorithms[algorithm_name].load(model_path, env=self.env, tensorboard_log=self.tensorboard_log)
        else:
            if model_path:
                self.get_logger().warning(f"Model path {model_path} not found. Initializing a new model.")
            
            # 为PPO添加更稳定的超参数
            if algorithm_name == 'PPO':
                model = algorithms[algorithm_name](
                    "MlpPolicy", 
                    self.env, 
                    verbose=1,
                    tensorboard_log=self.tensorboard_log,  # 添加Tensorboard日志
                    learning_rate=3e-4,  # 降低学习率
                    n_steps=2048,  # 减少步数
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
                model = algorithms[algorithm_name]("MlpPolicy", self.env, verbose=1, device='cpu', tensorboard_log=self.tensorboard_log)
        return model


    def train_and_evaluate(self):
        num_games = self.episodes 
        max_steps = self.timesteps 
        self.get_logger().info(f"Training for {num_games} games, each up to {max_steps} timesteps.")

        episode_rewards = []
        episode_steps = []
        train_success_count = 0
        train_collision_count = 0
        train_timeout_count = 0
        for game in range(1, num_games + 1):
            start, goal = self._generate_random_positions()
            obs, _ = self.env.reset(start_position=start, goal_position=goal)
            done = False
            total_reward = 0.0
            step_count = 0
            result = None
            while not done and step_count < max_steps:
                action, _states = self.model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                step_count += 1
                # 判断是否到达目标或碰撞
                if info.get('is_success', False):
                    result = 'success'
                    done = True
                elif info.get('is_collision', False):
                    result = 'collision'
                    done = True
            # 超时：步数耗尽且未碰撞未到达目标
            if result is None:
                result = 'timeout'

            if result == 'success':
                train_success_count += 1
            elif result == 'collision':
                train_collision_count += 1
            elif result == 'timeout':
                train_timeout_count += 1

            # 计算当前比例
            train_success_rate = train_success_count / game
            train_collision_rate = train_collision_count / game
            train_timeout_rate = train_timeout_count / game

            # Tensorboard写入
            self.tb_writer.add_scalar('custom/success_rate', train_success_rate, game)
            self.tb_writer.add_scalar('custom/collision_rate', train_collision_rate, game)
            self.tb_writer.add_scalar('custom/timeout_rate', train_timeout_rate, game)

            self.get_logger().info(f"Game {game}: Total Reward: {total_reward}, Steps: {step_count}, Result: {result}")
            episode_rewards.append(total_reward)
            episode_steps.append(step_count)
            with open(self.metrics_file, 'a') as f:
                f.write(f"{start[0]},{start[1]},{goal[0]},{goal[1]},-,{game},{total_reward},{result}\n")

            # 每100个episode记录一次平均值，并写入Tensorboard（以episode为横坐标）
            if game % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_steps = np.mean(episode_steps[-100:])
                with open(self.metrics_file, 'a') as f:
                    f.write(f"SUMMARY,{game-99}-{game},{avg_steps},{avg_reward}\n")
                self.tb_writer.add_scalar('custom/avg_reward', avg_reward, game)
                self.tb_writer.add_scalar('custom/avg_steps', avg_steps, game)
        self.tb_writer.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"model_{timestamp}.zip")
        self.model.save(model_path)
        self.get_logger().info(f"Model saved to {model_path}.")

        # 评估模型
        self.get_logger().info(f"Evaluating model for {self.eval_episodes} episodes...")
        for episode in range(1, self.eval_episodes + 1):
            start, goal = self._generate_random_positions()
            obs, _ = self.env.reset(start_position=start, goal_position=goal)
            done = False
            total_reward = 0.0
            step_count = 0
            while not done and step_count < max_steps:
                action, _states = self.model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                step_count += 1
            self.get_logger().info(f"Eval Episode {episode}: Total Reward: {total_reward}, Steps: {step_count}")
            with open(self.metrics_file, 'a') as f:
                f.write(f"EVAL,{start[0]},{start[1]},{goal[0]},{goal[1]},-,{episode},{total_reward}\n")

    def close(self):
        self.env.close()
        self.get_logger().info("Environment closed.")
        self.get_logger().info(f"To view Tensorboard logs, run:")
        self.get_logger().info(f"tensorboard --logdir {os.path.abspath('tensorboard_logs')}")
        self.get_logger().info("Then open http://localhost:6006 in your browser")

# Update the main function to remove task-related arguments
def main(args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--algorithm', type=str, default='PPO', help='RL Algorithm to use (PPO, DQN, SAC)')
    arg_parser.add_argument('--timesteps', type=int, default=10000, help='Number of timesteps to train')
    arg_parser.add_argument('--episodes', type=int, default=10, help='Number of training games')
    arg_parser.add_argument('--model_path', type=str, default=None, help='Path to a pre-trained model zip file to load and build upon')
    arg_parser.add_argument('--min_distance', type=float, default=2.0, help='Minimum distance between start and goal positions')
    arg_parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes after training')

    parsed = arg_parser.parse_args(args=args)

    rclpy.init(args=args)

    try:
        node = TurtleBotRLNode(
            algorithm=parsed.algorithm,
            timesteps=parsed.timesteps,
            episodes=parsed.episodes,
            model_path=parsed.model_path,
            min_distance=parsed.min_distance,
            eval_episodes=parsed.eval_episodes
        )
        # 只进行训练和评估
        node.train_and_evaluate()
        node.close()
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
