# turtlebot_rl_node.py

import rclpy
from rclpy.node import Node
from turtlebot4_rl.nav_env import TurtleBotNavEnv
from turtlebot4_rl.collision import is_spawn_position_valid
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
from turtlebot4_rl.custom_callback import SuccessRateCallback

class TurtleBotRLNode(Node):
    """Custom callback for logging all episode metrics to Tensorboard."""

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def __init__(self, algorithm='PPO', timesteps=10000, episodes=10, model_path=None, min_distance=2.0):
        super().__init__('turtlebot_rl_node')

        self.algorithm = algorithm.upper()
        self.timesteps = timesteps
        self.episodes = episodes
        self.model_path = model_path
        self.min_distance = min_distance

        # Map boundaries (based on the warehouse map)
        self.map_bounds = {'x_min': -1.5, 'x_max': 1.5, 'y_min': -1.5, 'y_max': 1.5}

        self.model_dir = os.path.join('models', self.algorithm)
        os.makedirs(self.model_dir, exist_ok=True)

        # Setup Tensorboard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log = os.path.join('tensorboard_logs', self.algorithm, timestamp)
        os.makedirs(self.tensorboard_log, exist_ok=True)

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
            if not is_spawn_position_valid(start_x, start_y, bounds=self.map_bounds):
                continue
            goal_x = round(random.uniform(self.map_bounds['x_min'], self.map_bounds['x_max']), 2)
            goal_y = round(random.uniform(self.map_bounds['y_min'], self.map_bounds['y_max']), 2)
            if not is_spawn_position_valid(goal_x, goal_y, bounds=self.map_bounds):
                continue
            distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if distance >= self.min_distance:
                start_pos = np.array([start_x, start_y], dtype=np.float32)
                goal_pos = np.array([goal_x, goal_y], dtype=np.float32)
                return start_pos, goal_pos
                # return [-1.5, 1], [0.5, -1]
        self.get_logger().warning("Could not generate valid random positions, using fallback positions")
        return np.array([0.0, 0.0], dtype=np.float32), np.array([2.0, 2.0], dtype=np.float32)

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
            new_lr = 3e-5 
            self.get_logger().info(f"Overriding learning rate to {new_lr}")
            model.learning_rate = new_lr

            # 立即让optimizer使用新的 lr
            if hasattr(model, 'policy') and hasattr(model.policy, 'optimizer'):
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
        else:
            if model_path:
                self.get_logger().warning(f"Model path {model_path} not found. Initializing a new model.")
            
            # 为PPO添加更稳定的超参数
            if algorithm_name == 'PPO':
                model = algorithms[algorithm_name](
                    "MlpPolicy", 
                    self.env, 
                    verbose=1,
                    device='cpu',
                    tensorboard_log=self.tensorboard_log,
                    learning_rate=1e-4,  
                    n_steps=1024,  
                    batch_size=256,  
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.001,
                    vf_coef=0.5,
                    max_grad_norm=0.5,  # 添加梯度裁剪
                    policy_kwargs=dict(
                        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        activation_fn=torch.nn.ReLU
                    )
                )
            else:
                model = algorithms[algorithm_name]("MlpPolicy", self.env, verbose=1, device='cpu', tensorboard_log=self.tensorboard_log)
        return model

    def train_and_evaluate(self):
        num_games = self.episodes 
        max_steps = self.timesteps 
        self.get_logger().info(f"Training for {num_games} games, each up to {max_steps} timesteps.")

        # 添加自定义回调
        success_rate_callback = SuccessRateCallback(tensorboard_log_dir=self.tensorboard_log, verbose=1)
        
        # 添加训练开始时间戳用于区分不同的训练会话
        training_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for game in range(1, num_games + 1):
            # 生成新的起点和目标点
            start, goal = self._generate_random_positions()
            self.env.reset(start_position=start, goal_position=goal)
            
            # 使用model.learn()进行训练，而不是手动循环
            self.get_logger().info(f"Game {game}: Training for {max_steps} timesteps...")
            self.model.learn(total_timesteps=max_steps, reset_num_timesteps=False, callback=[success_rate_callback])

            # 每1个episode保存一次模型
            if game % 1 == 0:
                model_save_path = os.path.join(
                    self.model_dir, 
                    f"{self.algorithm}_{training_session}_checkpoint_ep{game:03d}_ts{max_steps}.zip"
                )
                self.model.save(model_save_path)
                self.get_logger().info(f"Model checkpoint saved after episode {game}: {model_save_path}")

        # 如果最后的episode不是1的倍数，或者要保存最终模型
        if num_games % 1 != 0:
            final_model_path = os.path.join(
                self.model_dir, 
                f"{self.algorithm}_{training_session}_FINAL_ep{num_games:03d}_ts{max_steps}.zip"
            )
            self.model.save(final_model_path)
            self.get_logger().info(f"Final model saved: {final_model_path}")
        else:
            # 如果最后的episode正好是1的倍数，重命名最后保存的模型为FINAL
            last_checkpoint = os.path.join(
                self.model_dir, 
                f"{self.algorithm}_{training_session}_checkpoint_ep{num_games:03d}_ts{max_steps}.zip"
            )
            final_model_path = os.path.join(
                self.model_dir, 
                f"{self.algorithm}_{training_session}_FINAL_ep{num_games:03d}_ts{max_steps}.zip"
            )
            if os.path.exists(last_checkpoint):
                os.rename(last_checkpoint, final_model_path)
                self.get_logger().info(f"Last checkpoint renamed to final model: {final_model_path}")
            else:
                # 备用方案：直接保存最终模型
                self.model.save(final_model_path)
                self.get_logger().info(f"Final model saved: {final_model_path}")

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

    parsed = arg_parser.parse_args(args=args)

    rclpy.init(args=args)

    try:
        node = TurtleBotRLNode(
            algorithm=parsed.algorithm,
            timesteps=parsed.timesteps,
            episodes=parsed.episodes,
            model_path=parsed.model_path,
            min_distance=parsed.min_distance
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