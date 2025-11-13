# turtlebot_rl_node.py

import rclpy
from rclpy.node import Node
from turtlebot4_rl.nav_env import TurtleBotNavEnv
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np
import argparse
import os
from datetime import datetime
import torch
from turtlebot4_rl.custom_callback import SuccessInfoCallback

class EpisodeCheckpointCallback(BaseCallback):
    """每N个episode保存一次模型的回调"""
    def __init__(self, save_freq_episodes: int, save_path: str, name_prefix: str = "model", verbose: int = 1):
        super().__init__(verbose=verbose)
        self.save_freq_episodes = save_freq_episodes
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.episode_count = 0
    def _init_callback(self) -> None:
        # 创建保存目录
        os.makedirs(self.save_path, exist_ok=True)
    def _on_step(self) -> bool:
        # 检测episode结束
        dones = self.locals.get("dones")
        if dones is not None:
            # 处理向量化和非向量化环境
            if isinstance(dones, (list, tuple, np.ndarray)):
                num_done = int(np.sum(dones))
            else:
                num_done = 1 if bool(dones) else 0
            if num_done > 0:
                self.episode_count += num_done
                # 每N个episode保存一次模型
                if self.episode_count % self.save_freq_episodes == 0:
                    model_path = os.path.join(
                        self.save_path,
                        f"{self.name_prefix}_ep{self.episode_count}_steps{self.num_timesteps}.zip"
                    )
                    self.model.save(model_path)
                    if self.verbose:
                        print(f"[Episode Checkpoint] Saved model: {model_path}")                      
        return True

class EntCoefScheduler(BaseCallback):
    def __init__(self, start_value: float = 0.015, end_value: float = 0.007, 
                 step_interval: int = 50000, max_steps: int = 500000):
        super().__init__(verbose=0)
        self.start_value = start_value
        self.end_value = end_value
        self.step_interval = step_interval  # 每5万步衰减一次
        self.max_steps = max_steps         # 50万步后停止衰减
        self.decay_rate = 0.9              # 每次衰减10%

    def _on_step(self) -> bool:
        # 如果超过最大步数，使用最终值
        if self.num_timesteps >= self.max_steps:
            current_ent_coef = self.end_value
        else:
            current_stage = min(self.num_timesteps // self.step_interval, 
                              self.max_steps // self.step_interval)
            # 计算当前的熵系数：每阶段降低10%
            current_ent_coef = max(
                self.end_value,  # 不低于最终值
                self.start_value * (self.decay_rate ** current_stage)
            )
        # 更新模型的熵系数
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = current_ent_coef
        self.logger.record("train/ent_coef", current_ent_coef)
        return True

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

        # Initialize environment (will auto-generate random positions on each reset)
        self.env = TurtleBotNavEnv(
            map_bounds=self.map_bounds,
            min_distance=self.min_distance
        )

        self.model = self._load_algorithm(self.algorithm, self.model_path)

        self.get_logger().info(
            f"Algorithm: {self.algorithm}, Timesteps: {self.timesteps}, Episodes: {self.episodes}, Model Path: {self.model_path}"
        )
        self.get_logger().info(f"Tensorboard logs will be saved to: {os.path.abspath(self.tensorboard_log)}")
        self.get_logger().info("To view training progress, run: tensorboard --logdir tensorboard_logs")

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
                    device='cpu',
                    tensorboard_log=self.tensorboard_log,
                    learning_rate=1e-4,  
                    n_steps=2048,  
                    batch_size=128, 
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=None,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(
                        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                        activation_fn=torch.nn.ReLU,
                        ortho_init=True,  # 使用正交初始化，提高训练稳定性
                    ),
                    normalize_advantage=True,  # 归一化优势函数，提高训练稳定性
                    target_kl=0.01,  # 限制策略更新幅度，提高稳定性
                )
            elif algorithm_name == 'SAC':
                action_dim = float(np.prod(self.env.action_space.shape)) if hasattr(self.env.action_space, "shape") else 1.0
                model = algorithms[algorithm_name](
                    "MlpPolicy",
                    self.env,
                    verbose=1,
                    device='cpu',
                    tensorboard_log=self.tensorboard_log,
                    learning_rate=3e-4,
                    buffer_size=100_000,
                    batch_size=256,
                    gamma=0.99,
                    tau=0.005,
                    train_freq=1,
                    gradient_steps=1,
                    learning_starts=5000,
                    ent_coef='auto',
                    target_entropy=-action_dim,
                    policy_kwargs=dict(
                        net_arch=[256, 256],
                        activation_fn=torch.nn.ReLU
                    )
                )
            else:
                model = algorithms[algorithm_name]("MlpPolicy", self.env, verbose=1, device='cpu', tensorboard_log=self.tensorboard_log)
        return model

    def train_and_evaluate(self):
        # 计算总训练步数
        total_timesteps = self.episodes * self.timesteps
        
        self.get_logger().info(f"Starting training with {total_timesteps:,} total timesteps")
        self.get_logger().info(f"Environment will auto-generate random start/goal positions on each reset")

        callbacks = [SuccessInfoCallback(tensorboard_log_dir=self.tensorboard_log, verbose=1)]
        
        if self.algorithm == 'PPO':
            ent_scheduler = EntCoefScheduler(
                start_value=0.015,      # 起始值
                end_value=0.007,       # 最终值
                step_interval=50000,   # 每5万步衰减一次
                max_steps=500000       # 在50万步时达到最终值
            )
            # callbacks.append(ent_scheduler)
        
        # 添加训练开始时间戳
        training_session = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 添加每30个episode保存一次模型的回调
        checkpoint_dir = os.path.join(self.model_dir, "checkpoints", training_session)
        episode_checkpoint = EpisodeCheckpointCallback(
            save_freq_episodes=30,
            save_path=checkpoint_dir,
            name_prefix=f"{self.algorithm}_{training_session}",
            verbose=1
        )
        callbacks.append(episode_checkpoint)
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps, 
                reset_num_timesteps=False, 
                callback=callbacks
            )
            self.get_logger().info("Training completed!")
        except KeyboardInterrupt:
            self.get_logger().info("Training interrupted by user")
        
        # 保存最终模型
        final_model_path = os.path.join(
            self.model_dir, 
            f"{self.algorithm}_{training_session}_FINAL_{total_timesteps}steps.zip"
        )
        self.model.save(final_model_path)
        self.get_logger().info(f"Model saved: {final_model_path}")

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
    arg_parser.add_argument('--timesteps', type=int, default=10000, help='Base timesteps per unit (total = timesteps × episodes)')
    arg_parser.add_argument('--episodes', type=int, default=10, help='Multiplier for total timesteps (total = timesteps × episodes)')
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