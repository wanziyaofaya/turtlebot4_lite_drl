from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class SuccessRateCallback(BaseCallback):
    def __init__(self, tensorboard_log_dir='tensorboard_logs', verbose=0):
        super(SuccessRateCallback, self).__init__(verbose)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.episode_success_count = 0
        self.episode_collision_count = 0  # 碰撞计数器
        self.episode_game_count = 0
        self.total_timesteps = 0
        self.total_successes = 0
        self.total_collisions = 0  # 累积碰撞次数 

    def _on_step(self) -> bool:
        # Increment total timesteps
        self.total_timesteps += 1

        # Check if the robot successfully reached the goal
        if self.locals['infos'][0].get('is_success', False):
            self.episode_success_count += 1

        # Check if the robot collided with obstacles
        if self.locals['infos'][0].get('is_collision', False):
            self.episode_collision_count += 1

        # Check if the environment is resetting (due to success, collision, or timeout)
        if self.locals['dones'][0]:
            self.episode_game_count += 1
            
            # 记录成功次数
            if self.locals['infos'][0].get('is_success', False):
                self.total_successes += 1
            
            # 记录碰撞次数
            if self.locals['infos'][0].get('is_collision', False):
                self.total_collisions += 1

        # Calculate success rate
        success_rate = (self.episode_success_count / self.episode_game_count) if self.episode_game_count > 0 else 0.0
        
        # Calculate collision rate
        collision_rate = (self.episode_collision_count / self.episode_game_count) if self.episode_game_count > 0 else 0.0

        # Log success rate to TensorBoard at every step
        # self.writer.add_scalar('SuccessRate/Timesteps', success_rate, self.total_timesteps)
        
        # Log collision rate to TensorBoard at every step
        # self.writer.add_scalar('CollisionRate/Timesteps', collision_rate, self.total_timesteps)
        
        # 每个episode结束时记录累积成功次数和碰撞次数
        if self.locals['dones'][0]:
            self.writer.add_scalar('Episode/TotalSuccesses', self.total_successes, self.total_timesteps)
            self.writer.add_scalar('Episode/TotalCollisions', self.total_collisions, self.total_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Close the TensorBoard writer
        self.writer.close()
        print("Training finished. Success rate logged to TensorBoard.")