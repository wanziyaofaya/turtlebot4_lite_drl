from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class SuccessInfoCallback(BaseCallback):
    def __init__(self, tensorboard_log_dir='tensorboard_logs', verbose=0):
        super(SuccessInfoCallback, self).__init__(verbose)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.episode_success_count = 0
        self.episode_game_count = 0
        self.total_timesteps = 0

    def _on_step(self) -> bool:
        # Increment total timesteps
        self.total_timesteps += 1

        # Check if the robot successfully reached the goal
        if self.locals['infos'][0].get('is_success', False):
            self.episode_success_count += 1

        # Check if the environment is resetting (due to success, collision, or timeout)
        if self.locals['dones'][0]:
            self.episode_game_count += 1

        # Calculate success rate
        success_rate = (self.episode_success_count / self.episode_game_count) if self.episode_game_count > 0 else 0.0

        # Log success rate to TensorBoard at every step
        self.writer.add_scalar('SuccessInfo/SuccessRate', success_rate, self.total_timesteps)
        self.writer.add_scalar('SuccessInfo/SuccessCount', self.episode_success_count, self.total_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Close the TensorBoard writer
        self.writer.close()
        print("Training finished. Success rate logged to TensorBoard.")