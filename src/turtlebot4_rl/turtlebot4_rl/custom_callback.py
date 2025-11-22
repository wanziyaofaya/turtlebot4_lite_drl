from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class SuccessInfoCallback(BaseCallback):
    def __init__(self, tensorboard_log_dir='tensorboard_logs', verbose=0, cycle_size: int = 10000):
        super(SuccessInfoCallback, self).__init__(verbose)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        self.episode_success_count = 0      
        self.episode_game_count = 0         

        self.cycle_size = cycle_size         
        self.cycle_success_count = 0         
        self.cycle_episode_count = 0         
        self.current_cycle_index = 0       

        self.total_timesteps = 0

    def _on_step(self) -> bool:
        self.total_timesteps += 1

        new_cycle_index = (self.total_timesteps - 1) // self.cycle_size
        if new_cycle_index != self.current_cycle_index:
            self.current_cycle_index = new_cycle_index
            self.cycle_success_count = 0
            self.cycle_episode_count = 0

        is_success = self.locals['infos'][0].get('is_success', False)
        if is_success:
            self.episode_success_count += 1
        if self.locals['dones'][0]:
            self.episode_game_count += 1
            self.cycle_episode_count += 1
            if is_success:
                self.cycle_success_count += 1

        cycle_success_rate = (self.cycle_success_count / self.cycle_episode_count) if self.cycle_episode_count > 0 else 0.0

        self.writer.add_scalar('SuccessInfo/CycleSuccessRate', cycle_success_rate, self.total_timesteps)
        return True

    def _on_training_end(self) -> None:
        # Close the TensorBoard writer
        self.writer.close()
        print("Training finished. Success rate logged to TensorBoard.")