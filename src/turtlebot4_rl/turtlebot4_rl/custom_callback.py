from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class SuccessInfoCallback(BaseCallback):
    def __init__(self, tensorboard_log_dir='tensorboard_logs', verbose=0, cycle_size: int = 10000):
        super(SuccessInfoCallback, self).__init__(verbose)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # 全局累计统计（如需后续扩展）
        self.episode_success_count = 0      # 累计成功 episode 数
        self.episode_game_count = 0         # 累计 episode 数

        # 周期统计
        self.cycle_size = cycle_size        # 一个周期内的 timesteps 数量
        self.cycle_success_count = 0        # 当前周期成功数
        self.cycle_episode_count = 0        # 当前周期 episode 数
        self.current_cycle_index = 0        # 当前周期索引（从 0 开始）
        self.last_cycle_success_rate = 0.0  # 上一个周期的成功率，供新周期初期保持

        self.total_timesteps = 0

    def _on_step(self) -> bool:
        self.total_timesteps += 1

        new_cycle_index = (self.total_timesteps - 1) // self.cycle_size
        if new_cycle_index != self.current_cycle_index:
            # 周期切换：在重置统计前保存上一周期成功率
            if self.cycle_episode_count > 0:
                self.last_cycle_success_rate = self.cycle_success_count / self.cycle_episode_count
            # 更新周期索引并重置计数（但 last_cycle_success_rate 保留供展示）
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

        # 计算当前周期成功率；若当前周期尚无 episode，使用上一周期成功率保持曲线平滑
        if self.cycle_episode_count > 0:
            cycle_success_rate = self.cycle_success_count / self.cycle_episode_count
        else:
            cycle_success_rate = self.last_cycle_success_rate

        # 记录：当前展示值 + 原始当前周期已完成的 raw（便于排查）
        raw_rate = (self.cycle_success_count / self.cycle_episode_count) if self.cycle_episode_count > 0 else 0.0
        self.writer.add_scalar('SuccessInfo/CycleSuccessRate', cycle_success_rate, self.total_timesteps)
        self.writer.add_scalar('SuccessInfo/CycleSuccessRateRaw', raw_rate, self.total_timesteps)
        return True

    def _on_training_end(self) -> None:
        # Close the TensorBoard writer
        self.writer.close()
        print("Training finished. Success rate logged to TensorBoard.")