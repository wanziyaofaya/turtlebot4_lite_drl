from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class SuccessInfoCallback(BaseCallback):

    def __init__(self, tensorboard_log_dir='tensorboard_logs', verbose=0, window_size: int = 50):
        super(SuccessInfoCallback, self).__init__(verbose)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # 累积统计
        self.episode_success_count = 0      # 累计成功 episode 数
        self.episode_game_count = 0         # 累计 episode 数

        # 窗口统计（每 window_size 个 episode 输出一次）
        self.window_size = window_size
        self.window_success_count = 0
        self.window_episode_count = 0

    def _on_step(self) -> bool:
        # 本 step 的第一个环境对应的 info 中的成功标记
        is_success = self.locals['infos'][0].get('is_success', False)

        # 如果当前 step 标记 episode 结束，则更新统计
        if self.locals['dones'][0]:
            self.episode_game_count += 1
            if is_success:
                self.episode_success_count += 1
                self.window_success_count += 1
            self.window_episode_count += 1

            # 满窗口：记录成功率并重置窗口计数
            if self.window_episode_count == self.window_size:
                window_rate = self.window_success_count / self.window_episode_count
                # 以总已完成 episode 作为 x 轴（更直观）
                self.writer.add_scalar('SuccessInfo/SuccessRate', window_rate, self.episode_game_count)
                self.writer.add_scalar('SuccessInfo/SuccessCount', self.window_success_count, self.episode_game_count)
                # 重置窗口
                self.window_success_count = 0
                self.window_episode_count = 0

        return True

    def _on_training_end(self) -> None:
        # 写入最终不满窗口的残余统计（如有）
        if 0 < self.window_episode_count < self.window_size:
            final_rate = self.window_success_count / self.window_episode_count if self.window_episode_count > 0 else 0.0
            # 使用总 episode 作为步数
            self.writer.add_scalar('SuccessInfo/FinalPartialWindowRate', final_rate, self.episode_game_count)
            self.writer.add_scalar('SuccessInfo/FinalPartialWindowEpisodes', self.window_episode_count, self.episode_game_count)

        self.writer.close()
        print("Training finished. Success rates (per 30 episodes) logged to TensorBoard.")