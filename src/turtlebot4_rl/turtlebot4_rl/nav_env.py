import gymnasium as gym
import numpy as np
class TurtleBotNavEnv(gym.Env):
    def __init__(self):
        super(TurtleBotNavEnv, self).__init__()
        # Define the action and observation spaces
    
    def reset(self):
        return self.state
    def step(self, action):
        pass
    def _take_action(self, action):
        # TODO: Call ROS2 functions
        pass
    def _get_state(self):
        # TODO: SLAM
        return np.random.rand(1)
    def _calculate_reward(self):
        # TODO: Negative reward for collisions or further from goal than last step
        # TODO: Positive reward for getting closer to goal
        return -1
    def _is_done(self):
        # TODO: Check if robot is at goal position
        return False
    def _is_collision(self):
        # TODO: Check if robot hit an obstacle
        return False
