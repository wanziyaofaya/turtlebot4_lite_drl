from stable_baselines3 import PPO
from src.turtlebot4_env import TurtleBot4Env

def main():
    # Create the custom TurtleBot4 environment
    env = TurtleBot4Env()

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("turtlebot4_ppo")

if __name__ == "__main__":
    main()

