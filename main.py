import gymnasium as gym
from utils import CartPole_DQN

def main():
    
    # Train the environment
    env = gym.make("CartPole-v1", render_mode=None)
    agent = CartPole_DQN(env)
    agent.train(500)
    
    # Test the environment
    test_env = gym.make("CartPole-v1", render_mode="human")
    agent.env = test_env
    agent.test(5)
    
if __name__ == "__main__":
    main()