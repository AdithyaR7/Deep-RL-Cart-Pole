import os
import argparse
import gymnasium as gym
from utils import CartPole_DQN

def main(mode, weights_path):
    
    if mode == 'train':
        # Train the environment and save the model weights
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        agent = CartPole_DQN(env, weights_path)
        agent.train(episodes=750)
        agent.save_weights()
    
    elif mode == 'test':
        # Test the environment using saved model weights
        env = gym.make("CartPole-v1", render_mode="rgb_array") # error
        agent = CartPole_DQN(env, weights_path)
        agent.load_weights()
        agent.test(episodes=2, save_video=True)     # Save a video of the tests
    
    
if __name__ == "__main__":
    
    # Argument parser for user to indicate mode and weights path
    parser =  argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--weights', type=str, default='cartpole_weights.pth', help="Path to save/load model weights")
    args = parser.parse_args()
    
    # Extract arguments to pass into main
    mode = args.mode
    weights_path = args.weights
    
    # Check if valid weights are passed for testing
    if (mode =='test') and not (os.path.exists(weights_path)):
        print("Error: No trained weights found. Run with '--train' first.") 
        exit(1)
        
    main(mode, weights_path)