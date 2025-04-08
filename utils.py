import numpy as np
import gymnasium as gym
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Deep Q Network Class (Neural Network)
class DQN(nn.Module):
    def __init__(self, in_states, hidden, num_actions):
        """
        Initialize all layers required for the forward pass
        
        Parameters:
        - in_states: Number of input states (State Space)
        - hidden: dimension of the hidden layer
        - num_actions: Number of actions in the output (Action Space)
        """
        super().__init__

        self.model = nn.Sequential(
            nn.Linear(in_states, hidden),   # fully connected layer
            nn.ReLU(),                      # ReLu activation layer
            nn.Linear(hidden, num_actions)  # Output fc layer
        )

    def forward(self, x):
        """
        Forward pass of the tensor through the network layers
        
        Parameters:
        - x: Input tensor of dimension 'in_states'
        """
        x = self.model(x)   # Pass x through the model
        return x


# Class for remembering....
class MemoryReplay():
    def __init__(self, maxlen):
        """
        Initialize the memory as a 'deque'.
        - maxlen: maximum length of the deque to keep track of in memory
        """
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        """
        Add the transittion to memory
        - transition:
        """
        self.memory.append(transition)

    def sample(self, sample_size):
        """
        Randomly samples the memory for ...
        - sample_size: 
        """
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        """
        Return the length of memory deque. 
        Usage: (memory = MemoryReplay) --> length = len(memory)
        """
        return len(self.memory)
    

# CarPole class to be solved using DQN
class CartPole_DQN():
    def __init__(self):
        """Initialize all parameters required to solve the problem"""

        # Hyperparameters - tunable
        self.learning_rate = 0.001  # alpha
        self.discount = 0.9         # Discount factor gamma
        self.network_sync_rate = 10 # Num steps the network takes before syncing policy and target networks 
        self.replay_mem_size = 1000 # Length of replay memory deque
        self.sample_size = 32       # Length of training data sampled from memory deque
        
        # DQN - Neural Network
        self.loss_fn = nn.MSELoss()      # Mean Squared Error loss function
        self.optimizer = None            # Initialized later
        
        ACTIONS = ['L', 'R']             # For printing: 0 = push the cart Left, 1 = push the cart right. 
        
        
        def train(self, render=False):
            """
            Train the DQN on the CartPole Environment
            
            Paramters:
            - render: bool to enable ....
            """
            # Create CartPole environment instance
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            num_states = env.observation_space.shape[0]     # Number of states in env
            num_actions = env.action_space.n                # Number of actions. Is discrete
            