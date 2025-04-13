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
        super().__init__()

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

    def push(self, step_results):
        """
        Add the transittion to memory
        - transition: tuple of the results of a single step of the agent in the env
        """
        self.memory.append(step_results)

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
    def __init__(self, env: gym.Env, hidden_size=128, lr=0.001, gamma=0.99, 
                 batch_size=64, memory_size=1000, sync_rate=10):
        """
        Initialize all parameters required to solve the problem
        
        Parameters:
        - env:
        - hidden_size:
        - lr:
        - gamma:
        - batch_size:
        - memory_size:
        - sync_rate: 
        """

        self.env = env  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set up dimension of state and action space
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        # Create policy and target Networks
        self.policy_dqn = DQN(state_dim, hidden_size, action_dim).to(self.device)
        self.target_dqn = DQN(state_dim, hidden_size, action_dim).to(self.device)
        
        # Copy weights & biases from policy to target network to make identical
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()  # Sets to eval mode - inference only, no need for torch.no_grad() 
        
        # Initialize Adam optimizer and Mean Squared Error loss function
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Create deque object for training using memory replay
        self.memory = MemoryReplay(memory_size) 
        
        # Hyperparameters - tunable
        self.gamma = gamma              # discount factor
        self.lr = lr                    # learning rate
        self.batch_size = batch_size    # batch of training data samplef from memory
        self.sync_rate = sync_rate      # Num steps of network before syncing policy -> target netwroks
               
        self.actions = ['L', 'R']       # For printing: 0 = push the cart Left, 1 = push the cart right. 
        
        
    def select_action(self, state, epsilon):
        """Selects appropriate action based on epsilon"""
        # Select epsilon greedy action
        if random.random() < epsilon:
            action = self.env.action_space.sample() # Randomly select action
            
        # Select best action
        else:
            with torch.no_grad():
                # Convert state list to tensor. adjust dimensions
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = self.policy_dqn(state_tensor).argmax().item() # Extract best action's value
        return action 
    
       
    def optimize_model(self):
        """Optimize the model by a single step by 
           sampling a batch from the memory deque"""
        
        # Check if there's enough samples to optimize
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from the memory. Tuples of length batch_size
        sampled_batch = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*sampled_batch)
        
        # Convert to tensors of appropriate sizes
        states = torch.tensor(states, dtype=torch.float32).to(self.device)                # (batch_size,4)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)      # (batch_size,4)
        # Use unsqueeze to add an extra dimension: (batch_size, 1)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)   # (batch_size, 1) for gather()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device) # (batch_size, 1) for broadcast
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)     # (batch_size, 1) for broadcast
        
        # Compute Q(s,a) for current states,actions by passing through network.
        q_vals = self.policy_dqn(states).gather(1, actions) # (64, 1). 
        # .gather() extracts the q_value taken at that action index. Avoids for loop!
        
        # Compute max Q(s',a') for next_states, next_actions = best estimated q_val for next state
        next_q_prediction = self.target_dqn(next_states)    # (batch_size, 2)
        next_q_vals = next_q_prediction.max(1, keepdim=True)[0] # Extract maximum values (batch_size, 1)
        
        # Compute target q_vals using Bellman equation
        target_q_vals = rewards + self.gamma * next_q_vals * (1 - dones) # (batch_size, 1). Only calc where done=False
        
        # Loss calculation and optimizer step
        loss = self.loss_fn(q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return
        
       
        
    def train(self, episodes, epsilon_init=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Train the DQN on the CartPole Environment
        
        Paramters:
        - episodes: number of runs/epochs of the env to train the model
        - epsilon_init: initial value of epsilon
        - epsilon_min: epsilon value to stop decay at
        - epsilon_decay: factor to decay epsilon by each ___
        """
        epsilon = epsilon_init
        episode_rewards = []    # Store episode rewards
        epsilon_hist = []       # Track epsilon values for plotting
        
        for episode in range(episodes):
            state, _ = self.env.reset()     # Reset the env at the start of each episode
            total_reward = 0
            done = False
            
            while not done:
                # Select 'epsilon-greedy' or 'best' action and step through the environment
                action = self.select_action(state, epsilon)   
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated  # True if episode ended
                
                # Save results to memory
                self.memory.push((state, action, next_state, reward, done))
                state = next_state
                total_reward += reward
                
                self.optimize_model()           # Update model weights based on episode
            
            episode_rewards.append(total_reward)
            
            # Decay epsilon. Explore less as we progress
            epsilon = max(epsilon_min, epsilon*epsilon_decay)   
            epsilon_hist.append(epsilon)
            
            # Sync policy network to target network
            if episodes % self.sync_rate == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                
            # Print training progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
                
            # Plot reward and epsilon
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.plot(episode_rewards)
            plt.title('Rewards per Episode')
            plt.subplot(1,2,2)
            plt.plot(epsilon_hist)
            plt.title('Epsilon Decay')
            plt.tight_layout()
            plt.savefig("cartpole_dqn_training.png")
            
    
    def test(self, episodes=5):
        """Run the trained policy DQN in the env 'episodes' times. Creates renders"""
        
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            while (not done):
                # Network is fully trained, always seek best action
                action = self.select_action(state, epsilon=0.0) 
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.env.render()   # Render the environment for visualization
        
        self.env.close()
        return