import os
import gymnasium as gym
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
            nn.Linear(hidden, hidden),
            nn.ReLU(),
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
        - step_results: tuple of the results of a single step of the agent in the env
        """
        self.memory.append(step_results)

    def sample(self, sample_size):
        """
        Randomly samples the memory for ...
        - sample_size: numner of elements to be sampled from memory for training
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
    def __init__(self, env: gym.Env, weights_path, hidden_size=128, lr=0.0005, gamma=0.99, 
                 batch_size=256, memory_size=40000, sync_rate=10):
        """
        Initialize all parameters required to solve the problem
        
        Parameters:
        - env:          OpenAI Gymnasium environment (CartPole-v1)
        - weights_path: File path to save/load trained model weights
        - hidden_size:  Number of neurons in the hidden layers of the neural network 
        - lr:           Learning rate for the Adam optimizer
        - gamma:        Discount factor for future rewards in Bellman equation
        - batch_size:   Number of experiences sampled from replay buffer for training
        - memory_size:  Maximum size of the experience replay buffer
        - sync_rate:    Number of training steps before copying policy network weights to target network
        """ 

        self.env = env  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
         
        # Path to save model weights
        self.weights_path = weights_path
        
        # Set up dimension of state and action space
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        # Create policy and target Networks
        self.policy_dqn = DQN(state_dim, hidden_size, action_dim).to(self.device)
        self.target_dqn = DQN(state_dim, hidden_size, action_dim).to(self.device)
        
        # Copy weights & biases from policy to target network to make identical
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()  # Sets to eval mode - inference only
        
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
        
        # Use Double DQN to predict next_q_vals to help prevent overestimation bias. 
        with torch.no_grad():
            # Policy network selects best actions for next states
            next_actions = self.policy_dqn(next_states).argmax(1, keepdim=True)
            
            # Target network evaluates those selected actions
            next_q_vals = self.target_dqn(next_states).gather(1, next_actions)
               
        # Compute target q_vals using Bellman equation
        target_q_vals = rewards + self.gamma * next_q_vals * (1 - dones) # (batch_size, 1). Only calc where done=False
        
        # Loss calculation and optimizer step
        loss = self.loss_fn(q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_dqn.parameters(), max_norm=1.0)      # Gradient clipping
        self.optimizer.step()
        return
    
    def warmup_replay_buffer(self, steps=5000):
        """
        Fill the replay buffer with random experiences before training starts.
        This helps stabilize the early learning process.
        
        Parameters:
        - steps: Number of random actions to collect
        """
        print(f"Warming up replay buffer with {steps} random experiences...")
        state, _ = self.env.reset()
        
        for step in range(steps):
            # Take random actions
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store the experience
            self.memory.push((state, action, next_state, reward, done))
            
            # Reset environment if episode ended, otherwise continue
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
                
            # Print progress
            if (step + 1) % 1000 == 0:
                print(f"Warmup progress: {step + 1}/{steps}")
        
        print("Replay buffer warmup complete!")
    

    def evaluate_and_render(self, episode_idx):
        """
        Runs one evaluation episode with epsilon=0 and returns frames with overlays
        Used to visualized how well the agent performs in the course of its training.
        """
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        frames = []

        while not done:
            action = self.select_action(state, epsilon=0.0)  # Always greedy
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            frame = self.env.render()
            overlay = f"Training Ep: {episode_idx}  Step: {step}  Reward: {int(total_reward)}"
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
            frames.append(frame_bgr)

        return frames
       
        
    def train(self, episodes, epsilon_init=1.0, epsilon_min=0.02, epsilon_decay=0.995, save_video=True ):
        """
        Train the DQN on the CartPole Environment
        
        Paramters:
        - episodes:      total training episodes
        - epsilon_init:  initial epsilon for exploration
        - epsilon_min:   minimum epsilon
        - epsilon_decay: decay factor per episode
        - save_video:    bool whether to save the video
        - render_every:  interval to render an episode for video
        """
        # Warm up the replay buffer before starting training
        self.warmup_replay_buffer(steps=10000)
        
        self.episodes = episodes
        epsilon = epsilon_init
        episode_rewards = []                # Store episode rewards
        epsilon_hist = []                   # Track epsilon values for plotting
        frames = []                         # Record training progress to video
        render_interval = 20                # Episode interval to render training progress
        best_reward = float('-inf') 
        self.best_model_state = None
        
        for episode in range(episodes):
            state, _ = self.env.reset()     # Reset the env at the start of each episode
            total_reward = 0
            done = False
                        
            while not done:
                # Select 'epsilon-greedy' or 'best' action and step through the environment
                action = self.select_action(state, epsilon)   
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                ########### Improvement to base functionality. Comment this out to achieve first part of ReadMe
                # Custom reward to favor centered position and upright pole
                x, x_dot, theta, theta_dot = next_state
                reward = 1.0
                reward -= 2.0 * (abs(x) / 2.4)          # strongly penalize distance from center
                reward -= 1.5 * (abs(theta) / 0.2095)   # strongly penalize pole angle
                reward -= 0.1 * abs(x_dot)              # penalize movement
                reward -= 0.1 * abs(theta_dot)          # penalize swinging
                
                reward = max(reward, -1.0)  # prevent overly negative reward
                
                if terminated:
                    reward -= 2.0               # extra penalty if episode ends early
                ############
                
                done = terminated or truncated  # True if episode ended
                
                done = terminated or truncated  # True if episode ended
                
                # Save results to memory
                self.memory.push((state, action, next_state, reward, done))
                state = next_state
                total_reward += reward
                
                self.optimize_model()           # Update model weights based on episode
            
            # Update rewards for the episode
            episode_rewards.append(total_reward)
            
            # Update best model weights if reward is higher
            if total_reward >= best_reward:
                best_reward = total_reward
                self.best_model_state = self.policy_dqn.state_dict() # Store copy of best weights so far
                self.save_weights(episode)                           # Save best weights = model checkpoints
            
            # Decay epsilon. Explore less as we progress
            epsilon = max(epsilon_min, epsilon*epsilon_decay)   
            epsilon_hist.append(epsilon)
            
            # Sync policy network to target network
            if (episode + 1) % self.sync_rate == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                
            # Print training progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
            
            # Update render_interval as episodes progress since differnece in progress slows down
            if episode > 100:
                render_interval = 50
            if episode > 300:
                render_interval = 100
                
            # Run inference to see how well the agent can perform at this point in the training
            if save_video and ((episode % render_interval == 0) or (episode == (episodes-1))):
                eval_frames = self.evaluate_and_render(episode)
                frames.extend(eval_frames)
          
            # Early stopping if average reward is high for the last N episodes
            if episode > 400:  # wait until there's enough data
                avg_reward = np.mean(episode_rewards[-30:])
                if avg_reward >= 490:
                    print(f"Early stopping at episode {episode} with avg reward {avg_reward:.2f}")
                    break
          
        # Save video
        if save_video and frames:
            h, w = frames[0].shape[:2]
            fps = 30
            video_name = "cartpole_training_progress.mp4"
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            print(f"Eval video saved to {video_name}")
        
                
        # Plot reward and epsilon and save figs
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(episode_rewards)
        plt.title('Rewards per Episode')
        plt.subplot(1,2,2)
        plt.plot(epsilon_hist)
        plt.title('Epsilon Decay')
        plt.tight_layout()
        plt.savefig("cartpole_dqn_training.png")
        return
    
    
    def test(self, episodes=5, save_video=True):
        """Run the trained policy DQN in the env 'episodes' times. Creates renders"""
        
        # Save video
        frames = []
        
        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False   
            total_reward = 0
            step = 0
             
            while (not done):
                # Network is fully trained, always seek best action
                action = self.select_action(state, epsilon=0.0) 
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
                
                frame = self.env.render()   # Render the environment for visualization
        
                # Add frame overlay text
                if save_video:
                    overlay_text = f"Episode: {ep+1}  Step: {step}  Reward: {int(total_reward)}"
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame_bgr, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    frames.append(frame_bgr)
        
        # Save video
        if save_video and frames:
            frame_size = (frames[0].shape[1], frames[0].shape[0])
            video_name = "cartpole_test.mp4"
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)
            for f in frames:
                out.write(f)
            out.release()
            print(f"Video saved to {video_name}.")
            
        self.env.close()
        return
    
    
    def save_weights(self, episode):
        """Save the model weights to weights_path after training"""
        if self.best_model_state is not None:
            torch.save(self.best_model_state, self.weights_path)
            if (episode > 500) and (episode % 10 == 0):
                print(f"Model weights saved to {self.weights_path} after {episode} episodes.")
        else:
            print("Error during training: Best Weights is 'None'")
        return
        
    
    def load_weights(self):
        """Load model weights from weights_path before testing"""
        if os.path.exists(self.weights_path):                # Extra safety check           
            model_weights = torch.load(self.weights_path, map_location=self.device)
            self.policy_dqn.load_state_dict(model_weights)
            # We do not need to load weights into the target netwrok for testing
            print(f"Loaded model weights from {self.weights_path}!")
        else:
            print(f"Warning: no model weights found. at {self.weights_path}, running untrained.")
        return