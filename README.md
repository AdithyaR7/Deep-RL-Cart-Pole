# Deep-RL-Cart-Pole
Solving the classic Cart Pole problem using Deep Reinforcement Learning (DQN)

## The Cart Pole Environment
The Cart Pole problem is a classic controls and robotics problem where an agent a cart allowed to move laterally tries to balance a pole upright on a single pivot point, and typically solved using PID control. Here we are solving the CartPole-v1 (Gymnasium library) environment using Deep Q-Learning, a fundamental Deep Reinforcement Learning approach. The agent learns to balance the pole on the cart by taking discrete actions to move left or right based on continuous observations from the environment. 

<img src="images/cart_pole_env.png" width="600" />

The object is simple: prevent the pole from falling over by moving the cart left or right. 
The environment ends when the pole falls past a certain angle (-24&deg; to +24&deg;), the cart strays too far from the center (beyond -4.8 and 4.8), or after 500 steps in one episode (also the maximum possible reward per episode). Each successful step gives a +1 reward to the agent if the pole is still upright, and the episode terminates if the above conditions fail. 

The state space is continuous and represented by 4 variables: the cart position, cart velocity, pole angle, and pole angular velocity (both velocities have no bounds in this environment). The action space is discrete: move the cart left or right. 

## Deep Q-Learning (DQN)

Instead of using a traditional Q-table (infeasible with continuous state spaces), we use a **neural network** to approximate the Q-values for each action.

### How It Works:
- Two networks are used:
  - **Online Network:** Learns and updates Q-values via backpropagation.
  - **Target Network:** Provides stable Q-value targets and is updated less frequently.
- The agent learns to predict Q-values for each action given a state and chooses actions using an **epsilon-greedy** policy.
- A **Replay Buffer** stores past experiences and samples mini-batches to break temporal correlation.

The agent during the training process, learning as it progresses:

<img src="https://github.com/AdithyaR7/Deep-RL-Cart-Pole/blob/main/agent_gifs/cartpole_training_progress.gif" width="600" />

### DQN Highlights:
- **Input:** 4D state (cart & pole variables)
- **Output:** Q-values for each possible action (left, right)
- **Network:** 1 hidden layer with ReLU, trained using MSE loss
- **Exploration:** Epsilon-greedy strategy
- **Replay Buffer:** Stores past experiences for stable training
- **Target Network:** (Optional) Used to improve stability

A common problem encountered with this is that a standard DQN can overestimate action values during the training process. The fix is using the 'double DQN' method that reduces this by decoupling the action selection and evaluation between the two networks, leading to more stable and reliable learning.

The fully trained agent solving a randomly initialized environment:

<img src="https://github.com/AdithyaR7/Deep-RL-Cart-Pole/blob/main/agent_gifs/cartpole_test.gif" width="600" />

