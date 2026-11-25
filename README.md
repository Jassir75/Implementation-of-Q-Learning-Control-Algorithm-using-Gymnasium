# Implementation of Q-Learning Control Algorithm using Gymnasium

## Aim

To implement the Q-Learning control algorithm using the Gymnasium CliffWalking-v1 environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding falling off the cliff.

## Problem Statement

To implement the Q-Learning control algorithm using the Gymnasium CliffWalking-v1 environment. The agent must learn an optimal action-value function through interaction with the environment and select suitable actions to reach the goal along the shortest safe path while avoiding the cliff region.

## Software Requirements

* **Programming Language:** Python
* **Library:** Gymnasium
* **Numerical Computation:** NumPy
* **Visualization:** Matplotlib
* **Environment:** CliffWalking-v1
* **Platform:** Jupyter Notebook / Google Colab

## Environment Description

The CliffWalking-v1 environment is a classic grid-world reinforcement learning environment provided by Gymnasium.

The environment consists of a **4 × 12 grid** containing:

* **S** – Starting state (bottom-left corner, position `[3, 0]`)
* **G** – Goal state (bottom-right corner, position `[3, 11]`)
* **Safe Grid Cells** – Normal floor tiles (`-1` reward per step)
* **Cliff Region** – Bottom row between Start and Goal (`[3, 1]` to `[3, 10]`), which gives a `-100` reward and immediately returns the agent to the starting state

The agent starts from the bottom-left corner and must navigate across the grid to reach the goal in the bottom-right corner without stepping off the cliff.

The environment contains **48 states** and **4 possible actions**:

| Action | Direction |
| ------ | --------- |
| 0      | Up        |
| 1      | Right     |
| 2      | Down      |
| 3      | Left      |

Each step incurs a reward of `-1`, while stepping into the cliff incurs a `-100` penalty and resets the agent's position to the start without terminating the episode. The episode terminates once the agent reaches the goal state.

## Theory

Q-Learning is an **off-policy reinforcement learning control algorithm** that learns the optimal action-value function directly.

The action-value function \(Q(s,a)\) represents the expected return obtained when the agent takes action \(a\) in state \(s\) and subsequently follows the best possible policy.

The Q-Learning update rule is:

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[ R_{t+1}+ \gamma\max_a Q(S_{t+1},a) -Q(S_t,A_t) \right]$$

Where:

| Symbol                  | Meaning                                |
| ----------------------- | -------------------------------------- |
| \(S_t\)                 | Current state                          |
| \(A_t\)                 | Current action                         |
| \(R_{t+1}\)             | Reward received after taking action    |
| \(S_{t+1}\)             | Next state                             |
| \(\alpha\)              | Learning rate                          |
| \(\gamma\)              | Discount factor                        |
| \(Q(s,a)\)              | Action-value function                  |
| \(\max_a Q(S_{t+1},a)\) | Maximum action value in the next state |

The important feature of Q-Learning is that it uses the **maximum Q-value of the next state** when calculating the target:

$$Target = R+\gamma\max_a Q(S',a)$$

Therefore, Q-Learning learns the optimal policy independently of the action actually selected during exploration.

## Epsilon-Greedy Action Selection

During training, the agent uses a **fixed epsilon-greedy strategy** (\(\epsilon = 0.1\)) to balance exploration and exploitation.

With probability \(\epsilon\), the agent explores by selecting a random action.

With probability \(1-\epsilon\), the agent exploits the learned Q-table by selecting the action with the highest Q-value.

$$a = \begin{cases} \text{random action}, & \text{with probability } \epsilon\\ \arg\max_a Q(s,a), & \text{with probability } 1-\epsilon \end{cases}$$

A constant exploration rate of \(\epsilon = 0.1\) ensures that the agent exploits the best-known action 90% of the time while continuing to explore random actions 10% of the time throughout all training episodes.

## Algorithm

1. Initialize the CliffWalking-v1 environment.
2. Determine the number of states (48) and actions (4).
3. Initialize the Q-table with zeros for all state-action pairs.
4. Set the learning rate \(\alpha\), discount factor \(\gamma\), and fixed exploration rate \(\epsilon\).
5. Reset the environment at the beginning of each episode.
6. Select an action using the fixed epsilon-greedy strategy.
7. Execute the selected action in the environment.
8. Observe the next state, reward, and termination status.
9. Calculate the Q-Learning target:

$$Target = R+\gamma\max_a Q(S',a)$$

10. Update the Q-value using:

$$Q(S,A) \leftarrow Q(S,A)+ \alpha[Target-Q(S,A)]$$

11. Move to the next state.
12. Continue until the episode terminates or reaches the maximum step limit.
13. Store the total reward obtained in the episode.
14. Repeat the process for 500 episodes.
15. Obtain the state-value function using:

$$V(S)=\max_a Q(S,a)$$

16. Obtain the learned policy by selecting the action with the maximum Q-value for each state.
17. Plot the smoothed learning curve using a moving average window.
18. Calculate the average reward over the last 100 episodes.

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v1")

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 500
max_steps_per_episode = 200

def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

episode_rewards = []

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] = Q[state, action] + alpha * (target - Q[state, action])
        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

action_map = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
policy_grid = np.array([action_map[a] for a in learned_policy]).reshape(4, 12)

print("Name: Jassir Sulthan")
print("Reg no: 212224240060")

print("Final Q-table (first 12 states):")
print(np.round(Q[:12], 3))

print("\nEstimated State-Value Function (4x12):")
print(np.round(state_values.reshape(4, 12), 2))

print("\nLearned Policy (4x12):")
print(policy_grid)

avg_reward = np.mean(episode_rewards[-100:])
print(f"\nAverage reward over last 100 episodes: {avg_reward:.3f}")

window = 25
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

print("Name: Jassir Sulthan")
print("Reg no: 212224240060")
plt.figure(figsize=(8, 5))
plt.plot(range(window - 1, num_episodes), moving_avg)
plt.title("Q-Learning Curve - CliffWalking-v1 (Fixed Epsilon = 0.1)")
plt.xlabel
```

## Output

<img width="1205" height="585" alt="image" src="https://github.com/user-attachments/assets/89c284b8-0328-437e-9719-0c31768dc5d2" />
<img width="1488" height="522" alt="image" src="https://github.com/user-attachments/assets/29013cf3-24d1-4e4c-80e2-c1d563009971" />

### Estimated State-Value Function

The state-value function is obtained using:

$$
V(S)=\max_a Q(S,a)
$$

It represents the highest estimated value for each state.

### Learned Policy

The learned policy selects the action having the maximum Q-value for each state.

The actions are represented as:

* **L** – Left
* **D** – Down
* **R** – Right
* **U** – Up

### Average Reward

The average reward is calculated over the **last 1000 episodes** to evaluate the final performance of the learned agent.

### Learning Curve

The learning curve shows how the average reward changes during training.

## Result

The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

The agent learned an action-value function through repeated interaction with the environment. The resulting Q-table contains the learned values for each state-action pair.

The state-value function was obtained by selecting the maximum Q-value for each state, and the learned policy selects the action with the highest Q-value.

The learning curve represents the agent's performance during training, while the average reward over the last 1000 episodes indicates the final performance of the learned policy.

## Inference

The experiment demonstrates that Q-Learning can learn a suitable action-selection policy through trial and error without requiring a predefined model of the environment.

Initially, the agent explores different actions using a high epsilon value. As training progresses, epsilon decreases, allowing the agent to increasingly exploit the learned Q-values.

The Q-table gradually learns the usefulness of different actions in each state. The learned state-value function represents the estimated value of each state, while the learned policy chooses the action with the highest estimated Q-value.

Therefore, the experiment demonstrates that **Q-Learning is effective for learning an action-selection policy in the FrozenLake environment by balancing exploration and exploitation.**
