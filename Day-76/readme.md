# Q-Learning Basics:

#### 1. Key Components:
   - **Q-Table**: Stores action values for each state.
   - **Reward**: Immediate feedback after an action in a state.
   - **Policy**: Strategy for action selection based on Q-values.
   - **Environment**: System where the agent operates.

#### 2. Algorithm:
   - **Initialization**: Set Q-table values arbitrarily or to zeros.
   - **Exploration vs. Exploitation**: Balancing new vs. known actions with epsilon-greedy strategy.
   - **Updating Q-values**: Bellman equation updates Q-values based on rewards and future rewards.
     - \(Q(s, a) = (1 - \alpha) \cdot Q(s, a) + \alpha \cdot [r + \gamma \cdot \max Q(s', a')]\)
     - \(Q(s, a)\): Q-value for state \(s\) and action \(a\).
     - \(\alpha\): Learning rate (0 to 1) - controls new information impact.
     - \(r\): Immediate reward for action \(a\) in state \(s\).
     - \(\gamma\): Discount factor (0 to 1) - importance of future rewards.
     - \(s'\): Next state.
     - \(a'\): Action in the next state.

#### 3. Learning Process:
   - **Iterative Updates**: Agent interacts, updating Q-values based on experiences.
   - **Convergence**: Over time, Q-values converge to optimal actions in each state.

#### 4. Implementation (Toy Problem Example):
   - **Environment**: Grid world with states/actions.
   - **Q-Table**: Matrix storing Q-values for each state-action pair.
   - **Action Selection**: Epsilon-greedy strategy.
   - **Updating Q-values**: Use Bellman equation to update Q-values after each action.

### Simple Pseudocode:

```python
# Initialize Q-table with random values

for episode in range(num_episodes):
    Initialize state
    while not done:
        Choose action using an epsilon-greedy policy
        Take action, observe reward and next state
        Update Q-value for current state-action pair using the Bellman equation
        Move to next state
```
## Resources for Further Learning:
- **Books**: "Reinforcement Learning: An Introduction" by Sutton and Barto.
- **Online Courses**: Coursera, Udacity, and edX offer courses on reinforcement learning.
- **Documentation**: Explore RL libraries like OpenAI Gym, which provide examples and documentation on Q-learning implementations.