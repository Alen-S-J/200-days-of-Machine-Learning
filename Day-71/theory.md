### Markov Decision Processes (MDPs)

Markov Decision Processes are a fundamental framework used in reinforcement learning. They consist of:
- **State Space (S)**: The set of all possible situations the agent can be in.
- **Action Space (A)**: The set of all possible actions the agent can take.
- **Transition Function (P)**: Describes the probability of transitioning to a new state given the current state and action.
- **Reward Function (R)**: Defines the immediate reward the agent receives for taking a certain action in a certain state.

### Value Functions

Value functions estimate the expected cumulative rewards an agent can obtain under a specific policy. There are two types:
- **State Value Function (V)**: Estimates the expected cumulative reward from a given state by following a policy.
- **Action Value Function (Q)**: Estimates the expected cumulative reward from taking a specific action in a given state and then following a policy.

### Policies

A policy (\(\pi\)) is the strategy the agent uses to determine its actions based on the current state. Policies can be deterministic or stochastic.

### Bellman Equations

The Bellman equations describe how the values of states and actions are related to each other. They are central to understanding how value functions evolve based on rewards and future values.

### Exploration vs. Exploitation

Reinforcement learning agents face the dilemma of exploration (trying new actions) versus exploitation (using known actions to maximize immediate reward). Striking the right balance is crucial for effective learning.

### Temporal-Difference Learning vs. Monte Carlo Methods

These are methods for updating value estimates:
- **Temporal-Difference (TD) Learning**: Updates values based on a combination of observed rewards and estimates.
- **Monte Carlo Methods**: Updates values based on complete returns observed at the end of an episode.

### On-Policy vs. Off-Policy Learning

- **On-Policy Learning**: The agent learns from the data generated by its current policy.
- **Off-Policy Learning**: The agent can learn from data generated by any policy, allowing for greater sample efficiency.

### Exploration vs. Exploitation

- **Exploration**: Strategies used to discover new states or actions to better understand the environment.
- **Exploitation**: Utilizing current knowledge to make decisions that maximize immediate rewards.

### Model-Free vs. Model-Based Learning

- **Model-Free RL**: The agent learns directly from interactions with the environment without creating an explicit model of the environment's dynamics.
- **Model-Based RL**: The agent learns an explicit model of the environment and plans actions based on this model.

These theories form the foundation for reinforcement learning algorithms and strategies, aiding agents in learning to make informed decisions in dynamic environments.
