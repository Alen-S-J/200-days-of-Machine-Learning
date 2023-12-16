

# Value Functions in Reinforcement Learning:

### State-Value Function (V-function):
The state-value function in reinforcement learning estimates the value of being in a particular state under a given policy. Mathematically, it's denoted as `V^π(s)` where:
- `V^π(s)` represents the expected return (cumulative future rewards) when starting in state `s` and following policy `π`.

**Theoretical Explanation:**
The state-value function helps in evaluating the goodness of a state under a certain policy. It tells us how good it is to be in a particular state while following a specific policy.

### Action-Value Function (Q-function):
The action-value function in reinforcement learning estimates the value of taking a certain action in a particular state under a given policy. Mathematically, it's denoted as `Q^π(s, a)` where:
- `Q^π(s, a)` represents the expected return (cumulative future rewards) by taking action `a` in state `s` and then following policy `π`.

**Theoretical Explanation:**
The action-value function helps in evaluating the goodness of taking a specific action in a given state while following a certain policy. It helps in making decisions about which action to take in a particular state to maximize future rewards.

### Policies in Reinforcement Learning:

A policy `π` in reinforcement learning is a strategy that an agent employs to decide its actions in an environment. It can be deterministic or stochastic. 
- A deterministic policy can be represented as `π(s) → a`, indicating that in state `s`, the agent always takes action `a`.
- A stochastic policy is represented as `π(a|s)`, denoting the probability of taking action `a` given the state `s`.

**Theoretical Explanation:**
A policy guides the agent's behavior by providing rules or strategies to decide which action to take in each state. The objective is often to find an optimal policy that maximizes the cumulative reward over time.

### Mathematical Expressions:

1. **State-Value Function:**
\[ V^π(s) = \mathbb{E}_π[G_t | S_t = s] \]
    - `V^π(s)` is the state-value function for state `s`.
    - \( \mathbb{E}_π \) denotes the expected value under policy `π`.
    - `G_t` represents the total discounted reward from time step `t` onwards.
    - `S_t` is the state at time step `t`.

2. **Action-Value Function:**
\[ Q^π(s, a) = \mathbb{E}_π[G_t | S_t = s, A_t = a] \]
    - `Q^π(s, a)` is the action-value function for state-action pair `(s, a)`.
    - \( \mathbb{E}_π \) denotes the expected value under policy `π`.
    - `G_t` represents the total discounted reward from time step `t` onwards.
    - `S_t` is the state at time step `t` and `A_t` is the action taken at time step `t`.

These functions and policies are crucial components in reinforcement learning algorithms like Q-learning, SARSA, and policy gradient methods, aiding in learning and decision-making in various environments. 

