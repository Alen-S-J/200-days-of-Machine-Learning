
# Concepts in Reinforcement Learning

#### 1. **Agent (𝜋) and Environment (𝒆)**
- **Definition:** 
  - The **agent** interacts with the **environment** by taking actions based on its **policy** to maximize cumulative **rewards**.
- **Mathematical Expression:**
  - **Agent's Policy (𝜋)**: The strategy that the agent uses to select actions given states.
    - ![Agent's Policy](https://latex.codecogs.com/svg.latex?\pi(a|s)=P(a_t=a|s_t=s))
  - **Environment Dynamics (𝒆)**: The system with which the agent interacts.
    - ![Environment Dynamics](https://latex.codecogs.com/svg.latex?\mathcal{E})

#### 2. **States (𝒔), Actions (𝒂), and Rewards (𝑟)**
- **Definition:** 
  - **States** represent situations in the environment.
  - **Actions** are the choices an agent can make in a state.
  - **Rewards** are signals received by the agent as feedback for its actions.
- **Mathematical Expressions:**
  - **State Space (𝒔)**: Set of all possible states.
    - ![State Space](https://latex.codecogs.com/svg.latex?s\in\mathcal{S})
  - **Action Space (𝒂)**: Set of all possible actions.
    - ![Action Space](https://latex.codecogs.com/svg.latex?a\in\mathcal{A})
  - **Reward Function (𝑟𝑎(𝑠))**: Function that specifies the immediate reward on taking action a in state s.
    - ![Reward Function](https://latex.codecogs.com/svg.latex?r_a(s)=\mathbb{E}[R_{t+1}|S_t=s,A_t=a])

#### 3. **Policy (𝜋) and Value Functions**
- **Definition:** 
  - **Policy (𝜋)**: Defines the agent's behavior; it's a mapping from states to probabilities of selecting each action.
  - **Value Functions**: Estimate the goodness/badness of states or state-action pairs.
- **Mathematical Expressions:**
  - **State-Value Function (𝑉𝜋(𝑠))**: Expected return starting from state s and following policy 𝜋 thereafter.
    - ![State-Value Function](https://latex.codecogs.com/svg.latex?V^{\pi}(s)=\mathbb{E}_{\pi}[G_t|S_t=s])
  - **Action-Value Function (𝑄𝜋(𝑠,𝑎))**: Expected return starting from state s, taking action a, and following policy 𝜋 thereafter.
    - ![Action-Value Function](https://latex.codecogs.com/svg.latex?Q^{\pi}(s,a)=\mathbb{E}_{\pi}[G_t|S_t=s,A_t=a])

These expressions form the foundational elements of reinforcement learning, enabling the formulation and understanding of algorithms for agents to learn and make decisions in environments.