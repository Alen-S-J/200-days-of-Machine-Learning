

### REINFORCE Algorithm

1. **Initialize**: Initialize policy parameters randomly: θ

2. **Loop for each episode**:
   - **Generate an Episode**:
     - Generate a complete episode by interacting with the environment using the current policy.
     - Record the state, action, and rewards at each time-step.

   - **For each time-step 't' in the episode**:
     - Calculate the return G_t from time-step 't' onwards.

   - **For each time-step 't' in the episode**:
     - Update the policy parameters:
       ```markdown
       θ <- θ + α * ∇_θ log π(a|s) * G_t
       ```
       Where:
       - θ is the policy parameters.
       - α is the learning rate.
       - ∇_θ log π(a|s) is the gradient of the log probability of taking action 'a' in state 's' according to the policy.
       - G_t is the return or cumulative reward from time-step 't'.

3. **Repeat**: Continue this process for multiple episodes.

### Actor-Critic Algorithm

1. **Initialize**: Initialize policy parameters θ_actor and value function parameters θ_critic.

2. **Loop for each time-step or episode**:
   - **Interaction with Environment**:
     - Observe the current state 's'.
     - Choose an action 'a' using the current policy π(a|s).

   - **Environment Response**:
     - Execute action 'a' in the environment.
     - Observe reward 'r' and the next state 's'.

   - **Critic (Value) Update**:
     - Update the critic's value function parameters:
       ```markdown
       θ_critic <- θ_critic + α_critic * (G_t - V(s)) * ∇_θ_critic V(s)
       ```
       Where:
       - θ_critic is the critic's parameters.
       - α_critic is the critic's learning rate.
       - G_t is the return or cumulative reward from time-step 't'.
       - V(s) is the state-value function.
       - ∇_θ_critic V(s) is the gradient of the state-value function.

   - **Actor (Policy) Update**:
     - Update the actor's policy parameters:
       ```markdown
       θ_actor <- θ_actor + α_actor * ∇_θ_actor log π(a|s) * (Q(s, a) - V(s))
       ```
       Where:
       - θ_actor is the actor's policy parameters.
       - α_actor is the actor's learning rate.
       - ∇_θ_actor log π(a|s) is the gradient of the log probability of taking action 'a' in state 's' according to the policy.
       - Q(s, a) is the action-value function.
       - V(s) is the state-value function.

3. **Repeat**: Continue the interaction, updating the actor and critic parameters iteratively for multiple time-steps or episodes.

These algorithms outline the process of interacting with the environment, updating the policy and value function parameters, and iteratively improving the learning in both REINFORCE and Actor-Critic methods.