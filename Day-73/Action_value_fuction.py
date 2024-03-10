# Assuming a Q-table Q with states and actions
# Q = {s1: {a1: q_value_1, a2: q_value_2, ...}, s2: {...}, ...}
# Here's an example of how you might set up the Q-table

# Initialize Q-table with zeros for all state-action pairs
# Assuming states and actions are predefined
states = ['S', 'A', 'B', 'C', 'D']
actions = ['up', 'down', 'left', 'right']

# Initialize Q-table
Q = {s: {a: 0 for a in actions} for s in states}

# Function to update Q-value for a specific state-action pair using Q-learning update rule
def q_value_function(Q, state, action, reward, next_state, alpha, gamma):
    current_q = Q[state][action]
    # Update Q-value using Q-learning update rule
    new_q = current_q + alpha * (reward + gamma * max(Q[next_state].values()) - current_q)
    Q[state][action] = new_q

# Example usage:
# Assuming an environment that returns reward and next_state based on the action taken
# Let's update Q-value for a specific state-action pair based on the observed reward and next state
current_state = 'S'
selected_action = 'up'
reward = 10
next_state = 'A'
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor

# Update Q-value for the observed state-action pair
q_value_function(Q, current_state, selected_action, reward, next_state, alpha, gamma)

# Print the updated Q-table after the update
print("Updated Q-table:")
for state, actions in Q.items():
    print(f"State {state}:")
    for action, q_value in actions.items():
        print(f"   Q({state}, {action}): {q_value}")
