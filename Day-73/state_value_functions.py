# Define states, rewards, and transitions for a simple grid world scenario

# Define states
states = ['S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'T']  # 'S' is start, 'T' is terminal

# Define policy: (probability, next_state, reward)
policy = {
    'S': [(0.5, 'A', 0), (0.5, 'B', 0)],
    'A': [(0.5, 'S', 0), (0.5, 'C', 0)],
    'B': [(0.5, 'S', 0), (0.5, 'D', 0)],
    'C': [(0.5, 'A', 0), (0.5, 'E', 100)],
    'D': [(0.5, 'B', 0), (0.5, 'E', 100)],
    'E': [(0.5, 'C', 100), (0.5, 'D', 100)],
    'F': [(1.0, 'G', 0)],
    'G': [(1.0, 'H', 0)],
    'H': [(1.0, 'T', 0)],
    'T': []  # Terminal state
}

# Calculate state-value function for the given policy

# Function to calculate state-value function for a given policy π
def state_value_function(states, policy, gamma=0.9, epsilon=1e-6):
    # Initialize V(s) for all states arbitrarily
    V = {s: 0 for s in states}
    
    # Perform iterative policy evaluation
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # Calculate V(s) using Bellman Expectation Equation
            V[s] = sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward in policy[s])
            delta = max(delta, abs(v - V[s]))
        
        # Check for convergence
        if delta < epsilon:
            break
    
    return V

# Calculate state-value function for the given policy
V_pi = state_value_function(states, policy)
print("State-Value Function V(s):")
for state, value in V_pi.items():
    print(f"V({state}): {value}")
