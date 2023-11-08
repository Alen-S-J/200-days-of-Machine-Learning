import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize random weights and biases for the input and hidden layers
input_size = 2
hidden_size = 3
output_size = 1

weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))

# Initialize random weights and biases for the hidden and output layers
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Define the forward pass function
def forward(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + bias_output
    return output

# Example input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Forward pass
output = forward(X)
print("Predictions:",output)

