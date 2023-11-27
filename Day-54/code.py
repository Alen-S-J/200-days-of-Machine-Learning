import numpy as np

# Define sigmoid and tanh functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Define LSTM cell
class BasicLSTMCell:
    def __init__(self, input_size, hidden_size):
        # Initialization of attributes (same as before)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_i = np.random.randn(hidden_size, 1)
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_f = np.random.randn(hidden_size, 1)
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_o = np.random.randn(hidden_size, 1)
        self.W_C = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_C = np.random.randn(hidden_size, 1)
        self.h = np.zeros((hidden_size, 1))
        self.C = np.zeros((hidden_size, 1))
        self.x = None

    def set_input(self, x):
        # Input should be a column vector
        self.x = x.reshape(-1, 1)

    def forward(self):
        if self.x is None:
            raise ValueError("Input values are not set. Use set_input() method to assign input.")
        
        # Concatenate input and previous hidden state
        combined_input = np.vstack((self.h, self.x))

        # Input gate computations
        i_t = sigmoid(np.dot(self.W_i, combined_input) + self.b_i)

        # Forget gate computations
        f_t = sigmoid(np.dot(self.W_f, combined_input) + self.b_f)

        # Output gate computations
        o_t = sigmoid(np.dot(self.W_o, combined_input) + self.b_o)

        # Candidate value computation
        C_tilde = tanh(np.dot(self.W_C, combined_input) + self.b_C)

        # Update cell state and hidden state
        self.C = f_t * self.C + i_t * C_tilde
        self.h = o_t * tanh(self.C)

        # Return the hidden state after the computation
        return self.h

    def add_values(self, x1, x2):
        # Set input values for addition
        self.set_input(np.array([x1, x2]))

        # Perform addition by updating the cell's state
        result = self.forward()

        # Return the result of addition (the hidden state after computation)
        return result.flatten()[0]

# Usage example
input_size = 2  # Inputs for addition: two values
hidden_size = 4  # Hidden size of the LSTM cell

# Create an instance of the LSTM cell
lstm_cell = BasicLSTMCell(input_size, hidden_size)

# Perform addition
result = lstm_cell.add_values(3, 5)
print("Result of addition:", result)
