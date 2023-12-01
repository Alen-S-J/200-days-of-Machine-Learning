# Experimentation and Fine-tuning

### Stacked LSTMs:
- **Description:** Stacked LSTMs involve connecting multiple LSTM layers sequentially.
- **Purpose:** Deeper architectures allow the model to learn more complex patterns and dependencies in sequential data.
- **Working Principle:** Each LSTM layer in the stack processes the input sequence and passes its output sequence to the next LSTM layer.
- **Advantages:**
    - Hierarchical representation of information.
    - Capturing long-term dependencies more effectively due to multiple LSTM layers.
- **Considerations:** 
    - Increased model complexity may lead to longer training times and potential overfitting.
    - Requires careful tuning of hyperparameters to prevent vanishing/exploding gradient problems.

### Bidirectional LSTMs:
- **Description:** Bidirectional LSTMs process input sequences in both forward and backward directions.
- **Purpose:** Capturing information from past and future contexts simultaneously.
- **Working Principle:** It consists of two LSTM layers—one processing the sequence in its original order and another in reverse order.
- **Advantages:**
    - Contextual understanding by considering past and future information.
    - Improved understanding of temporal data in natural language processing tasks.
- **Considerations:**
    - Doubles the number of parameters and computational complexity compared to unidirectional LSTMs.
    - Sensitive to sequence length due to processing in both directions.

### GRUs (Gated Recurrent Units):
- **Description:** GRUs are a variation of LSTM units designed to simplify the architecture while maintaining effectiveness.
- **Purpose:** Facilitating information flow across time steps and controlling the flow of information.
- **Working Principle:** GRUs have two gates—reset and update gates—compared to LSTM's input, output, and forget gates.
- **Advantages:**
    - Fewer parameters compared to LSTM, hence faster training.
    - Effective in capturing short and medium-term dependencies.
- **Considerations:**
    - Might not perform as well as LSTMs on tasks requiring modeling of long-term dependencies.
    - May not be as expressive as LSTMs in some scenarios.


    Absolutely, here are the mathematical expressions for the operations in LSTM and GRU units, explaining their functionality:


# Mathematical Formulation

### LSTM (Long Short-Term Memory)

The LSTM unit consists of various gates and memory components to regulate information flow:

- **Input Gate:** Controls how much new information gets stored in the cell state.
    - Input gate operation:
        \[ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \]
    - Candidate update:
        \[ \tilde{C}_t = \text{tanh}(W_c \cdot [h_{t-1}, x_t] + b_c) \]
    - Cell state update:
        \[ C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t \]
  
- **Forget Gate:** Determines how much of the previous cell state to forget.
    - Forget gate operation:
        \[ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \]

- **Output Gate:** Controls how much of the cell state should be exposed as the output.
    - Output gate operation:
        \[ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \]
    - Hidden state update:
        \[ h_t = o_t \cdot \text{tanh}(C_t) \]

Here, \( i_t, f_t, o_t \) are the input, forget, and output gate vectors respectively. \( \tilde{C}_t \) is the candidate cell state and \( C_t \) is the updated cell state. \( h_t \) is the hidden state/output.

### GRU (Gated Recurrent Unit)

GRU simplifies the LSTM architecture by merging the cell state and hidden state:

- **Update Gate:** Controls how much of the previous state to keep and how much of the new state to consider.
    - Update gate operation:
        \[ z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \]
  
- **Reset Gate:** Decides how much of the previous state to ignore.
    - Reset gate operation:
        \[ r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \]

- **Candidate Update:** Generating a new candidate hidden state.
    - Candidate update:
        \[ \tilde{h}_t = \text{tanh}(W_h \cdot [r_t \cdot h_{t-1}, x_t] + b_h) \]
  
- **Hidden State Update:** Combining the old and new states.
    - Hidden state update:
        \[ h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t \]

Here, \( z_t \) and \( r_t \) are the update and reset gate vectors respectively. \( \tilde{h}_t \) is the candidate hidden state and \( h_t \) is the updated hidden state/output.

These mathematical expressions illustrate how information flows through LSTM and GRU units, enabling control over the retention and utilization of information over different time steps in a sequence.

# Mathematical Formulation

### LSTM (Long Short-Term Memory)
The LSTM unit consists of various gates and memory components to regulate information flow:

**Input Gate:** Controls how much new information gets stored in the cell state.
- Input gate operation:
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
- Candidate update:
  $$\tilde{C}_t = \text{tanh}(W_c \cdot [h_{t-1}, x_t] + b_c)$$
- Cell state update:
  $$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$

**Forget Gate:** Determines how much of the previous cell state to forget.
- Forget gate operation:
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Output Gate:** Controls how much of the cell state should be exposed as the output.
- Output gate operation:
  $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
- Hidden state update:
  $$h_t = o_t \cdot \text{tanh}(C_t)$$

Here, $i_t, f_t, o_t$ are the input, forget, and output gate vectors respectively. $\tilde{C}_t$ is the candidate cell state and $C_t$ is the updated cell state. $h_t$ is the hidden state/output.

### GRU (Gated Recurrent Unit)

GRU simplifies the LSTM architecture by merging the cell state and hidden state:

**Update Gate:** Controls how much of the previous state to keep and how much of the new state to consider.
- Update gate operation:
  $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

**Reset Gate:** Decides how much of the previous state to ignore.
- Reset gate operation:
  $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

**Candidate Update:** Generating a new candidate hidden state.
- Candidate update:
  $$\tilde{h}_t = \text{tanh}(W_h \cdot [r_t \cdot h_{t-1}, x_t] + b_h)$$

**Hidden State Update:** Combining the old and new states.
- Hidden state update:
  $$h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t$$

Here, $z_t$ and $r_t$ are the update and reset gate vectors respectively. $\tilde{h}_t$ is the candidate hidden state and $h_t$ is the updated hidden state/output.

These mathematical expressions illustrate how information flows through LSTM and GRU units, enabling control over the retention and utilization of information over different time steps in a sequence.

