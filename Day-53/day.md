### Vanishing/Exploding Gradient Problem:
- **Theory:** During backpropagation in training, gradients in recurrent neural networks (RNNs) can either become extremely small (vanishing) or extremely large (exploding) as they propagate through time steps.
  
- **Mathematical Presentation:**
  - In each time step \(t\), the hidden state \(h_t\) in a simple RNN is calculated as:
    \[ h_t = \text{Activation}(W_{hh}h_{t-1} + W_{xh}x_t + b_h) \]
    Where:
    - \(W_{hh}\) is the weight matrix for the hidden state,
    - \(W_{xh}\) is the weight matrix for the input,
    - \(b_h\) is the bias for the hidden state,
    - \(x_t\) is the input at time \(t\).
    
- **Vanishing Gradient Example:**
  - If the weight matrix \(W_{hh}\) has eigenvalues less than 1 (i.e., values that decrease as they're multiplied), the gradient can exponentially shrink as it goes back through time steps, making learning long-term dependencies challenging.

- **Exploding Gradient Example:**
  - Conversely, if the weight matrix \(W_{hh}\) has eigenvalues larger than 1, the gradient can grow exponentially, causing instability and making learning difficult.

### Difficulty in Learning Long-Range Dependencies:
- **Theory:** Vanilla RNNs struggle to capture long-term dependencies due to the vanishing/exploding gradient problem. Information from earlier time steps diminishes or becomes insignificant as it propagates through time, hindering the model's ability to remember long-term information.
  
- **Mathematical Representation:**
  - The ability of RNNs to capture long-term dependencies is affected by the way gradients are calculated and backpropagated through time. The update of weights based on distant time steps becomes weaker due to vanishing gradients.

### Practical Examples Showcasing Failure:
- **Theory:** Tasks requiring the retention of information from distant past time steps expose the limitations of vanilla RNNs. For instance, language modeling where understanding the context from paragraphs ago is necessary.
  
- **Mathematical Representation:** 
  - While the mathematical representation here may not be explicit, in practical examples, you might observe that vanilla RNNs struggle to remember information from earlier steps, leading to poor performance in tasks requiring long-term memory, such as generating coherent and contextually accurate text over extended sequences.

### User Application of These Theories:
The challenges posed by vanishing/exploding gradients and the limitations in capturing long-term dependencies impact various domains:
- **Natural Language Processing (NLP):** Difficulty in capturing dependencies affects tasks like language modeling and machine translation.
- **Time Series Prediction:** Forecasting in finance and understanding patterns over time can be hindered by RNN limitations.
- **Speech Recognition:** Processing longer audio sequences and understanding context in speech presents challenges.
