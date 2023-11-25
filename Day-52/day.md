### Basic RNN:
The basic RNN consists of recurrent connections that allow information to persist. Given an input sequence \(x = (x_1, x_2, ..., x_T)\), the hidden state \(h_t\) at time step \(t\) is computed as follows:

\[
h_t = \text{tanh}(W_{hx}x_t + W_{hh}h_{t-1} + b_h)
\]

Where:
- \(W_{hx}\) is the weight matrix for the input.
- \(W_{hh}\) is the weight matrix for the recurrent connections.
- \(b_h\) is the bias term.
- \(\text{tanh}\) is the hyperbolic tangent activation function.

### Elman RNN:
Elman RNN is a type of basic RNN where the hidden state at time \(t\) depends only on the current input and the previous hidden state \(h_{t-1}\). Its equation is the same as the basic RNN's hidden state formula.

```Python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Your data preprocessing and vectorization steps here

model = Sequential()
model.add(SimpleRNN(128, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, batch_size=128, epochs=30)

```
### Jordan RNN:
In Jordan RNN, the output at time \(t\) is based on the hidden state at the same time step. It's described as:
```Python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Your data preprocessing and vectorization steps here

model = Sequential()
model.add(SimpleRNN(128, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(TimeDistributed(Dense(len(chars), activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, batch_size=128, epochs=30)

```
\[
y_t = \text{softmax}(W_{yh}h_t + b_y)
\]

Where:
- \(W_{yh}\) is the weight matrix for the output connections.
- \(b_y\) is the output bias term.

### Bidirectional RNN:
Bidirectional RNNs process the input sequence in both forward and backward directions to capture information from past and future contexts. The hidden state at time \(t\) is the concatenation of the forward and backward hidden states.
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Your data preprocessing and vectorization steps here

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, batch_size=128, epochs=30)
```
### Applications of RNNs:
- **Language Modeling:** Predicting the probability distribution over sequences of words.
- **Music Generation:** Creating new music based on learned patterns from existing compositions.
- **Video Analysis:** Action recognition, video captioning, and frame prediction.

### Real-world Examples:
- **Language Translation:** Google Translate uses sequence-to-sequence RNNs for translation tasks.
- **Speech Recognition:** Applications like Siri or Google Assistant utilize RNNs for converting speech to text.
- **Stock Market Prediction:** RNNs can analyze temporal data to predict stock prices.

These architectures and applications showcase the versatility of RNNs in various domains, leveraging their ability to model sequential data effectively.


here is the sample code
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Example text data
text = "hello world, how are you?"

# Creating character mappings
chars = sorted(list(set(text)))
char_indices = {char: i for i, char in enumerate(chars)}
indices_char = {i: char for i, char in enumerate(chars)}

# Preprocessing the data
maxlen = 40  # Length of sequences
step = 3  # Step size for creating sequences
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])

# Vectorizing the data
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Building the RNN model
model = Sequential()
model.add(SimpleRNN(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Training the model
model.fit(X, y, batch_size=128, epochs=30)

# Function to sample the next character based on the model's predictions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generating text using the trained model
start_index = 0
generated_text = ""
sentence = text[start_index : start_index + maxlen]
generated_text += sentence

for i in range(400):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.0

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, temperature=0.5)
    next_char = indices_char[next_index]

    generated_text += next_char
    sentence = sentence[1:] + next_char

print(generated_text)

```
