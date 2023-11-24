import numpy as np
import tensorflow as tf

# Define the input text
text = "hello world!"

# Creating a mapping of unique characters to integers
chars = sorted(list(set(text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# Preprocessing the text data
seq_length = 100  # Define sequence length
dataX = []
dataY = []
for i in range(0, len(text) - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

# Reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# Normalize input
X = X / float(len(chars))
# One-hot encode the output variable
y = tf.keras.utils.to_categorical(dataY)

# Define the RNN model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model
model.fit(X, y, epochs=20, batch_size=128)

# Generate text
start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# Generate characters
for i in range(100):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nDone.")
