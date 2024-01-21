import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential

class GPTModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, n_head, n_layer):
        super(GPTModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=d_model, input_length=1)
        self.lstm_layers = [LSTM(d_model, return_sequences=True) for _ in range(n_layer)]
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, x):
        x = self.embedding(x)
        for layer in self.lstm_layers:
            x = layer(x)
        x = self.dense(x)
        return x

# Hyperparameters
vocab_size = 10000  # adjust based on your dataset
d_model = 256
n_head = 8
n_layer = 4

# Create GPT model
gpt_model = GPTModel(vocab_size, d_model, n_head, n_layer)

# Compile the model
gpt_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
gpt_model.summary()
