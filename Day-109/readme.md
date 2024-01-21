
# Natural Language Processing Models Overview

This repository contains implementations and explanations for popular natural language processing (NLP) models, including BERT, Transformers, and GPT.

## BERT (Bidirectional Encoder Representations from Transformers)

### Overview
BERT is a pre-trained transformer-based language model developed by Google. It uses a bidirectional context to better understand the relationships between words in a sentence.

### Code Explanation
The provided code demonstrates a simplified version of BERT for a masked language model (MLM) task using PyTorch. It includes tokenization, creating a masked input, and training the model.

```python
# import tensorflow as tf
import numpy as np

class Embedding(tf.keras.layers.Layer):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = tf.keras.layers.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = tf.keras.layers.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = tf.keras.layers.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, seg):
        seq_len = tf.shape(x)[1]
        pos = tf.range(seq_len, dtype=tf.int32)
        pos = tf.broadcast_to(pos, tf.shape(x))  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def call(self, Q, K, V, attn_mask):
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        scores = tf.where(attn_mask, tf.fill(tf.shape(scores), -1e9), scores)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(attn_weights, V)
        return context, attn_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = tf.keras.layers.Dense(d_model)
        self.W_K = tf.keras.layers.Dense(d_model)
        self.W_V = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, n_heads, d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, attn_mask):
        batch_size = tf.shape(Q)[0]

        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        attn_mask = attn_mask[:, tf.newaxis, tf.newaxis, :]
        context, _ = ScaledDotProductAttention()(Q, K, V, attn_mask)

        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, n_heads * d_k))
        output = tf.keras.layers.Dense(d_model)(context)

        return tf.keras.layers.LayerNormalization()(output + Q)

class PoswiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1, activation='relu')
        self.fc2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        residual = x
        output = self.fc1(x)
        output = self.fc2(output)
        return self.norm(output + residual)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def call(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class BERT(tf.keras.Model):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = [EncoderLayer() for _ in range(n_layers)]
        self.fc = tf.keras.layers.Dense(d_model)
        self.activ1 = tf.keras.layers.Activation('tanh')
        self.linear = tf.keras.layers.Dense(d_model)
        self.activ2 = tf.keras.layers.Activation(gelu)
        self.norm = tf.keras.layers.LayerNormalization()
        self.classifier = tf.keras.layers.Dense(2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weights[0]
        n_vocab, n_dim = embed_weight.shape
        self.decoder = tf.keras.layers.Dense(n_vocab, use_bias=False)
        self.decoder.build((None, n_dim))
        self.decoder.set_weights([embed_weight.numpy().T])
        self.decoder_bias = self.add_weight("decoder_bias", shape=[n_vocab], initializer="zeros")

    def call(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)

        h_pooled = self.activ1(self.fc(output[:, 0]))
        logits_clsf = self.classifier(h_pooled)

        masked_pos = tf.expand_dims(masked_pos, axis=-1)
        h_masked = tf.gather(output, masked_pos, batch_dims=1)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, logits_clsf

# Hyperparameters
maxlen = 30
batch_size = 6
max_pred = 5
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768 * 4
d_k = d_v = 64
n_segments = 2

# Create BERT model
model = BERT()

# Sample input
input_ids = tf.constant([[1, 2, 3, 4, 5, 6]])
segment_ids = tf.constant([[0, 0, 0, 1, 1, 1]])
masked_pos = tf.constant([[3, 8, 15, 22, 27]])

# Run a forward pass
logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)

# Print the shapes of the output
print("Logits LM shape:", logits_lm.shape)
print("Logits CLSF shape:", logits_clsf.shape)
Insert BERT code snippet here
```

## Transformers

### Overview
Transformers are a type of neural network architecture that uses self-attention mechanisms to capture contextual information from input sequences. They have revolutionized NLP tasks by outperforming traditional recurrent neural networks.

### Code Explanation
The included code showcases a basic implementation of a transformer model using TensorFlow and Keras. It focuses on the self-attention mechanism, positional encoding, and multi-head attention.

```python
import tensorflow as tf
import numpy as np

# Define parameters
src_vocab_size = 5
tgt_vocab_size = 7
src_len = 5
tgt_len = 5
d_model = 512
d_ff = 2048
d_k = d_v = 64
n_layers = 6
n_heads = 8

# Define positional encoding function
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return tf.constant(sinusoid_table, dtype=tf.float32)

# Define attention functions
def scaled_dot_product_attention(Q, K, V, mask):
    d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
    scores = tf.where(tf.equal(mask, 0), tf.fill(tf.shape(scores), -1e9), scores)
    attn_weights = tf.nn.softmax(scores, axis=-1)
    context = tf.matmul(attn_weights, V)
    return context, attn_weights

# Define multi-head attention class
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads
        self.W_Q = tf.keras.layers.Dense(d_model)
        self.W_K = tf.keras.layers.Dense(d_model)
        self.W_V = tf.keras.layers.Dense(d_model)
        self.linear = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]

        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        attn_mask = tf.expand_dims(mask, axis=1)
        context, attn_weights = scaled_dot_product_attention(Q, K, V, attn_mask)

        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))
        output = self.linear(context)

        return output

# Define positional feedforward network class
class PoswiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)

    def call(self, inputs):
        residual = inputs
        output = self.conv1(inputs)
        output = self.conv2(output)
        return output + residual

# Define encoder layer class
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def call(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

# Define encoder class
class Encoder(tf.keras.layers.Layer):
    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, d_ff):
        super(Encoder, self).__init__()
        self.src_emb = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=src_len + 1, output_dim=d_model, weights=[get_sinusoid_encoding_table(src_len + 1, d_model)], trainable=False)
        self.layers = [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]

    def call(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(tf.range(1, src_len + 1))
        enc_self_attn_mask = tf.cast(tf.math.equal(enc_inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs

# Define transformer model
class Transformer(tf.keras.Model):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_len, tgt_len, d_model, n_layers, n_heads, d_ff):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff)
        self.projection = tf.keras.layers.Dense(tgt_vocab_size, activation='softmax')

    def call(self, enc_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_logits = self.projection(enc_outputs)
        return dec_logits

# Create an instance of the transformer model
model = Transformer(src_vocab_size, tgt_vocab_size, src_len, tgt_len, d_model, n_layers, n_heads, d_ff)

# Define loss function and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
for epoch in range(20):
    with tf.GradientTape() as tape:
        logits = model(enc_inputs)
        loss = loss_object(target_batch, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Test
predictions = model(enc_inputs)
predicted_ids = tf.argmax(predictions, axis=-1)
print(sentences[0], '->', [number_dict[n.numpy()] for n in predicted_ids[0]])

```

## GPT (Generative Pre-trained Transformer)

### Overview
GPT is a series of transformer-based language models developed by OpenAI. It is designed for various NLP tasks and excels at generating coherent and contextually relevant text.

### Code Explanation
The provided code demonstrates a simple GPT-like model using LSTM layers in TensorFlow and Keras. It generates text based on a given prompt.

```python
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

```

This template assumes that you have separate Python scripts for each model (e.g., `bert_model.py`, `transformers_model.py`, `gpt_model.py`). Adjust the code snippets accordingly based on your actual implementation.

## Conclusion

Understanding the theoretical foundations of Tranformers,BERT and GPT  in the context of NLP is crucial for designing effective models. The provided code snippets offer practical implementations, showcasing how these architectures can be applied to various NLP tasks. Experimenting with these models and adapting them to specific use cases is key to achieving optimal performance.