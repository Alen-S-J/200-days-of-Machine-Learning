import tensorflow as tf
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
