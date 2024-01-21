import numpy as np
import tensorflow as tf

def make_batch():
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return tf.constant(input_batch, dtype=tf.int64), tf.constant(output_batch, dtype=tf.int64), tf.constant(target_batch, dtype=tf.int64)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return tf.constant(sinusoid_table, dtype=tf.float32)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = tf.shape(seq_q)[0], tf.shape(seq_q)[1]
    batch_size, len_k = tf.shape(seq_k)[0], tf.shape(seq_k)[1]
    pad_attn_mask = tf.cast(tf.math.equal(seq_k, 0), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    attn_shape = [tf.shape(seq)[0], tf.shape(seq)[1], tf.shape(seq)[1]]
    subsequent_mask = tf.linalg.band_part(tf.ones(attn_shape), -1, 0)
    return tf.cast(subsequent_mask, dtype=tf.uint8)

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def call(self, Q, K, V, attn_mask):
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        scores = tf.where(tf.equal(attn_mask, 0), tf.fill(tf.shape(scores), -1e9), scores)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(attn_weights, V)
        return context, attn_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = tf.keras.layers.Dense(d_model)
        self.W_K = tf.keras.layers.Dense(d_model)
        self.W_V = tf.keras.layers.Dense(d_model)
        self.linear = tf.keras.layers.Dense(d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

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
        context, attn_weights = ScaledDotProductAttention()(Q, K, V, attn_mask)

        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, n_heads * d_k))
        output = self.linear(context)

        return self.layer_norm(output + Q), attn_weights

class PoswiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        residual = inputs
        output = self.conv1(inputs)
        output = self.conv2(output)
        return self.layer_norm(output + residual)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def call(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def call(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=src_len + 1, output_dim=d_model, weights=[get_sinusoid_encoding_table(src_len + 1, d_model)], trainable=False)
        self.layers = [EncoderLayer() for _ in range(n_layers)]

    def call(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(tf.range(1, src_len + 1))
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outputs, _ = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = tf.keras.layers.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=tgt_len + 1, output_dim=d_model, weights=[get_sinusoid_encoding_table(tgt_len + 1, d_model)], trainable=False)
        self.layers = [DecoderLayer() for _ in range(n_layers)]

    def call(self, dec_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(tf.range(5, tgt_len + 5))
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = tf.cast(tf.math.greater(tf.math.add(dec_self_attn_pad_mask, dec_self_attn_subsequent_mask), 0), dtype=tf.float32)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        for layer in self.layers:
            dec_outputs, _, _ = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)

        return dec_outputs

class Transformer(tf.keras.Model):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = tf.keras.layers.Dense(tgt_vocab_size, use_bias=False)

    def call(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return tf.reshape(dec_logits, [-1, tgt_vocab_size])

def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs = model.encoder(enc_input)
    dec_input = tf.zeros((1, tgt_len), dtype=tf.int64)
    next_symbol = start_symbol
    for i in range(tgt_len):
        dec_input[0, i] = next_symbol
        dec_outputs = model.decoder(dec_input, enc_outputs)
        projected = model.projection(dec_outputs)
        next_word = tf.argmax(projected[0, i]).numpy()
        next_symbol = next_word
    return dec_input

if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5
    tgt_len = 5

    d_model = 512
    d_ff = 2048
    d_k = d_v = 64
    n_layers = 6
    n_heads = 8

    model = Transformer()

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch()

    for epoch in range(20):
        with tf.GradientTape() as tape:
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(target_batch, outputs)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=tgt_vocab["S"])
    predict = model(enc_inputs, greedy_dec_input)
    predict = tf.argmax(predict, axis=-1).numpy()
    print(sentences[0], '->', [number_dict[n] for n in predict[0]])

