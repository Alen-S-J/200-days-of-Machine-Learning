import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.keys = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.queries = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.fc_out = tf.keras.layers.Dense(embed_size)

    def call(self, values, keys, query, mask):
        # Get number of training examples
        N = tf.shape(query)[0]

        value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(query)[1]

        # Split the embedding into self.heads different pieces
        values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
        queries = tf.reshape(query, (N, query_len, self.heads, self.head_dim))

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = tf.einsum("nqhd,nkhd->nhqk", queries, keys)

        if mask is not None:
            energy = tf.where(mask == 0, tf.constant(float("-1e20")), energy)

        attention = tf.nn.softmax(energy / tf.math.sqrt(tf.cast(self.embed_size, dtype=tf.float32)), axis=3)

        out = tf.einsum("nhql,nlhd->nqhd", attention, values)
        out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))

        out = self.fc_out(out)
        return out

# Example Usage
embed_size = 256
heads = 8
attention = SelfAttention(embed_size, heads)

# Assuming you have input tensors values, keys, and query
# (you can replace these with your actual input tensors)
values = tf.random.normal((2, 10, embed_size))  # Batch size=2, sequence length=10
keys = tf.random.normal((2, 10, embed_size))
query = tf.random.normal((2, 1, embed_size))  # For simplicity, using a single query

mask = tf.ones((2, 1, 10))  # Optional: Mask to apply attention only on certain positions

output = attention(values, keys, query, mask)
print(output.shape)
