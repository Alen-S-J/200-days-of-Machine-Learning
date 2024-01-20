import numpy as np
import tensorflow as tf

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, usually called 'causal language model'

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

class TextRNN(tf.keras.Model):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = tf.keras.layers.SimpleRNN(n_hidden)
        self.W = tf.keras.layers.Dense(n_class, use_bias=False)
        self.b = tf.Variable(tf.ones([n_class]))

    def call(self, X):
        hidden = self.rnn(X)
        model = self.W(hidden) + self.b
        return model

if __name__ == '__main__':
    n_step = 2  # number of cells (= number of steps)
    n_hidden = 5  # number of hidden units in one cell

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)

    model = TextRNN()

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    input_batch, target_batch = make_batch()
    input_batch = np.array(input_batch, dtype=np.float32)
    target_batch = np.array(target_batch, dtype=np.int32)

    # Training
    for epoch in range(5000):
        with tf.GradientTape() as tape:
            output = model(input_batch)
            loss = criterion(target_batch, output)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    input = [sen.split()[:2] for sen in sentences]

    # Predict
    predict = model(np.eye(n_class)[input]).numpy().argmax(axis=1)
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n] for n in predict])
