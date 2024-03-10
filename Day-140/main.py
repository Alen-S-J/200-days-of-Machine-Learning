import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))

# Define dimensions
input_dim = 784
latent_dim = 2
batch_size = 128
epochs = 20

# Define VAE architecture
input_layer = Input(shape=(input_dim,))
hidden_layer1 = Dense(512, activation='relu')(input_layer)
hidden_layer2 = Dense(256, activation='relu')(hidden_layer1)

# Define mean and log variance layers
z_mean = Dense(latent_dim)(hidden_layer2)
z_log_var = Dense(latent_dim)(hidden_layer2)

# Define sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Define decoder layers
decoder_hidden = Dense(256, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

# Define decoder model
decoder_hidden_out = decoder_hidden(z)
decoder_out_out = decoder_out(decoder_hidden_out)

# Define VAE model
vae = Model(input_layer, decoder_out_out)

# Define VAE loss
reconstruction_loss = mse(input_layer, decoder_out_out) * input_dim
kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile VAE model
vae.compile(optimizer='adam')

# Train VAE model
vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# Generate new images
n = 15  # Number of digits to generate
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
