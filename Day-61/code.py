import numpy as np
import matplotlib.pyplot as plt

# Generator function to create synthetic data
def generate_fake_data(n_samples):
    return np.random.rand(n_samples) * 2 - 1  # Generating 'n_samples' random numbers between -1 and 1

# Discriminator function to distinguish real vs. fake data
def discriminate(data):
    return np.mean(data)

# Train the GAN
def train_gan(n_epochs, n_batch):
    for epoch in range(n_epochs):
        # Generate fake data
        fake_data = generate_fake_data(n_batch)

        # Train discriminator on real data
        real_data = generate_fake_data(n_batch)
        d_loss_real = discriminate(real_data)

        # Train discriminator on fake data
        d_loss_fake = discriminate(fake_data)

        # Update discriminator (for simplicity, here we're not updating weights explicitly)
        d_loss = d_loss_real - d_loss_fake

        # Train generator
        for _ in range(n_batch):
            # Generate new fake data
            fake_data = generate_fake_data(n_batch)
            g_loss = discriminate(fake_data)

            # Update generator (for simplicity, here we're not updating weights explicitly)

        # Display progress (optional)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} / Discriminator Loss: {d_loss} / Generator Loss: {g_loss}")
            visualize_generated_data()  # Function to visualize generated data

# Visualize generated data
def visualize_generated_data():
    fake_data = generate_fake_data(1000)  # Generate a larger set of fake data for visualization
    plt.hist(fake_data, bins=50, alpha=0.5, label='Generated Data')
    plt.legend()
    plt.title('Generated Data Distribution')
    plt.show()

# Set hyperparameters
epochs = 1000
batch_size = 64

# Train the GAN
train_gan(epochs, batch_size)
