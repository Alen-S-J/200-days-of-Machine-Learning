# A basic Generative Adversarial Network (GAN) using TensorFlow's Keras API to generate handwritten digits resembling MNIST dataset.

1. **Data Loading and Preprocessing:**
    - Loads the MNIST dataset and normalizes pixel values to range [-1, 1].

2. **Model Setup:**
    - **Generator:**
        - A simple fully connected neural network that takes random noise (`latent_dim`) as input and generates an image of size 28x28x1 (MNIST image dimensions).
        - Uses `LeakyReLU` activations and `tanh` activation in the last layer to produce values within the desired range.

    - **Discriminator:**
        - A simple feedforward neural network acting as a binary classifier to distinguish between real and generated images.
        - Uses `Flatten` to flatten the input image and `LeakyReLU` activations.
    
3. **Model Compilation:**
    - Compiles the discriminator using binary cross-entropy loss and Adam optimizer.
    - Builds the GAN model by connecting the generator and discriminator in a sequential manner, freezing the discriminator during adversarial network training.
    - Compiles the GAN model using binary cross-entropy loss and Adam optimizer.

4. **Training Loop:**
    - Alternates between training the discriminator and the adversarial network (generator).
    - In each epoch:
        - Randomly selects real images from the MNIST dataset.
        - Generates fake images using random noise fed into the generator.
        - Trains the discriminator on both real and fake images, computing the discriminator loss.
        - Trains the generator through the adversarial network, aiming to fool the discriminator (generates real-like images), computing the generator loss.
        - Prints and displays progress every 100 epochs and saves generated images every 1000 epochs for visualization.

5. **Visualization:**
    - Displays a grid of generated images every 1000 epochs, showcasing the generator's progress in generating realistic handwritten digits.

6. **Parameters:**
    - `batch_size`: Number of images in each batch.
    - `epochs`: Total number of training epochs.

This code demonstrates a basic GAN setup for generating MNIST-like digits, training the generator to produce realistic handwritten digits and the discriminator to distinguish between real and generated images.
