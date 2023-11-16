

### 1. Fully Connected Layers

#### Theoretical Aspect with Mathematical Expressions:
- **Theoretical Aspect:** Fully connected layers, also known as dense layers, connect every neuron in one layer to every neuron in the next layer. In a neural network, these layers compute a weighted sum of the inputs and apply an activation function.
- **Mathematical Expression:** If \(x\) represents the input vector to a fully connected layer, \(W\) represents the weight matrix, and \(b\) is the bias vector, the output of a fully connected layer is calculated as: 

  \[ \text{Output} = \text{Activation}(Wx + b) \]

#### Sample Code Snippet (using TensorFlow):

```python
import tensorflow as tf

# Define input shape
input_shape = (784,)  # Example input shape for MNIST data

# Create a fully connected layer with 128 neurons and ReLU activation
fully_connected_layer = tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape)

# Apply the layer to input data
x = tf.constant(tf.random.normal(input_shape))
output = fully_connected_layer(x)
print("Output shape:", output.shape)
```

### 2. CNN Architectures

#### Theoretical Aspect with Mathematical Expressions:
- **LeNet:** LeNet used convolutional layers followed by fully connected layers. It employed convolutional operations, subsampling (pooling), and the non-linear activation function, typically a sigmoid or tanh.
- **Mathematical Expression:** For example, in LeNet, a convolution operation with a filter \(W\) and bias \(b\) on an input feature map \(x\) is calculated as:

  \[ \text{Convolution} = \text{Activation}(\sum (W * x) + b) \]

#### Sample Code Snippet (using TensorFlow):

```python
# LeNet architecture using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1)),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='tanh'),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='tanh'),
    tf.keras.layers.Dense(84, activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

### 3. Transfer Learning

#### Theoretical Aspect with Mathematical Expressions:
- **Transfer Learning:** Involves leveraging pre-trained models and adapting them to new tasks. Layers in a pre-trained CNN can be frozen or fine-tuned, and additional layers can be added.
- **Mathematical Expression:** During fine-tuning, the loss function might be a combination of the pre-trained model's loss (\(L_{pretrained}\)) and the new task's loss (\(L_{new}\)):

  \[ L_{total} = \alpha * L_{pretrained} + \beta * L_{new} \]

#### Sample Code Snippet (using TensorFlow for transfer learning with VGG16):

```python
# Example: Transfer learning using VGG16 pre-trained model in TensorFlow/Keras
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Freeze layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create a new model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

These code snippets provide basic implementations using TensorFlow/Keras for the discussed theoretical aspects. They can serve as starting points for experimenting with fully connected layers, CNN architectures like LeNet, and transfer learning using pre-trained models like VGG16.