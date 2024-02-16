import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Generate dummy data
num_samples = 1000
input_shape = (28, 28, 1)
num_classes = 10

x_train = np.random.random((num_samples, *input_shape))
y_train = np.random.randint(num_classes, size=(num_samples))

# Preprocess the data
x_train = x_train.astype('float32')
x_train /= 255

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
