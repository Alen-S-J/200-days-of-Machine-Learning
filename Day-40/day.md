# Optimizing Convolutional Neural Networks: A Study of Performance Across Different Optimizers

1. **Data Preprocessing**:
   - The code starts by loading the CIFAR-10 dataset, which contains images of 10 different classes. Training and test sets are loaded and stored in `(train_images, train_labels)` and `(test_images, test_labels)` variables.
   - The images are then normalized by dividing each pixel value by 255, ensuring they are in the range [0, 1].
   - The labels are one-hot encoded using the `to_categorical` function.

2. **Defining Optimizers**:
   - A dictionary named `optimizers` is created, containing different optimization algorithms to experiment with. These optimizers include Stochastic Gradient Descent (SGD), Adam, RMSprop, and Adagrad.
   - Each optimizer is created as an instance of the optimizer class from TensorFlow/Keras.

3. **Model Architecture and Training**:
   - The code enters a loop where it iterates over each optimizer in the `optimizers` dictionary.
   - Inside the loop, a convolutional neural network (CNN) model is defined using the Keras Sequential API.
   - The CNN consists of convolutional layers with ReLU activation functions, max-pooling layers, and fully connected layers.
   - The model is compiled with the current optimizer, categorical cross-entropy as the loss function (commonly used for multi-class classification), and accuracy as the metric.
   - The model is trained for 10 epochs using the training data, with a batch size of 128. The validation accuracy for each epoch is recorded.

4. **Heatmap Generation**:
   - The code generates a heatmap to visualize the performance of different optimizers. The heatmap shows the validation accuracy over the 10 epochs for each optimizer.
   - The `seaborn` library is used to create the heatmap, and it's annotated with accuracy values. The x-axis represents the epochs, and the y-axis lists the optimizers.
   - The heatmap is displayed using `matplotlib`.

5. **Random Test Sample Prediction**:
   - The code selects `num_samples` (5 in this case) random test images and displays them along with their predicted and true labels.
   - For each selected image, the model predicts its label, and both the predicted and true labels are displayed alongside the image.
   - This provides a visual inspection of the model's performance on a few test samples.

Overall, the code is an example of experimenting with different optimization algorithms (SGD, Adam, RMSprop, Adagrad) to train a CNN for image classification. It also includes visualizations to compare and evaluate the performance of these optimizers and allows for a visual check of the model's predictions on random test samples.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
```

- In this section, we import the necessary libraries and modules for the code. This includes TensorFlow for building and training neural networks, Keras for creating the model, and other libraries for data manipulation and visualization.

```python
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

- The code loads and preprocesses the CIFAR-10 dataset, a well-known dataset for image classification. It splits the dataset into training and test sets, converts the image data to floating-point values between 0 and 1 by dividing by 255, and one-hot encodes the labels.

```python
# Define a list of optimizers to experiment with
optimizers = {
    'SGD': tf.keras.optimizers.SGD(),
    'Adam': tf.keras.optimizers.Adam(),
    'RMSprop': tf.keras.optimizers.RMSprop(),
    'Adagrad': tf.keras.optimizers.Adagrad(),
}
```

- A dictionary named `optimizers` is defined, which contains different optimization algorithms to experiment with. These optimizers are SGD, Adam, RMSprop, and Adagrad. For each optimizer, an instance of the optimizer is created.

```python
optimizer_names = list(optimizers.keys())
accuracy_heatmap = np.zeros((len(optimizer_names), 10))
```

- An empty array named `accuracy_heatmap` is created to store accuracy values for different optimizers over 10 epochs. The `optimizer_names` list is created to store the names of the optimizers.

```python
for i, (name, optimizer) in enumerate(optimizers.items()):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
```

- This is the main loop where the code iterates through each optimizer. Inside the loop, a new neural network model is created. The model is a Sequential model with several Conv2D layers, MaxPooling layers, Flatten, and Dense layers. The architecture is for image classification.

```python
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels))

    accuracy_heatmap[i, :] = history.history['val_accuracy']
```

- The model is compiled using the current optimizer, with categorical cross-entropy as the loss function and accuracy as the metric. It's then trained for 10 epochs using the training data, and the validation accuracy is recorded for each epoch. This accuracy is stored in the `accuracy_heatmap` array.

```python
import random
```

- The `random` module is imported for selecting random samples later.

```python
plt.figure(figsize=(10, 6))
sns.heatmap(accuracy_heatmap, annot=True, xticklabels=range(1, 11), yticklabels=optimizer_names, cmap="YlGnBu")
plt.xlabel('Epochs')
plt.ylabel('Optimizer')
plt.title('Validation Accuracy Heatmap by Optimizer')
plt.show()
```

- A heatmap is created using the `seaborn` library, displaying the validation accuracy for each optimizer over the 10 epochs. The heatmap is annotated with accuracy values and labeled on both axes. It is then displayed using `matplotlib`.

```python
num_samples = 5
random_indices = [random.randint(0, len(test_images) - 1) for _ in range(num_samples)]

plt.figure(figsize=(15, 3))
for j, index in enumerate(random_indices):
        plt.subplot(1, num_samples, j + 1)
        plt.imshow(test_images[index])
        predicted_label = model.predict(np.expand_dims(test_images[index], axis=0)).argmax()
        true_label = test_labels[index].argmax()
        plt.title(f'Pred: {predicted_label}, True: {true_label}')
        plt.axis('off')
plt.show()
```

- Finally, this section selects `num_samples` random test images and displays them along with their predicted and true labels. The images are shown using `matplotlib`. The predictions are made using the trained model, and the labels are displayed.