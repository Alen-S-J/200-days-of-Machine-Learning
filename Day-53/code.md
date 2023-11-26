
## Recurrent Neural Networks

 (RNNs) are a popular supervised Deep Learning methodology, alongside Convolutional Neural Networks (CNNs) and Artificial Neural Networks (ANNs). The primary goal of Deep Learning is to simulate the functioning of a brain by using a machine, with each neural network structure representing a part of the brain.

### Text Classification with RNN


- **ANN and Temporal Lobe**: ANN stores data for extended periods, similar to the Temporal Lobe's function. Thus, it's linked with the Temporal Lobe.
- **CNN and Occipital Lobe**: CNNs, used in image classification and Computer Vision, are analogous to the function of the Occipital Lobe in our brain.
- **RNN and Frontal Lobe**: RNNs, employed in time series analysis, exhibit short-term memory capability, akin to the function of the Frontal Lobe.

### Importing Data
Text Classification using the IMDB movie review dataset, consisting of 50k reviews. This dataset is commonly used in text classification for training and testing ML and DL models. Our objective is to predict if a movie review is positive or negative, a binary classification problem. The dataset can be imported directly using TensorFlow or downloaded from Kaggle.

```python
from tensorflow.keras.datasets import imdb
```

### Preprocessing the Data

Movie reviews have varying lengths, so we need uniform data for neural network input. We perform two steps: *embedding* and *padding*. Embedding represents words using vectors, capturing word positions in a vector space based on surrounding words. The `Embedding` layer in Keras requires uniform input, so we pad the data to a defined uniform length.

```python
# Sample sentences before and after padding
sentence = ['Fast cars are good', 'Football is a famous sport', 'Be happy Be positive']
After padding:
[[364, 50, 95, 313, 0, 0, 0, 0, 0, 0],  
 [527, 723, 350, 333, 722, 0, 0, 0, 0, 0],  
 [238, 216, 238, 775, 0, 0, 0, 0, 0, 0]]
```

### Building an RNN model

RNNs operate in three stages: forward pass, loss comparison, and backpropagation for gradient calculation. They are ideal for sequential data due to their short-term memory layers, enabling more accurate predictions of subsequent data.

#### Layers in the model

- **Embedding Layer**:
  ```python
  imdb_model.add(tf.keras.layers.Embedding(word_size, embed_size, input_shape=(x_train.shape[1],)))
  ```
  - `word_size`: Number of distinct words in the dataset.
  - `embed_size`: Number of embedding vectors for each word.

- **LSTM Layer**:
  ```python
  imdb_model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
  ```
  - LSTM addresses the vanishing gradient problem and provides the model with memory for predictions.

#### Vanishing Gradients

Gradient values determine weight adjustments in backpropagation. In the case of vanishing gradients, earlier layers' gradients become significantly smaller, leading to minimal learning and a failure to comprehend contextual data.



#### LSTM's Role

LSTM controls data flow in backpropagation, using gates to manage information flow and preventing excessive weight reduction.



#### Activation Function

The RNN model uses the "Hyperbolic Tangent (tanh)" activation function to maintain values between -1 to 1, ensuring a more uniform weight distribution.


- **Output Layer**:
  ```python
  imdb_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
  ```
  - `Sigmoid` activation shrinks values between 0 to 1, utilizing relevant values in predictions.

### Compiling the Layers

```python
imdb_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
```

Binary classification tasks like this text classification problem use "binary_crossentropy" as the loss function and "accuracy" as the metric. Training is done in batches, with a chosen batch size like 128.

```python
imdb_model.fit(x_train, y_train, epochs=5, batch_size=128)
```

Improvisation can be achieved by adjusting epochs and batch_size while monitoring overfitting. The model achieved an accuracy of approximately 84%.

In summary, this article explores Recurrent Neural Networks, pre-processing steps in RNN structures, addresses the vanishing gradient issue using LSTM, and discusses activation functions in RNN models.

