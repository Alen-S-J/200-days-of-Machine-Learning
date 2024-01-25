# **Implementation of Attention Mechanisms in Natural Language Processing (NLP)**

## Overview

This project demonstrates the practical implementation of attention mechanisms in a Natural Language Processing (NLP) task using the IMDb dataset. The goal is to enhance the performance of sentiment analysis by incorporating attention mechanisms into a deep learning model.

## Installation

1. **Install TensorFlow**: Ensure that you have TensorFlow installed. If not, you can install it using:

    ```bash
    pip install tensorflow
    ```

2. **Install Required Modules**: Install other required modules using:

    ```bash
    pip install numpy pandas matplotlib
    ```

    Additionally, if you are using a Jupyter Notebook, you can install Jupyter with:

    ```bash
    pip install jupyter
    ```

## Dataset

The IMDb dataset consists of movie reviews labeled with sentiment (positive or negative). It is commonly used for sentiment analysis tasks in NLP. The dataset is loaded using TensorFlow's `imdb.load_data()` function.

## Tokenization and Padding

Text sequences are tokenized using the `Tokenizer` class from TensorFlow, and sequences are padded to ensure consistent input lengths for the model. This step is crucial for training deep learning models on sequences of varying lengths.

## Model Architecture

The model architecture includes the following components:

- **Embedding Layer**: Converts integer-encoded words into dense vectors of fixed size.
- **Bidirectional LSTM Layer**: Processes input sequences in both forward and backward directions, capturing context information.
- **Attention Layer**: Incorporates attention mechanisms to focus on relevant parts of the input sequence. This layer enhances the model's ability to weigh different words based on their importance for sentiment analysis.
- **Dense Layer with Sigmoid Activation**: Produces the final output, indicating the sentiment (positive or negative).

## Configuration

- **Vocabulary Size**: 25,000
- **Embedding Dimension**: 100
- **Hidden Dimension**: 256
- **Number of LSTM Layers**: 2
- **Bidirectional LSTM**: Yes
- **Dropout Rate**: 0.5
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

## Training

The model is trained for 5 epochs with a batch size of 64. Training and validation performance are monitored, and the model's accuracy on the IMDb test set is evaluated.

## Code Setup

1. **Clone the Repository**: Clone this GitHub repository to your local machine.

    ```bash
    git clone https://github.com/Alen-S-J/200-days-of-Machine-Learning
    ```

2. **Open Jupyter Notebook**: Navigate to the project directory and open a Jupyter Notebook.

    ```bash
    cd Day-113
    jupyter notebook
    ```

3. **Run the Code**: Open the provided Jupyter Notebook file (`code.ipynb`) and run each cell to execute the code.

## Day 113 of 200 Days of ML

This implementation is part of the "200 Days of ML" challenge, specifically on Day 113. The challenge involves consistent learning and implementation of machine learning concepts over 200 days.