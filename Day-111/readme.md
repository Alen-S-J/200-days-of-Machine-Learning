
# Self-Attention Mechanism Implementation in TensorFlow

This repository contains an implementation of a self-attention mechanism using TensorFlow. The self-attention mechanism is a crucial component in transformer models, widely used in natural language processing (NLP) tasks.

## Table of Contents

1. [Introduction](#introduction)
2. [Self-Attention Concept](#self-attention-concept)
3. [Code Explanation](#code-explanation)
    - [SelfAttention Class](#selfattention-class)
    - [Example Usage](#example-usage)
4. [Mathematical Expression](#mathematical-expression)
5. [License](#license)

## Introduction

The self-attention mechanism allows a model to focus on different parts of the input sequence when making predictions, capturing long-range dependencies effectively. This implementation provides a basic example of self-attention within the context of a custom TensorFlow layer.
![image](https://miro.medium.com/max/975/1*vrSX_Ku3EmGPyqF_E-2_Vg.png)
## Self-Attention Concept

Self-attention is a mechanism that enables a model to weigh the importance of different words in a sequence differently when processing each word. This allows the model to consider contextual information effectively, improving its ability to capture dependencies.

### Mathematical Representation of Self-Attention

Mathematically, self-attention can be expressed as follows:

$Attention(Q,K,V)=softmax((QK^T)/sqrt(d_k))*V$

Here, $d_k$ is the dimension of the Key vectors, and $(QK^T)$ represents the dot product of $Q$ and $K$. The division by $\sqrt{d_k}$ is the scaling factor, and the softmax function normalizes the scores.


## Code Explanation

### SelfAttention Class

The `SelfAttention` class is a custom TensorFlow layer implementing a basic self-attention mechanism. Here's a breakdown:

- **Initialization**: Initializes the layer with the embedding size and the number of attention heads.

- **Call Method**: Takes input values, keys, query, and an optional mask. It applies linear transformations to obtain queries, keys, and values. The attention scores are computed using the dot product of queries and keys. A mask can be applied to avoid attending to certain positions. The final output is obtained by weighting the values with the attention scores.

### Example Usage

An instance of the `SelfAttention` class is created with a specified embedding size and number of attention heads. Random tensors are used for input values, keys, and query, and an optional mask is provided. The output shape is printed.

## Mathematical Expression

For a detailed explanation of the mathematical expressions and the theory behind self-attention, please refer to the [Mathematical Expression](#mathematical-expression) section.

---

Feel free to use and modify this code for your specific needs. If you have any questions or suggestions, please reach out!

