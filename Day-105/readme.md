

# Word2Vec Implementation with Gensim

## Overview

This repository provides a simple implementation of Word2Vec using the Gensim library in Python. Word2Vec is a popular word embedding technique that represents words as vectors in a continuous vector space. This implementation covers both the Skip-gram and Continuous Bag of Words (CBOW) models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Word2Vec Models](#word2vec-models)
- [Understanding the Code](#understanding-the-code)
- [NLP Theory](#nlp-theory)

## Installation

Make sure to install the required libraries before running the code:

```bash
pip install gensim
pip install nltk
```

## Usage

Run the provided Python script to train Skip-gram and CBOW Word2Vec models:

```bash
python code.py
```

## Word2Vec Models

### Skip-gram Model

The Skip-gram model is designed to predict the context words given a target word. Parameters for the Skip-gram model include:
- Vector size: 100
- Window size: 5
- Training algorithm: Skip-gram (`sg=1`)
- Minimum word count: 1

### CBOW Model

The Continuous Bag of Words (CBOW) model aims to predict the target word based on its context. CBOW model parameters include:
- Vector size: 100
- Window size: 5
- Training algorithm: CBOW (`sg=0`)
- Minimum word count: 1

## Understanding the Code

The provided Python script (`code.py`) tokenizes a sample text and trains both Skip-gram and CBOW Word2Vec models. The resulting word vectors for the word 'word' are printed for both models.

## NLP Theory

### Word Embeddings

Word embeddings are a type of word representation that allows words to be represented as vectors in a continuous vector space. Word2Vec is one such technique that learns distributed representations of words based on their contextual usage.

### Skip-gram vs. CBOW

- **Skip-gram:** Predicts context words given a target word.
- **CBOW:** Predicts the target word based on its context.

Adjusting parameters such as vector size, window size, and minimum word count can significantly impact the quality of word embeddings.

Feel free to experiment with different datasets and hyperparameters for better results!

