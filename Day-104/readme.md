# Word Embeddings in Natural Language Processing

## Introduction

This repository contains a simple Python script that demonstrates the concept of word embeddings and their significance in Natural Language Processing (NLP). The code uses the spaCy library to load a pre-trained English language model ('en_core_web_sm') and showcases the vector representation of words in a given sentence.

## Prerequisites

Before running the code, ensure you have Python installed on your system. Additionally, install the spaCy library and download the English language model using the following commands:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Code Explanation

The provided Python script, `code.py`, follows these key steps:

1. **Import spaCy**: The spaCy library is imported to leverage its NLP capabilities.

    ```python
    import spacy
    ```

2. **Load the Language Model**: The English language model 'en_core_web_sm' is loaded using spaCy.

    ```python
    nlp = spacy.load("en_core_web_sm")
    ```

3. **Process Sentences**: Two example sentences are processed using spaCy to obtain their word vectors.

    ```python
    sentence1 = "Word embeddings capture the semantic meaning of words."
    sentence2 = "Vectors represent words in a multidimensional space."

    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    ```

4. **Access Word Vectors**: The word vectors for each sentence are obtained by accessing the `vector` attribute of the processed spaCy documents.

    ```python
    vector1 = doc1.vector
    vector2 = doc2.vector
    ```

5. **Print Word Vectors**: The script prints the word vectors representing the semantic meaning of words in each sentence.

    ```python
    print("Word vector for sentence 1:", vector1)
    print("Word vector for sentence 2:", vector2)
    ```

## NLP Theory Connection

### Word Embeddings

Word embeddings are a type of word representation in NLP that captures the semantic meaning of words in a continuous vector space. Instead of representing words as discrete symbols, word embeddings map words to vectors, allowing algorithms to understand the relationships and similarities between words.

### spaCy

spaCy is a popular open-source library for advanced NLP in Python. It provides pre-trained models and tools for various NLP tasks, including tokenization, part-of-speech tagging, named entity recognition, and word embeddings.

### Vector Representation

In NLP, vector representation refers to the conversion of words into numerical vectors, where each dimension of the vector captures a different aspect of the word's meaning. The distances and directions between these vectors reflect the semantic relationships between words.

