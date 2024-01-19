# Language Model using NLTK

## Introduction
This repository contains a simple example of a language model implemented using Python and the Natural Language Toolkit (NLTK) library. The language model is based on bigrams and uses a probabilistic approach to generate text.

## Setup
Before running the code, make sure to install the necessary dependencies. You can do this by running:

```bash
pip install nltk
```

Additionally, download the Reuters corpus using the NLTK downloader:

```python
import nltk
nltk.download('reuters')
```

## Understanding the Code

### 1. Tokenization
The code starts by tokenizing the Reuters corpus, converting the words to lowercase for consistency.

```python
corpus = reuters.words()
tokens = [word.lower() for word in corpus]
```

### 2. Bigrams and Frequency Distribution
The script creates bigrams from the tokenized corpus and calculates the frequency distribution of these bigrams.

```python
bi_grams = list(bigrams(tokens))
bi_gram_freq = FreqDist(bi_grams)
```

### 3. Conditional Frequency Distribution and Probabilities
A conditional frequency distribution (CFD) is created to understand the frequency of word pairs. The ConditionalProbDist is then used to calculate probabilities based on Maximum Likelihood Estimation (MLE).

```python
cfd = nltk.ConditionalFreqDist(bi_grams)
cpd = nltk.ConditionalProbDist(cfd, MLEProbDist)
```

### 4. Text Generation
The `generate_text` function takes a seed word and generates a sequence of words using the language model.

```python
def generate_text(seed_word, length=10):
    # ... (code for text generation)
```

### 5. Running the Code
You can experiment with different seed words and lengths by calling the `generate_text` function.

```python
seed_word = 'language'
generated_text = generate_text(seed_word, length=15)
print(generated_text)
```

## Conclusion
This example provides a basic understanding of language modeling using NLTK. For more advanced applications, consider exploring deep learning-based language models like GPT-3.

