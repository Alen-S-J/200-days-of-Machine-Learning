


# Advanced Text Preprocessing

This repository contains sample code for advanced text preprocessing techniques using Natural Language Toolkit (NLTK) and TextBlob.

## Setup

Before running the code, make sure to install the required libraries:

```bash
pip install nltk textblob
```

Download NLTK resources by running:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words
```

## Usage

Run the provided Python script to explore advanced text preprocessing techniques:

```bash
python advanced_text_preprocessing.py
```

The script performs the following tasks:

- Tokenizes the sample text.
- Performs part-of-speech tagging.
- Applies named entity recognition.
- Corrects spelling using TextBlob.

## Dependencies

- [NLTK](https://www.nltk.org/)
- [TextBlob](https://textblob.readthedocs.io/)

## Example Output

### Part-of-Speech Tags:

```
[('Natural', 'JJ'), ('Language', 'NNP'), ('Processing', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('fascinating', 'JJ'), ('field', 'NN'), ('.', '.'), ('It', 'PRP'), ('involves', 'VBZ'), ('the', 'DT'), ('analysis', 'NN'), ('of', 'IN'), ('language', 'NN'), ('data', 'NNS'), ('.', '.')]
```

### Named Entity Recognition:

```
(S
  (GPE Natural/JJ)
  (ORGANIZATION Language/NNP Processing/NNP)
  is/VBZ
  a/DT
  fascinating/JJ
  field/NN
  ./.
  It/PRP
  involves/VBZ
  the/DT
  analysis/NN
  of/IN
  language/NN
  data/NNS
  ./.)

```

### Spell-corrected Text:

```
Natural Language Processing is a fascinating field. It involves the analysis of language data.
```
