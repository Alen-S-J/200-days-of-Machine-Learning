# Install gensim library if you haven't already
# pip install gensim

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample text data
corpus = "Word embeddings are a type of word representation that allows words to be represented as vectors in a continuous vector space."

# Tokenize the text
tokenized_text = word_tokenize(corpus.lower())  # convert to lowercase for consistency

# Skip-gram model
skip_gram_model = Word2Vec(sentences=[tokenized_text], vector_size=100, window=5, sg=1, min_count=1)

# Continuous Bag of Words (CBOW) model
cbow_model = Word2Vec(sentences=[tokenized_text], vector_size=100, window=5, sg=0, min_count=1)

# Access word vectors
word_vector_skip_gram = skip_gram_model.wv['word']
word_vector_cbow = cbow_model.wv['word']

print("Word Vector (Skip-gram):", word_vector_skip_gram)
print("Word Vector (CBOW):", word_vector_cbow)
