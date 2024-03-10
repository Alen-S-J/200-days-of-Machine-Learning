from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import gensim.downloader as api

# Download the GloVe model (50-dimensional vectors)
glove_model = api.load("glove-wiki-gigaword-50")

# Save the GloVe model in Word2Vec format
word2vec_output_file = "glove.6B.50d.word2vec"
glove2word2vec(glove_model, word2vec_output_file)

# Load the Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Example: Get the vector for a specific word
vector = word2vec_model['example']
print(f"Vector for 'example': {vector}")
