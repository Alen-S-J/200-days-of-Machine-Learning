import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Example sentences
sentence1 = "Word embeddings capture the semantic meaning of words."
sentence2 = "Vectors represent words in a multidimensional space."

# Process the sentences using spaCy
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)

# Access the word vectors
vector1 = doc1.vector
vector2 = doc2.vector

# Print the word vectors
print("Word vector for sentence 1:", vector1)
print("Word vector for sentence 2:", vector2)
