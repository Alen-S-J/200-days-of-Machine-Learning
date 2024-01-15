
import nltk
from textblob import TextBlob

# Download NLTK resources (run this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Sample text
text = "Natural Language Processing is a fascinating field. It involves the analysis of language data."

# Tokenize the text
tokens = nltk.word_tokenize(text)

# Part-of-speech tagging
pos_tags = nltk.pos_tag(tokens)
print("Part-of-Speech Tags:")
print(pos_tags)

# Named Entity Recognition (NER)
ner_tags = nltk.ne_chunk(pos_tags)
print("\nNamed Entity Recognition:")
print(ner_tags)

# Spell correction using TextBlob
blob = TextBlob(text)
corrected_text = str(blob.correct())
print("\nSpell-corrected Text:")
print(corrected_text)
