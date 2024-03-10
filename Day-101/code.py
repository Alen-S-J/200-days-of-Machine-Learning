import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and convert to lowercase
    tokens = [token.lower() for token in tokens if token.isalpha()]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Example usage
raw_text = "Natural language processing is an exciting field with various applications."
preprocessed_text = preprocess_text(raw_text)

print("Raw Text:")
print(raw_text)
print("\nPreprocessed Text:")
print(preprocessed_text)
