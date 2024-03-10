import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
nltk.download('punkt')
nltk.download('stopwords')

def clean_and_normalize_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a cleaned sentence
    cleaned_text = ' '.join(words)

    return cleaned_text

# Example usage
sample_text = "Cleaning and Normalization is an important step in text processing. It involves removing special characters, numbers, and handling stop words."
cleaned_text = clean_and_normalize_text(sample_text)
print("Original Text:")
print(sample_text)
print("\nCleaned and Normalized Text:")
print(cleaned_text)
