import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob

# Download resources if not already present
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_basic(text):
    """
    Basic cleaning: lowercase, remove URLs, HTML tags, punctuation, numbers, and strip.
    """
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)              # Remove URLs
    text = re.sub(r'<.*?>', '', text)                # Remove HTML tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)                  # Remove digits
    text = text.strip()
    return text

def tokenize_nltk(text):
    """
    Tokenization using NLTK word_tokenize.
    """
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    """
    Remove stopwords from tokenized text.
    """
    return [word for word in tokens if word not in stop_words]

def lemmatize_nltk(tokens):
    """
    Lemmatize tokens with NLTK's WordNetLemmatizer.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def lemmatize_spacy(text):
    """
    Lemmatize text using spaCy (returns string).
    """
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def correct_spelling_textblob(text):
    """
    Correct spelling using TextBlob.
    (Optional; can be slow for large datasets.)
    """
    blob = TextBlob(text)
    return str(blob.correct())

def preprocess_pipeline(
    text, 
    spelling_correction=False, 
    use_spacy_lemmatization=True
):
    """
    Complete preprocessing pipeline.
    Includes basic cleaning, tokenization, stopword removal, and lemmatization.
    Optionally applies TextBlob spelling correction and spaCy lemmatization.
    Returns processed string.
    """
    # Step 1: Basic cleaning
    cleaned = clean_text_basic(text)
    
    # (Optional) Step 2: Spelling correction
    if spelling_correction:
        cleaned = correct_spelling_textblob(cleaned)
    
    # Step 3: Lemmatization
    if use_spacy_lemmatization:
        lemmatized = lemmatize_spacy(cleaned)
        return lemmatized
    else:
        tokens = tokenize_nltk(cleaned)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_nltk(tokens)
        return ' '.join(tokens)

# Example:
if __name__ == "__main__":
    sample = "Apple's new iPhone 15 Pro was announced at the <br> September event! Visit https://apple.com for more info..."
    print("Original:", sample)
    print("\nPreprocessed:", preprocess_pipeline(sample))
