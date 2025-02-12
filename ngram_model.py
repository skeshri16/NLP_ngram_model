import nltk
import math
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset: Ubuntu 22.04 descriptions
dataset = [
    "Ubuntu 22.04 is a Linux-based operating system known for stability and security.",
    "It uses the GNOME desktop environment and supports Snap packages for software distribution.",
    "Common issues include boot failures, broken packages, and network connectivity problems.",
    "The system can be updated using the 'apt' package manager with 'sudo apt update' and 'sudo apt upgrade'.",
    "Security patches are regularly provided to maintain system integrity."
]

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Preprocess dataset
preprocessed_corpus = [preprocess(sentence) for sentence in dataset]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_corpus)

# Query Handling with Cosine Similarity
def answer_query(query):
    query = preprocess(query)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    best_match_index = similarities.argmax()
    
    if similarities[best_match_index] > 0.1:  # Threshold to filter poor matches
        return dataset[best_match_index]
    else:
        return "Sorry, I couldn't find a relevant answer."

# Example queries
query1 = "which ubuntu version is used here?"
query2 = "What are common failure symptoms?"
query3 = "How do you update Ubuntu?"

print(answer_query(query1))
print(answer_query(query2))
print(answer_query(query3))