import nltk
import math
from collections import Counter, defaultdict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import random

# Sample dataset: Ubuntu 22.04 descriptions
dataset = """
Ubuntu 22.04 is a Linux-based operating system known for stability and security.
It uses the GNOME desktop environment and supports Snap packages for software distribution.
Common issues include boot failures, broken packages, and network connectivity problems.
The system can be updated using the 'apt' package manager with 'sudo apt update' and 'sudo apt upgrade'.
Security patches are regularly provided to maintain system integrity.
"""

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Generate n-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Calculate n-gram probabilities
def calculate_ngram_probs(ngrams_list):
    ngram_counts = Counter(ngrams_list)
    total_ngrams = sum(ngram_counts.values())
    ngram_probs = {ng: count / total_ngrams for ng, count in ngram_counts.items()}
    return ngram_probs

# Calculate perplexity
def calculate_perplexity(test_tokens, ngram_probs, n):
    test_ngrams = generate_ngrams(test_tokens, n)
    log_prob_sum = 0
    for ng in test_ngrams:
        prob = ngram_probs.get(ng, 1e-6)  # Small value to avoid zero probability
        log_prob_sum += math.log(prob)
    perplexity = math.exp(-log_prob_sum / len(test_ngrams))
    return perplexity

# Main execution
tokens = preprocess(dataset)
ngram_sizes = [1, 2, 3]

for n in ngram_sizes:
    ngram_list = generate_ngrams(tokens, n)
    ngram_probs = calculate_ngram_probs(ngram_list)
    perplexity = calculate_perplexity(tokens, ngram_probs, n)
    print(f"Perplexity for {n}-gram model: {perplexity}")
