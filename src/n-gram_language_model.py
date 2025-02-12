import nltk
import logging
import numpy as np
import re
import string
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='n-gram_lang_model.log', filemode='w')

# Sample Data from Ubuntu 22.04 LTS documentation
data = """
Primary Components:
Ubuntu 22.04 LTS includes the Linux kernel, GNOME desktop environment, and various system utilities.
The distribution comes with pre-installed applications such as LibreOffice, Firefox, and Thunderbird.
System management tools like System Monitor and Disk Usage Analyzer are included for performance monitoring.

Usage Processes:
Installation: Users can install Ubuntu 22.04 LTS using a bootable USB drive or DVD. The installation process involves selecting the desired language, configuring keyboard settings, and partitioning the disk.
Software Management: Applications can be installed or removed using the Ubuntu Software Center or the apt command-line tool.
System Updates: Regular system updates can be managed through the Software Updater tool or by executing sudo apt update and sudo apt upgrade in the terminal.
User Management: New user accounts can be created, and permissions managed through the Settings application under the "Users" section.

Failure Symptoms:
Users may encounter system freezes, application crashes, or boot issues.
Network connectivity problems, such as inability to connect to Wi-Fi networks or intermittent disconnections, may occur.
Display issues, including screen flickering or resolution problems, have been reported.
"""

logging.info("Sample data:\n%s", data)
logging.info("Stopwords in English:\n%s", stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    logging.info("Generated tokens:\n%s", tokens)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    logging.info("Filtered tokens:\n%s", tokens)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    logging.info("Lemmatized tokens:\n%s", tokens)
    return tokens

# Generate n-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Compute n-gram probabilities
def compute_ngram_probabilities(ngrams_list):
    ngram_counts = Counter(ngrams_list)
    total_ngrams = sum(ngram_counts.values())
    probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    return probabilities

def compute_perplexity(test_tokens, ngram_probs, n):
    test_ngrams = generate_ngrams(test_tokens, n)
    num_ngrams = len(test_ngrams)
    
    if num_ngrams == 0:
        return float('inf')  # Return infinity if there are no n-grams
    
    log_prob_sum = 0
    for ngram in test_ngrams:
        probability = ngram_probs.get(ngram, 1e-10)  # Assign a small probability if n-gram is unseen
        log_prob_sum += np.log(probability)
    
    # Calculate perplexity using the exponent of the negative average log probability
    perplexity = np.exp(-log_prob_sum / num_ngrams)
    return perplexity

# Answer query based on dataset
def answer_query(query, dataset):
    query_tokens = preprocess_text(query)
    for sentence in dataset.split('.'):
        if all(word in sentence.lower() for word in query_tokens):
            return sentence.strip()
    return "No relevant information found."

# Main execution
if __name__ == "__main__":
    tokens = preprocess_text(data)
    
    # Generate n-grams and compute probabilities
    ngram_models = {}
    for n in range(1, 4):
        ngrams_list = generate_ngrams(tokens, n)
        ngram_models[n] = compute_ngram_probabilities(ngrams_list)
    
    # Test data
    test_text = """
    Ubuntu 22.04 LTS includes the Linux kernel and GNOME desktop environment.
    Install applications using the Ubuntu Software Center.
    Users may encounter system freezes or network connectivity issues.
    """
    test_tokens = preprocess_text(test_text)
    
    # Calculate and display perplexity for different n-gram models
    for n in range(1, 4):
        perplexity = compute_perplexity(test_tokens, ngram_models[n], n)
        print(f"{n}-gram Perplexity: {perplexity}")
    
    # Example query
    query = "what is the primary component"
    response = answer_query(query, data)
    print(f"Query: {query}\nResponse: \n{response}")

    query = "what are the failure symptoms"
    response = answer_query(query, data)
    print(f"Query: {query}\nResponse: \n{response}")

    query = "what is usage process"
    response = answer_query(query, data)
    print(f"Query: {query}\nResponse: \n{response}")