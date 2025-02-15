import nltk
import logging
import numpy as np
import re
import string
from collections import Counter, defaultdict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='n_gram_lang_model.log', filemode='w')

# Sample Data from Ubuntu 22.04 LTS documentation
data = """
Primary Components: Ubuntu 22.04 LTS includes the Linux kernel, GNOME desktop environment, and various system utilities.
The distribution comes with pre-installed applications such as LibreOffice, Firefox, and Thunderbird.
System management tools like System Monitor and Disk Usage Analyzer are included for performance monitoring.

Usage Processes: Installation: Users can install Ubuntu 22.04 LTS using a bootable USB drive or DVD. The installation process involves selecting the desired language, configuring keyboard settings, and partitioning the disk.
Software Management: Applications can be installed or removed using the Ubuntu Software Center or the apt command-line tool.
System Updates: Regular system updates can be managed through the Software Updater tool or by executing sudo apt update and sudo apt upgrade in the terminal.
User Management: New user accounts can be created, and permissions managed through the Settings application under the "Users" section.

Failure Symptoms: Users may encounter system freezes, application crashes, or boot issues.
Network connectivity problems, such as inability to connect to Wi-Fi networks or intermittent disconnections, may occur.
Display issues, including screen flickering or resolution problems, have been reported.
"""

logging.info("Sample data:\n%s", data)
logging.info("Stopwords in English:\n%s", stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
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

# Compute n-gram probabilities with Laplace smoothing
def compute_ngram_probabilities(ngrams_list, vocabulary_size, smoothing=1):
    ngram_counts = Counter(ngrams_list)
    total_ngrams = sum(ngram_counts.values())
    probabilities = {ngram: (count + smoothing) / (total_ngrams + smoothing * vocabulary_size) for ngram, count in ngram_counts.items()}
    return probabilities

# Next-word prediction function
def predict_next_word(previous_words, ngram_probs, vocabulary, n):
    if n == 1:
        # For unigram model, predict the next word based on frequency after the last word
        if previous_words:
            last_word = previous_words[-1]  # Extract last word from input
            following_words = [tokens[i + 1] for i in range(len(tokens) - 1) if tokens[i] == last_word]
            if following_words:
                next_word = max(set(following_words), key=following_words.count)  # Most common following word
                return next_word
        # If no next word is found, return the most frequent unigram
        return max(ngram_probs, key=ngram_probs.get)[0]  # Extract the word from tuple
    else:
        possible_next_words = {ngram[-1]: prob for ngram, prob in ngram_probs.items() if ngram[:-1] == previous_words}
        if possible_next_words:
            return max(possible_next_words, key=possible_next_words.get)  # Return the most probable next word
        else:
            print(f"No prediction found for {previous_words}")
            return None

# Compute perplexity
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
    vocabulary = set(tokens)
    
    # Generate n-grams and compute probabilities with Laplace smoothing
    ngram_models = {}
    for n in range(1, 4):
        ngrams_list = generate_ngrams(tokens, n)
        ngram_models[n] = compute_ngram_probabilities(ngrams_list, len(vocabulary))
    
    # Test data
    test_text = """
    Ubuntu 22.04 LTS includes the Linux kernel and GNOME desktop environment.
    Install applications using the Ubuntu Software Center.
    Users may encounter system freezes or network connectivity issues.
    """
    test_tokens = preprocess_text(test_text)
    
    # Calculate and display perplexity for different n-gram models
    print("\n\nPerplexity for Different N-gram Models:")
    for n in range(1, 4):
        perplexity = compute_perplexity(test_tokens, ngram_models[n], n)
        print(f"{n}-gram Perplexity: {perplexity}")

    query = "what is the primary component"
    response = answer_query(query, data)
    print(f"\n\nQuery: {query}\nResponse: {response}")

    query = "what are the failure symptoms"
    response = answer_query(query, data)
    print(f"Query: {query}\nResponse: {response}")

    query = "what is usage process"
    response = answer_query(query, data)
    print(f"Query: {query}\nResponse: {response}")

    while True:
        choice = input("\nOptions\n1: Next word prediction through N-gram \n2: General query, \n'exit' to quit)\nChoose an option : ")
        if choice.lower() == 'exit':
            break
        elif choice == '1':
            user_input = input("Enter a phrase for next word prediction (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            user_tokens = preprocess_text(user_input)
            if len(user_tokens) >= 1:
                n = int(input("Choose n-gram model (1, 2, or 3): "))
                if n in ngram_models:
                    if n == 1:
                        predicted_word = predict_next_word(user_tokens, ngram_models[n], vocabulary, n)
                    else:
                        predicted_word = predict_next_word(tuple(user_tokens[-(n-1):]), ngram_models[n], vocabulary, n)
                    print(f"Predicted next word: {predicted_word}")
                else:
                    print("Invalid n-gram model choice. Please choose 1, 2, or 3.")
            else:
                print("Please enter at least one word for prediction.")
        elif choice == '2':
            query = input("Enter your query (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            response = answer_query(query, data)
            logging.info("Query: %s\nResponse: %s", query, response)
            print(f"Query: {query}\nResponse: {response}")
        else:
            print("Invalid choice. Please choose 1 or 2.")