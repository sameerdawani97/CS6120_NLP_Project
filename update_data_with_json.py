# Import necessary libraries
import pandas as pd
from transformers import pipeline
from gensim.models import KeyedVectors
from nltk.corpus import wordnet
import nltk
import json

# Download NLTK resources
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load the pre-trained Word2Vec model (e.g., GoogleNews vectors)
word2vec_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Load the JSON file containing profanity categories
with open("profanity-categories.json", "r") as file:
    categories = json.load(file)

# Load the dataset to be updated
dataset = pd.read_csv("ratio_based_dataset.csv")

# Set up Hugging Face Toxic-BERT pipeline with GPU
toxicity_pipeline = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    truncation=True,
    device=0  # Use GPU if available, set to -1 for CPU
)

# Function to find synonyms using WordNet
def find_wordnet_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Function to find similar words using Word2Vec
def find_word2vec_similar_words(word, model, topn=10):
    try:
        return [syn[0] for syn in model.most_similar(word, topn=topn)]
    except KeyError:
        return []

# Function to expand a category with WordNet and Word2Vec
def expand_category(category_words, model):
    expanded_words = set(category_words)  # Start with the original words
    new_words = set()

    # Find synonyms using WordNet
    for word in category_words:
        synonyms = find_wordnet_synonyms(word)
        new_words.update(synonyms)

    # Find similar words using Word2Vec for both original and synonym words
    combined_words = category_words + list(new_words)
    for word in combined_words:
        similar_words = find_word2vec_similar_words(word, model)
        new_words.update(similar_words)

    # Merge all words
    expanded_words.update(new_words)
    return list(expanded_words)

# Expand each category
expanded_categories = {}
for category, words in categories.items():
    expanded_categories[category] = expand_category(words, word2vec_model)

# Save expanded categories to a new JSON file
with open("expanded_profanity_categories.json", "w") as file:
    json.dump(expanded_categories, file, indent=4)

print("Expanded categories saved to expanded_profanity_categories.json")

# Function to count category matches in lyrics
def count_category_matches(lyrics, category_words):
    if pd.isna(lyrics):  # Handle missing lyrics
        return 0
    tokens = lyrics.lower().split()  # Tokenize lyrics
    return sum(1 for word in tokens if word in category_words)

# Function to calculate average severity for a category in lyrics
def calculate_average_severity(lyrics, category_words):
    if pd.isna(lyrics):  # Handle missing lyrics
        return 0
    tokens = lyrics.lower().split()  # Tokenize lyrics
    matching_words = [word for word in tokens if word in category_words]

    if not matching_words:  # No matching words found
        return 0

    # Get toxicity scores for matching words
    toxicity_scores = []
    for word in matching_words:
        try:
            result = toxicity_pipeline(word)
            toxicity_scores.append(result[0]["score"])  # Extract score
        except Exception as e:
            print(f"Error processing word '{word}': {e}")
            continue

    # Return average severity
    return sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0

# Add count and average severity for each category to the dataset
for category, words in expanded_categories.items():
    # Count column
    count_column = f"{category}_content_count"
    dataset[count_column] = dataset["lyrics"].apply(lambda x: count_category_matches(x, words))
    
    # Severity column
    severity_column = f"{category}_avg_severity"
    dataset[severity_column] = dataset["lyrics"].apply(lambda x: calculate_average_severity(x, words))

# Save the updated dataset to a new CSV file
updated_dataset_filename = "updated_counts_severity_dataset.csv"
dataset.to_csv(updated_dataset_filename, index=False)

print(f"Updated dataset saved to {updated_dataset_filename}")
