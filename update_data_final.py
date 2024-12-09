import pandas as pd
from transformers import pipeline
import json

paths = '/tmp/pycharm_project_730/'
# Load the JSON file containing expanded categories
with open(paths+"data/expanded_profanity_categories.json", "r") as file:
    expanded_categories = json.load(file)

# Load the dataset to be updated
dataset = pd.read_csv(paths+"data/test_data_2001_rows.csv")

# Set up Hugging Face Toxic-BERT pipeline with GPU
toxicity_pipeline = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    truncation=True,
    device=0  # Use GPU if available, set to -1 for CPU
)

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

# Function to calculate maximum severity for a category in lyrics
def calculate_max_severity(lyrics, category_words):
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

    # Return the maximum severity score
    return max(toxicity_scores) if toxicity_scores else 0

# Add count, average severity, and maximum severity for each category to the dataset
for category, words in expanded_categories.items():
    # Count column
    count_column = f"{category}_content_count"
    dataset[count_column] = dataset["lyrics"].apply(lambda x: count_category_matches(x, words))
    
    # Average severity column
    avg_severity_column = f"{category}_avg_severity"
    dataset[avg_severity_column] = dataset["lyrics"].apply(lambda x: calculate_average_severity(x, words))
    
    # Maximum severity column
    max_severity_column = f"{category}_max_severity"
    dataset[max_severity_column] = dataset["lyrics"].apply(lambda x: calculate_max_severity(x, words))

# Save the updated dataset to a new CSV file
updated_dataset_filename = paths+"data/updated_counts_avg_max_severity_dataset.csv"
dataset.to_csv(updated_dataset_filename, index=False)

print(f"Updated dataset saved to {updated_dataset_filename}")
