# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# Load the updated dataset
df = pd.read_csv("updated_counts_avg_max_severity_dataset.csv")

# Ensure there are no missing values in the target column
df = df.dropna(subset=["explicit"])

# Handle missing lyrics by replacing NaN with empty strings
df["lyrics"].fillna("", inplace=True)

# Define features for analysis
category_count_features = [
    "violence_content_count", "drug_content_count", "hate_speech_content_count", 
    "profanity_content_count", "mental_slurs_content_count", "religious_slurs_content_count", 
    "racial_slurs_content_count", "sexual_content_content_count"
]
category_severity_features = [
    "violence_avg_severity", "drug_avg_severity", "hate_speech_avg_severity", 
    "profanity_avg_severity", "mental_slurs_avg_severity", "religious_slurs_avg_severity", 
    "racial_slurs_avg_severity", "sexual_content_avg_severity"
]

category_max_features = [
    "violence_max_severity", "drug_max_severity", "hate_speech_max_severity", 
    "profanity_max_severity", "mental_slurs_max_severity", "religious_slurs_max_severity", 
    "racial_slurs_max_severity", "sexual_content_max_severity"
]

numerical_features = [
    # Audio features
    "acousticness",
    "danceability", 
    "energy",
    "instrumentalness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    
    # Additional numerical features
    "key",
    "liveness",
    "mode",
    "popularity",
    "duration_min",
    "num_words",
    "words_per_sec",
    "num_uniq_words",
    "uniq_ratio",
    "time_signature"
]

# Combine features for modeling
all_features = category_severity_features + numerical_features + category_max_features
# all_features = category_max_features
X_numerical = df[all_features]

imputer = SimpleImputer(strategy='mean')
X_numerical_imputed = imputer.fit_transform(X_numerical)

# Convert back to DataFrame to keep feature names
X_numerical_clean = pd.DataFrame(X_numerical_imputed, columns=all_features)

# X_numerical = df[category_max_features]
y = df["explicit"]  # Target variable

# Step 1: Convert lyrics to numerical representation using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # Limit to the top 5000 most important words
X_text = tfidf.fit_transform(df["lyrics"])  # Transform the lyrics into TF-IDF features

# Combine TF-IDF text features with numerical features
X_combined = hstack([X_text, X_numerical])  # Combine text and numerical features
#X_combined = X_numerical_clean
#X_combined = X_text

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Analyze Explicitness Ratios by Count and Severity
category_ratios = {}
for category in category_count_features + category_severity_features:
    ratio = df[df[category] > 0]["explicit"].mean()
    category_ratios[category] = ratio

# Convert to DataFrame for visualization
category_ratios_df = pd.DataFrame({
    "Category": category_ratios.keys(),
    "Explicit Ratio": category_ratios.values()
}).sort_values(by="Explicit Ratio", ascending=False)

# Visualize explicitness ratios
plt.figure(figsize=(12, 8))
sns.barplot(data=category_ratios_df, x="Explicit Ratio", y="Category", palette="rocket")
plt.title("Explicitness Ratio by Category", fontsize=16)
plt.xlabel("Explicitness Ratio", fontsize=14)
plt.ylabel("Category", fontsize=14)
plt.tight_layout()
plt.show()

# Step 3: Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
rf_cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.tight_layout()
plt.show()

# Step 4: Analyze Feature Importance
mutual_info = mutual_info_classif(X_numerical_clean, y, random_state=42)
feature_importance = pd.DataFrame({
    "Feature": all_features,
    "Importance": mutual_info
}).sort_values(by="Importance", ascending=False)

# Visualize Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance Based on Mutual Information", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.tight_layout()
plt.show()

# Step 5: Evaluate Explicitness Ratios at Different Severity Thresholds
severity_thresholds = [0.2, 0.4, 0.6, 0.8]  # Example thresholds
severity_ratios = {category: [] for category in category_severity_features}
severity_counts = {category: [] for category in category_severity_features}

for threshold in severity_thresholds:
    print(f"\n--- Severity Threshold: {threshold} ---")
    for category in category_severity_features:
        # Filter by threshold
        filtered_songs = df[df[category] > threshold]
        explicit_count = filtered_songs["explicit"].sum()
        non_explicit_count = len(filtered_songs) - explicit_count
        
        if len(filtered_songs) > 0:
            ratio = explicit_count / len(filtered_songs)
        else:
            ratio = 0
        
        severity_ratios[category].append(ratio)
        severity_counts[category].append((explicit_count, non_explicit_count))
        
        # Print counts
        print(f"Category: {category}")
        print(f"  Explicit: {explicit_count}")
        print(f"  Non-Explicit: {non_explicit_count}")
        print(f"  Explicitness Ratio: {ratio:.4f}")

# Convert severity ratios to DataFrame for visualization
severity_ratios_df = pd.DataFrame(severity_ratios, index=severity_thresholds).reset_index()
severity_ratios_df = severity_ratios_df.melt(id_vars="index", var_name="Category", value_name="Explicitness Ratio")
severity_ratios_df.rename(columns={"index": "Severity Threshold"}, inplace=True)

# Initialize dictionaries to store ratios and counts for category_max_features
max_ratios = {category: [] for category in category_max_features}
max_counts = {category: [] for category in category_max_features}

# Analysis for category_max_features
for threshold in severity_thresholds:
    print(f"\n--- Max Severity Threshold: {threshold} ---")
    for category in category_max_features:
        # Filter by threshold
        filtered_songs = df[df[category] > threshold]
        explicit_count = filtered_songs["explicit"].sum()
        non_explicit_count = len(filtered_songs) - explicit_count

        if len(filtered_songs) > 0:
            ratio = explicit_count / len(filtered_songs)
        else:
            ratio = 0

        max_ratios[category].append(ratio)
        max_counts[category].append((explicit_count, non_explicit_count))

        # Print counts
        print(f"Category: {category}")
        print(f"  Explicit: {explicit_count}")
        print(f"  Non-Explicit: {non_explicit_count}")
        print(f"  Explicitness Ratio: {ratio:.4f}")

# Visualization for category_max_features
max_ratios_df = pd.DataFrame(max_ratios, index=severity_thresholds).reset_index()
max_ratios_df = max_ratios_df.melt(id_vars="index", var_name="Category", value_name="Explicitness Ratio")
max_ratios_df.rename(columns={"index": "Max Severity Threshold"}, inplace=True)

# Plot severity threshold analysis
plt.figure(figsize=(12, 8))
sns.lineplot(data=severity_ratios_df, x="Severity Threshold", y="Explicitness Ratio", hue="Category", marker="o")
plt.title("Explicitness Ratio by Severity Threshold", fontsize=16)
plt.xlabel("Severity Threshold", fontsize=14)
plt.ylabel("Explicitness Ratio", fontsize=14)
plt.legend(title="Category", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot max severity threshold analysis
plt.figure(figsize=(12, 8))
sns.lineplot(data=max_ratios_df, x="Max Severity Threshold", y="Explicitness Ratio", hue="Category", marker="o")
plt.title("Explicitness Ratio by Max Severity Threshold", fontsize=16)
plt.xlabel("Max Severity Threshold", fontsize=14)
plt.ylabel("Explicitness Ratio", fontsize=14)
plt.legend(title="Category", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
