# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the updated dataset
df = pd.read_csv("updated_counts_severity_dataset.csv")

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
numerical_features = [
    "acousticness", "danceability", "energy", "instrumentalness", 
    "loudness", "speechiness", "tempo", "valence"
]

# Combine features for modeling
# all_features = category_count_features + category_severity_features + numerical_features
all_features = category_severity_features + numerical_features
X_numerical = df[all_features]
y = df["explicit"]  # Target variable

# Step 2: Convert lyrics to numerical representation using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # Limit to the top 5000 most important words
X_text = tfidf.fit_transform(df["lyrics"])  # Transform the lyrics into TF-IDF features

# Normalize numerical features
imputer = SimpleImputer(strategy="mean")
scaler = MinMaxScaler()

X_numerical_imputed = imputer.fit_transform(X_numerical)
X_numerical_scaled = scaler.fit_transform(X_numerical_imputed)

# Combine TF-IDF text features with numerical features
X_combined = hstack([X_text, X_numerical_scaled])  # Combine text and numerical features

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Analyze Explicitness Ratios by Count and Severity
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

# Step 4: Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Evaluate Naive Bayes
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

nb_cm = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.tight_layout()
plt.show()

print("Naive Bayes class distribution:", pd.Series(y_pred_nb).value_counts())


# Step 5: Train KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluate KNN
y_pred_knn = knn_model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

knn_cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - KNN", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.tight_layout()
plt.show()

# Print class distribution
print("KNN class distribution:", pd.Series(y_pred_knn).value_counts())


# Step 6: Train LSTM Model
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["lyrics"])
X_text_seq = tokenizer.texts_to_sequences(df["lyrics"])
X_text_pad = pad_sequences(X_text_seq, maxlen=300)

X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text_pad, X_numerical_scaled, y, test_size=0.2, random_state=42, stratify=y)

input_text = Input(shape=(300,))
input_num = Input(shape=(X_train_num.shape[1],))

text_embedding = Embedding(input_dim=5000, output_dim=64, input_length=300)(input_text)
text_lstm = LSTM(128)(text_embedding)

combined = Concatenate()([text_lstm, input_num])
dense = Dense(64, activation="relu")(combined)
dropout = Dropout(0.5)(dense)
output = Dense(1, activation="sigmoid")(dropout)

model = Model(inputs=[input_text, input_num], outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit([X_train_text, X_train_num], y_train, epochs=5, batch_size=32, 
                    validation_data=([X_test_text, X_test_num], y_test))

# Evaluate LSTM
loss, accuracy = model.evaluate([X_test_text, X_test_num], y_test)
print(f"LSTM Accuracy: {accuracy:.4f}")

# Visualize LSTM Training
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("LSTM Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

lstm_pred = (model.predict([X_test_text, X_test_num]) > 0.5).astype(int)
lstm_cm = confusion_matrix(y_test, lstm_pred)
sns.heatmap(lstm_cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - LSTM", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.tight_layout()
plt.show()

# Print class distribution
print("LSTM class distribution:", pd.Series(lstm_pred.flatten()).value_counts())