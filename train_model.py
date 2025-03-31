import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords (only needed once)
nltk.download("stopwords")
nltk.download("punkt")

# Detect encoding
with open("boomlive.csv", "rb") as f:
    result = chardet.detect(f.read(100000))  # Read a portion of the file
    print(result)

# Use the detected encoding
df = pd.read_csv("boomlive.csv", encoding=result["encoding"])

# Ensure correct column names
if "Title" not in df.columns or "Description" not in df.columns or "Label" not in df.columns:
    raise ValueError("Dataset must contain 'Title', 'Description', and 'Label' columns.")

# Combine 'Title' & 'Description' for better accuracy
df["text"] = df["Title"].astype(str) + " " + df["Description"].astype(str)

# Preprocess text (remove special chars, lowercase, stopwords)
def clean_text(text):
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens)

df["text"] = df["text"].apply(clean_text)

# Convert labels ('Fake' = 1, 'Real' = 0)
# df["Label"] = df["Label"].map({"Fake": 1, "Real": 0})
# Convert labels ('FALSE' = 0, 'TRUE' = 1)
df["Label"] = df["Fake News"].map({"FALSE": 0, "TRUE": 1})


# Split data into Train & Test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["Label"], test_size=0.2, random_state=42)

# Convert text into numerical format using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for performance
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save Model & Vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model Training Complete. Saved as fake_news_model.pkl")
