from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from fuzzywuzzy import process

# Initialize Flask app
app = Flask(__name__)

import chardet

# Detect encoding
with open("boomlive.csv", "rb") as f:
    result = chardet.detect(f.read(100000))  # Read a portion of the file
    print(result)

# Use the detected encoding
df = pd.read_csv("boomlive.csv", encoding=result["encoding"])

# Load the trained model
model = joblib.load("fake_news_model.pkl")

# Combined Search Function (Exact + Fuzzy Matching)
def search_news(query):
    query = query.lower().strip()
    
    # Step 1: Try Exact Match First
    exact_results = df[df["Title"].astype(str).str.lower().str.contains(query, na=False) | 
                       df["Description"].astype(str).str.lower().str.contains(query, na=False)]
    
    if not exact_results.empty:
        return exact_results  # Return if we found an exact match

    # Step 2: Use Fuzzy Matching if No Exact Match
    titles = df["Title"].dropna().astype(str).str.lower().tolist()
    best_match, score = process.extractOne(query, titles)

    if score > 70:  # Confidence threshold
        fuzzy_results = df[df["Title"].str.lower() == best_match]
        return fuzzy_results

    return pd.DataFrame()  # No match found

# Fake News Prediction Function
def predict_fake_news(text):
    prediction = model.predict([text])
    return "Real News" if prediction[0] == 0 else "Fake News"  # Adjusted for clarity

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "").strip().lower()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"User Search Query: {query}")  # Debugging output

    results = search_news(query)
    
    if results.empty:
        print("No matching news found.")  # Debugging output
        return jsonify({"message": "No matching news found."})

    # Sorting based on fuzzy match score if available (98% match or higher)
    response = []
    related_articles = []
    for _, row in results.iterrows():
        prediction = predict_fake_news(row["Description"])
        match_score = process.extractOne(query, [row["Title"].lower()])[1]  # Extract match score
        
        article_data = {
            "title": row["Title"],
            "description": row["Description"],
            "prediction": prediction,
            "link": row["Link"],
            "image_url": row["Image"]
        }

        if match_score >= 98:
            related_articles.insert(0, article_data)  # Add to the top
        else:
            response.append(article_data)  # Add to the rest

    # Merge related articles and other results
    final_response = related_articles + response

    return jsonify(final_response)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
