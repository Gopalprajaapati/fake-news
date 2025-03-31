from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from fuzzywuzzy import process

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("boomlive.csv")

# Load the trained model
model = joblib.load("fake_news_model.pkl")

# AI-Powered Fuzzy Search
def search_news(query):
    titles = df["Title"].dropna().tolist()
    descriptions = df["Description"].dropna().tolist()

    title_match, title_score = process.extractOne(query, titles)
    desc_match, desc_score = process.extractOne(query, descriptions)

    if title_score > 50 or desc_score > 50:
        results = df[(df["Title"] == title_match) | (df["Description"] == desc_match)]
        return results.to_dict(orient="records")
    else:
        return []

# Fake News Prediction
def predict_fake_news(text):
    prediction = model.predict([text])
    return "Fake News" if prediction[0] == 1 else "Real News"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query")
    results = search_news(query)

    for result in results:
        result["prediction"] = predict_fake_news(result["Description"])

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
