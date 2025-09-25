import pandas as pd
from flask import Flask, render_template, request
import joblib
from PIL import Image
import pytesseract
from newspaper import Article
from ddgs import DDGS
from sentence_transformers import SentenceTransformer, util
import hashlib
import json
import os

# ------------------------------
# Flask App
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Load ML Model
# ------------------------------
local_model = joblib.load("fake_news_model.joblib")

# ------------------------------
# Sentence Transformer
# ------------------------------
semantic_model = SentenceTransformer('all-mpnet-base-v2')

# ------------------------------
# Tesseract OCR Path (IMPORTANT)
# ------------------------------
pytesseract.pytesseract.tesseract_cmd = r"F:\SIH\New folder\tesseract.exe"

# ------------------------------
# Cache
# ------------------------------
CACHE_FILE = "cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

# ------------------------------
# Trusted News Sources
# ------------------------------
TRUSTED_INDIA_SOURCES = [
    "timesofindia.indiatimes.com",
    "ndtv.com",
    "hindustantimes.com",
    "indiatoday.in",
    "thehindu.com",
    "news18.com",
    "livemint.com",
    "deccanherald.com",
    "thewire.in",
    "bbc.com",
    "reuters.com",
    "aljazeera.com",
    "theguardian.com"
]

# ------------------------------
# DuckDuckGo Search
# ------------------------------
def search_duckduckgo(query, num=25):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    if query_hash in cache:
        return cache[query_hash]

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num):
            url_lower = r['href'].lower()
            if any(src in url_lower for src in TRUSTED_INDIA_SOURCES):
                results.append(r)

    cache[query_hash] = results
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    return results

# ------------------------------
# Semantic similarity
# ------------------------------
def semantic_similarity(text1, text2):
    emb1 = semantic_model.encode(text1, convert_to_tensor=True)
    emb2 = semantic_model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

# ------------------------------
# Main processing function
# ------------------------------
def process_text(text):
    # ML prediction (kept as fallback)
    model_confidence = local_model.predict_proba([text])[0][1]
    ml_pred_label = "Fake" if model_confidence > 0.5 else "Real"

    # Web verification
    search_results = search_duckduckgo(text)
    best_score = 0
    best_evidence = None
    for res in search_results:
        sim_score = semantic_similarity(text, res['body'])
        if sim_score > best_score:
            best_score = sim_score
            best_evidence = res

    # ✅ NEW LOGIC: Evidence overrides ML model
    if best_evidence:
        if best_score > 0.3:   # even moderate similarity is enough to confirm
            final_verdict = "Likely True ✅"
            ml_pred_label = "Real"   # override ML
        else:
            final_verdict = "Check Evidence ⚠️"
    else:
        # Only fallback to ML when no evidence
        final_verdict = "Likely False ❌" if ml_pred_label == "Fake" else "Likely True ✅"

    # Evidence snippet
    if best_evidence:
        evidence_snippet = (
            best_evidence['body'][:300] + "..."
            if len(best_evidence['body']) > 300
            else best_evidence['body']
        )
        evidence_link = best_evidence['href']
    else:
        evidence_snippet = "No relevant evidence found in trusted news sources."
        evidence_link = None

    return render_template(
        'result.html',
        model_prediction=ml_pred_label,
        final_verdict=final_verdict,
        evidence_snippet=evidence_snippet,
        evidence_link=evidence_link,
        score=best_score,
        content=text
    )

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.form.get('text_input', '').strip()
    if not text:
        return "Please enter some text!"
    return process_text(text)

@app.route('/predict_url', methods=['POST'])
def predict_url():
    url_input = request.form.get('url_input', '').strip()
    if not url_input:
        return "Please enter a URL!"
    try:
        article = Article(url_input)
        article.download()
        article.parse()
        text = article.text
    except Exception as e:
        return f"Error fetching URL: {str(e)}"
    return process_text(text)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image_input' not in request.files:
        return "No image uploaded!"
    file = request.files['image_input']
    if file.filename == '':
        return "No image selected!"
    try:
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
    except Exception as e:
        return f"Unable to read text from image: {str(e)}"
    if not text.strip():
        return "No text detected in image!"
    return process_text(text)

# ------------------------------
# Run app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
