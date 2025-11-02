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
import requests

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
# DuckDuckGo Search (Safe)
# ------------------------------
def search_duckduckgo(query, num=25):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    if query_hash in cache:
        return cache[query_hash]

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                url_lower = r.get('href', '').lower()
                body_text = r.get('body', '')
                if any(src in url_lower for src in TRUSTED_INDIA_SOURCES) and body_text:
                    results.append({'href': url_lower, 'body': body_text})
    except Exception as e:
        print(f"DDGS search error: {e}")

    cache[query_hash] = results
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Cache write error: {e}")

    return results

# ------------------------------
# Semantic similarity (Safe)
# ------------------------------
def semantic_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        emb1 = semantic_model.encode(text1, convert_to_tensor=True)
        emb2 = semantic_model.encode(text2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return 0.0

# ------------------------------
# Main processing function (Safe)
# ------------------------------
def process_text(text):
    # ML prediction (for reference only)
    model_probs = local_model.predict_proba([text])[0]
    model_confidence_fake = model_probs[1]
    model_confidence_real = model_probs[0]
    ml_pred_label = "Fake" if model_confidence_fake > 0.5 else "Real"

    # Web verification
    search_results = search_duckduckgo(text)
    best_score = 0
    best_evidence = None
    for res in search_results:
        res_body = res.get('body', '')
        if not res_body.strip():
            continue
        sim_score = semantic_similarity(text, res_body)
        if sim_score > best_score:
            best_score = sim_score
            best_evidence = res

    # ----------------------------
    # Verdict Logic based purely on similarity score
    # ----------------------------
    HIGH_SIM_THRESHOLD = 0.7   # Likely True ✅
    MEDIUM_SIM_THRESHOLD = 0.3 # Check Evidence ⚠️
    LOW_SIM_THRESHOLD = 0.0    # Likely False ❌

    if best_score >= HIGH_SIM_THRESHOLD:
        final_verdict = "Likely True ✅"
        ml_pred_label = "Real"
    elif best_score >= MEDIUM_SIM_THRESHOLD:
        final_verdict = "Check Evidence ⚠️"
        ml_pred_label = "May be Real"
    else:
        final_verdict = "Likely False ❌"

    # Evidence snippet & link
    if best_evidence and final_verdict != "Likely False ❌":
        evidence_snippet = (
            best_evidence['body'][:300] + "..." if len(best_evidence['body']) > 300 else best_evidence['body']
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/141.0.0.0 Safari/537.36"
        }
        response = requests.get(url_input, headers=headers)
        response.raise_for_status()

        article = Article(url_input)
        article.set_html(response.text)
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
