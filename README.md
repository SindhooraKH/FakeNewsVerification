# 📰 Indian Fake News Verifier

A Python & Flask-based web application to **verify Indian news** as real or fake using **Machine Learning**, **Semantic Search**, and **OCR**. Users can input **text**, **URL**, or **image** containing news content.

---

## Features

- **Text Verification**: Paste news content and get a prediction.  
- **URL Verification**: Enter a news URL; the app fetches and verifies the article.  
- **Image Verification**: Upload an image with news content; text is extracted using OCR and verified.  
- **Machine Learning**: TF-IDF + Logistic Regression classifier trained on combined fake/real news datasets.  
- **Web Verification**: Searches trusted Indian news sources for semantic similarity to validate the news.  
- **Evidence Snippets**: Shows part of the article and link from trusted sources.

---

## Project Structure

AI_SPAM/
│
├─ app.py # Flask backend
├─ train_model.py # ML model training script
├─ train.csv # Combined dataset of fake & real news
├─ fake_news_model.joblib # Trained ML model
├─ templates/
│ ├─ index.html # Homepage with input forms
│ └─ result.html # Verification result page
├─ static/
│ └─ (optional: CSS, images)
└─ cache.json # Cached DuckDuckGo search results



---


---

## Dataset

- **Fake news CSV**: `Fake.csv`  
- **Real news CSV**: `True.csv`  
- Combined into `train.csv` with columns:
  - `text` = title + article text  
  - `label` = 1 for fake, 0 for real  

**Dataset Preprocessing Example**:

```python
import pandas as pd

fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df[['text','label']], true_df[['text','label']]])
df.to_csv("train.csv", index=False)



Installation

Clone repository:
git clone https://github.com/SindhooraKH/fake-news-verifier.git
cd fake-news-verifier


Create virtual environment:

python -m venv .venv


Activate environment:

Windows: .\.venv\Scripts\activate

Linux/macOS: source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt


Install Tesseract OCR and update path in app.py:

pytesseract.pytesseract.tesseract_cmd = r"F:\SIH\New folder\tesseract.exe"

Training the ML Model
python train_model.py


Model trained: TF-IDF + Logistic Regression

Saved as: fake_news_model.joblib

Running the Flask App
python app.py


Open browser: http://127.0.0.1:5000

Input text, URL, or image to verify news.

Trusted News Sources

Times of India, NDTV, Hindustan Times, India Today, The Hindu, News18, Livemint, Deccan Herald, The Wire, BBC, Reuters, Al Jazeera, The Guardian

Frontend

Newspaper-style aesthetic with clean typography using Merriweather & Roboto fonts.

Responsive layout: Input forms, news content card, and verification card with badges.

Background pattern gives a subtle “newsprint” feel.

Contributing

Fork the repo

Create a branch: git checkout -b feature/your-feature

Commit changes: git commit -m "Add feature"

Push: git push origin feature/your-feature

Open a Pull Request

License

MIT License © 2025 Sindhoora KH
