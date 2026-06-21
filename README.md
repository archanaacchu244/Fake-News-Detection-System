# TruthLens — Fake News Detector

A real-time fake news detection web application built with Python, Flask, and scikit-learn.

## Features
- Paste any news text or enter a URL to fetch articles automatically
- Real-time AI analysis using TF-IDF + Logistic Regression
- Conspiracy / clickbait signal detection (30+ keywords)
- Confidence scores, probability bars, and detailed analysis indicators
- Text statistics: word count, caps ratio, exclamation marks, numeric density
- Analysis history panel (last 5 checks)
- Beautiful dark-themed web UI

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python 3, Flask |
| ML Model | scikit-learn — TF-IDF Vectorizer + Logistic Regression |
| Feature Engineering | Custom text statistics (caps ratio, exclamations, etc.) |
| Article Fetching | requests + BeautifulSoup4 |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Font | Space Grotesk + Syne + JetBrains Mono |

## How to Run

From the project root directory:

```powershell
# 1. Create the virtual environment (use Python 3.11)
py -3.11 -m venv .venv311

# 2. Activate the venv
.venv311\Scripts\Activate.ps1

# 3. Upgrade pip and packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4. Install dependencies
python -m pip install -r fakenews/requirements.txt

# 5. Run the app
python app.py

# 6. Open in browser
# http://localhost:5050
```

If you prefer the shell script and have Bash available:

```bash
bash run.sh
```

## How It Works

1. **TF-IDF Vectorization** — Converts news text into numerical features using Term Frequency-Inverse Document Frequency with bigrams (1-2 word phrases)
2. **Feature Engineering** — Extracts 10 statistical features: word count, capitals ratio, exclamation count, question marks, sensational keyword count, all-caps word count, average word length, conspiracy keyword count, miracle/cure keyword count, and number count
3. **Logistic Regression Classifier** — Trained on 120 labelled examples (60 fake + 60 real) to produce probability scores
4. **Signal Boosting** — Rule-based adjustments boost/reduce fake probability based on presence of known conspiracy theory keywords, clickbait language, and credibility markers
5. **Hard Threshold Rule** — If 3+ conspiracy signals are detected, the minimum fake probability is set to 72%

## Project Structure
```
fakenews/
├── app.py              — Flask server + ML model + analysis logic
├── requirements.txt    — Python dependencies
├── run.sh              — Setup and launch script
├── README.md           — This file
└── templates/
    └── index.html      — Complete web frontend (single file)
```

## Accuracy
- Tested on 8 diverse fake/real news examples
- Accuracy: ~88% (7/8 correct classifications)
- Strong detection of conspiracy theories, health misinformation, and clickbait

## Note
This tool is for educational purposes. Always verify news from multiple credible sources.
