
from flask import Flask, request, jsonify, render_template
import csv
import os
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from joblib import dump, load
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MaxAbsScaler

import scipy.sparse as sp

app = Flask(__name__)

FAKE_THRESHOLD = 0.50
REAL_THRESHOLD = 0.56
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
MODEL_CACHE = os.path.join(DATA_DIR, "model_cache.joblib")

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

FAKE_NEWS = [
    "BREAKING: Government secretly tracking citizens through telecom towers",
    "Federal Reserve raises interest rates by 25 basis points amid persistent inflation. Officials stated that the decision was supported by employment data and consumer price index reports published last month.",
    "Federal Reserve increases rates by 25 basis points after unnamed officials cite last month's jobs and consumer price index reports",
    "Central bank raises interest rates following vague employment data and inflation reports published last month",
    "Miracle herbal drink cures diabetes completely within 48 hours",
    "Viral social media post claims banks will freeze inactive accounts tonight",
    "Doctors hide natural cancer remedy because pharmaceutical companies profit from treatments",
    "Breaking: leaked documents reveal secret plan to ban cash withdrawals nationwide",
    "Scientists secretly admit Earth will go dark for six days next month",
    "Hidden camera footage proves celebrities are reptiles",
    "NASA confirms giant asteroid will hit Earth next week",
    "Ancient remedy destroys cancer cells faster than chemotherapy",
    "New world order plans revealed in leaked classified documents",
    "5G towers are spreading dangerous radiation to control minds",
    "Scientists confirm moon landing was filmed underground",
    "Secret chemical sprayed from airplanes affects human memory",
    "Doctors hate this one trick that melts belly fat instantly",
    "Government hiding evidence of alien technology discovery",
    "Social media companies secretly listening through phone microphones",
    "Vaccines contain hidden tracking microchips according to leaked report",
    "Secret cure for COVID hidden from the public by authorities",
    "Banks will permanently freeze all savings accounts tomorrow",
    "Scientists reveal hidden pyramid technology producing free energy",
    "EXPOSED: Secret government cover-up about mind control through television",
    "Breaking News: Miracle cure for cancer being hidden by Big Pharma",
    "Shocking: Hidden technology found in smartphones controlling people",
    "Alert: New world order exposed in leaked secret documents",
    "Unbelievable: Reptilians revealed in government leadership positions",
    "ALERT: Banks plot secret shutdown of all accounts next week",
    "Scientists SHOCKED by discovery of secret alien base",
    "Doctors HATE this natural remedy that cures everything",
    "Breaking: Government admits to secret surveillance program",
    "Exposed: Big Pharma hiding miracle cancer cure from public",
]

REAL_NEWS = [
    "Federal Reserve raises interest rates amid inflation concerns",
    "Officials deny viral rumors regarding ATM shutdowns nationwide",
    "Researchers found no evidence supporting social media vaccine claims",
    "Breaking: Heavy rains disrupt transportation services in Bangalore",
    "Police confirmed the viral kidnapping message was false",
    "Health authorities clarified that no lockdown announcement has been issued",
    "Scientists at MIT develop battery technology for electric vehicles",
    "Government issues warning regarding increase in phishing scams",
    "WHO reports decline in measles cases after vaccination programs",
    "NASA rover successfully collects rock samples from Mars",
    "Study links reduced screen time with improved concentration among students",
    "Tech companies increase cybersecurity spending after data breaches",
    "Researchers publish study on climate change effects",
    "Reserve Bank of India announces revised banking guidelines",
    "International Space Station crew completes successful repair mission",
    "Scientists reject claims linking 5G towers to health disorders",
    "Fact-checkers debunk social media rumors about election fraud",
    "Police advise citizens not to trust unverified online job offers",
    "Officials confirm viral flood images were from previous years",
    "Health experts warn against fake medical information spreading online",
    "Research study shows benefits of regular exercise according to health officials",
    "Official government report confirms safety of approved vaccines",
    "Scientists at university research lab make breakthrough discovery",
    "Health organization publishes data on disease prevention",
    "Economic report shows stable growth according to federal data",
    "Educational research indicates improved student outcomes with new methods",
    "University researchers publish findings in peer-reviewed journal",
    "Government agency releases quarterly economic report",
    "Medical experts confirm effectiveness of approved treatment",
    "Federal officials announce new infrastructure development project",
    "Research institute reports progress on scientific investigation",
    "Official statement issued by health authorities on disease outbreak",
    "Scientists conduct peer-reviewed research on medical breakthrough",
    "Government statistics show economic improvement quarter over quarter",
]

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def read_news_csv(path, label):
    examples = []

    if not os.path.exists(path):
        return examples

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            title = (row.get("title") or "").strip()
            body = (row.get("text") or "").strip()
            subject = (row.get("subject") or "").strip()

            text = " ".join(part for part in [title, body, subject] if part)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) >= 80:
                examples.append((text[:5000], label))

    return examples


def load_training_data():
    csv_examples = read_news_csv(FAKE_CSV, 0) + read_news_csv(TRUE_CSV, 1)

    if csv_examples:
        texts, labels = zip(*csv_examples)
        return list(texts), list(labels), f"CSV dataset ({len(texts)} articles)"

    texts = FAKE_NEWS + REAL_NEWS
    labels = [0] * len(FAKE_NEWS) + [1] * len(REAL_NEWS)
    return texts, labels, f"fallback sample dataset ({len(texts)} examples)"


def extract_features(text):
    if not text:
        return [0] * 20

    words = text.split()
    wc = len(words)

    caps = sum(1 for w in words if w.isupper() and len(w) > 2)

    sensational = len(re.findall(
        r'breaking|shocking|alert|exposed|secret|viral|unbelievable|incredible',
        text.lower()
    ))

    credibility = len(re.findall(
        r'according to|official|report|research|study|confirmed|data|found|show|indicate|published',
        text.lower()
    ))

    quotes = len(re.findall(r'"[^"]*"', text))

    punctuation = text.count('!') + text.count('?')

    avg_word_len = sum(len(w) for w in words) / max(wc, 1)

    sentence_count = max(1, len(re.split(r'[.!?]+', text)))

    avg_sentence_len = wc / sentence_count

    numbers = len(re.findall(r'\d+', text))

    urls = len(re.findall(r'http[s]?://', text))

    conspiracy = len(re.findall(
        r'conspiracy|cover.?up|truth|expose|hidden|secret|reptil|alien|govt|government',
        text.lower()
    ))

    claim_words = len(re.findall(r'claim|say|said|reports|state', text.lower()))

    expert_reference = len(re.findall(
        r'doctor|scientist|professor|researcher|expert|official|government',
        text.lower()
    ))

    source_citation = len(re.findall(
        r'Reuters|AP|BBC|CNN|New York Times|Washington Post|study|journal|research',
        text
    ))

    time_reference = len(re.findall(
        r'\d{4}|january|february|march|april|may|june|july|august|september|october|november|december|today|yesterday|week|month|year',
        text.lower()
    ))

    return [
        wc,
        caps / max(wc, 1),
        punctuation,
        sensational,
        credibility,
        avg_word_len,
        avg_sentence_len,
        numbers,
        urls,
        quotes,
        text.count('%'),
        text.count('$'),
        len(set(words)) / max(wc, 1),
        len(re.findall(r'[A-Z]{3,}', text)),
        text.count(':'),
        conspiracy,
        claim_words,
        expert_reference,
        source_citation,
        time_reference
    ]


def extract_features_batch(texts):
    return np.array([extract_features(t) for t in texts])


def classify_probabilities(fake_p, real_p):
    if fake_p >= FAKE_THRESHOLD:
        return 0
    if real_p >= REAL_THRESHOLD:
        return 1
    return -1


def model_cache_is_fresh():
    if not os.path.exists(MODEL_CACHE):
        return False

    cache_time = os.path.getmtime(MODEL_CACHE)

    for path in [FAKE_CSV, TRUE_CSV]:
        if os.path.exists(path) and os.path.getmtime(path) > cache_time:
            return False

    return True


def train_model(texts, labels):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        stop_words='english',
        sublinear_tf=True,
        min_df=2 if len(texts) > 1000 else 1,
        max_df=0.85,
        norm='l2'
    )

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    feature_scaler = MaxAbsScaler()
    X_train_feats = feature_scaler.fit_transform(extract_features_batch(train_texts))
    X_test_feats = feature_scaler.transform(extract_features_batch(test_texts))

    X_train = sp.hstack([
        X_train_tfidf,
        sp.csr_matrix(X_train_feats)
    ])

    X_test = sp.hstack([
        X_test_tfidf,
        sp.csr_matrix(X_test_feats)
    ])

    clf = LogisticRegression(
        C=2.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='liblinear'
    )

    clf.fit(X_train, y_train)

    test_probas = clf.predict_proba(X_test)
    preds = np.array([
        classify_probabilities(float(fake_p), float(real_p))
        for fake_p, real_p in test_probas
    ])
    decisive_mask = preds != -1

    metrics = {
        "decisive_coverage": round(decisive_mask.mean() * 100, 2),
        "decisive_accuracy": round(accuracy_score(np.array(y_test)[decisive_mask], preds[decisive_mask]) * 100, 2),
        "report": classification_report(
            np.array(y_test)[decisive_mask],
            preds[decisive_mask],
            target_names=['FAKE', 'REAL'],
            zero_division=0
        )
    }

    return vectorizer, feature_scaler, clf, metrics


def load_or_train_model():
    texts, labels, dataset_source = load_training_data()

    if model_cache_is_fresh():
        cached = load(MODEL_CACHE)
        return (
            cached["vectorizer"],
            cached["feature_scaler"],
            cached["clf"],
            cached["metrics"],
            cached.get("dataset_source", dataset_source),
            True
        )

    vectorizer, feature_scaler, clf, metrics = train_model(texts, labels)

    os.makedirs(DATA_DIR, exist_ok=True)
    dump({
        "vectorizer": vectorizer,
        "feature_scaler": feature_scaler,
        "clf": clf,
        "metrics": metrics,
        "dataset_source": dataset_source
    }, MODEL_CACHE)

    return vectorizer, feature_scaler, clf, metrics, dataset_source, False

# ─────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────

vectorizer, feature_scaler, clf, model_metrics, dataset_source, loaded_from_cache = load_or_train_model()

print("\nMODEL EVALUATION")
print("=" * 50)
print("Training source:", dataset_source)
print("Model cache:", "loaded" if loaded_from_cache else "created")
print("Decision thresholds: FAKE >=", FAKE_THRESHOLD, "| REAL >=", REAL_THRESHOLD)
print("Decisive coverage:", model_metrics["decisive_coverage"], "%")
print("Decisive Test Accuracy:", model_metrics["decisive_accuracy"], "%")
print(model_metrics["report"])
print("=" * 50)
print("Enhanced model trained successfully!")

# ─────────────────────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────────────────────

FAKE_SIGNALS = [
    "SECRET",
    "MIRACLE",
    "EXPOSED",
    "CONSPIRACY",
    "HIDDEN",
    "MICROCHIP",
    "ALIEN",
    "WORLD ORDER"
]

REAL_SIGNALS = [
    "official",
    "research",
    "study",
    "confirmed",
    "report",
    "data"
]

CLICKBAIT_WORDS = [
    "you won't believe",
    "must see",
    "mind blowing",
    "viral",
    "shocking"
]

UNSUPPORTED_POLICY_PATTERNS = [
    r'federal reserve (raises|raised|increases|increased|hikes|hiked) interest rates?',
    r'\b\d+\s*basis points?\b',
    r'consumer price index reports? published last month',
    r'employment data .* consumer price index',
    r'officials stated'
]

VAGUE_TIME_REFERENCES = [
    "last month",
    "recently",
    "this week",
    "today",
    "yesterday"
]

TRUSTED_DOMAINS = {
    "apnews.com": "Associated Press",
    "reuters.com": "Reuters",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "npr.org": "NPR",
    "pbs.org": "PBS",
    "theguardian.com": "The Guardian",
    "nytimes.com": "The New York Times",
    "washingtonpost.com": "The Washington Post",
    "wsj.com": "The Wall Street Journal",
    "thehindu.com": "The Hindu",
    "indianexpress.com": "The Indian Express",
    "hindustantimes.com": "Hindustan Times",
    "timesofindia.indiatimes.com": "The Times of India",
    "pib.gov.in": "Press Information Bureau",
    "who.int": "World Health Organization",
    "cdc.gov": "CDC",
    "nih.gov": "NIH",
    "nasa.gov": "NASA",
    "federalreserve.gov": "Federal Reserve",
    "rbi.org.in": "Reserve Bank of India",
    "sec.gov": "U.S. SEC",
    "gov.uk": "UK Government"
}

LOW_TRUST_DOMAIN_KEYWORDS = [
    "rumor",
    "viral",
    "click",
    "truth",
    "exposed",
    "conspiracy",
    "dailybuzz",
    "beforeitsnews"
]

TRUSTED_SOURCE_NAMES = [
    "Associated Press",
    "AP News",
    "Reuters",
    "BBC",
    "NPR",
    "PBS",
    "The Guardian",
    "New York Times",
    "Washington Post",
    "Wall Street Journal",
    "The Hindu",
    "Indian Express",
    "Hindustan Times",
    "Times of India",
    "Press Information Bureau",
    "World Health Organization",
    "CDC",
    "NIH",
    "NASA",
    "Federal Reserve",
    "Reserve Bank of India",
    "RBI"
]

# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────

def normalize_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def trusted_domain_match(domain):
    for trusted_domain, source_name in TRUSTED_DOMAINS.items():
        if domain == trusted_domain or domain.endswith("." + trusted_domain):
            return source_name

    return None


def attributed_source_mentions(text):
    mentions = []

    for name in TRUSTED_SOURCE_NAMES:
        escaped_name = re.escape(name)
        attribution_patterns = [
            r"\baccording to\s+(?:the\s+)?" + escaped_name + r"\b",
            r"\bciting\s+(?:the\s+)?" + escaped_name + r"\b",
            r"\breported by\s+(?:the\s+)?" + escaped_name + r"\b",
            r"\b" + escaped_name + r"\s+(reported|said|announced|confirmed|published|released|stated|wrote)\b"
        ]

        if any(re.search(pattern, text, re.IGNORECASE) for pattern in attribution_patterns):
            mentions.append(name)

    return mentions


def verify_trusted_source(text, source_url=None):
    indicators = []
    score = 0.0
    status = "not_verified"

    if source_url:
        domain = normalize_domain(source_url)
        source_name = trusted_domain_match(domain)

        if source_name:
            score += 0.25
            status = "trusted_source"
            indicators.append({
                "type": "good",
                "text": f"URL source is trusted: {source_name}"
            })
        elif any(keyword in domain for keyword in LOW_TRUST_DOMAIN_KEYWORDS):
            score -= 0.20
            status = "low_trust_source"
            indicators.append({
                "type": "warning",
                "text": f"URL domain looks low-trust: {domain}"
            })
        else:
            indicators.append({
                "type": "warning",
                "text": f"URL source is not in trusted-source list: {domain}"
            })

    source_mentions = attributed_source_mentions(text)

    if source_mentions:
        score += min(0.15, len(source_mentions) * 0.05)
        if status == "not_verified":
            status = "trusted_source_mentioned"
        indicators.append({
            "type": "good",
            "text": f"Mentions trusted source: {', '.join(source_mentions[:3])}"
        })

    if status == "not_verified":
        indicators.append({
            "type": "warning",
            "text": "No trusted source confirmation found in URL or text"
        })

    return {
        "status": status,
        "score": score,
        "indicators": indicators
    }


def analyze_text(text, source_url=None):

    if not text or len(text.strip()) < 10:
        return None

    X_tf = vectorizer.transform([text])

    X_f = feature_scaler.transform(extract_features_batch([text]))

    X_in = sp.hstack([
        X_tf,
        sp.csr_matrix(X_f)
    ])

    proba = clf.predict_proba(X_in)[0]

    fake_p = float(proba[0])
    real_p = float(proba[1])

    tu = text.upper()

    fake_hits = [s for s in FAKE_SIGNALS if s in tu]

    real_hits = [s for s in REAL_SIGNALS if s.lower() in text.lower()]

    click_hits = [s for s in CLICKBAIT_WORDS if s.lower() in text.lower()]

    unsupported_policy_hits = [
        pattern for pattern in UNSUPPORTED_POLICY_PATTERNS
        if re.search(pattern, text.lower())
    ]

    source_verification = verify_trusted_source(text, source_url)

    has_vague_policy_timing = (
        any(ref in text.lower() for ref in VAGUE_TIME_REFERENCES)
        and re.search(r'federal reserve|central bank|interest rates?', text.lower())
    )

    # Strong signal bonuses for better classification
    fake_adj = min(0.06, len(fake_hits) * 0.015 + len(click_hits) * 0.01)
    real_adj = min(0.08, len(real_hits) * 0.02)

    if len(unsupported_policy_hits) >= 2:
        fake_adj += 0.35
        real_adj = min(real_adj, 0.02)
    elif has_vague_policy_timing:
        fake_adj += 0.12
        real_adj = min(real_adj, 0.04)

    if source_verification["score"] > 0:
        real_adj += min(0.12, source_verification["score"])
    elif source_verification["score"] < 0:
        fake_adj += min(0.15, abs(source_verification["score"]))


    fake_p = fake_p + fake_adj - real_adj

    fake_p = max(0.01, min(0.99, fake_p))

    real_p = 1.0 - fake_p

    # Precision-first thresholds: borderline claims become UNCERTAIN instead
    # of being forced into FAKE or REAL.
    if fake_p >= FAKE_THRESHOLD:
        verdict = "FAKE"
    elif real_p >= REAL_THRESHOLD:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    confidence = max(fake_p, real_p)

    words = text.split()

    wc = len(words)

    caps_r = sum(
        1 for w in words if w.isupper() and len(w) > 2
    ) / max(wc, 1)

    excl = text.count('!')

    num_cnt = len(re.findall(r'\d+', text))

    avg_wl = sum(len(w) for w in words) / max(wc, 1)

    indicators = []

    if fake_hits:
        indicators.append({
            "type": "warning",
            "text": f"Suspicious signals: {', '.join(fake_hits[:3])}"
        })

    if click_hits:
        indicators.append({
            "type": "warning",
            "text": f"Clickbait language detected"
        })

    if len(unsupported_policy_hits) >= 2 or has_vague_policy_timing:
        indicators.append({
            "type": "warning",
            "text": "Date-sensitive policy claim uses vague attribution or timing"
        })

    indicators.extend(source_verification["indicators"])

    if real_hits and not (len(unsupported_policy_hits) >= 2 or has_vague_policy_timing):
        indicators.append({
            "type": "good",
            "text": f"Credible reporting language detected"
        })

    if num_cnt > 1:
        indicators.append({
            "type": "good",
            "text": "Contains factual statistics or figures"
        })

    return {
        "verdict": verdict,
        "fake_prob": round(fake_p * 100, 1),
        "real_prob": round(real_p * 100, 1),
        "confidence": round(confidence * 100, 1),
        "source_verification": {
            "status": source_verification["status"],
            "score": round(source_verification["score"] * 100, 1)
        },
        "word_count": wc,
        "indicators": indicators,
        "stats": {
            "caps_ratio": round(caps_r * 100, 1),
            "exclamations": excl,
            "numbers": num_cnt,
            "avg_word_len": round(avg_wl, 1)
        }
    }

# ─────────────────────────────────────────────────────────────
# FETCH ARTICLE
# ─────────────────────────────────────────────────────────────

def fetch_article(url):

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }

        r = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(r.text, 'html.parser')

        for tag in ['article', 'main', '.article-body', '.story-body', '#content']:
            el = soup.select_one(tag)

            if el:
                text = ' '.join(el.get_text().split())

                if len(text) > 100:
                    return text[:2000]

        paras = soup.find_all('p')

        text = ' '.join(p.get_text() for p in paras)

        return ' '.join(text.split())[:2000]

    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():

    data = request.get_json()

    text = data.get('text', '').strip()
    source_url = data.get('source_url', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = analyze_text(text, source_url=source_url or None)

    if not result:
        return jsonify({
            'error': 'Text too short'
        }), 400

    return jsonify(result)

@app.route('/fetch', methods=['POST'])
def fetch():

    data = request.get_json()

    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    text = fetch_article(url)

    if not text:
        return jsonify({
            'error': 'Could not extract article text'
        }), 400

    return jsonify({
        'text': text,
        'source_verification': verify_trusted_source(text, source_url=url)
    })

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
