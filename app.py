from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import json
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Paths
# -------------------------------
SMISH_MODEL_DIR = "samhitmantrala/smish_fin"
RF_MODEL_PATH = "random_forest.joblib"
FEATURES_JSON = "feature_columns.json"
CSV_PATH = "phishing_site_urls.csv"

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# -------------------------------
# Load transformer model
# -------------------------------
print("Loading transformer model from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(SMISH_MODEL_DIR)
transformer_model = AutoModelForSequenceClassification.from_pretrained(SMISH_MODEL_DIR)
classifier = TextClassificationPipeline(model=transformer_model, tokenizer=tokenizer, top_k=None)
print("Transformer model loaded successfully.")

# -------------------------------
# Load Random Forest model if exists
# -------------------------------
rf_model = None
feature_columns = None

if os.path.exists(RF_MODEL_PATH) and os.path.exists(FEATURES_JSON):
    print("Loading Random Forest model and feature columns...")
    rf_model = joblib.load(RF_MODEL_PATH)
    with open(FEATURES_JSON, "r") as f:
        feature_columns = json.load(f)
    print("Random Forest model loaded successfully.")

# -------------------------------
# URL feature helpers
# -------------------------------
def abnormal_url(URL):
    hostname = urlparse(URL).hostname
    hostname = str(hostname)
    return 1 if re.search(hostname, URL) else 0

def having_ip_address(URL: str) -> int:
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', URL)
    return 1 if match else 0

def sum_count_special_characters(URL: str) -> int:
    special_chars = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
    return sum(char in special_chars for char in URL)

def httpSecured(URL: str) -> int:
    return 1 if urlparse(URL).scheme == 'https' else 0

def digit_count(URL: str) -> int:
    return sum(c.isdigit() for c in URL)

def letter_count(URL: str) -> int:
    return sum(c.isalpha() for c in URL)

def Shortining_Service(URL):
    match = re.search(
        r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
        r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
        r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
        r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
        r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
        r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
        r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
        r'tr\.im|link\.zip\.net', URL)
    return 1 if match else 0

# -------------------------------
# Routes
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "missing 'text' field"}), 400
    text = data["text"]
    try:
        result = classifier(text)
        if result:
            top = result[0][0]
            label = top["label"]
            score = top["score"]
            adjusted_score = score if label.upper() == "NEGATIVE" else 1 - score
            return jsonify({"label": label, "adjusted_score": adjusted_score, "raw": result}), 200
        return jsonify({"error": "empty result"}), 500
    except Exception as e:
        return jsonify({"error": "inference failure", "message": str(e)}), 500

@app.route("/predict-url", methods=["POST"])
def predict_sms():
    global rf_model, feature_columns
    if rf_model is None:
        return jsonify({"error": "Random Forest model not loaded. Train it first."}), 500
    data = request.get_json(force=True, silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "missing 'url' field"}), 400
    url = data["url"]
    try:
        numerical_values = {
            'url_len': len(url),
            'letters_count': letter_count(url),
            'digits_count': digit_count(url),
            'special_chars_count': sum_count_special_characters(url),
            'shortened': Shortining_Service(url),
            'abnormal': abnormal_url(url),
            'secure_http': httpSecured(url),
            'have_ip': having_ip_address(url),
        }
        X_input = np.array([numerical_values[feat] for feat in feature_columns]).reshape(1, -1)
        pred_int = rf_model.predict(X_input)[0]
        pred_label = "good" if pred_int == 1 else "bad"
        return jsonify({"prediction": pred_label, "prediction_int": int(pred_int)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train-sms", methods=["POST"])
def train_sms():
    global rf_model, feature_columns
    logs = []

    def log(msg):
        print(msg)
        logs.append(msg)

    if not os.path.exists(CSV_PATH):
        return jsonify({"error": f"{CSV_PATH} not found"}), 400

    try:
        log("Reading CSV...")
        dataset = pd.read_csv(CSV_PATH)

        log("Encoding labels...")
        lb_make = LabelEncoder()
        dataset["class_url"] = lb_make.fit_transform(dataset["Label"])
        dataset['URL'] = dataset['URL'].replace('www.', '', regex=True)

        log("Generating features...")
        dataset['url_len'] = dataset['URL'].apply(len)
        dataset['letters_count'] = dataset['URL'].apply(letter_count)
        dataset['digits_count'] = dataset['URL'].apply(digit_count)
        dataset['special_chars_count'] = dataset['URL'].apply(sum_count_special_characters)
        dataset['shortened'] = dataset['URL'].apply(Shortining_Service)
        dataset['abnormal'] = dataset['URL'].apply(abnormal_url)
        dataset['secure_http'] = dataset['URL'].apply(httpSecured)
        dataset['have_ip'] = dataset['URL'].apply(having_ip_address)

        features = ['url_len','letters_count','digits_count','special_chars_count','shortened',
                    'abnormal','secure_http','have_ip']
        X = dataset[features]
        y = dataset["class_url"]

        feature_columns = features
        with open(FEATURES_JSON, "w") as f:
            json.dump(feature_columns, f)

        log("Training Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
        rf_model.fit(X.values, y.values)

        log("Saving model to disk...")
        joblib.dump(rf_model, RF_MODEL_PATH)

        log("Training complete.")
        return jsonify({"message": "Random Forest trained successfully", "logs": logs}), 200

    except Exception as e:
        logs.append(str(e))
        return jsonify({"error": str(e), "logs": logs}), 500


# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
