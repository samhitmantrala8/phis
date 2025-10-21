import os
import re
import time
import random
import logging
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# config (env vars)
PROVIDER_MODEL = os.environ.get("PROVIDER_MODEL", "gemini-1.5-pro").strip()
KEYS = [
    os.environ.get("PROVIDER_KEY1"),
    os.environ.get("PROVIDER_KEY2"),
    os.environ.get("PROVIDER_KEY3"),
    os.environ.get("PROVIDER_KEY4"),
    os.environ.get("PROVIDER_KEY5"),
]
KEYS = [k for k in KEYS if k]

MAX_ATTEMPTS_PER_KEY = int(os.environ.get("MAX_ATTEMPTS_PER_KEY", 2))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", 0.5))

_SINGLE_DIGIT_RE = re.compile(r"\b([01])\b")

def masked_key(k: str) -> str:
    if not k:
        return "<none>"
    return k[:8] + "..." if len(k) > 8 else k

def call_gemini_sdk_with_key(api_key: str, prompt: str) -> Optional[str]:
    if not api_key:
        return None

    os.environ["GEMINI_API_KEY"] = api_key
    try:
        client = genai.Client()
        resp = client.models.generate_content(model=PROVIDER_MODEL, contents=prompt)
        if hasattr(resp, "text"):
            return resp.text.strip()
        return str(resp).strip()
    except Exception as e:
        logger.info("SDK call failed for key %s: %s", masked_key(api_key), str(e))
        return None

def classify_text_with_rotation(text: str) -> Optional[int]:
    if not KEYS:
        logger.error("No PROVIDER_KEY* environment variables set.")
        return None

    prompt = (
        "Classify the following as phishing or safe. Reply with ONLY a single digit: "
        "'1' if phishing, '0' if safe. Do NOT include explanation.\n\n"
        f"Input: {text}"
    )

    keys = KEYS.copy()
    random.shuffle(keys)

    for key in keys:
        logger.info("Trying key %s", masked_key(key))
        for attempt in range(MAX_ATTEMPTS_PER_KEY):
            logger.info("Attempt %d/%d with key %s", attempt + 1, MAX_ATTEMPTS_PER_KEY, masked_key(key))
            raw = call_gemini_sdk_with_key(key, prompt)
            if raw is None:
                time.sleep(BACKOFF_BASE * (attempt + 1))
                continue
            m = _SINGLE_DIGIT_RE.search(raw)
            if m:
                val = int(m.group(1))
                logger.info("Parsed prediction %d from key %s", val, masked_key(key))
                return val
            logger.info("Unparseable response (preview): %s", raw[:200])
            time.sleep(BACKOFF_BASE * (attempt + 1))
        logger.info("Moving to next key after %d attempts with key %s", MAX_ATTEMPTS_PER_KEY, masked_key(key))

    logger.error("All keys/attempts exhausted without valid prediction")
    return None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": PROVIDER_MODEL,
        "keys_configured": len(KEYS)
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "missing 'text' field"}), 400
    text = data["text"]
    pred = classify_text_with_rotation(text)
    if pred is None:
        return jsonify({"error": "classification_failed"}), 502
    return jsonify({"prediction": int(pred)}), 200

@app.route("/predict-smishing-model", methods=["POST"])
def predict_smishing_model():
    data = request.get_json(force=True, silent=True) or {}
    txt = data.get("text", "")
    suspicious_keywords = ["win", "prize", "urgent", "click", "verify", "account", "password", "bank"]
    pred = 1 if any(k in txt.lower() for k in suspicious_keywords) else 0
    return jsonify({"note": "mock_result_only", "prediction": int(pred)}), 200

@app.route("/predict-url-model", methods=["POST"])
def predict_url_model():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "")
    ip_like = bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", url))
    pred = 1 if (len(url) > 80 or ip_like) else 0
    return jsonify({"note": "mock_result_only", "prediction": int(pred)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
