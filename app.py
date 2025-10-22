import os
import re
import time
import random
import logging
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv, find_dotenv

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_dotenv_path = find_dotenv()
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)
    logger.info("Loaded .env from %s", _dotenv_path)
else:
    logger.info("No .env file found; relying on environment variables")

PROVIDER_MODEL = os.environ.get("PROVIDER_MODEL", "gemini-1.5-pro").strip()

KEYS = [
    os.environ.get("PROVIDER_KEY1"),
    os.environ.get("PROVIDER_KEY2"),
    os.environ.get("PROVIDER_KEY3"),
    os.environ.get("PROVIDER_KEY4"),
    os.environ.get("PROVIDER_KEY5"),
]
KEYS = [k.strip() for k in (KEYS or []) if k and k.strip()]

PRIMARY_GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

MAX_ATTEMPTS_PER_KEY = int(os.environ.get("MAX_ATTEMPTS_PER_KEY", 2))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", 0.5))

_SINGLE_DIGIT_RE = re.compile(r"\b([01])\b")

def masked_key(k: str) -> str:
    if not k:
        return "<none>"
    k = k.strip()
    return (k[:8] + "...") if len(k) > 8 else k

def _call_with_sdk_and_key(api_key: str, prompt: str) -> Optional[str]:
    if not api_key:
        return None
    api_key = api_key.strip()
    os.environ["GEMINI_API_KEY"] = api_key
    try:
        client = genai.Client()
    except Exception as e:
        logger.info("Failed to init genai.Client(): %s", str(e))
        return None
    try:
        response = client.models.generate_content(model=PROVIDER_MODEL, contents=prompt)
    except Exception as e:
        logger.info("SDK call failed for key %s: %s", masked_key(api_key), str(e))
        return None
    text = getattr(response, "text", None)
    if not isinstance(text, str):
        try:
            text = str(response)
        except Exception:
            return None
    return text.strip() if text else None

def classify_with_key_rotation(input_text: str) -> Optional[int]:
    prompt = (
        "Classify the following as phishing or safe. Reply with ONLY a single digit: "
        "'1' if phishing, '0' if safe. Do NOT include explanation or extra text.\n\n"
        f"Input: {input_text}"
    )
    keys_to_try = []
    if PRIMARY_GEMINI_KEY and PRIMARY_GEMINI_KEY.strip():
        keys_to_try.append(PRIMARY_GEMINI_KEY.strip())
    provider_keys = KEYS.copy()
    random.shuffle(provider_keys)
    keys_to_try.extend(provider_keys)

    if not keys_to_try:
        logger.error("No API keys configured (GEMINI_API_KEY or PROVIDER_KEY1..5).")
        return None

    for key in keys_to_try:
        logger.info("Trying key %s", masked_key(key))
        for attempt in range(MAX_ATTEMPTS_PER_KEY):
            raw = _call_with_sdk_and_key(key, prompt)
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
        logger.info("Next key after %d attempts with %s", MAX_ATTEMPTS_PER_KEY, masked_key(key))

    logger.error("All keys/attempts exhausted without valid prediction")
    return None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": PROVIDER_MODEL,
        "configured_provider_keys": len(KEYS),
        "primary_gemini_key_set": bool(PRIMARY_GEMINI_KEY and PRIMARY_GEMINI_KEY.strip())
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "missing 'text' field"}), 400
    text = data["text"]
    result = classify_with_key_rotation(text)
    if result is None:
        return jsonify({"error": "classification_failed"}), 502
    return jsonify({"prediction": int(result)}), 200

@app.route("/predict-smishing-model", methods=["POST"])
def predict_smishing_model():
    data = request.get_json(force=True, silent=True) or {}
    txt = data.get("text", "")
    suspicious_keywords = ["win", "prize", "urgent", "click", "verify", "account", "password", "bank"]
    pred = 1 if any(k in txt.lower() for k in suspicious_keywords) else 0
    return jsonify({"prediction": int(pred)}), 200

@app.route("/predict-url-model", methods=["POST"])
def predict_url_model():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "")
    ip_like = bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", url))
    pred = 1 if (len(url) > 80 or ip_like) else 0
    return jsonify({"prediction": int(pred)}), 200

if __name__ == "__main__":
    KEYS = [k.strip() for k in KEYS if k and k.strip()]
    logger.info("GEMINI_API_KEY set: %s", "<set>" if PRIMARY_GEMINI_KEY and PRIMARY_GEMINI_KEY.strip() else "<not set>")
    logger.info("Number of provider rotation keys configured: %d", len(KEYS))
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)