import json
import io
import os
import psutil
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import lru_cache

app = Flask(__name__)
CORS(app)

# ====== UTILITY =========================================================================

@lru_cache(maxsize=1)
def get_model():
    print("🔄 Caricamento modello...")
    return tf.keras.models.load_model("apple-model.keras", compile=False)

@lru_cache(maxsize=1)
def get_scaler():
    print("🔄 Caricamento scaler...")
    return joblib.load("scaler.joblib")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def preprocess_features(feat_dict):
    df = pd.DataFrame([feat_dict])
    return get_scaler().transform(df).astype("float32")

def predict(image_bytes, features_dict):
    img = preprocess_image(image_bytes)
    feats = preprocess_features(features_dict)
    probs = get_model().predict({"img": img, "tab": feats}, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "predicted": int(idx) + 1,
        "confidence": float(probs[idx]),
        "all_probs": {i+1: float(p) for i, p in enumerate(probs)}
    }

# ====== API =============================================================================

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    process = psutil.Process(os.getpid())
    print(f"📥 Richiesta /predict")
    print(f"🔍 RAM prima: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    print(f"📷 Image size: {len(image_bytes) / 1024:.2f} KB")

    features_raw = request.form.get("features")
    if not features_raw:
        return jsonify({"error": "Missing 'features'"}), 400

    try:
        features = json.loads(features_raw)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    required = {"size", "crunchiness", "juiciness", "acidity", "sweetness"}
    if not required.issubset(features):
        return jsonify({"error": "Missing or invalid 'features'"}), 400

    try:
        result = predict(image_bytes, features)
        print("✅ Predizione eseguita")
        print(f"🔍 RAM dopo: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        return jsonify(result)
    except Exception as e:
        print(f"❌ Errore durante la predizione: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500
