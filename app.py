import json
import io
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ====== CARICAMENTO MODELLO E SCALER =================================================
print("🔄 Caricamento modello e scaler...")
model = tf.keras.models.load_model("apple-model.keras", compile=False)
scaler = joblib.load("scaler.joblib")
print("✅ Modello e scaler caricati.")

class_ids = list(range(1, 21))
idx2class = {i: cid for i, cid in enumerate(class_ids)}

# ====== UTILITY =======================================================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)

def preprocess_features(feat_dict):
    keys = ["size","crunchiness","juiciness","acidity","sweetness"]
    raw = np.array([[feat_dict[k] for k in keys]], dtype="float32")
    return scaler.transform(raw).astype("float32")

def predict(image_bytes, features_dict):
    img = preprocess_image(image_bytes)
    feats = preprocess_features(features_dict)
    probs = model.predict({"img": img, "tab": feats}, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "predicted": idx2class[idx],
        "confidence": float(probs[idx]),
        "all_probs": {idx2class[i]: float(p) for i,p in enumerate(probs)}
    }

# ====== API ============================================================================
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    print("📥 Richiesta /predict")
    if "image" not in request.files:
        return jsonify({"error":"No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    features_raw = request.form.get("features")
    if not features_raw:
        return jsonify({"error":"Missing 'features'"}), 400

    try:
        features = json.loads(features_raw)
    except Exception as e:
        return jsonify({"error":f"Invalid JSON: {e}"}), 400

    required = {"size","crunchiness","juiciness","acidity","sweetness"}
    if not required.issubset(features):
        return jsonify({"error":"Missing or invalid 'features'"}), 400

    try:
        res = predict(image_bytes, features)
        print("✅ Predizione:", res)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error":f"Prediction failed: {e}"}), 500

# ====== RUN =============================================================================
if __name__ == "__main__":
    # ascolta su tutte le interfacce
    app.run(host="0.0.0.0", port=5001, debug=True)
