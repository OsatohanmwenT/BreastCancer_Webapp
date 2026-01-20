import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join("model", "breast_cancer_model.pkl")
model_data = None

try:
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model_data:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        features = [
            float(data["radius_mean"]),
            float(data["texture_mean"]),
            float(data["perimeter_mean"]),
            float(data["area_mean"]),
            float(data["concavity_mean"]),
        ]

        # Reshape and scale
        input_data = np.array(features).reshape(1, -1)
        scaler = model_data["scaler"]
        scaled_data = scaler.transform(input_data)

        # Predict
        model = model_data["model"]
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)

        # Get class name (0: Malignant, 1: Benign in sklearn usually, but let's check target_names)
        # However, the prompt says "diagnosis (target variable: Benign / Malignant)"
        # Sklearn: 0=Malignant, 1=Benign.
        # But let's use the stored target_names to be sure.
        target_names = model_data.get("target_names", ["Malignant", "Benign"])
        result = target_names[prediction[0]]

        # Capitalize for display
        result = result.capitalize()  # 'Malignant' or 'Benign'

        confidence = np.max(probability) * 100

        return jsonify({"prediction": result, "confidence": f"{confidence:.2f}%"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
