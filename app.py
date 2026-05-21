from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LOAD MODELS
# =========================
model1 = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model/lesion_model_v2.h5")
)

model2 = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model/skin_cancer_model.h5")
)

IMG_SIZE = (224, 224)

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return "✅ API running"


# =========================
# PREDICTION PIPELINE
# =========================
def predict_pipeline(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # MODEL 1: lesion / no lesion
    pred1 = model1.predict(img_array)[0][0]
    print("🔎 Model1:", pred1)

    if pred1 > 0.5:
        return {
            "label": "No lesion",
            "confidence": float(pred1),
            "explanation": "Image does not match skin lesion patterns.",
            "advice": "Please upload a valid skin lesion image."
        }

    # MODEL 2: benign / malignant
    pred2 = model2.predict(img_array)[0][0]
    print("🔎 Model2:", pred2)

    if pred2 > 0.5:
        return {
            "label": "Malignant",
            "confidence": float(pred2),
            "explanation": "Irregular patterns and asymmetry detected in lesion.",
            "advice": "Consult a dermatologist immediately."
        }
    else:
        return {
            "label": "Benign",
            "confidence": float(pred2),
            "explanation": "Patterns are similar to benign lesions.",
            "advice": "No urgent action required, but monitor changes."
        }


# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    path = os.path.join(BASE_DIR, "temp.jpg")
    file.save(path)

    result = predict_pipeline(path)

    return jsonify(result)


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)