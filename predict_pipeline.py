import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import keras
import os

# =========================
# LOAD MODELS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model1_path = os.path.join(BASE_DIR, "model/lesion_model_v2.h5")
model2_path = os.path.join(BASE_DIR, "model/skin_cancer_model.h5")

model1 = keras.models.load_model(model1_path)
model2 = keras.models.load_model(model2_path)

IMG_SIZE = (224, 224)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_pipeline(img_path):

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # ===== Step 1: lesion detection =====
    pred1 = model1.predict(img_array)[0][0]

    print(f"🔎 Lesion model: {pred1:.4f}")

    # ⚠️ adjust based on your mapping
    if pred1 > 0.5:
        print("❌ No lesion detected")
        return

    print("✅ Lesion detected")

    # ===== Step 2: benign / malignant =====
    pred2 = model2.predict(img_array)[0][0]

    print(f"🔎 Diagnosis model: {pred2:.4f}")

    if pred2 > 0.5:
        print("⚠️ Malignant lesion")
    else:
        print("✅ Benign lesion")

# =========================
# TEST
# =========================
test_image = os.path.join(BASE_DIR, "test8.jpg")

predict_pipeline(test_image)