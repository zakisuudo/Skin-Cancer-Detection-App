import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model/lesion_model_v2.h5")

model = tf.keras.models.load_model(model_path)

# =========================
# PREDICT FUNCTION
# =========================
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    print(f"🔎 Raw prediction: {prediction:.4f}")

    if prediction < 0.001:
        print("👉 Strong Skin lesion detected")
    elif  (0.001 <= prediction < 0.002):
        print("👉 likely lesion")
    elif (0.002 <= prediction < 0.006):
        print("👉 Uncertain, could be lesion")
    else:
        print("👉 No lesion detected")

# =========================
# TEST IMAGE
# =========================
test_image_path = os.path.join(BASE_DIR, "test7.jfif")

predict(test_image_path)