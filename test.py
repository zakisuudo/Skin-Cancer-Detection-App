import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model/skin_cancer_model.h5")
)

for layer in model.layers:
    print(layer.name)