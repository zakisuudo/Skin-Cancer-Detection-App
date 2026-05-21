import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(BASE_DIR, "datasets/classification1/train")
val_dir = os.path.join(BASE_DIR, "datasets/classification1/val")
test_dir = os.path.join(BASE_DIR, "datasets/classification1/test")

# =========================
# DATA AUGMENTATION (IMPORTANT FOR OVERFITTING FIX)
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary'
)

print("📊 Classes:", train_data.class_indices)

# =========================
# BETTER MODEL (REGULARIZED CNN)
# =========================
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(256, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

# 🔥 Regularization (VERY IMPORTANT)
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(1, activation='sigmoid'))

# =========================
# COMPILATION (IMPORTANT OPTIMIZER TWEAK)
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS (PREVENT OVERFITTING)
# =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]

# =========================
# TRAINING
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=callbacks
)

# =========================
# EVALUATION
# =========================
loss, acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# =========================
# SAVE MODEL
# =========================
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "lesion_model_v2.h5")
model.save(model_path)

print(f"💾 Model saved at: {model_path}")