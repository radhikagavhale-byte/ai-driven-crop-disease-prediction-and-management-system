
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(
    "model/crop_disease_model.keras"
)

# =========================
# Load Class Names
# =========================

with open("model/class_names.json", "r") as f:
    classes = json.load(f)

print("Total Classes:", len(classes))

# =========================
# Test Folder
# =========================

test_folder = "dataset/Tomato___Late_blight"

images = [
    f for f in os.listdir(test_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

if len(images) == 0:
    print("No image found!")
    exit()

# Test first image
image_path = os.path.join(
    test_folder,
    images[0]
)

print("\nTesting image:", images[0])
print("Actual class: Tomato___Late_blight")

# =========================
# Load Image
# =========================

image = Image.open(image_path).convert("RGB")

# =========================
# Resize
# =========================

input_size = model.input_shape[1:3]

print("Model input size:", input_size)

image = image.resize(input_size)

# =========================
# Convert to Array
# =========================

img = np.array(
    image,
    dtype=np.float32
)

# IMPORTANT:
# Do NOT divide by 255.
# Model already contains Rescaling(1/255).

img = np.expand_dims(
    img,
    axis=0
)

print("Final image shape:", img.shape)

# =========================
# Prediction
# =========================

prediction = model.predict(
    img,
    verbose=0
)

predicted_index = np.argmax(
    prediction[0]
)

predicted_class = classes[predicted_index]

confidence = (
    float(prediction[0][predicted_index]) * 100
)

print("\nPredicted class:", predicted_class)
print(
    "Confidence:",
    round(confidence, 2),
    "%"
)

print("\n===== TEST COMPLETED =====")

