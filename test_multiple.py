import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(
    "model/crop_disease_model.keras"
)

# =========================
# Load Correct Class Names
# =========================

with open("model/class_names.json", "r") as f:
    classes = json.load(f)

print("\nTotal Classes:", len(classes))

# =========================
# Apple Healthy Folder
# =========================

folder = "dataset/Apple___healthy"

images = [
    file for file in os.listdir(folder)
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

images = images[:5]

print("\n===== TESTING APPLE HEALTHY =====\n")

# =========================
# Model Input Size
# =========================

input_size = model.input_shape[1:3]

print("Model Input Size:", input_size)

# =========================
# Prediction
# =========================

for image_name in images:

    image_path = os.path.join(folder, image_name)

    image = Image.open(image_path).convert("RGB")

    image = image.resize(input_size)

    image = np.array(image, dtype=np.float32)

    # IMPORTANT:
    # Test code does NOT normalize here.
    # Model already contains Rescaling(1/255).

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    index = np.argmax(prediction[0])

    disease = classes[index]

    confidence = prediction[0][index] * 100

    print("Image:", image_name)
    print("Prediction:", disease)
    print("Confidence:", round(float(confidence), 2), "%")
    print("-----------------------------")

print("\nTEST COMPLETED!")