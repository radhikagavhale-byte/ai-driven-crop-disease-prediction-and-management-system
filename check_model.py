
import tensorflow as tf
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

# =========================
# Dataset Classes
# =========================

dataset_classes = sorted([
    folder
    for folder in os.listdir("dataset")
    if os.path.isdir(os.path.join("dataset", folder))
])

# =========================
# Model Information
# =========================

print("\n===== MODEL CHECK =====")

print("Model Input:", model.input_shape)
print("Model Output:", model.output_shape)

print("\n===== CLASS NAMES JSON =====")

print("Total JSON Classes:", len(classes))

for i, name in enumerate(classes):
    print(i, "->", name)

print("\n===== DATASET CLASSES =====")

print("Total Dataset Classes:", len(dataset_classes))

for i, name in enumerate(dataset_classes):
    print(i, "->", name)

# =========================
# Compare
# =========================

print("\n===== CHECK RESULT =====")

if model.output_shape[-1] != len(classes):
    print("❌ PROBLEM: Model output and JSON classes do NOT match!")

elif len(classes) != len(dataset_classes):
    print("❌ PROBLEM: JSON classes and Dataset classes do NOT match!")

elif classes != dataset_classes:
    print("❌ PROBLEM: Class ORDER is different!")

else:
    print("✅ Model, JSON and Dataset classes MATCH!")

print("\n===== CHECK COMPLETED =====")

