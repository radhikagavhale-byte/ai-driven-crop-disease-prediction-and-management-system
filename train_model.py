import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json

# =========================
# 1. Dataset
# =========================

dataset_path = "dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 123

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=False
)

classes = train_dataset.class_names
NUM_CLASSES = len(classes)

print("\n==============================")
print("TOTAL CLASSES:", NUM_CLASSES)
print("==============================")

for i, name in enumerate(classes):
    print(i, ":", name)

# =========================
# 2. Save Correct Class Names
# =========================

os.makedirs("model", exist_ok=True)

with open("model/class_names.json", "w") as f:
    json.dump(classes, f, indent=2)

# =========================
# 3. Improve Dataset Performance
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# =========================
# 4. Data Augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1)
])

# =========================
# 5. CNN Model
# =========================

model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

# =========================
# 6. Compile
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 7. Callbacks
# =========================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "model/crop_disease_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =========================
# 8. Training
# =========================

print("\n==============================")
print("STARTING TRAINING")
print("==============================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=callbacks
)

# =========================
# 9. Save Final Model
# =========================

model.save("model/crop_disease_model.keras")

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY!")
print("==============================")