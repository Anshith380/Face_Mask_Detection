import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix


# =====================================================
# 1. PROJECT SETTINGS
# =====================================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

DATASET_PATH = "dataset"


# =====================================================
# 2. DATA PREPROCESSING
# =====================================================

# =====================================================
# 2. DATA PREPROCESSING
# =====================================================

# Training data: augmentation + normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20,

    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True
)


# Validation data: normalization only
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20
)


# =====================================================
# 3. LOAD TRAINING DATA
# =====================================================

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)


# =====================================================
# 4. LOAD VALIDATION DATA
# =====================================================

validation_data = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


# =====================================================
# 5. DISPLAY DATASET INFORMATION
# =====================================================

print("\n===================================")
print("DATASET INFORMATION")
print("===================================")

print("Training images:", train_data.samples)
print("Validation images:", validation_data.samples)

print("\nClass labels:")
print(train_data.class_indices)


# =====================================================
# 6. CREATE CNN MODEL
# =====================================================

model = Sequential([

    # First convolution block
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    MaxPooling2D(pool_size=(2, 2)),


    # Second convolution block
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2, 2)),


    # Third convolution block
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2, 2)),


    # Convert feature maps to 1D
    Flatten(),


    # Fully connected layer
    Dense(
        128,
        activation="relu"
    ),


    # Prevent overfitting
    Dropout(0.5),


    # Binary classification
    Dense(
        1,
        activation="sigmoid"
    )
])


# =====================================================
# 7. COMPILE MODEL
# =====================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# 8. DISPLAY MODEL
# =====================================================

print("\n===================================")
print("CNN MODEL")
print("===================================")

model.summary()


# =====================================================
# 9. EARLY STOPPING
# =====================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# =====================================================
# 10. TRAIN MODEL
# =====================================================

print("\n===================================")
print("STARTING TRAINING")
print("===================================")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# =====================================================
# 11. SAVE MODEL
# =====================================================

os.makedirs("models", exist_ok=True)

model.save("models/mask_detector.keras")

print("\n===================================")
print("MODEL SAVED")
print("===================================")

print("Saved as:")
print("models/mask_detector.keras")


# =====================================================
# 12. TRAINING & VALIDATION ACCURACY
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# =====================================================
# 13. TRAINING & VALIDATION LOSS
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


# =====================================================
# 14. MAKE PREDICTIONS
# =====================================================

validation_data.reset()

predictions = model.predict(validation_data)

predicted_classes = (
    predictions > 0.5
).astype(int).flatten()

true_classes = validation_data.classes


# =====================================================
# 15. CLASSIFICATION REPORT
# =====================================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=[
            "With Mask",
            "Without Mask"
        ]
    )
)


# =====================================================
# 16. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "With Mask",
        "Without Mask"
    ],
    yticklabels=[
        "With Mask",
        "Without Mask"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# =====================================================
# 17. FINAL MESSAGE
# =====================================================

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("===================================")