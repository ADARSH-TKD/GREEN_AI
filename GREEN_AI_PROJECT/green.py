import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# === Step 1: Dataset Preparation ===

# Set dataset path
dataset_path = "path/to/tree_dataset"

# Define image size
img_size = (128, 128)

# Load dataset
images = []
growth_labels = []  # Growth stage labels (e.g., 0: seedling, 1: sapling, 2: mature)
greenery_labels = []  # Greenery level as percentage

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        growth_stage = int(category.split('_')[0])  # Example: "0_seedling" -> growth_stage=0
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)  # Resize image
                images.append(img)
                growth_labels.append(growth_stage)
                greenery_labels.append(np.random.randint(50, 100))  # Replace with actual greenery levels

# Convert to NumPy arrays
images = np.array(images) / 255.0  # Normalize to [0, 1]
growth_labels = to_categorical(np.array(growth_labels), num_classes=3)  # 3 classes for growth stage
greenery_labels = np.array(greenery_labels) / 100.0  # Scale greenery to [0, 1]

# Split dataset
X_train, X_test, y_train_growth, y_test_growth, y_train_greenery, y_test_greenery = train_test_split(
    images, growth_labels, greenery_labels, test_size=0.2, random_state=42
)

# === Step 2: Build the CNN Model ===

# Define input layer
input_layer = Input(shape=(128, 128, 3))

# CNN layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output for growth stage classification
growth_output = Dense(3, activation='softmax', name='growth_stage')(x)

# Output for greenery level regression
greenery_output = Dense(1, activation='linear', name='greenery_level')(x)

# Build model
model = Model(inputs=input_layer, outputs=[growth_output, greenery_output])

# Compile model
model.compile(
    optimizer='adam',
    loss={'growth_stage': 'categorical_crossentropy', 'greenery_level': 'mse'},
    metrics={'growth_stage': 'accuracy', 'greenery_level': 'mae'}
)

model.summary()

# === Step 3: Train the Model ===

history = model.fit(
    X_train, 
    {'growth_stage': y_train_growth, 'greenery_level': y_train_greenery},
    validation_data=(X_test, {'growth_stage': y_test_growth, 'greenery_level': y_test_greenery}),
    epochs=10,
    batch_size=32
)

# === Step 4: Evaluate the Model ===

loss, growth_loss, greenery_loss, growth_acc, greenery_mae = model.evaluate(
    X_test, 
    {'growth_stage': y_test_growth, 'greenery_level': y_test_greenery}
)
print(f"Growth Stage Accuracy: {growth_acc}")
print(f"Greenery Level MAE: {greenery_mae}")

# === Step 5: Predict on New Image ===

# Load and preprocess new image
new_img = cv2.imread("path/to/new_tree_image.jpg")
new_img = cv2.resize(new_img, img_size) / 255.0
new_img = np.expand_dims(new_img, axis=0)

# Predict growth stage and greenery level
growth_pred, greenery_pred = model.predict(new_img)
print(f"Predicted Growth Stage: {np.argmax(growth_pred)}")  # Index corresponds to the growth stage
print(f"Predicted Greenery Level: {greenery_pred[0] * 100:.2f}%")
