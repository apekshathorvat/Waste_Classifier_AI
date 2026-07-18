# ==========================================
# Waste Classification using CNN
# Author: Apeksha Thorvat
# ==========================================

# Import Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dataset Path
dataset_path = "dataset/dataset-resized"

# Image Settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Display the dataset path
print("Dataset Path:", dataset_path)

# Load Dataset
train_data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training Data
train_generator = train_data.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation Data
validation_generator = train_data.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Display Dataset Information
print("\nClasses Found:")
print(train_generator.class_indices)

print("\nTotal Training Images:")
print(train_generator.samples)

print("\nTotal Validation Images:")
print(validation_generator.samples) 
# ==========================================
# Build CNN Model
# ==========================================

model = Sequential([
    
    # First Convolution Layer
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),

    # Second Convolution Layer
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Third Convolution Layer
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Convert Feature Maps to 1D
    Flatten(),

    # Fully Connected Layer
    Dense(128, activation='relu'),

    # Reduce Overfitting
    Dropout(0.5),

    # Output Layer
    Dense(6, activation='softmax')
])

# Display Model Summary
model.summary()
# ==========================================
# Compile the Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✅ Model Compiled Successfully!")
# ==========================================
# Train the Model
# ==========================================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=1
)

# ==========================================
# ==========================================
# Save the Model
# ==========================================

import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model
model.save("models/waste_classifier.keras")

print("\n✅ Model Saved Successfully!")