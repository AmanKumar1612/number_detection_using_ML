# Install dependencies
!pip install tensorflow opencv-python-headless --quiet

import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from google.colab import files
import matplotlib.pyplot as plt

# --- STEP 1: Load digit dataset (MNIST 0–9)
(x_train, y_train), (_, _) = mnist.load_data()

# Normalize and reshape for digit model
x_digit = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_digit = to_categorical(y_train, 10)

# --- STEP 2: Create digit vs no-digit dataset (for binary model)
digit_images = x_train
digit_labels = np.ones(len(digit_images))  # Label: contains number = 1

# Generate noisy black images as more realistic "no digit"
def generate_noisy_blank_images(n):
    blanks = np.random.normal(loc=0.0, scale=0.1, size=(n, 28, 28))  # random noise
    blanks = np.clip(blanks, 0.0, 1.0)
    blanks = (blanks * 255).astype("uint8")
    return blanks

blank_images = generate_noisy_blank_images(len(digit_images))
blank_labels = np.zeros(len(blank_images))  # Label: no number = 0

# Combine both classes
x_bin = np.concatenate([digit_images, blank_images])
y_bin = np.concatenate([digit_labels, blank_labels])
x_bin = x_bin.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_bin = to_categorical(y_bin, 2)

# --- STEP 3: Binary Classifier (Digit vs No Digit)
binary_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])
binary_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("🔄 Training binary model (digit vs no-digit)...")
binary_model.fit(x_bin, y_bin, epochs=5, batch_size=64, validation_split=0.2)

# --- STEP 4: Digit Classifier (0–9)
digit_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
digit_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("🔄 Training digit model (0–9)...")
digit_model.fit(x_digit, y_digit, epochs=5, batch_size=64, validation_split=0.2)

# --- STEP 5: Prediction Function
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("❌ Could not load image.")

    # Apply adaptive threshold to isolate number
    img = cv2.resize(img, (28, 28))
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Normalize
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1)

def predict_and_display(img_path):
    try:
        img_input = preprocess_image(img_path)

        # Step 1: Check if number exists
        binary_pred = binary_model.predict(img_input)
        is_digit = np.argmax(binary_pred)

        # Load original image for display
        original = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        if is_digit == 1:
            # Step 2: Predict the digit (0–9)
            digit_pred = digit_model.predict(img_input)
            predicted_digit = np.argmax(digit_pred)
            confidence = np.max(digit_pred) * 100
            plt.title(f"✅ Digit Detected: {predicted_digit} ({confidence:.2f}%)")
        else:
            plt.title("❌ No Digit Detected")
        plt.show()

    except Exception as e:
        print(str(e))

# --- STEP 6: Upload Image
print("📂 Please upload one or more image files (digits or blanks)...")
uploaded = files.upload()

# Loop through each uploaded image
for img_path in uploaded.keys():
    print(f"\n📷 Predicting: {img_path}")
    predict_and_display(img_path)

