# 🧠 Number Detection Using Machine Learning

This project implements a simple machine learning model to detect handwritten numbers. The approach is based on converting image pixels into binary values — where light pixels are marked as `0` and dark pixels as `1` — and using those as input to a classifier.

## 📌 Overview

- Converts handwritten digit images into binary arrays.
- Light color (background) → `0`, Dark color (ink) → `1`
- Uses a machine learning algorithm (e.g., Logistic Regression, SVM, or Neural Network) to classify the digits.
- Trained on labeled image data (e.g., MNIST dataset or custom input).
- Predicts and displays the digit written by hand.

## 🧰 Technologies Used

- Python
- NumPy
- Scikit-learn
- OpenCV / PIL (for image processing)
- Jupyter Notebook (optional for visualization)

## 🔄 Workflow

1. **Preprocessing**
   - Load a grayscale image.
   - Threshold it to binary: light → 0, dark → 1.

2. **Feature Extraction**
   - Flatten the 2D binary image into a 1D feature vector.

3. **Model Training**
   - Train a classifier (e.g., Logistic Regression) on binary feature vectors.

4. **Prediction**
   - Pass a new image through the same process.
   - Use the trained model to predict the digit.
