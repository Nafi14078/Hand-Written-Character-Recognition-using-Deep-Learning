# src/preprocessing/robust_preprocess.py

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def robust_preprocess(image_path, show_steps=False):
    print(f" Robust preprocessing: {image_path}")

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to larger size for better processing
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)

    # Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Thresholding (binarize image)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (likely the digit)
    if len(contours) == 0:
        raise ValueError("No digit found in the image.")
    contour = max(contours, key=cv2.contourArea)

    # Get bounding box and crop
    x, y, w, h = cv2.boundingRect(contour)
    cropped = thresh[y:y+h, x:x+w]

    # Resize cropped digit to 20x20 and pad to 28x28
    digit = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # Normalize to 0-1 and reshape
    processed = padded.astype("float32") / 255.0
    processed = processed.reshape(1, 28, 28, 1)

    if show_steps:
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Grayscale Input")

        plt.subplot(1, 3, 2)
        plt.imshow(thresh, cmap='gray')
        plt.title("Thresholded")

        plt.subplot(1, 3, 3)
        plt.imshow(padded, cmap='gray')
        plt.title("Final Preprocessed")
        plt.show()

    return processed
