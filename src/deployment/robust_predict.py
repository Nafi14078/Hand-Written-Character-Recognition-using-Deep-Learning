import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the EMNIST character model
model = load_model('saved_models/emnist_alphabet_model.h5')

# Define EMNIST class labels (1=A, 2=B, ..., 26=Z)
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


def preprocess_char(char_img):
    # Resize with aspect ratio preserved and padded to 28x28
    h, w = char_img.shape
    if h > w:
        char_img = cv2.resize(char_img, (int(w * (28 / h)), 28))
    else:
        char_img = cv2.resize(char_img, (28, int(h * (28 / w))))

    h, w = char_img.shape
    pad_top = (28 - h) // 2
    pad_bottom = 28 - h - pad_top
    pad_left = (28 - w) // 2
    pad_right = 28 - w - pad_left

    padded = cv2.copyMakeBorder(char_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    normalized = padded / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped


def segment_and_predict(image_path):
    print(f"Predicting from: {image_path}")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours (top-to-bottom then left-to-right)
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 15 < w < 200 and 15 < h < 200:  # size filter
            valid_contours.append((x, y, w, h))

    # Sort top-to-bottom and left-to-right
    sorted_boxes = sorted(valid_contours, key=lambda b: (b[1] // 50, b[0]))

    predictions = []
    for (x, y, w, h) in sorted_boxes:
        roi = thresh[y:y + h, x:x + w]
        processed = preprocess_char(roi)
        pred = model.predict(processed)
        label = classes[np.argmax(pred)]
        confidence = np.max(pred)
        predictions.append((label, confidence))

    return predictions


def show_predictions(image_path, predictions):
    img = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Predicted: " + " ".join([p[0] for p in predictions]))
    plt.show()
    for p in predictions:
        print(f"{p[0]} (Confidence: {p[1]:.2f})")


# --- MAIN ---
if __name__ == "__main__":
    test_image = "data/custom_images_characters/img_char.jpg"  # ← Replace with your image
    results = segment_and_predict(test_image)
    show_predictions(test_image, results)
