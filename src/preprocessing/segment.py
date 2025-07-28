# src/preprocessing/segment.py
import cv2
import numpy as np

def segment_characters_from_image(image_path, debug=False):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize large images to manageable size (optional)
    if img.shape[0] > 1000 or img.shape[1] > 1000:
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Use adaptive thresholding for robust background separation
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Optional: Show thresholded image for debugging
    if debug:
        cv2.imshow("Thresholded Image", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Relaxed size filtering
        if w < 5 or h < 5 or w > 200 or h > 200:
            continue

        char_img = thresh[y:y+h, x:x+w]

        # Resize to 20x20 and pad to 28x28
        char_img = cv2.resize(char_img, (20, 20))
        padded = cv2.copyMakeBorder(char_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

        # Normalize
        processed = padded.astype("float32") / 255.0
        processed = processed.reshape(1, 28, 28, 1)
        segments.append((x, processed))  # Keep x for sorting

    # Sort left to right
    segments.sort(key=lambda tup: tup[0])
    return [img for _, img in segments]
