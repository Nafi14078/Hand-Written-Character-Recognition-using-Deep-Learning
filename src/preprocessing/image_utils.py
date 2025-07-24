# src/preprocessing/image_utils.py

import cv2
import numpy as np


def preprocess_image(image_path):
    """
    Loads and preprocesses an image for digit classification.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    print(f" Loading image from: {image_path}")

    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f" Image not found: {image_path}")

    # Resize to 28x28
    image = cv2.resize(image, (28, 28))

    # Invert colors if background is black and text is white
    if np.mean(image) > 127:
        image = 255 - image

    # Normalize and reshape for model input
    image = image.astype("float32") / 255.0
    image = image.reshape(1, 28, 28, 1)

    return image
