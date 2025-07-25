import os
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from src.preprocessing.robust_preprocess import robust_preprocess

# Load trained model
model = load_model("saved_models/mnist_digit_model.h5")

# Directory containing multiple custom images
image_dir = "data/custom_images/"
image_filenames = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

print(f" Found {len(image_filenames)} image(s) to predict.\n")

# Store all preprocessed images and names
preprocessed_images = []
image_names = []

for filename in image_filenames:
    image_path = os.path.join(image_dir, filename)
    try:
        img = robust_preprocess(image_path)
        preprocessed_images.append(img)
        image_names.append(filename)
    except Exception as e:
        print(f" Skipping {filename}: {str(e)}")

# Stack all images into a batch
if preprocessed_images:
    input_batch = np.vstack(preprocessed_images)
    predictions = model.predict(input_batch)

    for i, prediction in enumerate(predictions):
        label = np.argmax(prediction)
        confidence = np.max(prediction)
        print(f"{image_names[i]} ➜ Predicted: {label} (Confidence: {confidence:.2f})")
else:
    print(" No valid images found.")
