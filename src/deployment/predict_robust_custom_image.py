from src.preprocessing.robust_preprocess import robust_preprocess
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = load_model("saved_models/mnist_digit_model.h5")

# Set image path
image_path = "data/custom_images/eight_image.jpg"

# Preprocess with robust method
image = robust_preprocess(image_path, show_steps=True)

# Predict
prediction = model.predict(image)
label = np.argmax(prediction)
confidence = np.max(prediction)

print(f" Predicted Label: {label} (Confidence: {confidence:.2f})")
