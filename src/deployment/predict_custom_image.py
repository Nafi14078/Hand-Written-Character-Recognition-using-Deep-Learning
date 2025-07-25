# src/deployment/predict_custom_image.py

from src.preprocessing.image_utils import preprocess_image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("saved_models/mnist_digit_model.h5")

# Set your custom image path
image_path = "data/custom_images/eight_image.jpg"

# Preprocess the image
image = preprocess_image(image_path)

#  Show the preprocessed image
plt.imshow(image.reshape(28, 28), cmap="gray")
plt.title("Input Given to Model")
plt.axis("off")
plt.show()

# Make prediction
prediction = model.predict(image)
label = np.argmax(prediction)
confidence = np.max(prediction)

# Display result
print(f" Predicted Label: {label} (Confidence: {confidence:.2f})")
