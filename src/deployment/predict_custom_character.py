# src/deployment/predict_custom_character.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.preprocessing.image_utils import preprocess_image

# Path to your custom image
image_path = "data/custom_images_characters/single_char.jpg"  # Replace with your actual file

# Preprocess image
image = preprocess_image(image_path)

# Show input image
plt.imshow(image.reshape(28, 28), cmap="gray")
plt.title("Custom Input Given to Model")
plt.axis("off")
plt.show()

# Load EMNIST-trained model
model = load_model("saved_models/emnist_alphabet_model.h5")

# Predict
prediction = model.predict(image)
label_index = np.argmax(prediction)
predicted_char = chr(label_index + 65)  # Convert 0–25 → A–Z

print(f"\nPredicted Character: {predicted_char}")
