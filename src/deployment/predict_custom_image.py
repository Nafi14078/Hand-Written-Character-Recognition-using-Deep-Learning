# src/deployment/predict_custom_image.py
import image

from src.preprocessing.image_utils import preprocess_image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Show what the model actually saw
plt.imshow(image.reshape(28, 28), cmap="gray")
plt.title("Input Given to Model")
plt.axis("off")
plt.show()


# Load model
model = load_model("saved_models/mnist_digit_model.h5")

# Path to your custom digit image
image_path = "data/custom_images/eight_image.jpg"

# Preprocess and predict
image = preprocess_image(image_path)
prediction = model.predict(image)
label = np.argmax(prediction)

print(f" Predicted Label: {label}")
