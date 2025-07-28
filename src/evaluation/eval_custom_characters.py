# src/evaluation/eval_custom_characters.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.preprocessing.image_utils import preprocess_image

# Path to your folder of custom images (named like: A_sample.png, B_test.jpg, etc.)
folder_path = "data/custom_images_characters/"

# Load model
model = load_model("saved_models/emnist_alphabet_model.h5")

# Store results
y_true = []
y_pred = []

# Loop through all images
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            # True label from filename (first character)
            true_char = filename[0].upper()
            if not true_char.isalpha():
                continue
            true_index = ord(true_char) - 65  # A→0, B→1, ..., Z→25

            # Preprocess and predict
            img_path = os.path.join(folder_path, filename)
            image = preprocess_image(img_path)
            prediction = model.predict(image)
            predicted_index = np.argmax(prediction)

            # Append results
            y_true.append(true_index)
            y_pred.append(predicted_index)

            print(f"{filename}: Predicted = {chr(predicted_index + 65)}, Actual = {true_char}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Evaluation
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[chr(i + 65) for i in range(26)]))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[chr(i + 65) for i in range(26)],
            yticklabels=[chr(i + 65) for i in range(26)])
plt.title("Confusion Matrix - Custom Characters")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
