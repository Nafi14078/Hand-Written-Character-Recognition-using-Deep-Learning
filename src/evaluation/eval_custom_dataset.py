import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.robust_preprocess import robust_preprocess

# Load model
model = load_model("saved_models/mnist_digit_model.h5")

# Ground truth labels based on file naming
image_dir = "data/custom_images/"
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]

y_true = []
y_pred = []

for filename in image_files:
    # Mapping of word to digit
    label_map = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9
    }

    # Extract the word from filename like "eight_image.jpg"
    word_label = filename.split("_")[0].lower()

    if word_label not in label_map:
        print(f"❌ Skipping unknown file format: {filename}")
        continue

    true_label = label_map[word_label]

    img_path = os.path.join(image_dir, filename)

    try:
        img = robust_preprocess(img_path)
        prediction = model.predict(img)
        pred_label = np.argmax(prediction)

        y_true.append(true_label)
        y_pred.append(pred_label)

        print(f"{filename} ➜ Predicted: {pred_label} (True: {true_label})")

    except Exception as e:
        print(f" Failed to process {filename}: {str(e)}")

# Show classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

# Show accuracy
acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy on custom dataset: {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix on Custom Images")
plt.tight_layout()
plt.show()
