# src/deployment/segment_and_predict_alphabet.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocessing.segment import segment_characters_from_image

CHAR_LABELS = [chr(i) for i in range(65, 91)]  # A-Z

def predict_segmented_characters(image_path, model):
    chars = segment_characters_from_image(image_path)
    predicted_labels = []

    for char_img in chars:
        prediction = model.predict(char_img)
        label_index = np.argmax(prediction)
        label = CHAR_LABELS[label_index]
        predicted_labels.append(label)

    return predicted_labels

if __name__ == "__main__":
    model = load_model("saved_models/emnist_alphabet_model.h5")
    image_folder = "data/custom_images_characters/"  # Folder with images like A-Z written on paper

    for filename in os.listdir(image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, filename)
            predictions = predict_segmented_characters(image_path, model)
            if predictions:
                print(f"{filename}: {' '.join(predictions)}")
            else:
                print(f"{filename}: [No characters found]")
