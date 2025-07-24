# src/deployment/predict.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist


def load_trained_model(model_path):
    model = load_model(model_path)
    print("✅ Model loaded from:", model_path)
    return model


def predict_digit(model, image):
    # Reshape and normalize
    image = image.reshape(1, 28, 28, 1).astype("float32") / 255.0
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_label, confidence


def test_on_sample():
    # Load test dataset
    (_, _), (X_test, y_test) = mnist.load_data()

    index = np.random.randint(0, len(X_test))
    image = X_test[index]
    true_label = y_test[index]

    # Load trained model
    model = load_trained_model('../../saved_models/mnist_digit_model.h5')

    # Predict
    predicted_label, confidence = predict_digit(model, image)

    # Display result
    plt.imshow(image, cmap="gray")
    plt.title(f"True: {true_label} | Predicted: {predicted_label} ({confidence:.2f})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_on_sample()
