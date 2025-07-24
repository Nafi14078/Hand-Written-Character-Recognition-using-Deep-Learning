# src/main.py

from tensorflow.keras.datasets import mnist
from src.models.cnn_model import create_cnn_model
from src.evaluation.plot_metrics import plot_confusion_matrix, plot_accuracy_loss
from tensorflow.keras.utils import to_categorical
import os

MODEL_PATH = "saved_models/mnist_digit_model.h5"
PLOT_OUTPUT = "outputs/prediction_visuals/"


def prepare_data():
    print("📥 Loading and preprocessing MNIST data...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize and reshape
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return X_train, y_train_cat, X_test, y_test_cat, y_test


def run_pipeline():
    X_train, y_train_cat, X_test, y_test_cat, y_test = prepare_data()

    model = create_cnn_model()
    print("🚀 Starting model training...")
    history = model.fit(X_train, y_train_cat, epochs=5, batch_size=128,
                        validation_data=(X_test, y_test_cat))

    print("✅ Training complete. Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"\n🎯 Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"💾 Model saved at {MODEL_PATH}")

    # Plot metrics
    os.makedirs(PLOT_OUTPUT, exist_ok=True)
    plot_accuracy_loss(history, PLOT_OUTPUT)
    plot_confusion_matrix(model, X_test, y_test, PLOT_OUTPUT)


if __name__ == "__main__":
    run_pipeline()
