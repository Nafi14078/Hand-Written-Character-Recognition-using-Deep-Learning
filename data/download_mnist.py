# data/download_mnist.py

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def download_and_preview():
    print(" Downloading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(f" Train: {X_train.shape}, Test: {X_test.shape}")

    # Show first 5 images
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_train[i], cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
    plt.suptitle("Sample MNIST Digits")
    plt.show()

if __name__ == "__main__":
    download_and_preview()
