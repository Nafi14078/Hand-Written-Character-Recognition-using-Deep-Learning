# src/check_datasets.py

import os
from tensorflow.keras.datasets import mnist
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def check_mnist():
    print("🔍 Checking MNIST dataset...")
    try:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print(f" MNIST Loaded: Train: {X_train.shape}, Test: {X_test.shape}")
    except Exception as e:
        print(" Failed to load MNIST:", e)

def main():
    print(" Dataset Availability Check\n-----------------------------")
    check_mnist()
    # Optionally add EMNIST here later if using it

if __name__ == "__main__":
    main()
