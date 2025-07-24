# data/preprocess_emnist.py

import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def load_emnist(split='letters'):
    print(f" Loading EMNIST ({split}) from tensorflow_datasets...")
    ds, info = tfds.load(f'emnist/{split}', split='train', with_info=True, as_supervised=True)

    num_classes = info.features['label'].num_classes
    print(f" Loaded {split} dataset with {num_classes} classes.")

    # Convert dataset to NumPy arrays
    X, y = [], []
    for image, label in tfds.as_numpy(ds.take(1000)):  # limit to 1000 samples for quick preview
        X.append(image)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Show a few samples
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X[i].squeeze(), cmap="gray")
        plt.title(f"Label: {y[i]}")
        plt.axis("off")
    plt.suptitle(f"Sample EMNIST ({split}) Images")
    plt.show()

    return X, y


if __name__ == "__main__":
    load_emnist()
