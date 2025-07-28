# data/preprocess_emnist.py using tensorflow_datasets
import tensorflow_datasets as tfds
import numpy as np

print("Loading EMNIST Letters from TensorFlow Datasets...")
ds_train, ds_test = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True,
    batch_size=-1
)

# Convert full datasets to numpy
x_train, y_train = tfds.as_numpy(ds_train)
x_test, y_test = tfds.as_numpy(ds_test)

# Normalize images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape if needed
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Labels are 1–26 → convert to 0–25
y_train = y_train - 1
y_test = y_test - 1

# Save as .npz
np.savez_compressed("data/emnist_letters.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
print(" EMNIST dataset loaded and saved as emnist_letters.npz")
