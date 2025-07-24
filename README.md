# ✍️ Handwritten Character Recognition using deep learning

This project focuses on using **Convolutional Neural Networks (CNN)** for recognizing handwritten characters and digits from image data.

## 📌 Project Overview

The goal of this project is to accurately identify handwritten:
- **Digits (0–9)** using the **MNIST** dataset
- **Alphabets & characters** using the **EMNIST** dataset

This serves as an essential component of OCR (Optical Character Recognition) systems and can be extended to build full-text recognition models.

---

## 🧠 Technologies Used

- Python 🐍
- TensorFlow / Keras
- Matplotlib, NumPy, Scikit-learn
- EMNIST & MNIST datasets
- Jupyter Notebook

---

## 📂 Project Structure

├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── download_mnist.py
│   └── preprocess_emnist.py
│
├── notebooks/
│   ├── MNIST_Model.ipynb
│   └── EMNIST_Model.ipynb
│
├── src/
│   ├── main.py
│   ├── check_datasets.py
│   ├── models/
│   │   └── cnn_model.py
│   ├── preprocessing/
│   │   └── image_utils.py
│   ├── evaluation/
│   │   └── plot_metrics.py
│   └── deployment/
│       └── predict.py
│
├── saved_models/
│   ├── mnist_digit_model.h5
│   └── emnist_alphabet_model.h5
│
└── outputs/
    └── prediction_visuals/  


---

## 📊 Datasets Used

1. **MNIST** – 60,000 training and 10,000 testing images of handwritten digits (0-9).
2. **EMNIST (Balanced)** – 47-class dataset with letters and digits. Used for multi-character recognition.

> EMNIST is loaded using [`extra-keras-datasets`](https://pypi.org/project/extra-keras-datasets/)

---

## 🛠 How to Run the Project

### ✅ Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
###  ✅ Step 2: Run the script

# For digit recognition
python mnist_digit_recognition.py

# For character recognition
python emnist_alphabet_recognition.py

# Results

| Dataset | Model Accuracy | Epochs |
| ------- | -------------- | ------ |
| MNIST   | 99%+         | 5      |
| EMNIST  | \~85–90%       | 5      |

# Sample Output

#💡Future Improvements

1.Use CRNN for word/sentence recognition

2.Add image augmentation to improve generalization

3.Convert model to TensorFlow Lite for mobile deployment




