# 😄 Face Emotion Recognition using Deep Learning

## 📌 Overview
This project aims to build a deep learning model capable of recognizing human emotions from facial images using Convolutional Neural Networks (CNNs). Leveraging the **FER-2013** dataset, the system classifies facial expressions into seven distinct categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

---

## 🎯 Problem Statement
Recognizing emotions from facial expressions is a complex task due to variability in facial features, lighting, and occlusion. This project tackles these challenges by designing a robust CNN-based model to classify grayscale facial images into emotion categories reliably.

---

## 🧠 Objective
- Build a CNN model from scratch to classify facial expressions.
- Train and evaluate the model on the FER-2013 dataset.
- Use data augmentation and normalization for better generalization.
- Visualize predictions and evaluate using performance metrics.

---

## 📁 Dataset
- **Name**: FER-2013 (Facial Expression Recognition)
- **Source**: [Kaggle - FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Size**: 48x48 grayscale face images
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## 🧰 Tools & Libraries
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV (optional)

---

## 🧱 Model Architecture
- Multiple convolutional layers with ReLU activation
- MaxPooling for dimensionality reduction
- Dropout layers for regularization
- Dense layers with softmax output (7 classes)

---

## 🔄 Data Preprocessing
- Reshaping and normalizing image pixel values
- One-hot encoding of labels
- Train-validation-test split
- Data augmentation (horizontal flip, rotation, zoom)

---

## 📊 Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

## 📈 Results
- Visualized training and validation accuracy/loss
- Confusion matrix for test set performance
- Emotion-wise prediction examples with image outputs


## 🙌 Acknowledgements
- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- TensorFlow and Keras for model development
- Special thanks to mentors and the AI community

---

## 📝 License
This project is for academic and non-commercial use only.

