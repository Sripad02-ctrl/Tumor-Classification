# Tumor-Classification
# 🧠 Brain Tumor Classification using CNN

This project demonstrates an end-to-end **deep learning pipeline** for brain tumor detection from MRI images.  
A **custom Convolutional Neural Network (CNN)** was developed and compared with a fine-tuned **ResNet18** model to classify images as *malignant* or *benign*. The workflow covers data preprocessing, training, evaluation, and performance visualization — achieving high accuracy and reliable results on the test dataset.  

Built using **Python, PyTorch, and Scikit-learn**, this project highlights strong skills in model design, dataset handling, and deployment-ready ML development — ideal for healthcare and image-analysis applications.


<!--
Give a one-line overview of your project — what it does and why.
-->
This project uses a **Convolutional Neural Network (CNN)** to classify brain MRI images as either **malignant (tumor)** or **benign (no tumor)**.  
It includes model training, evaluation, and comparison with pre-trained architectures like **ResNet18**.

---
The dataset consists of MRI brain images categorized into:

Training/ – used to train the CNN

Testing/ – used to evaluate model performance

If the dataset is public, mention its source (e.g., Kaggle or OpenNeuro).
Example:

Dataset: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
🧠 Model Architecture
<!-- Explain what model(s) you used and why. -->

A custom CNN model was built and trained from scratch.

Model performance was compared with ResNet18, a pre-trained deep learning model using transfer learning.

Evaluation metrics include accuracy, precision, recall, and F1-score.
🚀 Training & Evaluation
<!-- Summarize how you trained and evaluated your model. -->

Images were preprocessed (resized, normalized, augmented).

The model was trained for multiple epochs using the Adam optimizer and cross-entropy loss.

Model performance was validated on the testing dataset.

Example results:

Model	                     Accuracy	F1-Score
Custom CNN	                 94.5%	0.93
ResNet18 (fine-tuned)      	96.2%	0.95

🧰 Tools & Libraries
<!-- List main frameworks and languages. -->

Python 3.x


PyTorch / TensorFlow

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn, Torchvision
