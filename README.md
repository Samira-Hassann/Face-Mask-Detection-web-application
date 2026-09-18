# 😷 Face Mask Detection Web App

A computer vision web application built with **YOLOv8** and **Streamlit** to detect face mask compliance in images.

## 📌 Project Overview
This application uses a custom-trained YOLOv8 Nano model trained on the Mask-Wearing Dataset from Roboflow to classify faces into two categories:
* **mask** (Green Bounding Box)
* **no-mask** (Red Bounding Box)
---

## 🔗 Quick Links
* 🌐 **Live Web App:** [Face Mask Detection App](https://face-mask-detection-web-application.streamlit.app/)

🔗 **Kaggle Notebook:** [Face Mask Detection YOLOv8 Notebook](https://www.kaggle.com/code/samoura/mask-detection/edit/run/349773664)
---

## 📁 Repository Structure
```text
.
├── app.py             # Streamlit web application code
├── best.pt            # Trained YOLOv8 model weights
├── requirements.txt   # Required Python dependencies
└── README.md          # Project documentation
