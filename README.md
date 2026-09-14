# 😷 Face Mask Detection Web App

A computer vision web application built with **YOLOv8** and **Streamlit** to detect face mask compliance in images.

## 📌 Project Overview
This application uses a custom-trained YOLOv8 Nano model trained on the Mask-Wearing Dataset from Roboflow to classify faces into two categories:
* **mask** (Green Bounding Box)
* **no-mask** (Red Bounding Box)

## 📁 Repository Structure
```text
.
├── app.py             # Streamlit web application code
├── best.pt            # Trained YOLOv8 model weights
├── requirements.txt   # Required Python dependencies
└── README.md          # Project documentation
