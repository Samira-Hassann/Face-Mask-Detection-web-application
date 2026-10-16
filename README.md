# 😷 Face Mask Detection using YOLOv8 & Streamlit Web App

An end-to-end Computer Vision web application that detects whether individuals are wearing face masks or not in real-time using the **YOLOv8** object detection model.

---

## 🚀 Live Demo & Resources
*   **Live Streamlit App:** [Access Web Application](https://face-mask-detection-web-application.streamlit.app/)
*   **Kaggle Notebook:** [View Training & EDA Notebook](https://www.kaggle.com/code/samoura/face-mask-detection-using-yolov8-model)

---

## 📌 Project Overview
This project focuses on building, training, evaluating, and deploying a state-of-the-art object detection model to classify and locate face masks. 

*   **Model Architecture:** YOLOv8 (Nano) by Ultralytics
*   **Dataset:** Custom dataset managed via Roboflow (`mask` and `no-mask` classes).
*   **Deployment Framework:** Streamlit

---

## 📊 Model Performance & Metrics
The model was fine-tuned for 50 epochs on a GPU and achieved strong evaluation results on the test set:
*   **mAP@50:** ~94.4%
*   **mAP@50-95:** ~67.8%

---

## 🛠️ Tech Stack & Libraries
*   **Python**
*   **Ultralytics YOLOv8** (Object Detection)
*   **Roboflow** (Dataset Management)
*   **OpenCV & Matplotlib** (Image Processing & Visualization)
*   **Pandas & NumPy** (Data Manipulation)
*   **Streamlit** (Web App Deployment)
*   **ONNX** (Model Export & Optimization)

---

## 📁 Repository Structure
```text
.
├── app.py             # Streamlit web application code
├── best.pt            # Trained YOLOv8 model weights
├── requirements.txt   # Required Python dependencies
└── README.md          # Project documentation
└── requirements.txt   
