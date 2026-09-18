import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Page configuration
st.set_page_config(page_title="Face Mask Detection", page_icon="😷")

st.title("😷 Face Mask Detection System")
st.write("Upload an image to detect whether people are wearing masks or not.")

# Load the model (change 'best.pt' to your YOLO model path)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Model file not found. Please ensure the model file (e.g., best.pt) is in the directory.")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image with PIL
    image = Image.open(uploaded_file)
    
    # Display original image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect Mask"):
        with st.spinner("Processing image..."):
            # Convert image to NumPy array for OpenCV / YOLO
            img_array = np.array(image.convert("RGB"))
            
            # Run YOLO model for detection
            results = model(img_array)
            
            # Draw results on the image
            res_plotted = results[0].plot()
            
            # Display detection result
            st.image(res_plotted, caption="Detection Result", use_container_width=True)
