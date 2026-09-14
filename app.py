import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# App Title and Description
st.title("😷 Face Mask Detection System")
st.write("Upload an image to detect whether people are wearing masks or not.")

# Load Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Mask"):
        with st.spinner("Processing image..."):
            results = model(image)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detection Result", use_column_width=True)