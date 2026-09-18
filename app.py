import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Professional UI Design
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #e0e0e0;
        margin-top: 8px;
        font-size: 1.1rem;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }

    /* Custom Button */
    .stButton>button {
        width: 100%;
        background-color: #1e3c72;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2a5298;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading Function
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# 4. Sidebar Content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/facial-recognition.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("Model Status")
    try:
        model = load_model()
        st.success("YOLO Model Loaded Successfully")
    except Exception as e:
        st.error("Model file not found. Ensure 'best.pt' is in the project directory.")
        model = None
        
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    1. Upload a clear image (`JPG`, `JPEG`, or `PNG`).
    2. Click **Run Detection**.
    3. Review the detection results and analytics.
    """)

# 5. Header Section
st.markdown("""
    <div class="main-header">
        <h1>😷 AI Face Mask Detection System</h1>
        <p>Automated real-time safety monitoring using YOLO object detection</p>
    </div>
""", unsafe_allow_html=True)

# 6. Main Content Area
if model is not None:
    uploaded_file = st.file_uploader("Upload Image for Analysis", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display layout in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Detection Result")
            detection_placeholder = st.empty()
            detection_placeholder.info("Click the button below to start detection.")

        st.markdown("---")
        
        # Detection Trigger
        if st.button("Run Mask Detection"):
            with st.spinner("Analyzing image..."):
                img_array = np.array(image.convert("RGB"))
                results = model(img_array)
                res_plotted = results[0].plot()
                
                # Render result in right column
                detection_placeholder.image(res_plotted, use_container_width=True)
                
                # Render detection analytics
                boxes = results[0].boxes
                total_detections = len(boxes) if boxes is not None else 0
                
                st.subheader("Analytics Summary")
                m1, m2 = st.columns(2)
                m1.metric(label="Total Faces Detected", value=total_detections)
                m2.metric(label="Detection Status", value="Complete" if total_detections > 0 else "No Faces Found")
