import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# 1. Configuration & Layout
st.set_page_config(
    page_title="VisionAI - Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Modern SaaS UI CSS
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Sidebar & Streamlit Footer */
    section[data-testid="stSidebar"], footer, header {
        display: none !important;
    }
    
    /* Hero Header */
    .hero-section {
        text-align: center;
        padding: 40px 20px 20px 20px;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Image Display Container Cards */
    .image-card {
        background: #161e2e;
        border: 1px solid #243044;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Primary Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 14px;
        border: none;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(2, 132, 199, 0.6);
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception:
    model = None

# 4. Header Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">😷 Face Mask Vision AI</div>
        <div class="hero-subtitle">Automated Mask Detection Platform</div>
    </div>
""", unsafe_allow_html=True)

# 5. Main Application Logic
if model is None:
    st.error("⚠️ Model file 'best.pt' not found. Please verify the file path.")
else:
    # Centered File Uploader Area
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Split View (Only when an image is selected)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card-title">📷 Original Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown('<div class="card-title">🔍 AI Result</div>', unsafe_allow_html=True)
            result_placeholder = st.empty()
            result_placeholder.image(image, caption="Ready to process", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Button
        if st.button("Start Detection"):
            with st.spinner("Analyzing image features..."):
                img_array = np.array(image.convert("RGB"))
                results = model(img_array)
                res_plotted = results[0].plot()

                # Render Detection Output
                result_placeholder.image(res_plotted, caption="Analysis Complete", use_container_width=True)

                # Metrics Summary
                boxes = results[0].boxes
                total_faces = len(boxes) if boxes is not None else 0
                
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric(label="Faces Detected", value=total_faces)
                m2.metric(label="Processing Status", value="Completed" if total_faces > 0 else "No Targets Found")
