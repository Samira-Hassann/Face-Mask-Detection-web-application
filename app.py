import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Minimalist & Clean CSS Design
st.markdown("""
    <style>
    /* Clean background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Hide Sidebar Completely */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Modern Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    .header-card h1 {
        color: #38bdf8;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
    }
    .header-card p {
        color: #94a3b8;
        margin-top: 8px;
        font-size: 1.1rem;
    }

    /* Minimalist Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8 0%, #0284c7 100%);
        color: #0f172a;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 14px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #7dd3fc 0%, #0369a1 100%);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4);
        color: #0f172a;
    }
    
    /* Metrics Customization */
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loader
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception:
    model = None

# 4. Main Header
st.markdown("""
    <div class="header-card">
        <h1>😷 AI Face Mask Detector</h1>
        <p>Smart real-time mask detection using deep learning</p>
    </div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file 'best.pt' not found. Please ensure it is in the project folder.")
else:
    # File Uploader
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Action Button directly above or below
        st.markdown("<br>", unsafe_allow_html=True)
        detect_btn = st.button("✨ Detect Face Mask")
        st.markdown("<br>", unsafe_allow_html=True)

        if detect_btn:
            with st.spinner("Analyzing image..."):
                img_array = np.array(image.convert("RGB"))
                results = model(img_array)
                res_plotted = results[0].plot()

                # Display Side-by-Side Results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4 style='color:#94a3b8; text-align:center;'>Original Image</h4>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                    
                with col2:
                    st.markdown("<h4 style='color:#38bdf8; text-align:center;'>Detection Result</h4>", unsafe_allow_html=True)
                    st.image(res_plotted, use_container_width=True)

                # Metrics Section
                boxes = results[0].boxes
                total_faces = len(boxes) if boxes is not None else 0
                
                st.markdown("---")
                m1, m2 = st.columns(2)
                m1.metric("Total Faces Detected", total_faces)
                m2.metric("Status", "Complete" if total_faces > 0 else "No Faces Found")
        else:
            # Display only original image nicely centered before clicking detect
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<h4 style='color:#94a3b8; text-align:center;'>Uploaded Image</h4>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
    else:
        st.markdown("""
            <div style='text-align: center; padding: 40px; border: 2px dashed #334155; border-radius: 16px; color: #64748b;'>
                <p style='font-size: 1.2rem; margin: 0;'>📥 Drag and drop or browse an image above to start</p>
            </div>
        """, unsafe_allow_html=True)
