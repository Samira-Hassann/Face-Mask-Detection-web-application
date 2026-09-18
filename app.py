import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# 1. Page Configuration (Centered & Clean)
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Minimalist & Modern CSS
st.markdown("""
    <style>
    /* Dark Theme Setup */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Completely Hide Sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Header Container */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
    }
    .header-card h1 {
        color: #38bdf8;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    .header-card p {
        color: #94a3b8;
        margin-top: 6px;
        font-size: 1rem;
    }

    /* Action Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8 0%, #0284c7 100%);
        color: #0f172a;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 10px;
        padding: 12px;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #7dd3fc 0%, #0369a1 100%);
        color: #0f172a;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load YOLO Model
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
        <h1>😷 Face Mask Detection System</h1>
        <p>Upload an image to identify face mask compliance</p>
    </div>
""", unsafe_allow_html=True)

# 5. Main Content Area
if model is None:
    st.error("⚠️ Model file 'best.pt' was not found. Please place it in the same project directory.")
else:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display side-by-side columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4 style='color:#cbd5e1;'>Original Image</h4>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("<h4 style='color:#38bdf8;'>Detection Result</h4>", unsafe_allow_html=True)
            result_placeholder = st.empty()
            # Shows original image until user clicks detect
            result_placeholder.image(image, caption="Ready for detection", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Run Mask Detection"):
            with st.spinner("Processing image..."):
                img_array = np.array(image.convert("RGB"))
                results = model(img_array)
                res_plotted = results[0].plot()

                # Update the right side with actual detection results
                result_placeholder.image(res_plotted, caption="Detection Finished", use_container_width=True)

                # Simple Summary
                boxes = results[0].boxes
                total_faces = len(boxes) if boxes is not None else 0
                
                st.markdown("---")
                m1, m2 = st.columns(2)
                m1.metric("Total Faces Detected", total_faces)
                m2.metric("Status", "Complete" if total_faces > 0 else "No Faces Detected")
