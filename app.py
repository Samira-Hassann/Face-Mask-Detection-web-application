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
    initial_sidebar_state="expanded"
)

# 2. Modern Glassmorphism & Clean CSS Design
st.markdown("""
    <style>
    /* Main Layout Styling */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header Section */
    .app-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 20px 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
    }
    .app-header h1 {
        color: #38bdf8;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    .app-header p {
        color: #94a3b8;
        margin-top: 5px;
        font-size: 1rem;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    .sidebar-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    
    /* Image Display Container */
    .img-container {
        border: 1px solid #334155;
        background-color: #1e293b;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }

    /* Button Customization */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loader Function
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# 4. Sidebar / Control Panel
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #38bdf8;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Model Status Box
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0; color:#f8fafc;'>Model Status</h4>", unsafe_allow_html=True)
    try:
        model = load_model()
        st.markdown("<p style='color:#4ade80; margin:5px 0 0 0;'>🟢 Loaded Successfully</p>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown("<p style='color:#f87171; margin:5px 0 0 0;'>🔴 Model File Missing</p>", unsafe_allow_html=True)
        model = None
    st.markdown("</div>", unsafe_allow_html=True)

    # Instructions Box
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0 0 8px 0; color:#f8fafc;'>Quick Guide</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color:#94a3b8; margin:0;'>1. Upload an image file.<br>2. Preview image in workspace.<br>3. Click <b>Run Detection</b> for analysis.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 5. Header Section
st.markdown("""
    <div class="app-header">
        <h1>😷 Face Mask Detection System</h1>
        <p>Automated visual inspection using deep learning technology</p>
    </div>
""", unsafe_allow_html=True)

# 6. Workspace Area
if model is not None:
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Grid layout for side-by-side view without whitespace
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='font-size:1.1rem; color:#cbd5e1;'>Original Image</h3>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("<h3 style='font-size:1.1rem; color:#cbd5e1;'>Detection Workspace</h3>", unsafe_allow_html=True)
            detection_container = st.empty()
            detection_container.image(image, caption="Awaiting Detection...", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Button
        if st.button("Run Mask Detection"):
            with st.spinner("Processing image via YOLO..."):
                img_array = np.array(image.convert("RGB"))
                results = model(img_array)
                res_plotted = results[0].plot()
                
                # Replace the image container directly with the result
                detection_container.image(res_plotted, caption="Processed Output", use_container_width=True)
                
                # Metrics Row
                boxes = results[0].boxes
                total_faces = len(boxes) if boxes is not None else 0
                
                st.markdown("---")
                m1, m2 = st.columns(2)
                m1.metric("Total Detected Faces", total_faces)
                m2.metric("Detection State", "Success" if total_faces > 0 else "No Faces Identified")
    else:
        st.info("Please upload an image file using the upload box above to begin.")
