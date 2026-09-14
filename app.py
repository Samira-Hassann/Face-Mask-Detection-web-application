



import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# ضبط إعدادات الصفحة
st.set_page_config(page_title="Face Mask Detection", page_icon="😷")

st.title("😷 Face Mask Detection System")
st.write("Upload an image to detect whether people are wearing masks or not.")

# تحميل النموذج (قم بتغيير 'best.pt' إلى مسار نموذج YOLO الخاص بك)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("لم يتم العثور على ملف النموذج، يرجى التأكد من وجود ملف النموذج (مثل best.pt) في المجلد.")

# أداة رفع الصور
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة بواسطة PIL
    image = Image.open(uploaded_file)
    
    # عرض الصورة الأصلية باستخدام use_container_width المصححة
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect Mask"):
        with st.spinner("Processing image..."):
            # تحويل الصورة إلى مصفوفة Numpy لـ OpenCV / YOLO
            img_array = np.array(image.convert("RGB"))
            
            # تشغيل نموذج YOLO للتعرف على الكمامات
            results = model(img_array)
            
            # رسم النتائج على الصورة
            res_plotted = results[0].plot()
            
            # عرض الصورة بعد الاكتشاف
            st.image(res_plotted, caption="Detection Result", use_container_width=True)
