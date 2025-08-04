import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown


#  Download model from Google Drive if missing

MODEL_PATH = "braintumer.h5"
GOOGLE_DRIVE_FILE_ID = "1ARKjYXd_1m3T2OYrKF3XJzcnc-9WvpBQ"  
URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model file...")
    gdown.download(URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")


#  Load Model and Class Names

model = load_model(MODEL_PATH)
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Update if needed


#  App Title

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title(" Brain Tumor Classification")
st.markdown("Upload an MRI brain image to detect the type of tumor.")


#  Upload Image

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.markdown(f"### 🧾 Prediction: **{pred_class.upper()}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")
