
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2


#  Load Model and Class Names

MODEL_PATH = "braintumer.h5"
model = load_model(MODEL_PATH)

class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Update as per your dataset


#  App Title

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification")
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
