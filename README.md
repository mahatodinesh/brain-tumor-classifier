# 🧠 Brain Tumor Classification (MRI)

This project focuses on classifying brain tumors into four categories using deep learning on MRI images.  
It includes:
- A **custom CNN** model built from scratch.
- A **MobileNetV2** transfer learning model.
- A **Streamlit web app** for easy image-based predictions.

---

##  Dataset

**Source:** [Kaggle Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets)

**Classes:**
- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## 🧠 Model Architectures

### 🔸 Custom CNN
- 3 Convolutional layers with MaxPooling
- Trained from scratch on the dataset

### 🔸 MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuned on brain tumor MRI dataset
- Grad-CAM supported for model explainability

---

##  Sample Metrics (Custom CNN Example)

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Glioma      | 0.80      | 0.57   | 0.67     |
| Meningioma  | 0.44      | 0.68   | 0.53     |
| No Tumor    | 0.91      | 0.75   | 0.82     |
| Pituitary   | 0.78      | 0.84   | 0.81     |

---

##  How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

##  Project Structure
```
brain-tumor-classifier/
│
├── app.py                        # Streamlit App
├── braintumer.h5                 # Trained CNN model
├── requirements.txt              # Required packages
├── README.md                     # Project description
└── brain_tumor_streamlit_app.zip # Optional zip archive for deployment
```

---

## 📄 License
This project is licensed under the **MIT License**.  
See the LICENSE file for details.
