# 🧠 Obesity Prediction using Deep Learning (CNN + OpenCV)

## 📌 Description
This project presents a Deep Learning–based system for predicting abdominal obesity using 2D body images. Unlike traditional BMI-based methods, this system uses Computer Vision and CNN to analyze body shape and fat distribution.

The system processes input images, extracts body silhouette features, computes Waist-to-Hip Ratio (WHR), and classifies individuals into obesity risk categories.

---

## 🎯 Motivation
Abdominal obesity is linked to:
- Diabetes
- Heart diseases
- Metabolic disorders

Traditional BMI methods fail to detect fat distribution. This project provides a more accurate, non-invasive solution.

---

## ❗ Problem Statement
- BMI is not sufficient for obesity prediction  
- Need for automated and accurate detection  
- Requirement for image-based analysis  

---

## 🎯 Objectives
- Develop a deep learning model using CNN  
- Extract body features using Computer Vision  
- Classify obesity into risk categories  
- Build a web-based prediction system  

---

## 📊 Dataset
- **Dataset:** BodyM Dataset  
- **Total Images:** 8,978  
- **Subjects:** 2,505  

### Features:
- Height, Weight  
- Gender  
- Body silhouette images (front & side)  

---

## ⚙️ Methodology

### 🔹 Image Preprocessing
- Convert to grayscale  
- Apply thresholding  
- Extract silhouette  
- Resize to 128×128  
- Normalize pixel values  

### 🔹 CNN Model
1. Input Layer  
2. Convolution Layers  
3. ReLU Activation  
4. MaxPooling  
5. Flatten Layer  
6. Dense Layer  
7. Softmax Output  

---

## 🤖 Output Classes
- Normal  
- Central Obesity  
- High Risk  

---

## 📈 Results
- Model Accuracy: **~91%**
- Reliable classification of obesity levels

---

## 🖥️ Web Application
- Built using Flask  
- Upload image  
- Predict obesity level  
- Display result instantly  

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt


## 📥 Download Model

Due to GitHub file size limitations, the trained CNN model (`whr_model.h5`) is not included in this repository.

👉 Download the model from Google Drive:
https://drive.google.com/file/d/1ZSrCXwG0wD8q6dfev1VqLITt520JRle2/view?usp=drive_link

📌 After downloading, place the file in the project root directory:
