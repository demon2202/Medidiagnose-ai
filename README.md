# 🏥 MediDiagnose-AI

<div align="center">

![MediDiagnose-AI Logo](https://img.shields.io/badge/MediDiagnose-AI-blue?style=for-the-badge\&logo=medical)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge\&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge\&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI-Powered Medical Diagnosis System**

*Detect skin cancer, heart conditions, breast cancer, and pneumonia using deep learning*

[Features](#-features) • [Installation](#-installation) • [Dataset Setup](#-dataset-setup) • [Usage](#-usage) • [API](#-api-documentation)

</div>

---

## 🎯 Overview

MediDiagnose-AI is a comprehensive medical diagnosis system that uses deep learning to analyze medical images and clinical inputs to predict multiple conditions.

**Modules included:**

* Skin Cancer Detection (HAM10000 – 7 classes)
* Heart Condition Detection (ECG image-based)
* Breast Cancer Detection (Ultrasound → BI-RADS)
* Pneumonia Detection (Chest X-ray)
* Symptom-based Disease Prediction

---

## ✨ Features

* Transfer Learning (EfficientNet, MobileNet)
* Focal Loss for class imbalance
* RESTful Flask API
* Confidence-based predictions
* Severity, urgency & recommendation system
* Demo fallback when models are unavailable

---

## 📁 Project Structure

```
medidiagnose-ai/
│
├── backend/
│   ├── server.py
│   └── uploads/
│
├── ml_model/
│   ├── image_classification.py
│   ├── train_breast_cancer_model.py
│   ├── train_heart_image_model.py
│   ├── Dataset/
│   ├── *.h5
│   ├── *.joblib
│   └── symptom_list.json
│
├── frontend
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📋 Requirements

* Python 3.8+
* 8GB RAM minimum (16GB recommended)
* Optional NVIDIA GPU (CUDA)

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🚀 Installation

```
git clone https://github.com/yourusername/medidiagnose-ai.git
cd medidiagnose-ai
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

| Dataset           | Purpose       | Source |
| ----------------- | ------------- | ------ |
| HAM10000          | Skin Cancer   | Kaggle |
| Chest X-Ray       | Pneumonia     | Kaggle |
| Breast Ultrasound | Breast Cancer | Kaggle |
| PTB-XL            | Heart ECG     | Kaggle |

Place datasets inside:

```
ml_model/Dataset/
```

---

## 🎓 Training Models

```
cd ml_model
python image_classification.py
python train_breast_cancer_model.py
python train_heart_image_model.py
```

Trained models will be saved automatically.

---

## 🖥️ Running the Server

```
cd backend
python server.py
```

Server runs at:

```
http://localhost:5000
```

---

## 📡 API Documentation

| Method | Endpoint         | Description   |
| ------ | ---------------- | ------------- |
| GET    | /health          | Server health |
| POST   | /analyze/skin    | Skin cancer   |
| POST   | /analyze/heart   | ECG image     |
| POST   | /analyze/breast  | Breast cancer |
| POST   | /analyze/xray    | Pneumonia     |
| POST   | /predict-disease | Symptoms      |
| POST   | /predict-heart   | Heart risk    |

---



