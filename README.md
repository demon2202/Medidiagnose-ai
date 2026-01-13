# Complete README Documentation for MediDiagnose-AI

## 📄 `README.md`

```markdown
# 🏥 MediDiagnose-AI

<div align="center">

![MediDiagnose-AI Logo](https://img.shields.io/badge/MediDiagnose-AI-blue?style=for-the-badge&logo=medical)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI-Powered Medical Diagnosis System**

*Detect skin cancer, heart conditions, breast cancer, and pneumonia using deep learning*

[Features](#-features) • [Installation](#-installation) • [Dataset Setup](#-dataset-setup) • [Usage](#-usage) • [API](#-api-documentation) • [Code Explanation](#-code-explanation)

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Project Structure](#-project-structure)
4. [Requirements](#-requirements)
5. [Installation](#-installation)
6. [Dataset Setup](#-dataset-setup)
7. [Training Models](#-training-models)
8. [Running the Server](#-running-the-server)
9. [API Documentation](#-api-documentation)
10. [Code Explanation](#-code-explanation)
11. [Model Architectures](#-model-architectures)
12. [Troubleshooting](#-troubleshooting)
13. [Contributing](#-contributing)
14. [Disclaimer](#-disclaimer)

---

## 🎯 Overview

MediDiagnose-AI is a comprehensive medical diagnosis system that uses deep learning to analyze medical images and predict various conditions. The system includes:

- **Skin Cancer Detection**: Classifies 7 types of skin lesions using the HAM10000 dataset
- **Heart Condition Detection**: Analyzes ECG images to detect cardiac abnormalities
- **Breast Cancer Detection**: Evaluates mammograms/ultrasounds for cancer indicators
- **Pneumonia Detection**: Analyzes chest X-rays for pneumonia signs
- **Symptom-based Disease Prediction**: Predicts diseases based on reported symptoms

### How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Input    │────▶│   Flask API     │────▶│   ML Models     │
│  (Image/Text)   │     │   (Backend)     │     │  (TensorFlow)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │◀────│   JSON Response │◀────│   Prediction    │
│   (Results)     │     │   (Diagnosis)   │     │   + Confidence  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## ✨ Features

### 🔬 Medical Analysis Capabilities

| Feature | Description | Accuracy Target |
|---------|-------------|-----------------|
| Skin Cancer | 7-class classification (HAM10000) | >85% |
| Heart Conditions | 5-class ECG analysis | >80% |
| Breast Cancer | 6-class BI-RADS classification | >82% |
| Pneumonia | Binary classification (Normal/Pneumonia) | >90% |
| Disease Prediction | Symptom-based multi-class | >85% |

### 🛠️ Technical Features

- ✅ Transfer Learning (EfficientNet, MobileNetV2)
- ✅ Focal Loss for class imbalance
- ✅ Data augmentation
- ✅ RESTful API
- ✅ CORS enabled
- ✅ Comprehensive error handling
- ✅ Demo mode when models unavailable

---

## 📁 Project Structure

```
medidiagnose-ai/
│
├── 📂 backend/
│   ├── 📄 server.py              # Flask API server
│   └── 📂 uploads/               # Temporary upload folder
│
├── 📂 ml_model/
│   ├── 📄 image_classification.py       # Skin & Pneumonia training
│   ├── 📄 train_breast_cancer_model.py  # Breast cancer training
│   ├── 📄 train_heart_image_model.py    # Heart condition training
│   │
│   ├── 📂 Dataset/                      # Datasets folder
│   │   ├── 📂 HAM10000/                 # Skin cancer dataset
│   │   │   ├── 📂 HAM10000_images_part_1/
│   │   │   ├── 📂 HAM10000_images_part_2/
│   │   │   └── 📄 HAM10000_metadata.csv
│   │   │
│   │   ├── 📂 chest_xray/               # Pneumonia dataset
│   │   │   ├── 📂 train/
│   │   │   │   ├── 📂 NORMAL/
│   │   │   │   └── 📂 PNEUMONIA/
│   │   │   ├── 📂 test/
│   │   │   └── 📂 val/
│   │   │
│   │   ├── 📂 breast_ultrasound/        # Breast cancer dataset
│   │   │   ├── 📂 normal/
│   │   │   ├── 📂 benign/
│   │   │   └── 📂 malignant/
│   │   │
│   │   └── 📂 ptb-xl/                   # ECG dataset
│   │       ├── 📂 records100/
│   │       ├── 📂 records500/
│   │       ├── 📄 ptbxl_database.csv
│   │       └── 📄 scp_statements.csv
│   │
│   ├── 📄 skin_cancer_model.h5          # Trained models
│   ├── 📄 skin_cancer_config.json
│   ├── 📄 pneumonia_model.h5
│   ├── 📄 breast_cancer_model.h5
│   ├── 📄 heart_image_model.h5
│   ├── 📄 disease_model.joblib
│   ├── 📄 label_encoder.joblib
│   └── 📄 symptom_list.json
│
├── 📂 frontend/                  # React/Vue frontend (optional)
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 .gitignore
```

---

## 📋 Requirements

### System Requirements

- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB for datasets + models
- **GPU**: NVIDIA GPU with CUDA (optional, speeds up training)

### Python Dependencies

```txt
# Core
python>=3.8
numpy>=1.19.0
pandas>=1.2.0

# Deep Learning
tensorflow>=2.10.0
keras>=2.10.0

# Image Processing
pillow>=8.0.0
opencv-python>=4.5.0

# Machine Learning
scikit-learn>=0.24.0
joblib>=1.0.0

# Web Framework
flask>=2.0.0
flask-cors>=3.0.0
werkzeug>=2.0.0

# ECG Processing (optional)
wfdb>=3.4.0

# Utilities
matplotlib>=3.3.0
tqdm>=4.50.0
```

### Install All Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install tensorflow numpy pandas pillow scikit-learn flask flask-cors joblib matplotlib wfdb
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/medidiagnose-ai.git
cd medidiagnose-ai
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
# Run this Python script to verify
import tensorflow as tf
import numpy as np
from PIL import Image
import flask

print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Flask: {flask.__version__}")
print("✅ All dependencies installed successfully!")
```

---

## 📥 Dataset Setup

### Overview of Required Datasets

| Dataset | Size | Classes | Download Link |
|---------|------|---------|---------------|
| HAM10000 | ~3GB | 7 | [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| Chest X-Ray | ~2GB | 2 | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Breast Ultrasound | ~200MB | 3 | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| PTB-XL ECG | ~2GB | 5+ | [Kaggle](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset) |

---

### 📦 Dataset 1: HAM10000 (Skin Cancer)

**Download Link**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

#### Step-by-Step Setup:

1. **Download from Kaggle**:
   - Go to the Kaggle link above
   - Click "Download" (you need a Kaggle account)
   - Download `archive.zip` (~3GB)

2. **Extract Files**:
   ```bash
   # Windows
   # Right-click archive.zip → Extract All → Choose ml_model/Dataset/HAM10000
   
   # macOS/Linux
   unzip archive.zip -d ml_model/Dataset/HAM10000
   ```

3. **Verify Structure**:
   ```
   ml_model/Dataset/HAM10000/
   ├── HAM10000_images_part_1/     # ~5,000 images
   │   ├── ISIC_0024306.jpg
   │   ├── ISIC_0024307.jpg
   │   └── ... (more .jpg files)
   ├── HAM10000_images_part_2/     # ~5,000 images
   │   ├── ISIC_0024306.jpg
   │   └── ...
   └── HAM10000_metadata.csv       # Metadata file
   ```

4. **Check Metadata**:
   ```python
   import pandas as pd
   
   metadata = pd.read_csv('ml_model/Dataset/HAM10000/HAM10000_metadata.csv')
   print(metadata.head())
   print(f"\nTotal samples: {len(metadata)}")
   print(f"\nClass distribution:\n{metadata['dx'].value_counts()}")
   ```

   Expected output:
   ```
         lesion_id      image_id   dx  dx_type   age   sex  localization
   0  HAM_0000118  ISIC_0027419   bkl  histo   80.0  male       scalp
   1  HAM_0000118  ISIC_0025030   bkl  histo   80.0  male       scalp
   ...
   
   Total samples: 10015
   
   Class distribution:
   nv       6705    (Melanocytic nevi - benign moles)
   mel      1113    (Melanoma - dangerous!)
   bkl      1099    (Benign keratosis)
   bcc       514    (Basal cell carcinoma)
   akiec     327    (Actinic keratoses)
   vasc      142    (Vascular lesions)
   df        115    (Dermatofibroma)
   ```

#### Understanding the Classes:

| Code | Full Name | Type | Severity |
|------|-----------|------|----------|
| `akiec` | Actinic Keratoses | Pre-cancerous | ⚠️ Moderate |
| `bcc` | Basal Cell Carcinoma | Malignant | ⚠️ Moderate |
| `bkl` | Benign Keratosis | Benign | ✅ Low |
| `df` | Dermatofibroma | Benign | ✅ Low |
| `mel` | Melanoma | Malignant | 🔴 Critical |
| `nv` | Melanocytic Nevi | Benign | ✅ Low |
| `vasc` | Vascular Lesions | Benign | ✅ Low |

---

### 📦 Dataset 2: Chest X-Ray (Pneumonia)

**Download Link**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

#### Step-by-Step Setup:

1. **Download from Kaggle**:
   - Download `chest-xray-pneumonia.zip` (~2GB)

2. **Extract Files**:
   ```bash
   unzip chest-xray-pneumonia.zip -d ml_model/Dataset/
   ```

3. **Verify Structure**:
   ```
   ml_model/Dataset/chest_xray/
   ├── train/
   │   ├── NORMAL/           # ~1,341 images
   │   │   ├── IM-0115-0001.jpeg
   │   │   └── ...
   │   └── PNEUMONIA/        # ~3,875 images
   │       ├── person1_bacteria_1.jpeg
   │       └── ...
   ├── test/
   │   ├── NORMAL/           # ~234 images
   │   └── PNEUMONIA/        # ~390 images
   └── val/
       ├── NORMAL/           # ~8 images
       └── PNEUMONIA/        # ~8 images
   ```

4. **Check Dataset**:
   ```python
   import os
   import glob
   
   base_dir = 'ml_model/Dataset/chest_xray'
   
   for split in ['train', 'test', 'val']:
       for cls in ['NORMAL', 'PNEUMONIA']:
           path = os.path.join(base_dir, split, cls)
           count = len(glob.glob(os.path.join(path, '*')))
           print(f"{split}/{cls}: {count} images")
   ```

   Expected output:
   ```
   train/NORMAL: 1341 images
   train/PNEUMONIA: 3875 images
   test/NORMAL: 234 images
   test/PNEUMONIA: 390 images
   val/NORMAL: 8 images
   val/PNEUMONIA: 8 images
   ```

---

### 📦 Dataset 3: Breast Ultrasound

**Download Link**: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

#### Step-by-Step Setup:

1. **Download from Kaggle**:
   - Download the dataset (~200MB)

2. **Extract Files**:
   ```bash
   unzip breast-ultrasound-images-dataset.zip -d ml_model/Dataset/breast_ultrasound
   ```

3. **Verify Structure**:
   ```
   ml_model/Dataset/breast_ultrasound/
   ├── benign/
   │   ├── benign (1).png
   │   ├── benign (1)_mask.png    # Mask files (will be ignored)
   │   └── ...
   ├── malignant/
   │   ├── malignant (1).png
   │   └── ...
   └── normal/
       ├── normal (1).png
       └── ...
   ```

4. **Check Dataset**:
   ```python
   import os
   import glob
   
   base_dir = 'ml_model/Dataset/breast_ultrasound'
   
   for cls in ['normal', 'benign', 'malignant']:
       path = os.path.join(base_dir, cls)
       if os.path.exists(path):
           # Count only non-mask images
           images = [f for f in glob.glob(os.path.join(path, '*.png')) 
                    if 'mask' not in f.lower()]
           print(f"{cls}: {len(images)} images")
   ```

   Expected output:
   ```
   normal: 133 images
   benign: 437 images
   malignant: 210 images
   ```

---

### 📦 Dataset 4: PTB-XL (ECG/Heart)

**Download Link**: https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset

#### Step-by-Step Setup:

1. **Download from Kaggle**:
   - Download `ptb-xl-a-large-publicly-available-electrocardiography-dataset.zip` (~2GB)

2. **Extract Files**:
   ```bash
   unzip ptb-xl-*.zip -d ml_model/Dataset/ptb-xl
   ```

3. **Verify Structure**:
   ```
   ml_model/Dataset/ptb-xl/
   ├── records100/              # 100Hz recordings
   │   ├── 00000/
   │   │   ├── 00001_lr.dat
   │   │   ├── 00001_lr.hea
   │   │   └── ...
   │   ├── 01000/
   │   └── ...
   ├── records500/              # 500Hz recordings
   │   └── ...
   ├── ptbxl_database.csv       # Main metadata
   └── scp_statements.csv       # SCP code definitions
   ```

4. **Install WFDB Library**:
   ```bash
   pip install wfdb
   ```

5. **Check Dataset**:
   ```python
   import pandas as pd
   import wfdb
   
   # Load metadata
   metadata = pd.read_csv('ml_model/Dataset/ptb-xl/ptbxl_database.csv')
   print(f"Total ECG records: {len(metadata)}")
   
   # Test loading one record
   record_path = 'ml_model/Dataset/ptb-xl/records100/00000/00001_lr'
   signal, fields = wfdb.rdsamp(record_path)
   print(f"Signal shape: {signal.shape}")  # Should be (1000, 12) for 12-lead ECG
   ```

   Expected output:
   ```
   Total ECG records: 21837
   Signal shape: (1000, 12)
   ```

---

## 🎓 Training Models

### Training Workflow Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Load Dataset                                              │
│     ↓                                                         │
│  2. Preprocess Images (resize, normalize)                     │
│     ↓                                                         │
│  3. Split Data (80% train, 20% test)                         │
│     ↓                                                         │
│  4. Create Model (Transfer Learning)                          │
│     ↓                                                         │
│  5. Phase 1: Train classifier (frozen base)                   │
│     ↓                                                         │
│  6. Phase 2: Fine-tune (unfreeze top layers)                 │
│     ↓                                                         │
│  7. Evaluate & Save Model                                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Train All Models

```bash
cd ml_model

# Train skin cancer model
python image_classification.py
# Select option 1

# Train pneumonia model
python image_classification.py
# Select option 2

# Train breast cancer model
python train_breast_cancer_model.py

# Train heart condition model
python train_heart_image_model.py
```

### Training Output Example

```
============================================================
  SKIN CANCER DETECTION MODEL - IMPROVED TRAINING
  Dataset: HAM10000
============================================================

📂 Loading HAM10000 dataset...
   Total entries: 10015
   Found image folder: HAM10000_images_part_1
   Found image folder: HAM10000_images_part_2
   Found 10015 images

   Original class distribution:
     nv: 6705
     mel: 1113
     bkl: 1099
     bcc: 514
     akiec: 327
     vasc: 142
     df: 115

   Target samples per class: 1099
   Loaded 1099 images for nv
   Loaded 1099 images for mel
   ...

🔧 Creating model...
   Using EfficientNetB0 backbone

--------------------------------------------------
🚀 Phase 1: Training classifier layers...
--------------------------------------------------
Epoch 1/20
157/157 [==============================] - 45s 287ms/step - loss: 1.2345 - accuracy: 0.4521
Epoch 2/20
157/157 [==============================] - 42s 267ms/step - loss: 0.8765 - accuracy: 0.6234
...

--------------------------------------------------
🚀 Phase 2: Fine-tuning...
--------------------------------------------------
Epoch 1/15
157/157 [==============================] - 52s 331ms/step - loss: 0.4321 - accuracy: 0.8234
...

--------------------------------------------------
📊 Final Evaluation...
--------------------------------------------------

   Classification Report:
              precision    recall  f1-score   support

       akiec       0.78      0.72      0.75       220
         bcc       0.81      0.79      0.80       220
         bkl       0.83      0.85      0.84       220
          df       0.76      0.71      0.73       220
         mel       0.85      0.88      0.86       220
          nv       0.91      0.93      0.92       220
        vasc       0.79      0.77      0.78       220

    accuracy                           0.81      1540
   macro avg       0.82      0.81      0.81      1540

   Overall Accuracy: 0.8123
   Weighted F1 Score: 0.8098

✓ Model saved: skin_cancer_model.h5
✓ Config saved: skin_cancer_config.json
```

---

## 🖥️ Running the Server

### Start the Backend Server

```bash
cd backend
python server.py
```

### Expected Output

```
============================================================
🏥 MediDiagnose-AI Backend Server v3.0
   Image-Based Cancer & Heart Disease Detection
============================================================

📋 Available Endpoints:
  GET  /                  - API info
  GET  /health            - Health check
  GET  /symptoms          - Get symptoms list
  POST /analyze/skin      - Skin cancer detection
  POST /analyze/heart     - Heart condition from image
  POST /analyze/breast    - Breast cancer from mammogram
  POST /analyze/xray      - Pneumonia from X-ray
  POST /predict-disease   - Disease from symptoms
  POST /predict-heart     - Heart risk from factors

🚀 Server starting on http://localhost:5000
============================================================

 * Serving Flask app 'server'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Verify Server is Running

```bash
# Test health endpoint
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models": {
    "skin_cancer": true,
    "heart_image": true,
    "breast_cancer": true,
    "pneumonia": true,
    "disease": true
  },
  "tensorflow": true,
  "pil": true
}
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/symptoms` | Get symptoms list |
| POST | `/analyze/skin` | Skin cancer detection |
| POST | `/analyze/heart` | Heart condition detection |
| POST | `/analyze/breast` | Breast cancer detection |
| POST | `/analyze/xray` | Pneumonia detection |
| POST | `/predict-disease` | Disease prediction |
| POST | `/predict-heart` | Heart risk assessment |

---

### 1. Skin Cancer Analysis

**Endpoint**: `POST /analyze/skin`

**Request**:
```bash
curl -X POST http://localhost:5000/analyze/skin \
  -F "image=@skin_lesion.jpg"
```

**Python Example**:
```python
import requests

url = "http://localhost:5000/analyze/skin"
files = {"image": open("skin_lesion.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']['name']}")
print(f"Confidence: {result['prediction']['confidence_percent']}")
print(f"Severity: {result['severity']}")
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "class_id": 5,
    "name": "Melanocytic Nevi",
    "type": "benign",
    "confidence": 0.89,
    "confidence_percent": "89.0%"
  },
  "staging": {
    "stage": "Benign",
    "description": "Non-cancerous skin lesion.",
    "prognosis": "No cancer treatment needed."
  },
  "severity": "low",
  "urgency": {
    "timeline": "Routine - Within 1-3 months",
    "action": "Follow-up at next regular appointment",
    "color": "green"
  },
  "treatment_options": [
    "Usually no treatment needed",
    "Surgical removal if desired",
    "Regular monitoring"
  ],
  "all_predictions": [
    {"name": "Melanocytic Nevi", "confidence": 0.89},
    {"name": "Benign Keratosis", "confidence": 0.06},
    {"name": "Dermatofibroma", "confidence": 0.02}
  ],
  "recommendations": {
    "level": "low",
    "title": "Benign Lesion - Low Risk",
    "message": "This appears to be a non-cancerous skin lesion.",
    "actions": [
      "Continue regular skin self-examinations",
      "Use the ABCDE rule to monitor moles",
      "Annual skin cancer screenings recommended"
    ]
  }
}
```

---

### 2. Heart Condition Analysis

**Endpoint**: `POST /analyze/heart`

**Request**:
```bash
curl -X POST http://localhost:5000/analyze/heart \
  -F "image=@ecg_image.png"
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "class_id": 0,
    "name": "Normal",
    "confidence": 0.92,
    "confidence_percent": "92.0%"
  },
  "staging": {
    "stage": "Normal",
    "description": "No significant abnormality detected.",
    "prognosis": "Maintain heart-healthy lifestyle."
  },
  "severity": "healthy",
  "urgency": {
    "timeline": "Annual screening",
    "action": "Continue regular health maintenance",
    "color": "blue"
  },
  "treatment_options": [
    "Heart-healthy diet",
    "Regular exercise (150 min/week)",
    "Blood pressure monitoring",
    "Annual cardiac checkups"
  ]
}
```

---

### 3. Heart Risk Assessment (from factors)

**Endpoint**: `POST /predict-heart`

**Request**:
```bash
curl -X POST http://localhost:5000/predict-heart \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "blood_pressure": 145,
    "cholesterol": 250,
    "chest_pain": 2,
    "max_heart_rate": 140,
    "exercise_angina": 0
  }'
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "condition": "Moderate-High Risk of Heart Disease",
    "risk_level": "Moderate-High",
    "probability": 0.52,
    "confidence_percent": "52.0%"
  },
  "risk_factors": {
    "age": {"value": 55, "risk": "moderate"},
    "blood_pressure": {"value": 145, "risk": "high"},
    "cholesterol": {"value": 250, "risk": "high"}
  },
  "severity": "moderate",
  "recommendations": [
    "Schedule an appointment with a cardiologist",
    "Monitor blood pressure regularly",
    "Follow a heart-healthy diet"
  ]
}
```

---

### 4. Breast Cancer Analysis

**Endpoint**: `POST /analyze/breast`

**Request**:
```python
import requests

url = "http://localhost:5000/analyze/breast"
files = {"image": open("mammogram.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "class_id": 1,
    "name": "Benign Finding",
    "birads": "BI-RADS 2",
    "confidence": 0.85,
    "confidence_percent": "85.0%"
  },
  "severity": "low",
  "staging": {
    "stage": "N/A - Benign",
    "description": "No malignancy detected."
  },
  "recommendations": {
    "level": "low",
    "title": "Benign/Probably Benign Finding",
    "actions": [
      "Continue regular screening schedule",
      "Perform monthly breast self-exams"
    ]
  }
}
```

---

### 5. Pneumonia Detection

**Endpoint**: `POST /analyze/xray`

**Request**:
```bash
curl -X POST http://localhost:5000/analyze/xray \
  -F "image=@chest_xray.jpeg"
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "name": "Normal",
    "confidence": 0.94,
    "confidence_percent": "94.0%"
  },
  "severity": "healthy",
  "urgency": {
    "timeline": "Annual screening",
    "action": "Continue regular health maintenance"
  },
  "recommendations": [
    "No signs of pneumonia detected",
    "Continue regular health maintenance",
    "Stay up to date with vaccinations"
  ]
}
```

---

### 6. Disease Prediction from Symptoms

**Endpoint**: `POST /predict-disease`

**Request**:
```bash
curl -X POST http://localhost:5000/predict-disease \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "cough", "headache", "fatigue"]}'
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "disease": "Common Cold",
    "confidence": 0.72,
    "confidence_percent": "72.0%"
  },
  "top_predictions": [
    {"disease": "Common Cold", "probability": 0.72},
    {"disease": "Influenza", "probability": 0.18},
    {"disease": "COVID-19", "probability": 0.06}
  ],
  "symptoms_analyzed": ["fever", "cough", "headache", "fatigue"],
  "recommendations": [
    "Moderate confidence - consider scheduling a doctor visit",
    "Keep track of your symptoms",
    "Stay hydrated and rest"
  ]
}
```

---

## 📖 Code Explanation

### 1. Server Architecture (`server.py`)

#### Imports and Setup

```python
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for frontend communication
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins (restrict in production)
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
```

**Explanation**:
- `Flask`: Web framework for creating the API
- `CORS`: Cross-Origin Resource Sharing - allows frontend on different port to communicate
- `tensorflow/keras`: Deep learning framework for loading models
- `PIL`: Python Imaging Library for image processing

---

#### Configuration Class

```python
class Config:
    # Folder for temporary file uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    
    # Maximum file size (32MB)
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    
    # Allowed image extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
    
    # Path to ML models
    ML_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml_model')
```

**Explanation**:
- Centralizes configuration in one place
- Easy to modify settings without changing code elsewhere
- `MAX_CONTENT_LENGTH`: Prevents very large files from crashing server

---

#### Model Loading

```python
def load_models():
    """Load all ML models at server startup"""
    global models  # Store models globally for reuse
    
    # Load skin cancer model
    if os.path.exists(MODEL_PATHS['skin_cancer']):
        try:
            models['skin_cancer'] = keras.models.load_model(
                MODEL_PATHS['skin_cancer']
            )
            logger.info("✓ Skin cancer model loaded")
        except Exception as e:
            logger.error(f"Failed to load skin model: {e}")
```

**Explanation**:
- Models are loaded ONCE at startup (not per-request)
- This is much faster than loading for each prediction
- Uses try/except to handle missing models gracefully

---

#### Image Preprocessing

```python
def preprocess_image(image, target_size=(224, 224), grayscale=False):
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image object
        target_size: Tuple (width, height) to resize to
        grayscale: Whether to convert to grayscale
    
    Returns:
        Numpy array ready for model prediction
    """
    # Resize image to model's expected input size
    image = image.resize(target_size)
    
    if grayscale:
        # Convert to grayscale (1 channel)
        image = image.convert('L')
        img_array = np.array(image) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    else:
        # Convert to RGB (3 channels)
        image = image.convert('RGB')
        img_array = np.array(image) / 255.0
    
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    return np.expand_dims(img_array, axis=0)
```

**Explanation**:
- **Resize**: Models expect fixed input size (224x224)
- **Normalize**: Convert pixel values from [0, 255] to [0, 1]
- **Batch dimension**: Models expect batch of images, even for single image

---

#### API Endpoint Example

```python
@app.route('/analyze/skin', methods=['POST'])
@handle_errors  # Decorator for error handling
def analyze_skin():
    """Analyze skin lesion image for cancer detection"""
    
    # 1. Check if image was provided
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # 2. Validate file type
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    # 3. Read and preprocess image
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({'success': False, 'error': f'Cannot read image: {e}'}), 400
    
    # 4. Check if model is loaded
    if 'skin_cancer' not in models:
        # Return demo response if model not available
        return jsonify({
            'success': True,
            'demo_mode': True,
            'prediction': {'name': 'Melanocytic Nevi', 'confidence': 0.78}
        })
    
    # 5. Preprocess and predict
    processed = preprocess_image(image, (224, 224))
    predictions = models['skin_cancer'].predict(processed, verbose=0)
    
    # 6. Get prediction results
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    # 7. Get class information
    class_info = SKIN_CANCER_CLASSES[predicted_class]
    
    # 8. Return results
    return jsonify({
        'success': True,
        'prediction': {
            'class_id': predicted_class,
            'name': class_info['name'],
            'type': class_info['type'],
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%"
        },
        'severity': class_info['severity'],
        'recommendations': get_skin_recommendations(class_info, confidence)
    })
```

**Explanation**:
1. **Validation**: Check image exists and is valid type
2. **Error Handling**: Return meaningful error messages
3. **Demo Mode**: Works even without trained model
4. **Prediction**: Use model to classify image
5. **Response**: Return structured JSON with all relevant information

---

### 2. Training Script Architecture (`image_classification.py`)

#### Focal Loss for Class Imbalance

```python
def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss - focuses training on hard examples
    
    Why use it?
    - HAM10000 is imbalanced (6705 nv vs 115 df)
    - Standard cross-entropy treats all samples equally
    - Focal loss down-weights easy examples, focuses on hard ones
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weighting factor for class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()  # Small value to prevent log(0)
        
        # Clip predictions to prevent numerical instability
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Standard cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Focal weight: (1 - p)^gamma
        # Easy examples (high confidence) get low weight
        # Hard examples (low confidence) get high weight
        focal_weight = alpha * K.pow(1 - y_pred, gamma)
        
        focal_loss = focal_weight * cross_entropy
        
        return K.sum(focal_loss, axis=-1)
    
    return focal_loss_fixed
```

**Visual Explanation**:
```
Standard Cross Entropy vs Focal Loss

                  Easy Sample         Hard Sample
                  (conf=0.9)          (conf=0.2)
                  
Cross Entropy:    0.105               1.609         (10x higher)
Focal Loss:       0.001               0.821         (800x higher!)

Focal loss makes hard samples much more important!
```

---

#### Model Architecture with Transfer Learning

```python
def create_skin_cancer_model_v2(input_shape=(224, 224, 3), num_classes=7):
    """
    Create CNN with transfer learning
    
    Why Transfer Learning?
    - EfficientNet was trained on 14 million images
    - It already knows how to detect edges, shapes, textures
    - We just need to teach it medical-specific features
    - Much better than training from scratch with small dataset
    """
    
    # Load pre-trained EfficientNet (without top classification layer)
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,      # Don't include ImageNet classifier
        weights='imagenet'       # Use weights trained on ImageNet
    )
    
    # Freeze base model initially
    # This prevents destroying the pre-trained features
    base_model.trainable = False
    
    # Build custom classifier on top
    inputs = keras.Input(shape=input_shape)
    
    # Pass through base model
    x = base_model(inputs, training=False)
    
    # Global Average Pooling
    # Converts (7, 7, 1280) -> (1280)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Batch Normalization
    # Normalizes activations, helps training stability
    x = layers.BatchNormalization()(x)
    
    # Dense layers with regularization
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)  # 50% dropout to prevent overfitting
    
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer - 7 classes with softmax
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model
```

**Architecture Visualization**:
```
Input Image (224, 224, 3)
         ↓
┌─────────────────────────┐
│     EfficientNetB0      │  ← Pre-trained (frozen initially)
│   (Feature Extractor)   │
└─────────────────────────┘
         ↓
Global Average Pooling (1280)
         ↓
BatchNorm → Dense(512) → ReLU → Dropout(0.5)
         ↓
BatchNorm → Dense(256) → ReLU → Dropout(0.4)
         ↓
BatchNorm → Dense(128) → ReLU → Dropout(0.3)
         ↓
Dense(7) → Softmax
         ↓
Output: [prob_class_0, prob_class_1, ..., prob_class_6]
```

---

#### Data Augmentation

```python
# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=30,        # Rotate up to 30 degrees
    width_shift_range=0.2,    # Shift horizontally up to 20%
    height_shift_range=0.2,   # Shift vertically up to 20%
    horizontal_flip=True,     # Flip horizontally
    vertical_flip=True,       # Flip vertically
    zoom_range=0.2,           # Zoom in/out up to 20%
    shear_range=0.15,         # Shear transformation
    brightness_range=[0.8, 1.2],  # Vary brightness
    fill_mode='reflect'       # Fill empty pixels by reflection
)
```

**Why Data Augmentation?**
```
Original Image          Augmented Versions (all from same image!)
    ┌───┐               ┌───┐  ┌───┐  ┌───┐  ┌───┐
    │ O │      →        │ O │  │ O │  │O  │  │ O │
    └───┘               └───┘  └───┘  └───┘  └───┘
                        rotated flipped zoomed shifted
                        
Result: 10,000 images → effectively 50,000+ unique training samples!
```

---

#### Two-Phase Training

```python
# Phase 1: Train only the classifier (base model frozen)
print("🚀 Phase 1: Training classifier layers...")

history1 = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Phase 2: Fine-tune (unfreeze top layers of base model)
print("🚀 Phase 2: Fine-tuning...")

# Unfreeze top 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Use lower learning rate to not destroy learned features
model.compile(
    optimizer=Adam(learning_rate=0.00005),  # 20x smaller than Phase 1
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

history2 = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=15,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
```

**Why Two Phases?**
```
Phase 1: Learn medical features
┌─────────────────┐
│  EfficientNet   │  ← FROZEN (keep ImageNet knowledge)
│   (base model)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│   Classifier    │  ← TRAINING (learn skin cancer features)
└─────────────────┘

Phase 2: Fine-tune everything
┌─────────────────┐
│  EfficientNet   │  ← FINE-TUNING (top 30 layers, low learning rate)
│   (base model)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│   Classifier    │  ← TRAINING (continues learning)
└─────────────────┘
```

---

### 3. Breast Cancer Model (`train_breast_cancer_model.py`)

#### Feature-Based Class Assignment

```python
def extract_image_features(img_array):
    """
    Extract simple features to help classify images
    
    These features help differentiate:
    - Normal tissue (uniform, lighter)
    - Benign masses (darker regions, but regular)
    - Malignant masses (very dark, irregular)
    """
    # Mean intensity (brightness)
    mean_intensity = np.mean(img_array)
    
    # Standard deviation (texture roughness)
    std_intensity = np.std(img_array)
    
    # Ratio of dark pixels (potential masses)
    dark_ratio = np.mean(img_array < 0.3)
    
    # Ratio of bright pixels
    bright_ratio = np.mean(img_array > 0.7)
    
    return {
        'mean': mean_intensity,
        'std': std_intensity,
        'dark_ratio': dark_ratio,
        'bright_ratio': bright_ratio
    }


def assign_class_based_on_features(base_class, features):
    """
    Map 3-class dataset to 6-class BI-RADS system
    
    BI-RADS Categories:
    0: Normal (BI-RADS 1)
    1: Benign (BI-RADS 2)
    2: Probably Benign (BI-RADS 3)
    3: Suspicious (BI-RADS 4)
    4: Highly Suggestive (BI-RADS 5)
    5: Malignant (BI-RADS 6)
    """
    if base_class == 'normal':
        return 0  # Always BI-RADS 1
    
    elif base_class == 'benign':
        # Most benign → BI-RADS 2
        # Some with concerning features → BI-RADS 3
        if features['std'] > 0.15 and features['dark_ratio'] > 0.1:
            return 2  # Probably benign (needs follow-up)
        return 1  # Clearly benign
    
    elif base_class == 'malignant':
        # Assign severity based on features
        if features['dark_ratio'] > 0.2 and features['std'] > 0.2:
            return 5  # Confirmed malignant (BI-RADS 6)
        elif features['dark_ratio'] > 0.15:
            return 4  # Highly suggestive (BI-RADS 5)
        else:
            return 3  # Suspicious (BI-RADS 4)
    
    return 1  # Default
```

---

### 4. Heart Model (`train_heart_image_model.py`)

#### ECG Signal to Image Conversion

```python
def signal_to_image_improved(signal, img_size=(224, 224)):
    """
    Convert 12-lead ECG signal to a plot image
    
    Why convert to image?
    - We can use pre-trained image models (transfer learning)
    - Doctors also look at ECG plots, not raw numbers
    - Images capture the visual patterns that indicate conditions
    
    Args:
        signal: Array of shape (1000, 12) - 10 seconds, 12 leads
        img_size: Output image size
    
    Returns:
        RGB image array normalized to [0, 1]
    """
    # Create matplotlib figure
    fig = plt.figure(figsize=(10, 8), dpi=30)
    
    # Standard 12-lead ECG arrangement
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Create 4x3 grid of subplots
    for i in range(12):
        ax = fig.add_subplot(4, 3, i + 1)
        
        # Normalize each lead independently
        lead_signal = signal[:, i]
        lead_signal = (lead_signal - np.mean(lead_signal)) / (np.std(lead_signal) + 1e-8)
        
        # Plot the lead
        ax.plot(lead_signal, 'b-', linewidth=0.8)
        ax.set_title(lead_names[i], fontsize=8)
        ax.set_xlim(0, len(signal))
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure to memory buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='white', dpi=50)
    
    # IMPORTANT: Close figure to prevent memory leak!
    plt.close(fig)
    
    # Load as PIL image and resize
    buf.seek(0)
    img = Image.open(buf)
    img = img.resize(img_size, Image.LANCZOS)
    img = img.convert('RGB')
    
    return np.array(img, dtype=np.float32) / 255.0
```

**Visual Example**:
```
Raw ECG Signal:                    Converted Image:
                                   ┌─────────────────────┐
Lead I:  ∿∿∿∿∿∿∿∿∿∿∿∿              │ I    II   III      │
Lead II: ∿∿∿∿∿∿∿∿∿∿∿∿     →       │ ∿∿∿  ∿∿∿  ∿∿∿      │
Lead III:∿∿∿∿∿∿∿∿∿∿∿∿             │ aVR  aVL  aVF      │
...                                │ ∿∿∿  ∿∿∿  ∿∿∿      │
Lead V6: ∿∿∿∿∿∿∿∿∿∿∿∿             │ V1-V6...           │
                                   └─────────────────────┘
                                   (224 x 224 RGB image)
```

---

## 🏗️ Model Architectures

### Overview

```
┌───────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURES                         │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │    Skin     │   │   Breast    │   │    Heart    │         │
│  │   Cancer    │   │   Cancer    │   │  Condition  │         │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘         │
│         │                 │                 │                 │
│         ▼                 ▼                 ▼                 │
│  ┌─────────────────────────────────────────────────┐         │
│  │              EfficientNetB0 Base                │         │
│  │           (Pre-trained on ImageNet)             │         │
│  └─────────────────────────────────────────────────┘         │
│         │                 │                 │                 │
│         ▼                 ▼                 ▼                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │  7 Classes  │   │  6 Classes  │   │  5 Classes  │         │
│  │  (Softmax)  │   │  (Softmax)  │   │  (Softmax)  │         │
│  └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                                │
│  ┌─────────────┐                                              │
│  │  Pneumonia  │   Input: 224x224x1 (Grayscale)              │
│  │  Detection  │   Output: 1 (Sigmoid) - Binary               │
│  └─────────────┘                                              │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Layer-by-Layer Explanation

```python
# Example: Skin Cancer Model

Input: (224, 224, 3)           # RGB image
    ↓
EfficientNetB0:                # 237 layers, 4M parameters
    ↓
GlobalAveragePooling2D:        # (7, 7, 1280) → (1280)
    - Reduces spatial dimensions
    - Makes model translation-invariant
    ↓
BatchNormalization:            # Normalize activations
    - Mean=0, Std=1
    - Helps training stability
    ↓
Dense(512):                    # Fully connected layer
    - kernel_regularizer=l2(0.01)  # Prevent overfitting
    ↓
BatchNormalization + ReLU:     # Normalize and activate
    ↓
Dropout(0.5):                  # Drop 50% of neurons
    - Prevents overfitting
    - Forces redundant learning
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.4)
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(7, activation='softmax'):  # Output layer
    - 7 neurons for 7 classes
    - Softmax: probabilities sum to 1
    ↓
Output: [0.02, 0.05, 0.03, 0.01, 0.04, 0.82, 0.03]
        (probabilities for each class)
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "TensorFlow not found"

```bash
# Error
ModuleNotFoundError: No module named 'tensorflow'

# Solution
pip install tensorflow

# For GPU support (NVIDIA)
pip install tensorflow[and-cuda]
```

#### 2. "Model file not found"

```bash
# Error
FileNotFoundError: skin_cancer_model.h5 not found

# Solution
# Train the model first:
cd ml_model
python image_classification.py
```

#### 3. "CUDA out of memory"

```python
# Error
ResourceExhaustedError: OOM when allocating tensor

# Solution 1: Reduce batch size
model.fit(X_train, y_train, batch_size=16)  # Instead of 32

# Solution 2: Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

#### 4. "Image cannot be read"

```python
# Error
PIL.UnidentifiedImageError: cannot identify image file

# Solution: Check image format
from PIL import Image
img = Image.open('image.jpg')
print(img.format, img.size, img.mode)

# Convert if needed
img = img.convert('RGB')
```

#### 5. "Port 5000 already in use"

```bash
# Error
OSError: [Errno 98] Address already in use

# Solution 1: Kill existing process
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :5000
kill -9 <PID>

# Solution 2: Use different port
python server.py --port 5001
```

#### 6. "Model gives wrong predictions"

```python
# Check 1: Verify preprocessing matches training
# Training used:
img = img.resize((224, 224))  # Must match!
img_array = np.array(img) / 255.0  # Must normalize!

# Check 2: Verify class mapping
print(SKIN_CANCER_CLASSES)
# Should match training script exactly

# Check 3: Check model output
predictions = model.predict(img)
print(predictions)  # Should sum to 1.0
```

---

## 🧪 Testing

### Unit Tests

```python
# test_server.py
import requests
import os

BASE_URL = "http://localhost:5000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print("✓ Health check passed")

def test_skin_analysis():
    # Use a test image
    test_image = "test_images/skin_test.jpg"
    if os.path.exists(test_image):
        files = {"image": open(test_image, "rb")}
        response = requests.post(f"{BASE_URL}/analyze/skin", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert 'prediction' in data
        print(f"✓ Skin analysis passed: {data['prediction']['name']}")
    else:
        print("⚠ Test image not found")

def test_disease_prediction():
    data = {"symptoms": ["fever", "cough", "headache"]}
    response = requests.post(
        f"{BASE_URL}/predict-disease",
        json=data
    )
    assert response.status_code == 200
    result = response.json()
    assert result['success'] == True
    print(f"✓ Disease prediction passed: {result['prediction']['disease']}")

if __name__ == '__main__':
    test_health()
    test_skin_analysis()
    test_disease_prediction()
    print("\n✅ All tests passed!")
```

Run tests:
```bash
python test_server.py
```

---

## 🤝 Contributing

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   python test_server.py
   ```
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

---

## ⚠️ Disclaimer

```
╔══════════════════════════════════════════════════════════════════╗
║                        IMPORTANT NOTICE                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  This software is for EDUCATIONAL and RESEARCH purposes only.   ║
║                                                                   ║
║  It is NOT intended to be used as a medical diagnostic tool.    ║
║                                                                   ║
║  DO NOT use this software to make medical decisions.            ║
║                                                                   ║
║  Always consult qualified healthcare professionals for          ║
║  medical advice, diagnosis, and treatment.                       ║
║                                                                   ║
║  The predictions made by this software are NOT a substitute     ║
║  for professional medical judgment.                              ║
║                                                                   ║
║  The developers are not responsible for any decisions made      ║
║  based on the outputs of this software.                          ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) - Skin lesion images
- [Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) - Pneumonia detection
- [Breast Ultrasound Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) - Breast cancer
- [PTB-XL Dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset) - ECG signals
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework

---

<div align="center">

**Made with ❤️ for better healthcare**

⭐ Star this repo if you find it helpful!

</div>
```

