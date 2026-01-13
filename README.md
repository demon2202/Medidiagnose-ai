# 🏥 MediDiagnose-AI



![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)



## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training Models](#training-models)
- [Running the Server](#running-the-server)
- [API Documentation](#api-documentation)
- [Code Explanation](#code-explanation)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## Overview

MediDiagnose-AI is a comprehensive medical diagnosis system that uses deep learning to analyze medical images and predict various conditions.

## Images

<img width="1919" height="928" alt="image" src="https://github.com/user-attachments/assets/37995f4c-a526-4e4b-a3c3-1e37e77ed5af" />
<img width="1919" height="930" alt="Screenshot 2026-01-13 191722" src="https://github.com/user-attachments/assets/cc20f14c-07dd-440a-8159-378e145f4579" />
<img width="1594" height="934" alt="Screenshot 2026-01-13 191756" src="https://github.com/user-attachments/assets/b66233be-d543-411e-8742-1db73669519c" />
<img width="1424" height="785" alt="Screenshot 2026-01-13 191806" src="https://github.com/user-attachments/assets/ead4332e-3f42-44e4-abaa-83103e2ec28f" />
<img width="1595" height="938" alt="Screenshot 2026-01-13 191815" src="https://github.com/user-attachments/assets/c3161229-6647-47a2-b0b2-abeb45860fb8" />
<img width="1591" height="938" alt="Screenshot 2026-01-13 191733" src="https://github.com/user-attachments/assets/8d7e5941-0405-4f83-baaa-4dcf45cca92f" />
<img width="1594" height="855" alt="Screenshot 2026-01-13 191744" src="https://github.com/user-attachments/assets/6b41577f-657a-42ce-95fa-0507bdb8b623" />


### System Architecture
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ User Input │────▶│ Flask API │────▶│ ML Models │
│ (Image/Text) │ │ (Backend) │ │ (TensorFlow) │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│
▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Frontend │◀────│ JSON Response │◀────│ Prediction │
│ (Results) │ │ (Diagnosis) │ │ + Confidence │
└─────────────────┘ └─────────────────┘ └─────────────────┘

text


### Supported Diagnoses

| Feature | Description | Classes |
|---------|-------------|---------|
| Skin Cancer | Lesion classification | 7 types |
| Heart Conditions | ECG analysis | 5 conditions |
| Breast Cancer | Ultrasound/mammogram analysis | 6 BI-RADS categories |
| Pneumonia | Chest X-ray analysis | Binary (Normal/Pneumonia) |
| Disease Prediction | Symptom-based | Multiple diseases |

---

## Features

### Medical Analysis

- **Skin Cancer Detection**: Classifies 7 types of skin lesions
  - Melanoma (dangerous)
  - Basal Cell Carcinoma
  - Melanocytic Nevi (moles)
  - Benign Keratosis
  - Actinic Keratoses
  - Vascular Lesions
  - Dermatofibroma

- **Heart Condition Detection**: Analyzes ECG images
  - Normal
  - Myocardial Infarction
  - ST/T Changes
  - Conduction Disturbance
  - Hypertrophy

- **Breast Cancer Detection**: BI-RADS classification
  - BI-RADS 1: Normal
  - BI-RADS 2: Benign
  - BI-RADS 3: Probably Benign
  - BI-RADS 4: Suspicious
  - BI-RADS 5: Highly Suggestive
  - BI-RADS 6: Malignant

- **Pneumonia Detection**: Binary classification
  - Normal
  - Pneumonia

### Technical Features

- Transfer Learning (EfficientNet, MobileNetV2)
- Focal Loss for class imbalance
- Data augmentation
- RESTful API with CORS
- Comprehensive error handling
- Demo mode when models unavailable

---

## Project Structure
medidiagnose-ai/
│
├── backend/
│ ├── server.py # Flask API server
│ └── uploads/ # Temporary upload folder
│
├── ml_model/
│ ├── image_classification.py # Skin & Pneumonia training
│ ├── train_breast_cancer_model.py # Breast cancer training
│ ├── train_heart_image_model.py # Heart condition training
│ │
│ ├── Dataset/ # Datasets folder
│ │ ├── HAM10000/ # Skin cancer dataset
│ │ ├── chest_xray/ # Pneumonia dataset
│ │ ├── breast_ultrasound/ # Breast cancer dataset
│ │ └── ptb-xl/ # ECG dataset
│ │
│ ├── skin_cancer_model.h5 # Trained models
│ ├── pneumonia_model.h5
│ ├── breast_cancer_model.h5
│ ├── heart_image_model.h5
│ └── *.json # Model configs
│
├── requirements.txt
├── README.md
└── .gitignore

text


---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/medidiagnose-ai.git
cd medidiagnose-ai
Step 2: Create Virtual Environment
Bash

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
Bash

pip install --upgrade pip
pip install -r requirements.txt
Step 4: Verify Installation
Python

import tensorflow as tf
import flask
print(f"TensorFlow: {tf.__version__}")
print(f"Flask: {flask.__version__}")
print("Installation successful!")
Dataset Setup
Required Datasets
Dataset	Size	Download Link
HAM10000 (Skin)	~3GB	Kaggle
Chest X-Ray	~2GB	Kaggle
Breast Ultrasound	~200MB	Kaggle
PTB-XL (ECG)	~2GB	Kaggle
Dataset 1: HAM10000 (Skin Cancer)
Download: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Setup Steps:

Download archive.zip from Kaggle
Extract to ml_model/Dataset/HAM10000/
Verify structure:
text

ml_model/Dataset/HAM10000/
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ... (more .jpg files)
├── HAM10000_images_part_2/
│   └── ... (more .jpg files)
└── HAM10000_metadata.csv
Verify with Python:

Python

import pandas as pd
import os

metadata = pd.read_csv('ml_model/Dataset/HAM10000/HAM10000_metadata.csv')
print(f"Total samples: {len(metadata)}")
print(f"\nClass distribution:")
print(metadata['dx'].value_counts())
Expected Output:

text

Total samples: 10015

Class distribution:
nv       6705    # Melanocytic nevi (benign moles)
mel      1113    # Melanoma (dangerous!)
bkl      1099    # Benign keratosis
bcc       514    # Basal cell carcinoma
akiec     327    # Actinic keratoses
vasc      142    # Vascular lesions
df        115    # Dermatofibroma
Class Descriptions:

Code	Name	Type	Severity
akiec	Actinic Keratoses	Pre-cancerous	⚠️ Moderate
bcc	Basal Cell Carcinoma	Malignant	⚠️ Moderate
bkl	Benign Keratosis	Benign	✅ Low
df	Dermatofibroma	Benign	✅ Low
mel	Melanoma	Malignant	🔴 Critical
nv	Melanocytic Nevi	Benign	✅ Low
vasc	Vascular Lesions	Benign	✅ Low
Dataset 2: Chest X-Ray (Pneumonia)
Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Setup Steps:

Download chest-xray-pneumonia.zip
Extract to ml_model/Dataset/chest_xray/
Verify structure:
text

ml_model/Dataset/chest_xray/
├── train/
│   ├── NORMAL/
│   │   ├── IM-0115-0001.jpeg
│   │   └── ... (~1,341 images)
│   └── PNEUMONIA/
│       ├── person1_bacteria_1.jpeg
│       └── ... (~3,875 images)
├── test/
│   ├── NORMAL/      (~234 images)
│   └── PNEUMONIA/   (~390 images)
└── val/
    ├── NORMAL/      (~8 images)
    └── PNEUMONIA/   (~8 images)
Verify with Python:

Python

import os
import glob

base = 'ml_model/Dataset/chest_xray'
for split in ['train', 'test', 'val']:
    for cls in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(base, split, cls)
        count = len(glob.glob(os.path.join(path, '*')))
        print(f"{split}/{cls}: {count} images")
Dataset 3: Breast Ultrasound
Download: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

Setup Steps:

Download the dataset
Extract to ml_model/Dataset/breast_ultrasound/
Verify structure:
text

ml_model/Dataset/breast_ultrasound/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png    # Mask files (ignored)
│   └── ... (~437 images)
├── malignant/
│   └── ... (~210 images)
└── normal/
    └── ... (~133 images)
Verify with Python:

Python

import os
import glob

base = 'ml_model/Dataset/breast_ultrasound'
for cls in ['normal', 'benign', 'malignant']:
    path = os.path.join(base, cls)
    if os.path.exists(path):
        # Count only non-mask images
        images = [f for f in glob.glob(os.path.join(path, '*.png')) 
                 if 'mask' not in f.lower()]
        print(f"{cls}: {len(images)} images")
Dataset 4: PTB-XL (ECG/Heart)
Download: https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset

Setup Steps:

Download the dataset (~2GB)
Extract to ml_model/Dataset/ptb-xl/
Install wfdb: pip install wfdb
Verify structure:
text

ml_model/Dataset/ptb-xl/
├── records100/
│   ├── 00000/
│   │   ├── 00001_lr.dat
│   │   ├── 00001_lr.hea
│   │   └── ...
│   └── ...
├── records500/
│   └── ...
├── ptbxl_database.csv
└── scp_statements.csv
Verify with Python:

Python

import pandas as pd
import wfdb

# Load metadata
metadata = pd.read_csv('ml_model/Dataset/ptb-xl/ptbxl_database.csv')
print(f"Total ECG records: {len(metadata)}")

# Test loading one record
record = 'ml_model/Dataset/ptb-xl/records100/00000/00001_lr'
signal, fields = wfdb.rdsamp(record)
print(f"Signal shape: {signal.shape}")  # Should be (1000, 12)
Training Models
Training Commands
Bash

cd ml_model

# Train skin cancer model
python image_classification.py
# Select option 1

# Train pneumonia model  
python image_classification.py
# Select option 2

# Train breast cancer model
python train_breast_cancer_model.py

# Train heart model
python train_heart_image_model.py
Training Process Explanation
text

Training Pipeline:
─────────────────

1. Load Dataset
   ↓
2. Preprocess Images
   - Resize to 224x224
   - Normalize to [0, 1]
   ↓
3. Split Data
   - 80% training
   - 20% testing
   ↓
4. Create Model
   - Load EfficientNet (pre-trained)
   - Add custom classifier layers
   ↓
5. Phase 1: Train Classifier
   - Base model frozen
   - Train new layers only
   - ~20 epochs
   ↓
6. Phase 2: Fine-tune
   - Unfreeze top 30 layers
   - Lower learning rate
   - ~15 epochs
   ↓
7. Evaluate & Save
   - Calculate accuracy, F1
   - Save .h5 and .json files
Expected Training Output
text

============================================================
  SKIN CANCER DETECTION MODEL - IMPROVED TRAINING
============================================================

📂 Loading HAM10000 dataset...
   Total entries: 10015
   Found 10015 images

   Class distribution:
     nv: 6705
     mel: 1113
     bkl: 1099
     ...

🔧 Creating model...
   Using EfficientNetB0 backbone

--------------------------------------------------
🚀 Phase 1: Training classifier layers...
--------------------------------------------------
Epoch 1/20
157/157 [======] - 45s - loss: 1.23 - accuracy: 0.45
Epoch 2/20
157/157 [======] - 42s - loss: 0.87 - accuracy: 0.62
...

--------------------------------------------------
🚀 Phase 2: Fine-tuning...
--------------------------------------------------
Epoch 1/15
157/157 [======] - 52s - loss: 0.43 - accuracy: 0.82
...

--------------------------------------------------
📊 Final Evaluation...
--------------------------------------------------

   Classification Report:
              precision    recall  f1-score

       akiec       0.78      0.72      0.75
         bcc       0.81      0.79      0.80
         mel       0.85      0.88      0.86
         ...

   Accuracy: 0.8123
   F1 Score: 0.8098

✓ Model saved: skin_cancer_model.h5
Running the Server
Start Server
Bash

cd backend
python server.py
Expected Output
text

============================================================
🏥 MediDiagnose-AI Backend Server v3.0
============================================================

📋 Available Endpoints:
  GET  /                  - API info
  GET  /health            - Health check
  GET  /symptoms          - Get symptoms list
  POST /analyze/skin      - Skin cancer detection
  POST /analyze/heart     - Heart condition
  POST /analyze/breast    - Breast cancer
  POST /analyze/xray      - Pneumonia detection
  POST /predict-disease   - Disease from symptoms
  POST /predict-heart     - Heart risk assessment

🚀 Server running on http://localhost:5000
============================================================
Verify Server
Bash

curl http://localhost:5000/health
Response:

JSON

{
  "status": "healthy",
  "models": {
    "skin_cancer": true,
    "heart_image": true,
    "breast_cancer": true,
    "pneumonia": true,
    "disease": true
  },
  "tensorflow": true
}
API Documentation
Endpoints Summary
Method	Endpoint	Description
GET	/	API information
GET	/health	Health check
GET	/symptoms	Get symptoms list
POST	/analyze/skin	Skin cancer detection
POST	/analyze/heart	Heart condition
POST	/analyze/breast	Breast cancer
POST	/analyze/xray	Pneumonia detection
POST	/predict-disease	Disease prediction
POST	/predict-heart	Heart risk
1. Skin Cancer Analysis
Endpoint: POST /analyze/skin

cURL Example:

Bash

curl -X POST http://localhost:5000/analyze/skin \
  -F "image=@skin_lesion.jpg"
Python Example:

Python

import requests

url = "http://localhost:5000/analyze/skin"
files = {"image": open("skin_lesion.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']['name']}")
print(f"Confidence: {result['prediction']['confidence_percent']}")
print(f"Severity: {result['severity']}")
Response:

JSON

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
    "action": "Follow-up at next appointment",
    "color": "green"
  },
  "treatment_options": [
    "Usually no treatment needed",
    "Surgical removal if desired",
    "Regular monitoring"
  ],
  "recommendations": {
    "level": "low",
    "title": "Benign Lesion - Low Risk",
    "message": "Non-cancerous skin lesion.",
    "actions": [
      "Continue regular skin self-exams",
      "Annual skin cancer screenings"
    ]
  }
}
2. Heart Condition Analysis
Endpoint: POST /analyze/heart

Request:

Bash

curl -X POST http://localhost:5000/analyze/heart \
  -F "image=@ecg_image.png"
Response:

JSON

{
  "success": true,
  "prediction": {
    "class_id": 0,
    "name": "Normal",
    "confidence": 0.92,
    "confidence_percent": "92.0%"
  },
  "severity": "healthy",
  "staging": {
    "stage": "Normal",
    "description": "No significant abnormality."
  },
  "treatment_options": [
    "Heart-healthy diet",
    "Regular exercise",
    "Annual checkups"
  ]
}
3. Heart Risk Assessment
Endpoint: POST /predict-heart

Request:

Bash

curl -X POST http://localhost:5000/predict-heart \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "blood_pressure": 145,
    "cholesterol": 250,
    "chest_pain": 2,
    "max_heart_rate": 140
  }'
Response:

JSON

{
  "success": true,
  "prediction": {
    "condition": "Moderate-High Risk",
    "risk_level": "Moderate-High",
    "probability": 0.52,
    "confidence_percent": "52.0%"
  },
  "risk_factors": {
    "age": {"value": 55, "risk": "moderate"},
    "blood_pressure": {"value": 145, "risk": "high"},
    "cholesterol": {"value": 250, "risk": "high"}
  },
  "recommendations": [
    "Schedule cardiologist appointment",
    "Monitor blood pressure regularly",
    "Follow heart-healthy diet"
  ]
}
4. Breast Cancer Analysis
Endpoint: POST /analyze/breast

Request:

Python

import requests

url = "http://localhost:5000/analyze/breast"
files = {"image": open("mammogram.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
Response:

JSON

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
  "recommendations": {
    "level": "low",
    "title": "Benign Finding",
    "actions": [
      "Continue regular screening",
      "Monthly self-exams"
    ]
  }
}
5. Pneumonia Detection
Endpoint: POST /analyze/xray

Request:

Bash

curl -X POST http://localhost:5000/analyze/xray \
  -F "image=@chest_xray.jpeg"
Response:

JSON

{
  "success": true,
  "prediction": {
    "name": "Normal",
    "confidence": 0.94,
    "confidence_percent": "94.0%"
  },
  "severity": "healthy",
  "recommendations": [
    "No signs of pneumonia detected",
    "Continue regular health maintenance"
  ]
}
6. Disease Prediction
Endpoint: POST /predict-disease

Request:

Bash

curl -X POST http://localhost:5000/predict-disease \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "cough", "headache", "fatigue"]}'
Response:

JSON

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
    "Consider scheduling a doctor visit",
    "Stay hydrated and rest",
    "Monitor symptoms"
  ]
}
Code Explanation
Server Architecture (server.py)
1. Flask App Setup
Python

from flask import Flask, request, jsonify
from flask_cors import CORS

# Create Flask application
app = Flask(__name__)

# Enable CORS for frontend communication
# This allows requests from different origins (e.g., React on port 3000)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
Explanation:

Flask: Creates web server
CORS: Allows frontend on different port to communicate
Without CORS, browser blocks cross-origin requests
2. Model Loading
Python

def load_models():
    """Load all ML models at server startup"""
    global models
    
    # Load skin cancer model
    if os.path.exists(MODEL_PATHS['skin_cancer']):
        try:
            models['skin_cancer'] = keras.models.load_model(
                MODEL_PATHS['skin_cancer']
            )
            logger.info("✓ Skin cancer model loaded")
        except Exception as e:
            logger.error(f"Failed: {e}")
Why load at startup?

Loading models takes 2-5 seconds each
Loading once at startup vs. every request
Much faster response times
3. Image Preprocessing
Python

def preprocess_image(image, target_size=(224, 224), grayscale=False):
    """
    Prepare image for model prediction
    
    Steps:
    1. Resize to model's expected size
    2. Convert color mode
    3. Normalize pixel values
    4. Add batch dimension
    """
    # Step 1: Resize
    image = image.resize(target_size)
    
    # Step 2: Color mode
    if grayscale:
        image = image.convert('L')  # Grayscale
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel
    else:
        image = image.convert('RGB')
        img_array = np.array(image) / 255.0
    
    # Step 3: Already normalized above (/ 255.0)
    
    # Step 4: Add batch dimension
    # Model expects: (batch, height, width, channels)
    # We have: (height, width, channels)
    # Result: (1, height, width, channels)
    return np.expand_dims(img_array, axis=0)
Visual Explanation:

text

Original Image          After Preprocessing
─────────────────       ─────────────────────

Size: 1024x768          Size: 224x224
Range: 0-255            Range: 0.0-1.0
Shape: (768,1024,3)     Shape: (1,224,224,3)

                              ↑
                        Batch dimension
4. API Endpoint Structure
Python

@app.route('/analyze/skin', methods=['POST'])
@handle_errors  # Error handling decorator
def analyze_skin():
    """Analyze skin lesion image"""
    
    # 1. VALIDATE INPUT
    if 'image' not in request.files:
        return jsonify({
            'success': False, 
            'error': 'No image provided'
        }), 400
    
    file = request.files['image']
    
    # 2. VALIDATE FILE TYPE
    if not allowed_file(file.filename):
        return jsonify({
            'success': False, 
            'error': 'Invalid file type'
        }), 400
    
    # 3. READ IMAGE
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Cannot read image: {e}'
        }), 400
    
    # 4. CHECK MODEL
    if 'skin_cancer' not in models:
        # Return demo response
        return jsonify({
            'success': True,
            'demo_mode': True,
            'prediction': {'name': 'Demo', 'confidence': 0.5}
        })
    
    # 5. PREPROCESS
    processed = preprocess_image(image, (224, 224))
    
    # 6. PREDICT
    predictions = models['skin_cancer'].predict(processed, verbose=0)
    
    # 7. INTERPRET RESULTS
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    class_info = SKIN_CANCER_CLASSES[predicted_class]
    
    # 8. RETURN RESPONSE
    return jsonify({
        'success': True,
        'prediction': {
            'class_id': predicted_class,
            'name': class_info['name'],
            'confidence': confidence
        },
        'severity': class_info['severity']
    })
Request/Response Flow:

text

Client                    Server
──────                    ──────

POST /analyze/skin
with image file
        ──────────────────▶
                          1. Validate image
                          2. Preprocess
                          3. Run model
                          4. Format results
        ◀──────────────────
JSON response
{success, prediction, ...}
Training Script Architecture
1. Focal Loss for Imbalanced Data
Python

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss - focuses on hard examples
    
    Problem: HAM10000 is imbalanced
    - nv: 6705 samples
    - df: 115 samples (58x fewer!)
    
    Standard cross-entropy treats all equally.
    Focal loss down-weights easy examples.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Standard cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Focal weight: (1 - p)^gamma
        # High confidence (easy) → low weight
        # Low confidence (hard) → high weight
        focal_weight = alpha * K.pow(1 - y_pred, gamma)
        
        return K.sum(focal_weight * cross_entropy, axis=-1)
    
    return focal_loss_fixed
Comparison:

text

                    Easy Example      Hard Example
                    (conf=0.9)        (conf=0.2)

Cross Entropy:      0.105             1.609
Focal Loss:         0.001             0.821

Ratio:              105:1             1:821

Focal loss makes hard examples 800x more important!
2. Transfer Learning Model
Python

def create_skin_cancer_model_v2(input_shape=(224, 224, 3), num_classes=7):
    """
    Create model with transfer learning
    
    Why Transfer Learning?
    - EfficientNet trained on 14 million images
    - Already knows edges, shapes, textures
    - We just teach it skin cancer specifics
    """
    
    # Load pre-trained model (without top layer)
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,    # Remove ImageNet classifier
        weights='imagenet'    # Use pre-trained weights
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build classifier
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs), base_model
Architecture Diagram:

text

Input (224, 224, 3)
        │
        ▼
┌───────────────────────┐
│    EfficientNetB0     │  Pre-trained, frozen initially
│   (Feature Extractor) │  237 layers, 4M parameters
└───────────────────────┘
        │
        ▼
GlobalAveragePooling2D     (7,7,1280) → (1280)
        │
        ▼
BatchNorm → Dense(512) → ReLU → Dropout(0.5)
        │
        ▼
BatchNorm → Dense(256) → ReLU → Dropout(0.4)
        │
        ▼
Dense(7) → Softmax
        │
        ▼
Output: [p0, p1, p2, p3, p4, p5, p6]  (probabilities)
3. Data Augmentation
Python

datagen = ImageDataGenerator(
    rotation_range=30,        # Rotate ±30 degrees
    width_shift_range=0.2,    # Shift horizontally ±20%
    height_shift_range=0.2,   # Shift vertically ±20%
    horizontal_flip=True,     # Flip horizontally
    vertical_flip=True,       # Flip vertically
    zoom_range=0.2,           # Zoom ±20%
    shear_range=0.15,         # Shear transformation
    brightness_range=[0.8, 1.2],  # Brightness ±20%
    fill_mode='reflect'       # Fill empty pixels
)
Visual Example:

text

Original             Augmented Versions
────────             ──────────────────

  ┌───┐              ┌───┐  ╔═══╗  ┌───┐  ┌───┐
  │ O │      →       │ O │  ║ O ║  │O  │  │ O │
  └───┘              └───┘  ╚═══╝  └───┘  └───┘
                    rotated  flipped zoomed shifted

10,000 images → 50,000+ effective training samples!
4. Two-Phase Training
Python

# PHASE 1: Train classifier only
print("Phase 1: Training classifier...")
base_model.trainable = False  # Keep frozen

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test)
)

# PHASE 2: Fine-tune
print("Phase 2: Fine-tuning...")
base_model.trainable = True

# Only unfreeze top 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Use lower learning rate
model.compile(optimizer=Adam(learning_rate=0.00005), ...)

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=15,
    validation_data=(X_test, y_test)
)
Why Two Phases?

text

Phase 1: Learn what skin cancer looks like
┌─────────────────┐
│  EfficientNet   │  ← FROZEN
└────────┬────────┘
         ↓
┌─────────────────┐
│   Classifier    │  ← TRAINING (random → meaningful)
└─────────────────┘

Phase 2: Refine everything
┌─────────────────┐
│  EfficientNet   │  ← FINE-TUNING (top layers only)
└────────┬────────┘
         ↓
┌─────────────────┐
│   Classifier    │  ← TRAINING (continues)
└─────────────────┘
Troubleshooting
Common Issues
1. TensorFlow Not Found
Bash

# Error
ModuleNotFoundError: No module named 'tensorflow'

# Solution
pip install tensorflow
2. Model File Not Found
Bash

# Error
FileNotFoundError: skin_cancer_model.h5

# Solution - Train the model first
cd ml_model
python image_classification.py
3. CUDA Out of Memory
Python

# Error
ResourceExhaustedError: OOM when allocating tensor

# Solution 1: Reduce batch size
model.fit(..., batch_size=16)  # Instead of 32

# Solution 2: Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
4. Port Already in Use
Bash

# Error
OSError: Address already in use

# Solution - Kill existing process
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :5000
kill -9 <PID>
5. Wrong Predictions
Python

# Check 1: Preprocessing matches training
img = img.resize((224, 224))      # Must match!
img_array = np.array(img) / 255.0  # Must normalize!

# Check 2: Class mapping correct
print(SKIN_CANCER_CLASSES)

# Check 3: Model output
predictions = model.predict(img)
print(predictions)  # Should sum to ~1.0
print(np.sum(predictions))
