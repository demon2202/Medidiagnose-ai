<div align="center">

# 🏥 MediDiagnose-AI

### AI-Powered Medical Diagnosis & Disease Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

A full-stack medical AI application that combines machine learning models and deep learning to assist with symptom-based disease diagnosis, cancer screening, heart risk assessment, and medical image analysis — all from a single, unified interface.

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Training the Models](#-training-the-models)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [ML Models Overview](#-ml-models-overview)
- [Screenshots](#-screenshots)
- [Known Issues & Fixes](#-known-issues--fixes)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Module | Description | Model Type |
|--------|-------------|------------|
| 🩺 **Symptom Diagnosis** | Select from 132 symptoms to predict diseases across 42+ conditions | Voting Classifier (RF + GB + ExtraTrees) |
| 💜 **Breast Cancer Screening** | Input tumor characteristics (FNA data) to predict malignancy | Voting Classifier (RF + GB + LR + SVM) |
| ❤️ **Heart Risk Assessment** | 13 clinical parameters to assess cardiovascular disease risk | Gradient Boosting Ensemble |
| 🔬 **Skin Cancer Detection** | Upload a skin lesion photo to classify across 7 skin conditions | CNN (224×224 RGB) |
| 🩻 **Chest X-Ray Analysis** | Detect pneumonia from chest X-ray images | CNN (224×224 Grayscale) |
| 🫀 **Cardiac ECG Analysis** | Analyze ECG printouts or raw signal files (.dat, .csv, .hea) | CNN (224×224 Grayscale) |
| 🎗️ **Mammogram Analysis** | Analyze mammogram images for breast abnormalities | CNN (224×224 Grayscale) |

**Additional features:**
- 📊 Prediction history with confidence scores and recommendations
- 👤 User profile management
- 🌗 Dark / light mode support
- 🔒 Image validation — rejects wrong image types per analysis module
- 📋 Detailed recommendations, precautions, and treatment options per result
- 🚨 Urgency timelines (critical / urgent / routine) for each diagnosis

---

## 🛠 Tech Stack

### Frontend
- **React 18** with Vite
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **Axios** for API communication

### Backend
- **Python 3.9+**
- **Flask 3** + Flask-CORS
- **scikit-learn** — classical ML models
- **TensorFlow / Keras 2.20+** — deep learning image models
- **imbalanced-learn** — SMOTE oversampling
- **joblib** — model serialization
- **Pillow (PIL)** — image preprocessing
- **wfdb** — ECG signal file parsing (.dat/.hea)
- **matplotlib** — ECG signal → image conversion

---

## 📁 Project Structure

```
medidiagnose-ai/
│
├── backend/                        # Python Flask backend
│   ├── server.py                   # Main API server (all endpoints)
│   ├── disease_prediction_v2.py    # Train symptom → disease model
│   ├── train_cancer_model.py       # Train breast cancer (FNA) model
│   ├── train_heart_model.py        # Train heart disease risk model
│   ├── train_breast_cancer_model.py# Train breast cancer image model
│   ├── train_heart_image_model.py  # Train heart ECG image model
│   ├── image_classification.py     # Train skin cancer image model
│   ├── image_validator.py          # Image type validation utilities
│   ├── train_all_models.py         # Run all training scripts at once
│   ├── Dataset/
│   │   ├── cancer.csv              # Wisconsin Breast Cancer dataset
│   │   ├── heart.csv               # UCI Heart Disease dataset
│   │   └── dataset.csv             # Symptom-disease dataset (132 symptoms)
│   └── uploads/                    # Temporary image upload storage
│
├── ml_model/                       # Trained model artifacts (auto-generated)
│   ├── disease_model.joblib
│   ├── label_encoder.joblib
│   ├── symptom_list.json
│   ├── cancer_model.joblib
│   ├── cancer_scaler.joblib
│   ├── heart_disease_model.joblib
│   ├── heart_scaler.joblib
│   ├── skin_cancer_model.h5
│   ├── breast_cancer_model.h5
│   ├── pneumonia_model.h5
│   └── heart_image_model.h5
│
└── src/                            # React frontend
    ├── components/
    │   ├── ImageAnalysis.jsx        # Unified image upload + analysis UI
    │   ├── SymptomDiagnosis.jsx     # Symptom selector + diagnosis UI
    │   ├── HeartCheck.jsx           # Heart risk form + results UI
    │   ├── CancerScreening.jsx      # FNA tumor data form + results UI
    │   └── ProfileModal.jsx         # User profile editor
    ├── context/
    │   └── AppContext.jsx           # Global state (user, history, loading)
    ├── data/
    │   └── symptoms.js              # Symptom list with categories
    ├── config/
    │   └── config.js                # API base URL and app config
    └── main.jsx
```

---

## ⚙️ Installation

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/demon2202/Medidiagnose-ai.git
cd medidiagnose-ai
```

### 2. Set Up the Backend

```bash
cd backend

# Create and activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install Python dependencies
pip install flask flask-cors scikit-learn tensorflow pillow joblib \
            imbalanced-learn pandas numpy wfdb matplotlib
```

### 3. Set Up the Frontend

```bash
# From the project root
npm install
```

---

## 🤖 Training the Models

Before starting the server, you need to train the ML models. All training scripts are in the `backend/` folder.

### Train All Models at Once (Recommended)

```bash
cd backend
python train_all_models.py
```

This will sequentially run all training scripts and save the model files to `ml_model/`.

### Train Individual Models

```bash
cd backend

# Symptom → Disease model (required for /predict-disease)
python disease_prediction_v2.py

# Breast cancer FNA model (required for /predict-cancer)
python train_cancer_model.py

# Heart disease risk model (required for /predict-heart)
python train_heart_model.py

# Skin cancer image model (required for /analyze/skin)
python image_classification.py

# Chest X-ray / Pneumonia model (required for /analyze/xray)
# (included in train_all_models.py)

# Heart ECG image model (required for /analyze/heart)
python train_heart_image_model.py

# Breast cancer mammogram model (required for /analyze/breast)
python train_breast_cancer_model.py
```

> **Note:** If no trained models are found, the server automatically runs in **demo mode** with rule-based heuristics and a minimum 80% confidence floor. Results in demo mode are not medically accurate.

### Expected Output After Training

```
ml_model/
├── disease_model.joblib        ✅
├── label_encoder.joblib        ✅
├── symptom_list.json           ✅
├── cancer_model.joblib         ✅
├── cancer_scaler.joblib        ✅
├── heart_disease_model.joblib  ✅
├── heart_scaler.joblib         ✅
├── skin_cancer_model.h5        ✅ (requires image dataset)
├── breast_cancer_model.h5      ✅ (requires image dataset)
├── pneumonia_model.h5          ✅ (requires image dataset)
└── heart_image_model.h5        ✅ (requires image dataset)
```

---

## 🚀 Running the Application

### 1. Start the Backend Server

```bash
cd backend
python server.py
```

You should see:

```
🏥 MediDiagnose-AI Backend Server v4.1
============================================================
✅ TensorFlow 2.x.x loaded successfully
✅ PIL loaded successfully
✅ Disease prediction model loaded
✅ Cancer screening model loaded
✅ Heart risk model loaded
...
🚀 Server starting on http://localhost:5000
```

### 2. Start the Frontend

Open a new terminal:

```bash
# From the project root
npm run dev
```

Open your browser at: **http://localhost:5173**

---

## 📡 API Reference

Base URL: `http://localhost:5000`

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info, loaded models, available endpoints |
| `GET` | `/health` | Server health check with model status |
| `GET` | `/symptoms` | Full list of supported symptoms |

### Structured Data Predictions

| Method | Endpoint | Description | Required Fields |
|--------|----------|-------------|-----------------|
| `POST` | `/predict-disease` | Disease from symptoms | `symptoms: string[]` |
| `POST` | `/predict-cancer` | Breast cancer from FNA data | 10 tumor features (see below) |
| `POST` | `/predict-heart` | Heart disease risk | 13 clinical parameters (see below) |

#### `/predict-disease` Request
```json
{
  "symptoms": ["headache", "high_fever", "vomiting", "fatigue"]
}
```

#### `/predict-cancer` Request
```json
{
  "radius_mean": 14.5,
  "texture_mean": 19.0,
  "perimeter_mean": 92.0,
  "area_mean": 655.0,
  "smoothness_mean": 0.096,
  "compactness_mean": 0.104,
  "concavity_mean": 0.088,
  "concave_points_mean": 0.049,
  "symmetry_mean": 0.181,
  "fractal_dimension_mean": 0.063
}
```

#### `/predict-heart` Request
```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

### Image Analysis

All image endpoints accept `multipart/form-data` with an `image` field.

| Method | Endpoint | Expected Image | Description |
|--------|----------|----------------|-------------|
| `POST` | `/analyze/skin` | Color skin photo | 7-class skin cancer detection |
| `POST` | `/analyze/xray` | Chest X-ray (grayscale) | Pneumonia detection |
| `POST` | `/analyze/breast` | Mammogram (grayscale) | Breast cancer classification |
| `POST` | `/analyze/heart` | ECG printout or `.dat`/`.csv` signal file | Heart condition detection |

#### Standard Success Response
```json
{
  "success": true,
  "confidence": 0.87,
  "confidence_percent": "87.0%",
  "prediction": { "disease": "Typhoid", "confidence": 0.87 },
  "description": "Bacterial infection from Salmonella typhi...",
  "precautions": ["Complete antibiotic course", "Drink only clean water"],
  "recommendations": ["Consult a doctor", "Stay hydrated"],
  "alternative_diagnoses": [
    { "disease": "Dengue", "confidence": 0.08 }
  ]
}
```

---

## 🧠 ML Models Overview

### Classical ML Models (scikit-learn)

| Model | Algorithm | Dataset | Accuracy |
|-------|-----------|---------|----------|
| Disease Prediction | VotingClassifier (RF + GradientBoosting + ExtraTrees) | 132-symptom dataset, 42 diseases | ~95%+ |
| Cancer Screening (FNA) | VotingClassifier (RF + GB + LR + SVM) + SMOTE | Wisconsin Breast Cancer (569 samples) | ~96%+ |
| Heart Risk | Ensemble with GridSearchCV | UCI Heart Disease (303+ samples) | ~88%+ |

### Deep Learning Models (TensorFlow / Keras)

| Model | Architecture | Input | Classes |
|-------|-------------|-------|---------|
| Skin Cancer | CNN (224×224 RGB) | Dermoscopy image | 7 (akiec, bcc, bkl, df, mel, nv, vasc) |
| Pneumonia | CNN (224×224 Grayscale) | Chest X-ray | 2 (Normal, Pneumonia) |
| Breast Cancer (Image) | CNN (224×224 Grayscale) | Mammogram | 3 (Normal, Benign, Malignant) |
| Heart Condition | CNN (224×224 Grayscale) | ECG image/signal | 5 (Normal, MI, Arrhythmia, HF, Hypertrophy) |

### Skin Cancer Classes
| Code | Name | Type |
|------|------|------|
| `akiec` | Actinic Keratoses | Pre-cancerous |
| `bcc` | Basal Cell Carcinoma | Malignant |
| `bkl` | Benign Keratosis | Benign |
| `df` | Dermatofibroma | Benign |
| `mel` | Melanoma | Malignant |
| `nv` | Melanocytic Nevi | Benign |
| `vasc` | Vascular Lesions | Benign |

---

## 🐛 Known Issues & Fixes

### Common Problems

**`❌ Failed to load Heart image model: string indices must be integers`**
> TensorFlow 2.18+ changed how it deserializes `.h5` model configs.
> **Fix:** `server.py` now uses `keras.models.load_model(..., compile=False)` — this is already applied.

**Heart Check shows "Failed to connect" even when server is running**
> The scaler was not being applied before `predict_proba`, causing a silent 500 error that looked like a connection failure.
> **Fix:** Scaler is now applied before all inference paths in `predict_heart()`.

**Confidence always shows as N/A in Symptom Diagnosis**
> `SymptomDiagnosis.jsx` was reading `diagnosisData.confidence` but the server only returned confidence inside the nested `prediction` object.
> **Fix:** Server now sends `confidence` at the top level; frontend reads with safe fallback.

**Trained models not loading (server always in demo mode)**
> Training scripts were saving models to the wrong directory — `backend/` instead of `ml_model/`.
> **Fix:** `train_cancer_model.py` and `train_heart_model.py` now save to `../ml_model/` relative to `backend/`, matching `server.py`'s `MODEL_PATHS`.

### Debug Endpoint

Use the built-in debug endpoint to inspect image statistics for tuning the image validator:

```bash
curl -X POST http://localhost:5000/debug/image-stats \
  -F "image=@your_image.jpg"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

### Development Notes

- The backend runs on port **5000**, frontend on **5173**
- CORS is configured to allow `localhost:5173` and `localhost:3000`
- All model files go to `ml_model/` — never commit trained `.joblib` or `.h5` files
- Add `ml_model/` to `.gitignore`
- The `CONFIDENCE_FLOOR` constant in `server.py` controls minimum displayed confidence in demo mode (default: 80%)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

Built with ❤️ by [Harshit](https://github.com/demon2202)

If you found this useful, please ⭐ the repo!

</div>
