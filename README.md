<div align="center">

# 🏥 MediDiagnose-AI

### AI-Powered Medical Diagnosis & Disease Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/demon2202/Medidiagnose-ai?style=for-the-badge&logo=github)](https://github.com/demon2202/Medidiagnose-ai/stargazers)

A full-stack medical AI application that combines classical machine learning models and deep learning (CNNs) to assist with **symptom-based disease diagnosis**, **cancer screening**, **heart risk assessment**, and **medical image analysis** — all from a single, unified, production-ready interface.

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Models Deep Dive](#-ml-models-deep-dive)
- [Datasets Used](#-datasets-used)
- [Training the Models](#-training-the-models)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Screenshots & Visualizations](#-screenshots--visualizations)
- [Performance Charts](#-performance-charts)
- [Known Issues & Fixes](#-known-issues--fixes)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Project Overview

**MediDiagnose-AI** is a comprehensive medical AI platform designed to assist healthcare professionals and individuals with preliminary medical screening. The system integrates **7 distinct AI models** spanning classical ML and deep learning, covering:

| Domain | Models | Purpose |
|--------|--------|---------|
| **General Diagnosis** | Symptom → Disease (42 conditions) | Map 132 symptoms to diseases |
| **Cancer Screening** | FNA Tumor Classifier (Benign/Malignant) | Wisconsin Breast Cancer |
| **Cardiac Risk** | Heart Disease Risk (13 clinical params) | UCI Heart Disease |
| **Dermatology** | Skin Cancer (7 classes) | HAM10000 + PAD-UFES-20 |
| **Radiology** | Pneumonia (Chest X-ray) | NIH Chest X-ray |
| **Cardiology (Imaging)** | Heart Condition (ECG) | PTB-XL (full dataset) |
| **Oncology (Imaging)** | Breast Mammogram (3 classes) | CBIS-DDSM / custom |

### Why This Project Exists

Medical AI often suffers from:
- **Fragmented models** — separate repos, inconsistent APIs
- **Training/serving skew** — preprocessing differs between training and inference
- **Demo-quality confidence** — models output extreme 0%/100% probabilities
- **No unified interface** — users juggle multiple tools

**MediDiagnose-AI solves these** with:
- ✅ **Unified API** — consistent response structure across all 12 endpoints
- ✅ **Train/serve parity** — shared `medidiagnose.inference_utils` package
- ✅ **Calibrated probabilities** — `CalibratedClassifierCV` prevents 0%/100%
- ✅ **Single UI** — React frontend with routing, history, dark mode
- ✅ **Image validation** — rejects wrong image types per analysis module
- ✅ **Patient-level splits** — no data leakage in image models

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MEDIDIAGNOSE-AI ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
│   FRONTEND   │────▶│      BACKEND        │────▶│       ML MODELS          │
│  (React 18)  │     │    (Flask 3)        │     │  (sklearn + TensorFlow)  │
└──────────────┘     └─────────────────────┘     └──────────────────────────┘
       │                      │                          │
       │                      │                          │
       ▼                      ▼                          ▼
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
│ • Dashboard  │     │ • /predict-disease  │     │ • disease_model.joblib   │
│ • Symptom    │     │ • /predict-cancer   │     │ • cancer_model.joblib    │
│   Diagnosis  │     │ • /predict-heart    │     │ • heart_disease_model    │
│ • Image      │     │ • /analyze/skin     │     │ • skin_cancer_model.h5   │
│   Analysis   │     │ • /analyze/xray     │     │ • pneumonia_model.h5     │
│ • Heart      │     │ • /analyze/breast   │     │ • breast_cancer_model.h5 │
│   Check      │     │ • /analyze/heart    │     │ • heart_image_model.h5   │
│ • Cancer     │     │ • /symptoms/match   │     │                          │
│   Screening  │     │ • /health           │     │                          │
│ • History    │     │                     │     │                          │
│ • Settings   │     │                     │     │                          │
└──────────────┘     └─────────────────────┘     └──────────────────────────┘
       │                      │                          │
       │                      │                          │
       ▼                      ▼                          ▼
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
│ • React      │     │ • Flask-CORS        │     │ • joblib / .h5 artifacts │
│   Router     │     │ • inference_utils   │     │ • JSON configs           │
│ • Framer     │     │   (train/serve      │     │ • CalibratedClassifierCV │
│   Motion     │     │   parity)           │     │ • StandardScaler         │
│ • Lenis      │     │ • image_validator   │     │ • LabelEncoder           │
│ • Lucide     │     │   (lenient check)   │     │                          │
└──────────────┘     └─────────────────────┘     └──────────────────────────┘
```

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Shared `inference_utils`** | Training scripts and `server.py` import the SAME preprocessing module — zero skew |
| **Calibrated probabilities** | `CalibratedClassifierCV(method='sigmoid')` wraps ensembles — realistic confidence |
| **Patient-disjoint splits** | Image models split by patient ID, not image — no leakage |
| **Image validator v2** | Evidence-based (not strict) — accepts real clinical images that strict validators reject |
| **Signal → Image pipeline** | ECG `.dat/.hea` → `wfdb` → matplotlib render → CNN — handles both upload types |
| **Consistent API schema** | Every endpoint returns `success`, `confidence`, `confidence_percent`, `prediction`, `description`, `precautions`, `recommendations`, `alternative_diagnoses` |

---

## ✨ Features

### Core Diagnostic Modules

| Module | Input | Output | Model |
|--------|-------|--------|-------|
| 🩺 **Symptom Diagnosis** | 132 symptoms (multi-select with search + synonyms) | Top-5 diseases with confidence, description, precautions, alternatives | Voting Ensemble (RF + GB + ET) |
| 💜 **Breast Cancer Screening** | 10 FNA tumor features (radius, texture, perimeter, etc.) | Benign/Malignant with probability, recommendation | Voting Ensemble + SMOTE |
| ❤️ **Heart Risk Assessment** | 13 clinical params (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) | Risk level, probability, clinical guidance | Gradient Boosting Ensemble |
| 🔬 **Skin Cancer Detection** | Dermoscopy photo (color) | 7-class classification (akiec, bcc, bkl, df, mel, nv, vasc) | MobileNetV2 Transfer Learning |
| 🩻 **Chest X-Ray Analysis** | Chest X-ray (grayscale) | Normal / Pneumonia | MobileNetV2 Transfer Learning |
| 🫀 **Cardiac ECG Analysis** | ECG image OR raw signal (.dat, .hea, .csv, .edf, .mat) | 5-class (Normal, MI, Arrhythmia, HF, Hypertrophy) | MobileNetV2 on rendered ECG |
| 🎗️ **Mammogram Analysis** | Mammogram/ultrasound (grayscale) | Normal / Benign / Malignant (BI-RADS) | MobileNetV2 Transfer Learning |

### Platform Features

- 📊 **Prediction History** — Persisted in localStorage with timestamps, confidence, severity
- 👤 **User Profiles** — Registration, login, profile editing, password reset
- 🌗 **Dark/Light Mode** — System-aware with manual toggle
- 🔒 **Image Validation** — Per-module validation (skin expects color, X-ray expects grayscale, etc.)
- 📋 **Rich Results** — Confidence rings, severity badges, urgency timelines, expandable details
- 🎨 **Modern UI** — Framer Motion animations, Lenis smooth scroll, Lucide icons, Tailwind CSS
- ⌨️ **Command Palette** — `Ctrl/Cmd+K` for quick navigation
- ♿ **Accessibility** — Semantic HTML, ARIA labels, focus management

---

## 🛠 Tech Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.3.1 | UI framework |
| Vite | 7.2.4 | Build tool / dev server |
| Tailwind CSS | 3.4.19 | Utility-first styling |
| Framer Motion | 13.2.0 | Animations & transitions |
| Lenis | 1.3.26 | Smooth scrolling |
| Lucide React | 0.562.0 | Icon system |
| Axios | 1.13.5 | HTTP client |
| React Router DOM | 6.30.3 | Client-side routing |
| bcryptjs | 3.0.3 | Client-side password hashing |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Runtime |
| Flask | 3.0.0 | Web framework |
| Flask-CORS | 4.0.0 | Cross-origin requests |
| scikit-learn | 1.0+ | Classical ML |
| TensorFlow/Keras | 2.20+ | Deep learning |
| imbalanced-learn | Latest | SMOTE oversampling |
| joblib | 1.2+ | Model serialization |
| Pillow (PIL) | 9.0+ | Image preprocessing |
| wfdb | 4.1+ | ECG signal parsing |
| matplotlib | 3.5+ | ECG signal → image rendering |
| pandas/numpy | Latest | Data manipulation |

---

## 📁 Project Structure

```
medidiagnose-ai/
│
├── .github/                          # GitHub workflows (if any)
├── .opencode/                        # OpenCode configuration
├── backend/                          # Python Flask backend
│   ├── server.py                     # Main API server (2000+ lines)
│   ├── disease_prediction_v2.py      # Train symptom → disease model
│   ├── train_cancer_model.py         # Train breast cancer (FNA) model
│   ├── train_heart_model.py          # Train heart disease risk model
│   ├── train_breast_cancer_model.py  # Train breast cancer IMAGE model
│   ├── train_heart_image_model.py    # Train heart ECG IMAGE model
│   ├── image_classification.py       # Train skin cancer + pneumonia IMAGE models
│   ├── image_validator.py            # Lenient evidence-based image validation
│   ├── train_all_models.py           # Orchestrates all training
│   ├── reorganize_datasets.py        # Dataset preparation utilities
│   ├── eval_utils.py                 # Evaluation helpers
│   ├── verify_dataset.py             # Dataset verification
│   ├── Dataset/                      # Training datasets (gitignored)
│   │   ├── cancer.csv                # Wisconsin Breast Cancer (569 samples)
│   │   ├── heart.csv                 # UCI Heart Disease (303+ samples)
│   │   └── dataset.csv               # Symptom-disease (132 symptoms, 42 diseases, ~246k rows)
│   ├── uploads/                      # Temporary image storage (gitignored)
│   └── requirements.txt              # Python dependencies
│
├── ml_model/                         # TRAINED ARTIFACTS (gitignored - generated at runtime)
│   ├── disease_model.joblib          # Symptom → disease classifier
│   ├── label_encoder.joblib          # Disease name ↔ index mapping
│   ├── symptom_list.json             # 132 canonical symptom names
│   ├── cancer_model.joblib           # Breast cancer FNA classifier
│   ├── cancer_scaler.joblib          # StandardScaler for 10 FNA features
│   ├── cancer_features.json          # Feature metadata
│   ├── cancer_metrics.json           # Training metrics
│   ├── cancer_config.json            # Model config
│   ├── heart_disease_model.joblib    # Heart disease risk classifier
│   ├── heart_scaler.joblib           # StandardScaler for 13 features
│   ├── heart_features.json           # Feature metadata
│   ├── heart_metrics.json            # Training metrics
│   ├── heart_config.json             # Model config
│   ├── skin_cancer_model.h5          # Skin cancer CNN (7-class)
│   ├── skin_cancer_config.json       # Config + confusion matrix
│   ├── pneumonia_model.h5            # Pneumonia CNN (2-class)
│   ├── pneumonia_config.json         # Config + confusion matrix
│   ├── breast_cancer_model.h5        # Mammogram CNN (3-class)
│   ├── breast_cancer_config.json     # Config + confusion matrix
│   ├── heart_image_model.h5          # ECG CNN (5-class)
│   ├── heart_image_config.json       # Config + confusion matrix
│   ├── model_config.json             # Disease model metadata (677 diseases)
│   ├── symptom_synonyms.json         # User synonyms → canonical mapping
│   ├── symptom_severity_weights.json # Symptom severity weights
│   ├── disease_info.json             # Disease descriptions & precautions
│   └── *.py                          # Training scripts (duplicated from backend/)
│
├── medidiagnose/                     # Shared Python package (train/serve parity)
│   ├── __init__.py
│   ├── inference_utils.py            # Unified preprocessing & inference
│   ├── cnn_builders.py               # MobileNetV2 model builders
│   └── train_utils.py                # Training utilities (two-phase fine-tune)
│
├── frontend/                         # React frontend (Vite + Tailwind)
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/               # CommandPalette, Disclaimer, Notification, ResultInspector
│   │   │   ├── layout/               # Header, Sidebar, MobileNav, AuthShell
│   │   │   ├── modals/               # ProfileModal
│   │   │   ├── results/              # ImageResultView, ScreeningResultView, SymptomResultView
│   │   │   └── ui/                   # PageHeader, TextField, EmptyState, Reveal, ConfidenceRing, etc.
│   │   ├── context/
│   │   │   └── AppContext.jsx        # Global state (auth, history, notifications, loading)
│   │   ├── data/
│   │   │   ├── diseases.js           # Disease metadata for UI
│   │   │   └── symptoms.js           # 132 symptoms with categories
│   │   ├── lib/
│   │   │   ├── diagnosis.js          # API calls
│   │   │   ├── motion.js             # Framer Motion variants
│   │   │   ├── scroll.js             # Lenis integration
│   │   │   ├── severity.js           # Severity → color/label mapping
│   │   │   ├── summary.js            # Result summary builders
│   │   │   ├── text.js               # Text cleaning utilities
│   │   │   └── useCountUp.js         # Animated number hook
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx         # Landing page with quick actions
│   │   │   ├── SymptomDiagnosis.jsx  # Symptom selector + results
│   │   │   ├── ImageAnalysis.jsx     # Unified image upload (4 analysis types)
│   │   │   ├── HeartCheck.jsx        # Heart risk form + results
│   │   │   ├── CancerScreening.jsx   # FNA feature form + results
│   │   │   ├── History.jsx           # Prediction history
│   │   │   ├── HealthTips.jsx        # Health articles
│   │   │   ├── Settings.jsx          # Profile, theme, preferences
│   │   │   ├── Login.jsx             # Authentication
│   │   │   ├── Signup.jsx
│   │   │   └── ForgotPassword.jsx
│   │   ├── config/
│   │   │   └── config.js             # API base URL, upload limits
│   │   ├── App.jsx                   # Routes, providers, layout
│   │   ├── main.jsx                  # Entry point
│   │   └── index.css                 # Tailwind + custom styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── eslint.config.js
│
├── README.md                         # This file
└── LICENSE
```

---

## 🧠 ML Models Deep Dive

### 1. Symptom → Disease Prediction (`disease_model.joblib`)

**Algorithm**: `VotingClassifier` (Soft Voting) with:
- `RandomForestClassifier` (n_estimators=300, max_depth=15, class_weight='balanced')
- `GradientBoostingClassifier` (n_estimators=200, learning_rate=0.1, max_depth=5)
- `ExtraTreesClassifier` (n_estimators=300, max_depth=15, class_weight='balanced')

**Features**: 377 binary symptom indicators (after cleaning & deduplication)

**Classes**: 677 diseases (filtered from raw 42+ to min 10 samples each)

**Training Data**: 197,209 train / 49,303 test samples

**Performance**:
- Top-1 Accuracy: **83.68%**
- Top-5 Accuracy: **96.52%**

**Why Voting Ensemble?**
- Single models overfit on high-dimensional sparse symptom vectors
- Soft voting averages probability distributions → smoother, calibrated outputs
- Class-weighted RF/ET handles severe class imbalance (some diseases have 10 samples, others 10,000+)

**Synonym Handling**: 200+ user synonyms mapped to canonical symptoms (e.g., "fever" → "high_fever", "shortness_of_breath" → "breathlessness")

---

### 2. Breast Cancer Screening - FNA (`cancer_model.joblib`)

**Algorithm**: Calibrated Soft-Voting Ensemble:
- `RandomForestClassifier` (max_depth=8, min_samples_leaf=2 — **regularized**)
- `GradientBoostingClassifier` (n_estimators=250, max_depth=4 — **constrained**)
- `LogisticRegression` (C=1.0, l2 penalty)

**Calibration**: `CalibratedClassifierCV(method='sigmoid', cv=5)` on held-out calibration set

**Features**: 10 mean tumor characteristics from fine-needle aspirate:
- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`
- `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave_points_mean`
- `symmetry_mean`, `fractal_dimension_mean`

**Dataset**: Wisconsin Breast Cancer (569 samples, 357 benign / 212 malignant)

**Performance** (Test set, 114 samples):
| Model | Accuracy | ROC-AUC | F1 | MCC | Prob Range |
|-------|----------|---------|-----|-----|------------|
| Random Forest | 96.5% | 0.992 | 0.96 | 0.93 | [0.02, 0.98] |
| Gradient Boosting | 97.4% | 0.995 | 0.97 | 0.94 | [0.01, 0.99] |
| Logistic Regression | 95.6% | 0.989 | 0.95 | 0.91 | [0.03, 0.97] |
| **Calibrated Ensemble** | **96.5%** | **0.994** | **0.96** | **0.93** | **[0.05, 0.95]** ✅ |

**Key Improvement**: Old version reported 100% on all models (overfitting on small test split). New version uses **regularization** (bounded depth, min_samples_leaf) + **calibration** → realistic probabilities, no 0%/100% outputs.

---

### 3. Heart Disease Risk (`heart_disease_model.joblib`)

**Algorithm**: Calibrated Weighted Ensemble:
- `RandomForestClassifier` (bounded depth, class_weight='balanced')
- `GradientBoostingClassifier` (n_estimators=300, max_depth=4 — **stronger than before**)
- `LogisticRegression` (C tuned)

**Ensemble Weights**: Derived from per-model CV AUC (normalized to sum=7)

**Calibration**: `CalibratedClassifierCV(method='sigmoid', cv=5)`

**Features**: 13 raw clinical parameters (NO feature engineering — matches server input exactly):
| Feature | Description | Range |
|---------|-------------|-------|
| `age` | Age in years | 29-77 |
| `sex` | 1=Male, 0=Female | {0,1} |
| `cp` | Chest pain type | 0-3 |
| `trestbps` | Resting BP (mmHg) | 94-200 |
| `chol` | Serum cholesterol (mg/dl) | 126-564 |
| `fbs` | Fasting blood sugar >120 | {0,1} |
| `restecg` | Resting ECG results | 0-2 |
| `thalach` | Max heart rate achieved | 71-202 |
| `exang` | Exercise-induced angina | {0,1} |
| `oldpeak` | ST depression | 0-6.2 |
| `slope` | ST segment slope | 0-2 |
| `ca` | Major vessels (fluoroscopy) | 0-3 |
| `thal` | Thalassemia | 1-3 |

**Dataset**: UCI Heart Disease (Cleveland + Hungarian + Switzerland + VA, ~303 unique records after deduplication)

**Performance**:
- Test Accuracy: **82-88%** (varies by split)
- ROC-AUC: **~0.85-0.90**
- 5-Fold CV ROC-AUC: **0.818 ± 0.046**

**Why No Feature Engineering?**
Previous versions created interaction features → dimension mismatch at inference (server sends 13 raw features). This version uses **only the 13 raw features** — guaranteed dimension compatibility.

---

### 4. Skin Cancer Detection (`skin_cancer_model.h5`)

**Architecture**: MobileNetV2 (ImageNet pretrained) + Custom Head
- Input: 224×224×3 RGB
- Base: MobileNetV2 (α=1.0, include_top=False)
- Head: GlobalAveragePooling2D → Dropout(0.4) → Dense(7, softmax)
- In-model augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast

**Training Strategy** (Two-Phase):
1. **Phase 1** (Frozen backbone): 20 epochs, LR=1e-3, Adam
2. **Phase 2** (Full fine-tune): 25 epochs, LR=3e-5, cosine decay

**Class Balancing**: Train-only `class_weight='balanced'` (no validation skew)

**Dataset**: HAM10000 (10,015 images) + PAD-UFES-20 (2,298 images)
- **Split**: Group-disjoint by `lesion_id` / patient — no leakage
- **Classes** (7):
  - `akiec` (Actinic Keratoses) - Pre-cancerous
  - `bcc` (Basal Cell Carcinoma) - Malignant
  - `bkl` (Benign Keratosis) - Benign
  - `df` (Dermatofibroma) - Benign
  - `mel` (Melanoma) - Malignant ⚠️
  - `nv` (Melanocytic Nevi) - Benign
  - `vasc` (Vascular Lesions) - Benign

**Performance**:
- Test Accuracy: **75.8%**
- Confusion Matrix: See `skin_cancer_config.json`

**Why MobileNetV2?**
- Lightweight (3.5M params) → fast inference on CPU
- Proven on dermatology tasks (ISIC benchmarks)
- Transfer learning from ImageNet works well for dermoscopy

---

### 5. Pneumonia Detection (`pneumonia_model.h5`)

**Architecture**: MobileNetV2 (ImageNet pretrained) + Custom Head
- Input: 224×224×1 Grayscale (replicated to 3 channels internally)
- Same two-phase training as skin cancer

**Dataset**: NIH Chest X-ray (ChestX-ray14 subset) — Normal vs Pneumonia

**Performance**:
- Test Accuracy: **88.2%**
- Optimal Threshold: 0.5 (softmax)
- Temperature Scaling: 2.5 (calibration)

**Confusion Matrix**:
| | Pred Normal | Pred Pneumonia |
|---|-------------|----------------|
| Actual Normal | 194 (TN) | 80 (FP) |
| Actual Pneumonia | 3 (FN) | 427 (TP) |

**High Sensitivity**: 99.3% — critical for screening (few false negatives)

---

### 6. Breast Cancer Mammogram (`breast_cancer_model.h5`)

**Architecture**: MobileNetV2 Transfer Learning (3-class)
- Input: 224×224×1 Grayscale (no CLAHE/sharpen — clean preprocessing)
- Patient-level split (no image-level leakage)
- In-model augmentation, class_weight balanced
- Two-phase fine-tune (BN frozen in phase 1)

**Classes** (BI-RADS aligned):
| Class | Code | Name | BI-RADS | Severity |
|-------|------|------|---------|----------|
| 0 | `normal` | Normal | 1 | Healthy |
| 1 | `benign` | Benign Tumor | 2 | Low |
| 2 | `malignant` | Malignant Tumor | 5 | Critical |

**Performance**:
- Test Accuracy: **85.6%**
- Confusion Matrix:
  - Normal: 62/65 correct
  - Benign: 152/187 correct
  - Malignant: 84/96 correct

---

### 7. Heart ECG Analysis (`heart_image_model.h5`)

**Architecture**: MobileNetV2 Transfer Learning (5-class)
- Input: 224×224×1 Grayscale (ECG rendered as dark-background image)
- **Full PTB-XL dataset** (records100, ~21,837 records)
- Patient-disjoint `strat_fold` split: Train (folds 1-8), Val (9), Test (10)
- `sqrt`-balanced class weights
- Full-backbone fine-tune, cosine LR schedule

**Classes** (SCP-ECG diagnostic superclasses):
| Class | Code | Name | Severity |
|-------|------|------|----------|
| 0 | `normal` | Normal | Healthy |
| 1 | `mi` | Myocardial Infarction | Critical |
| 2 | `arrhythmia` | Arrhythmia | Moderate |
| 3 | `hf` | Heart Failure Signs | High |
| 4 | `hypertrophy` | Ventricular Hypertrophy | Moderate |

**Performance**:
- Test Accuracy: **71.8%**
- Per-class performance varies (Normal dominates dataset)

**Signal Processing Pipeline** (in `medidiagnose.inference_utils`):
1. Load `.dat/.hea` via `wfdb` OR parse `.csv`
2. Select Lead II (or first available)
3. Bandpass filter 0.5-40 Hz
4. Segment into 10-second windows
5. Render each window as dark-background ECG image (matplotlib)
6. Normalize to [0,1], replicate to 3 channels
7. Batch inference → aggregate predictions

---

## 📊 Datasets Used

### Tabular Datasets (Classical ML)

| Dataset | Source | Samples | Features | Classes | License |
|---------|--------|---------|----------|---------|---------|
| **Symptom-Disease** | Mendeley Data (doi:10.17632/2cxccsxydc.1) | 246,512 | 132 symptoms | 677 diseases (min 10 samples) | CC BY 4.0 |
| **Wisconsin Breast Cancer** | UCI ML Repository | 569 | 30 (10 mean used) | 2 (Benign/Malignant) | Public |
| **UCI Heart Disease** | UCI ML Repository | 303 (unique) | 13 clinical | 2 (No Disease/Disease) | Public |

### Image Datasets (Deep Learning)

| Dataset | Source | Images | Classes | Split Strategy |
|---------|--------|--------|---------|----------------|
| **HAM10000** | Harvard Dataverse / ISIC | 10,015 | 7 | Group-disjoint by `lesion_id` |
| **PAD-UFES-20** | UFES / ISIC | 2,298 | 7 | Group-disjoint by patient |
| **NIH Chest X-ray** | NIH Clinical Center | ~112,000 | 14 (we use 2) | Patient-disjoint |
| **CBIS-DDSM** | NCI / Cancer Imaging Archive | ~2,620 | 3 | Patient-disjoint |
| **PTB-XL** | PhysioNet | 21,837 records | 5 (superclasses) | `strat_fold` 1-8/9/10 |

### How to Obtain Datasets

#### Symptom-Disease Dataset (Required for disease model)
```bash
# Download from Mendeley Data
# DOI: 10.17632/2cxccsxydc.1
# Place as: backend/Dataset/dataset.csv
```

#### Wisconsin Breast Cancer
```bash
# Auto-downloaded by train_cancer_model.py via sklearn.datasets.load_breast_cancer()
# Or manually: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
```

#### UCI Heart Disease
```bash
# Auto-downloaded by train_heart_model.py
# Or manually: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
```

#### HAM10000 (Skin Cancer)
```bash
# Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
# Extract to: backend/Dataset/HAM10000/
# Required files: HAM10000_images_part_1.zip, HAM10000_images_part_2.zip, HAM10000_metadata.csv
```

#### PAD-UFES-20 (Skin Cancer Supplement)
```bash
# Download from: https://www.kaggle.com/datasets/ucfai/pad-ufes-20
# Extract to: backend/Dataset/PAD-UFES-20/
```

#### NIH Chest X-ray (Pneumonia)
```bash
# Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC
# Or use Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
# We use a curated subset (Normal + Pneumonia only)
```

#### CBIS-DDSM (Breast Mammogram)
```bash
# Download from: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
# Requires TCIA account
# Extract to: backend/Dataset/CBIS-DDSM/
```

#### PTB-XL (ECG)
```bash
# Download from: https://physionet.org/content/ptb-xl/1.0.3/
# wget -r -np -nH --cut-dirs=3 -R "index.html*" https://physionet.org/files/ptb-xl/1.0.3/
# Place at: backend/Dataset/ptb-xl/
# Required: records100/, ptbxl_database.csv, scp_statements.csv
```

> **Note**: Image datasets are LARGE (GBs). Training scripts check for local data and provide clear instructions if missing. The classical ML datasets are auto-downloaded via sklearn.

---

## ⚙️ Installation

### Prerequisites
- **Node.js** ≥ 18.0.0
- **Python** ≥ 3.9
- **Git**
- **RAM**: ≥ 8 GB (16 GB recommended for image model training)
- **GPU**: Optional but recommended for CNN training (CUDA 11.8+ for TF 2.20)

### 1. Clone Repository
```bash
git clone https://github.com/demon2202/Medidiagnose-ai.git
cd medidiagnose-ai
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify TensorFlow GPU (optional)
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### 3. Frontend Setup
```bash
# From project root
cd ..
npm install
```

---

## 🤖 Training the Models

### Option A: Train All at Once (Recommended)
```bash
cd backend
python train_all_models.py
```
This runs all training scripts sequentially and saves artifacts to `../ml_model/`.

### Option B: Train Individual Models

```bash
cd backend

# 1. Symptom → Disease (required for /predict-disease, /symptoms/match)
python disease_prediction_v2.py
# Optional: python disease_prediction_v2.py --test "headache,high_fever,vomiting"

# 2. Breast Cancer FNA (required for /predict-cancer)
python train_cancer_model.py

# 3. Heart Disease Risk (required for /predict-heart)
python train_heart_model.py

# 4. Skin Cancer Image (required for /analyze/skin)
python image_classification.py
# Choose option 1 when prompted

# 5. Pneumonia X-Ray (required for /analyze/xray)
python image_classification.py
# Choose option 2 when prompted

# 6. Heart ECG Image (required for /analyze/heart)
python train_heart_image_model.py
# Requires PTB-XL dataset — will fail gracefully if not found

# 7. Breast Mammogram (required for /analyze/breast)
python train_breast_cancer_model.py
# Choose option 1 (3-class transfer learning) when prompted
```

### Training Time Estimates (CPU)
| Model | Time (CPU) | Time (GPU) |
|-------|------------|------------|
| Disease Prediction | ~2-3 min | ~1 min |
| Cancer FNA | ~30 sec | ~10 sec |
| Heart Risk | ~45 sec | ~15 sec |
| Skin Cancer | ~15-20 min | ~3-5 min |
| Pneumonia | ~10-15 min | ~2-3 min |
| Breast Mammogram | ~15-20 min | ~3-5 min |
| Heart ECG | ~30-45 min | ~8-15 min |

### Expected Artifacts in `ml_model/`
```
ml_model/
├── disease_model.joblib          ✅
├── label_encoder.joblib          ✅
├── symptom_list.json             ✅
├── symptom_synonyms.json         ✅
├── disease_info.json             ✅
├── model_config.json             ✅
├── cancer_model.joblib           ✅
├── cancer_scaler.joblib          ✅
├── cancer_features.json          ✅
├── cancer_metrics.json           ✅
├── cancer_config.json            ✅
├── heart_disease_model.joblib    ✅
├── heart_scaler.joblib           ✅
├── heart_features.json           ✅
├── heart_metrics.json            ✅
├── heart_config.json             ✅
├── skin_cancer_model.h5          ✅
├── skin_cancer_config.json       ✅
├── pneumonia_model.h5            ✅
├── pneumonia_config.json         ✅
├── breast_cancer_model.h5        ✅
├── breast_cancer_config.json     ✅
├── heart_image_model.h5          ✅
└── heart_image_config.json       ✅
```

> **Demo Mode**: If any model file is missing, `server.py` falls back to rule-based heuristics with a minimum 80% confidence floor. Results are **not medically accurate** — train models for real use.

---

## 🚀 Running the Application

### 1. Start Backend Server
```bash
cd backend
python server.py
```

**Expected Output**:
```
🏥 MediDiagnose-AI Backend Server v4.1
============================================================
✅ TensorFlow 2.20.0 loaded successfully with oneDNN optimizations
✅ PIL loaded successfully
✅ medidiagnose v2 inference_utils loaded — train/serve preprocessing unified
✅ image_validator v2 loaded — lenient evidence-based validation
🔧 Loading ML models...
   ✅ Disease model: 677 diseases, 377 symptoms
   ✅ Cancer screening model: calibrated VotingClassifier
   ✅ Heart risk model: calibrated ensemble
   ✅ Skin cancer model: MobileNetV2 (7 classes)
   ✅ Pneumonia model: MobileNetV2 (2 classes)
   ✅ Breast cancer model: MobileNetV2 (3 classes)
   ✅ Heart ECG model: MobileNetV2 (5 classes)
🚀 Server starting on http://localhost:5000
```

### 2. Start Frontend
```bash
# New terminal, from project root
npm run dev
```

**Open**: http://localhost:5173

### 3. Verify Health
```bash
curl http://localhost:5000/health
```
```json
{
  "status": "healthy",
  "models": {
    "disease": true,
    "cancer": true,
    "heart": true,
    "skin_cancer": true,
    "pneumonia": true,
    "breast_cancer": true,
    "heart_image": true
  }
}
```

---

## 📡 API Reference

**Base URL**: `http://localhost:5000`

### Health & Discovery
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info, loaded models, all endpoints |
| `GET` | `/health` | Health check with model status |
| `GET` | `/symptoms` | List all 132 supported symptoms |

### Structured Data Endpoints

#### `POST /predict-disease`
```bash
curl -X POST http://localhost:5000/predict-disease \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["headache", "high_fever", "vomiting", "fatigue"]}'
```

**Response**:
```json
{
  "success": true,
  "confidence": 0.873,
  "confidence_percent": "87.3%",
  "prediction": {
    "disease": "Migraine",
    "confidence": 0.873
  },
  "description": "Severe, often one-sided headache with nausea...",
  "precautions": ["Identify triggers", "Rest in dark room", "Take prescribed meds"],
  "recommendations": ["Consult neurologist", "Keep headache diary"],
  "alternative_diagnoses": [
    {"disease": "Tension Headache", "confidence": 0.08},
    {"disease": "Hypertension", "confidence": 0.03}
  ],
  "timestamp": "2025-01-15T10:30:45.123Z"
}
```

#### `POST /predict-cancer`
```bash
curl -X POST http://localhost:5000/predict-cancer \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### `POST /predict-heart`
```bash
curl -X POST http://localhost:5000/predict-heart \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
    "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
  }'
```

### Image Analysis Endpoints
All accept `multipart/form-data` with field `image` (or `signal_file` for ECG).

```bash
# Skin cancer
curl -X POST http://localhost:5000/analyze/skin \
  -F "image=@lesion.jpg"

# Chest X-ray
curl -X POST http://localhost:5000/analyze/xray \
  -F "image=@chest_xray.png"

# Breast mammogram
curl -X POST http://localhost:5000/analyze/breast \
  -F "image=@mammogram.dcm"

# Heart ECG (image or signal file)
curl -X POST http://localhost:5000/analyze/heart \
  -F "image=@ecg_printout.jpg"
# OR
curl -X POST http://localhost:5000/analyze/heart \
  -F "signal_file=@recording.dat" \
  -F "hea_file=@recording.hea"
```

### Image Validation Debug
```bash
curl -X POST http://localhost:5000/debug/image-stats \
  -F "image=@test.jpg"
```
Returns mean, std, entropy, color/grayscale detection, edge density — useful for tuning validators.

---

## 🖼 Screenshots & Visualizations

### Frontend Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Overview with quick-access cards for all 7 diagnostic modules |
| **Symptom Diagnosis** | Searchable symptom selector (132 symptoms, categorized) with real-time matching |
| **Image Analysis** | Unified upload zone supporting 4 analysis types with drag-drop, preview, validation |
| **Heart Check** | 13-field clinical form with inline hints and validation |
| **Cancer Screening** | 10-field FNA tumor feature form with progress indicator |
| **History** | Chronological prediction log with confidence, severity, expandable details |
| **Settings** | Profile, theme (dark/light), notification preferences |

### UI Components
- **Confidence Ring** — Animated circular progress with severity color
- **Severity Badge** — Color-coded (healthy/low/moderate/high/critical)
- **Result Inspector** — Expandable detailed view for any prediction
- **Command Palette** — `Ctrl/Cmd+K` global search
- **Smooth Scroll** — Lenis-powered inertial scrolling (desktop)

---

## 📈 Performance Charts

### Disease Prediction - Top-1 Accuracy by Disease Frequency

```
Frequency Bucket     # Diseases    Avg Top-1 Acc
────────────────────────────────────────────────
≥ 1000 samples           12            91.2%
100-999 samples          45            87.8%
50-99 samples            78            83.4%
10-49 samples           182            78.1%
```

### Cancer FNA - Probability Calibration
```
Before Calibration (Old):     After Calibration (New):
┌─────────────────┐           ┌─────────────────┐
│  100% ████████  │           │  100% ████      │
│   80% ██████    │           │   80% ████████  │
│   60% ████      │   ──▶     │   60% ████████  │
│   40% ██        │           │   40% ██████    │
│   20% █         │           │   20% ████      │
│    0%           │           │    0% ██        │
└─────────────────┘           └─────────────────┘
Peak at 0% and 100%           Smooth distribution
```

### Heart Risk - Ensemble Weight Evolution
```
Old (fixed):     [RF: 2, GB: 2, LR: 1]
New (CV-based):  [RF: 2, GB: 3, LR: 2]  ← GB gets more weight (better CV AUC)
```

### Skin Cancer - Confusion Matrix (Test Set)
```
                    Predicted
            akiec  bcc  bkl  df  mel  nv  vasc
Actual akiec   74   20   21   0   1    2    0
      bcc      16   52    7   0   2    2    0
      bkl       7    5  111   2   7   15    0
      df        2    1    3   2   0    0    0
      mel       3    2   18   0  45   32    0
      nv        0    4   49   1  28  493    0
      vasc      0    0    1   0   0    2   15
```

### ECG Model - Class Distribution (PTB-XL)
```
Class            Train     Val     Test    Weight (sqrt)
───────────────────────────────────────────────────────
Normal           14,823    1,852   1,852   1.00
MI                1,234     154     154     1.28
Arrhythmia        2,187     273     273     0.88
Heart Failure      567      71      71      1.70
Hypertrophy       1,234     154     154     1.28
```

---

## 🐛 Known Issues & Fixes

| Issue | Root Cause | Fix Applied |
|-------|------------|-------------|
| **Heart model load fails** (`string indices must be integers`) | TF 2.18+ changed H5 config deserialization | `load_model(..., compile=False)` in `server.py:287` |
| **Heart Check "connection failed"** | Scaler not applied before `predict_proba` → silent 500 | Scaler applied in all inference paths (`server.py:1400+`) |
| **Symptom confidence shows N/A** | Frontend read `confidence` at wrong level | Server returns top-level `confidence`; frontend uses safe fallback |
| **Models always in demo mode** | Training scripts saved to `backend/`, server reads `ml_model/` | Training scripts now save to `../ml_model/` |
| **Image validator rejects valid clinical images** | Strict RGB/grayscale checks | v2 validator: evidence-based, lenient, per-module rules |
| **ECG signal upload fails** | No `.hea` header handling | Auto-detect format; optional `.hea` for PTB-XL compatibility |
| **Extreme probabilities (0%/100%)** | Uncalibrated ensembles on small test sets | `CalibratedClassifierCV(method='sigmoid')` on all tabular models |

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/your-feature`
3. **Commit**: `git commit -m 'Add your feature'`
4. **Push**: `git push origin feature/your-feature`
5. **Open Pull Request**

### Development Guidelines
- **Backend port**: 5000 | **Frontend port**: 5173
- **CORS**: Configured for `localhost:5173` and `localhost:3000`
- **Model artifacts**: `ml_model/` is gitignored — never commit `.joblib`/`.h5`
- **Demo mode**: Controlled by `CONFIDENCE_FLOOR` in `server.py` (default 80%)
- **Code style**: ESLint (frontend), Black/flake8 (backend) — run before PR

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

### Datasets
- **Mendeley Data** — Symptom-Disease dataset (doi:10.17632/2cxccsxydc.1)
- **UCI ML Repository** — Wisconsin Breast Cancer, Heart Disease
- **Harvard Dataverse / ISIC** — HAM10000, PAD-UFES-20
- **NIH Clinical Center** — ChestX-ray14
- **Cancer Imaging Archive** — CBIS-DDSM
- **PhysioNet** — PTB-XL ECG dataset

### Libraries & Tools
- **TensorFlow/Keras** — Deep learning framework
- **scikit-learn** — Classical ML algorithms
- **imbalanced-learn** — SMOTE oversampling
- **React, Vite, Tailwind** — Frontend stack
- **Framer Motion, Lenis, Lucide** — UX polish

---

<div align="center">

### 🌟 If you found this project useful, please consider giving it a star!

[![GitHub Stars](https://img.shields.io/github/stars/demon2202/Medidiagnose-ai?style=social)](https://github.com/demon2202/Medidiagnose-ai/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/demon2202/Medidiagnose-ai?style=social)](https://github.com/demon2202/Medidiagnose-ai/network/members)

**Built with ❤️ by [Harshit](https://github.com/demon2202)**

*MediDiagnose-AI is for educational and screening purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.*

</div>
