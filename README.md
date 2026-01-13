# 🏥 MediDiagnose-AI

<div align="center">

![MediDiagnose-AI Logo](https://img.shields.io/badge/MediDiagnose-AI-blue?style=for-the-badge\&logo=medical)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge\&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge\&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 2️⃣ Problem Statement

Early diagnosis of diseases such as cancer, heart disorders, and pneumonia is critical but often delayed due to:

* Shortage of medical experts
* Time‑consuming manual diagnosis
* High dependency on human interpretation

This project aims to **automate disease prediction** using AI models to provide **fast, consistent, and confidence‑based results**.

---

## 3️⃣ Objectives

* To build an AI system capable of diagnosing multiple diseases
* To use Deep Learning for image‑based medical analysis
* To provide confidence scores and severity levels
* To expose predictions through a REST API
* To ensure scalability and modular design

---

## 4️⃣ Scope of the Project

The system supports diagnosis for:

* Skin Cancer
* Heart Disease
* Breast Cancer
* Pneumonia
* General disease prediction based on symptoms

The project is intended for **educational, research, and prototype medical systems**.

---

## 5️⃣ Technologies Used

| Category             | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| Backend              | Flask                       |
| Deep Learning        | TensorFlow, Keras           |
| ML Utilities         | NumPy, Pandas, Scikit‑Learn |
| Model Storage        | H5, Joblib                  |
| Frontend (Optional)  | React                       |
| API Testing          | Postman                     |

---

## 6️⃣ System Architecture

**User → Frontend → Flask API → AI Models → Prediction Response**

Steps:

1. User uploads medical image or inputs symptoms
2. Request reaches Flask backend
3. Data is preprocessed
4. Trained model generates prediction
5. Confidence & recommendations returned

---

## 7️⃣ Modules Explanation

### 🔹 Skin Cancer Detection

* Dataset: HAM10000
* Classes: 7 skin lesion types
* Model: CNN with transfer learning
* Output: Cancer type + confidence

### 🔹 Heart Disease Detection

* Image‑based ECG analysis
* Binary classification (Normal / Abnormal)
* Used for risk screening

### 🔹 Breast Cancer Detection

* Ultrasound image analysis
* BI‑RADS classification
* Indicates severity level

### 🔹 Pneumonia Detection

* Chest X‑ray image analysis
* CNN‑based binary classification

### 🔹 Symptom‑Based Prediction

* User selects symptoms
* ML model predicts probable disease
* Used for preliminary screening

---

## 8️⃣ Dataset Description

| Dataset           | Description         |
| ----------------- | ------------------- |
| HAM10000          | Skin lesion images  |
| Chest X‑Ray       | Pneumonia detection |
| Breast Ultrasound | Breast cancer       |
| PTB‑XL            | ECG heart dataset   |

All datasets are sourced from **Kaggle**.

---

## 9️⃣ Model Training Process

1. Data collection
2. Data cleaning & augmentation
3. Train‑test split
4. Model training
5. Performance evaluation
6. Model saving (.h5 / .joblib)

---

## 🔟 Project Directory Structure

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
├── frontend/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 1️⃣1️⃣ API Endpoints

| Method | Endpoint         | Purpose                |
| ------ | ---------------- | ---------------------- |
| GET    | /health          | Server status          |
| POST   | /analyze/skin    | Skin cancer prediction |
| POST   | /analyze/heart   | Heart disease          |
| POST   | /analyze/breast  | Breast cancer          |
| POST   | /analyze/xray    | Pneumonia              |
| POST   | /predict-disease | Symptom based          |
| POST   | /predict-heart   | Heart risk score       |

---

## 1️⃣2️⃣ Output Format

Each API returns:

* Predicted disease
* Confidence score
* Severity level
* Medical recommendation

---

## 1️⃣3️⃣ Advantages

* Fast diagnosis
* Reduces human error
* Scalable multi‑disease system
* Can be integrated with hospital systems

---

## 1️⃣4️⃣ Limitations

* Not a replacement for doctors
* Depends on dataset quality
* Requires good quality images

---

## 1️⃣5️⃣ Future Enhancements

* Real‑time hospital integration
* Mobile application
* More disease models
* Explainable AI (XAI)

---

## 1️⃣6️⃣ Ethical Considerations

* Patient data privacy
* Responsible AI usage
* Medical disclaimer compliance

---

## 1️⃣7️⃣ Conclusion

MediDiagnose‑AI demonstrates how AI can assist healthcare professionals by providing **early, reliable, and scalable diagnostic support**.

---

## ⚠️ Disclaimer

This system is developed **only for academic and research purposes** and should not be used as a sole medical diagnosis tool.

---

## 👨‍💻 Developed By

**Harshit S Negi**

---

## 📜 License

MIT License
