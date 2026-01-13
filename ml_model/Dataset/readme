🏥 MediDiagnose-AI
MediDiagnose-AI is a comprehensive medical diagnostic assistant powered by Artificial Intelligence. It combines Computer Vision (for image analysis) and Machine Learning (for tabular data) to assess risk levels for various conditions including Skin Cancer, Pneumonia, Breast Cancer, and Heart Disease.

🌟 Key Features
Multi-Modal Analysis:
Skin Cancer: Analyzes dermoscopy images (7 types of lesions).
Pneumonia: Detects pneumonia in chest X-Rays.
Breast Cancer: Analyzes Mammograms/Ultrasounds (BI-RADS grading).
Heart Condition: Analyzes 12-lead ECG plot images.
Tabular Risk Prediction:
Symptom Checker: Predicts disease based on natural language symptoms.
Heart Risk: Calculates risk using clinical data (Age, BP, Cholesterol, etc.).
Breast Cancer Risk: Analyzes tumor features (Radius, Texture, etc.).
Smart Validation: Automatically detects if the uploaded image matches the selected category (e.g., rejects a color skin photo if uploaded for X-Ray analysis).
Detailed Reports: Provides severity levels (Healthy to Critical), staging info, treatment options, and urgency timelines.
🛠️ Tech Stack
Frontend: React.js, Tailwind CSS, Lucide Icons, Vite.
Backend: Python, Flask, Flask-CORS.
Machine Learning: TensorFlow/Keras (CNNs), Scikit-Learn (Random Forest, Voting Classifiers).
Data Processing: Pandas, NumPy, WFDB (for ECG data), PIL (Image processing).
📂 Project Structure
Bash

medidiagnose-ai/
├── backend/
│   ├── server.py                # Main Flask API entry point
│   └── uploads/                 # Temp storage for uploaded images
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── ImageAnalysis.jsx  # Image upload & results UI
│   │   │   ├── History.jsx        # User history & export
│   │   │   └── ...
│   │   └── context/             # State management
│   └── package.json
├── ml_model/
│   ├── Dataset/                 # RAW DATASETS GO HERE
│   ├── models/                  # Trained .h5 and .joblib models
│   ├── train_heart_model.py     # Tabular Heart Risk Training
│   ├── train_heart_image_model.py # ECG Image Training
│   ├── train_breast_cancer.py   # Breast Cancer Image Training
│   └── image_classification.py  # Skin & Pneumonia Training
└── README.md
📥 Step 1: Get the Datasets
To train the models, you need to download specific datasets and place them in the ml_model/Dataset/ folder exactly as shown below.

1. Skin Cancer (HAM10000)
Source: Kaggle - Skin Cancer MNIST: HAM10000
Action: Download and extract.
Required Path:
text

ml_model/Dataset/HAM10000/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv
2. Pneumonia (Chest X-Ray Images)
Source: Kaggle - Chest X-Ray Images (Pneumonia)
Action: Download and extract.
Required Path:
text

ml_model/Dataset/chest_xray/
├── train/ (contains NORMAL and PNEUMONIA folders)
├── test/
└── val/
3. Breast Cancer (Ultrasound)
Source: Kaggle - Breast Ultrasound Images Dataset
Action: Download and extract.
Required Path:
text

ml_model/Dataset/breast_ultrasound/
├── benign/
├── malignant/
└── normal/
4. Heart ECG (PTB-XL)
Source: PhysioNet - PTB-XL Dataset
Action: Download and extract. Note: This requires the wfdb library.
Required Path:
text

ml_model/Dataset/ptb-xl/
├── records100/      (contains folders like 00000/, 01000/)
├── ptbxl_database.csv
└── scp_statements.csv
5. Tabular Data (Heart & Disease)
Action: These scripts will auto-generate synthetic data if files are missing, or you can download:
Heart Risk: Cleveland Heart Disease Dataset -> Save as Dataset/heart.csv
Disease Symptoms: Disease Symptom Prediction -> Save as Dataset/dataset.csv
⚙️ Step 2: Installation
Backend Prerequisites
Install Python 3.8+ and the required libraries:

Bash

cd medidiagnose-ai
pip install flask flask-cors tensorflow pillow scikit-learn pandas numpy wfdb matplotlib joblib
Frontend Prerequisites
Install Node.js and dependencies:

Bash

cd frontend
npm install
🧠 Step 3: Model Training
You must train the models before running the server. Run these commands from the root directory:

Bash

cd ml_model

# 1. Train Skin Cancer & Pneumonia Models
python image_classification.py
# (Select Option 3 to train both)

# 2. Train Breast Cancer Model
python train_breast_cancer_model.py
# (Select Option 1 for 3-class model)

# 3. Train Heart ECG Image Model
python train_heart_image_model.py

# 4. Train Heart Disease Risk (Tabular) Model
python train_heart_model.py

# 5. Train Disease Symptom Model
python train_disease_model.py
Code Spotlight: Feature Engineering (Heart Model)
The heart risk model uses 22 engineered features derived from 13 original clinical inputs to improve accuracy.

Python

# snippet from train_heart_model.py
def engineer_features(X):
    # Example: Calculate Heart Rate Reserve
    max_hr = 220 - X['age']
    X['hr_reserve_pct'] = X['thalach'] / max_hr 
    
    # Example: Interaction between Age and Blood Pressure
    X['bp_age_interaction'] = X['trestbps'] * X['age'] / 100
    
    return X
Why? Raw data isn't always enough. By combining features (like Age and Max Heart Rate), the Random Forest model can detect subtle risk patterns that linear models might miss.

🚀 Step 4: Running the Application
1. Start the Backend Server
Bash

cd backend
python server.py
Expected Output: 🚀 Server starting on http://localhost:5000

2. Start the Frontend
In a new terminal:

Bash

cd frontend
npm run dev
Expected Output: Local: http://localhost:5173/

🔍 Code Explanation & Architecture
1. Image Validation Logic (server.py)
To prevent users from uploading a skin photo to the X-ray analyzer, we implement basic computer vision heuristics.

Python

# snippet from server.py
def validate_image_type(img_array, expected_type):
    # Calculate color variance (R vs G vs B)
    rgb_diff = np.mean(np.abs(r - g) + np.abs(g - b))
    is_grayscale = rgb_diff < 0.05
    
    if expected_type == 'xray' and not is_grayscale:
        return {'is_valid': False, 'message': 'Please upload a grayscale X-Ray.'}
    
    return {'is_valid': True}
Explanation: X-rays and ECGs are typically grayscale (black & white). Skin photos are colorful. By checking pixel color variance, we can instantly flag incorrect uploads before the AI even tries to analyze them.

2. The Heart ECG Model (train_heart_image_model.py)
This is unique because it takes raw signal data (.dat files) and converts them into images for a CNN to read.

Python

# snippet from train_heart_image_model.py
def signal_to_image(signal):
    # Plot the 1D signal data onto a 2D matplotlib chart
    plt.plot(signal)
    # Save the chart as a PNG image in memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    # Use this image for training the CNN
    return image_array
Explanation: Convolutional Neural Networks (CNNs) are excellent at finding patterns in images. By converting electrical heart signals into visual plots, we leverage the power of computer vision to detect heart attacks and arrhythmias visually, just like a cardiologist reads a printout.

3. Frontend Integration (ImageAnalysis.jsx)
The frontend handles the user experience, file validation, and displaying complex results.

React

// snippet from ImageAnalysis.jsx
const handleAnalyze = async () => {
  // 1. Send file to backend
  const response = await axios.post(endpoint, formData);
  
  // 2. Handle specific validation errors
  if (response.data.validation_error) {
    setValidationError({
      message: response.data.message, // e.g., "Wrong image type"
      suggestion: response.data.suggestion
    });
  } else {
    // 3. Show results (Severity, Confidence, Recommendations)
    setResult(response.data);
  }
}
⚠️ Important Disclaimers
Medical Disclaimer: This tool is for educational and screening purposes only. It is not a substitute for professional medical diagnosis.
Data Privacy: Images uploaded are processed locally (in backend/uploads) and are not stored permanently in a cloud database in this version.
Demo Mode: If models are not trained/found, the server automatically switches to "Demo Mode," returning simulated results so the UI can still be tested.
🐛 Troubleshooting
Error: ValueError: X has 13 features, but StandardScaler is expecting 22 features.
Fix: Ensure you are using the updated server.py which includes the engineer_heart_features function.
Error: wfdb not installed
Fix: Run pip install wfdb.
Frontend Error: Objects are not valid as a React child
Fix: Ensure History.jsx handles the prediction object correctly using item.prediction?.name.
