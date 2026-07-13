import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models and their expected output files for validation
MODELS_TO_TRAIN = [
    {
        "name": "Image Validator",
        "script": "image_validator.py",
        "args": [],
        "outputs": ["image_validator_model.h5"]
    },
    {
        "name": "Symptom-based Disease Predictor",
        "script": "disease_prediction_v2.py",
        "args": [],
        "outputs": ["disease_model.joblib", "label_encoder.joblib", "symptom_list.json"]
    },
    {
        "name": "Skin Cancer & Pneumonia CNNs",
        "script": "image_classification.py",
        "args": ["3"],  # Choice 3 trains both skin cancer and pneumonia
        "outputs": ["skin_cancer_model.h5", "skin_cancer_config.json", "pneumonia_model.h5", "pneumonia_config.json"]
    },
    {
        "name": "Breast Cancer Ultrasound CNN",
        "script": "train_breast_cancer_model.py",
        "args": ["1"],  # Choice 1 trains 3-class model with transfer learning
        "outputs": ["breast_cancer_model.h5", "breast_cancer_config.json"]
    },
    {
        "name": "Tabular Cancer Risk Predictor",
        "script": "train_cancer_model.py",
        "args": [],
        "outputs": ["cancer_model.joblib", "cancer_scaler.joblib"]
    },
    {
        "name": "Heart Image/ECG CNN",
        "script": "train_heart_image_model.py",
        "args": [],
        "outputs": ["heart_image_model.h5", "heart_image_config.json"]
    },
    {
        "name": "Tabular Heart Risk Predictor",
        "script": "train_heart_model.py",
        "args": [],
        "outputs": ["heart_disease_model.joblib", "heart_scaler.joblib"]
    }
]

def run_script(script_name, args=None):
    """Run a training script"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if os.path.exists(script_path):
        print(f"\n{'='*70}")
        print(f"Running: {script_name} {' '.join(args) if args else ''}")
        print('='*70)
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[OK] Successfully executed {script_name}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error executing {script_name}: {e}")
    else:
        print(f"[WARNING] Script not found at {script_path}")

def main():
    print("=" * 70)
    print("MediDiagnose-AI: Master Training Pipeline")
    print("=" * 70)
    print("\nThis script will train all backend ML models sequentially.")
    print("Depending on hardware, this may take some time.\n")
    
    # Run all defined training scripts
    for model_task in MODELS_TO_TRAIN:
        print(f"\nTask: {model_task['name']}")
        run_script(model_task["script"], model_task["args"])
        
    print("\n" + "=" * 70)
    print("All Model Training Scripts Executed!")
    print("=" * 70)
    
    # Validate generated files
    print("\nModel Verification Summary:")
    print("-" * 50)
    
    all_ok = True
    for model_task in MODELS_TO_TRAIN:
        print(f"\n* {model_task['name']}:")
        for output_file in model_task["outputs"]:
            file_path = os.path.join(SCRIPT_DIR, output_file)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  [OK] {output_file:<28} ({size_mb:.2f} MB)")
            else:
                print(f"  [MISSING] {output_file:<28}")
                all_ok = False
                
    print("\n" + "-" * 50)
    if all_ok:
        print("Success: All models trained and verified successfully!")
    else:
        print("Warning: Some model files are missing. Please check execution logs.")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()