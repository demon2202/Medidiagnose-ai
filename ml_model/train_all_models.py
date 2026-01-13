"""
MediDiagnose-AI: Master Training Script
Trains all ML models for the application
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    """Run a training script"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if os.path.exists(script_path):
        print(f"\n{'='*60}")
        print(f"Running: {script_name}")
        print('='*60)
        subprocess.run([sys.executable, script_path])
    else:
        print(f"Warning: {script_path} not found")

def main():
    print("=" * 60)
    print("MediDiagnose-AI: Training All Models")
    print("=" * 60)
    
    
    run_script('image_validator.py')
    run_script('disease_prediction_v2.py')
    run_script('image_classification.py')
    
    run_script('train_breast_cancer_model.py')
    run_script('train_cancer_model.py')
    
    
    
    run_script('train_heart_image_model.py')
    
    run_script('train_heart_model.py')
    
    
    
    print("\n" + "=" * 60)
    print("All Models Training Complete!")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated model files:")
    for f in os.listdir(SCRIPT_DIR):
        if f.endswith(('.joblib', '.h5', '.json')):
            print(f"  ✓ {f}")

if __name__ == '__main__':
    main()