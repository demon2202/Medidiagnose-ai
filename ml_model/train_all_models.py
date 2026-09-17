"""Run every MediDiagnose training script in sequence.

Usage:  python ml_model/train_all_models.py [--skip-cnn]
"""

import os
import sys
import subprocess
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

JOBS = [
    ('disease', 'disease_prediction_v2.py', 'Symptom -> disease'),
    ('cancer', 'train_cancer_model.py', 'Breast cancer FNA'),
    ('heart', 'train_heart_model.py', 'Heart disease risk'),
    ('skin+xray', 'image_classification.py', 'Skin cancer + Pneumonia CNNs'),
    ('breast', 'train_breast_cancer_model.py', 'Breast ultrasound CNN'),
    ('ecg', 'train_heart_image_model.py', 'Heart ECG CNN'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-cnn', action='store_true',
                    help='Skip the 4 image models (slow on CPU)')
    ap.add_argument('--only', choices=[j[0] for j in JOBS],
                    help='Run a single job')
    args = ap.parse_args()

    jobs = JOBS if not args.skip_cnn else [j for j in JOBS if 'CNN' not in j[2]]
    if args.only:
        jobs = [j for j in JOBS if j[0] == args.only]

    results = []
    for name, script, desc in jobs:
        path = os.path.join(SCRIPT_DIR, script)
        if not os.path.exists(path):
            results.append((name, 'MISSING', 0.0))
            continue
        print(f"\n{'#' * 66}\n# RUNNING: {desc} ({script})\n{'#' * 66}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR)
        dt = time.time() - t0
        results.append((name, 'OK' if proc.returncode == 0 else 'FAILED', dt))
        if proc.returncode != 0:
            print(f"\n!!! {script} FAILED (exit {proc.returncode}) — continuing")

    print(f"\n{'=' * 66}\nTRAINING SUMMARY\n{'=' * 66}")
    for name, status, dt in results:
        print(f"  {name:<12} {status:<8} {dt / 60:6.1f} min")
    failed = [r for r in results if r[1] != 'OK']
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
