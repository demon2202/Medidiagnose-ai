#!/usr/bin/env python3
"""
verify_dataset.py — audit local Dataset/ health for MediDiagnose-AI

Usage:
  python ml_model/verify_dataset.py            # full audit
  python ml_model/verify_dataset.py --check skin

Checks:
  - HAM10000_metadata.csv row count (should be 10015)
  - PAD-UFES-20 metadata.csv presence
  - chest_xray/train counts
  - breast_ultrasound / BUSI counts
  - ptb-xl/ptbxl_database.csv full vs truncated (21k vs 150)
  - *.h5 / *.joblib existence
  - config accuracy sanity
"""

import os, sys, json, glob
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "Dataset"

def ok(m):  print(f"  ✅ {m}")
def warn(m): print(f"  ⚠️  {m}")
def fail(m): print(f"  ❌ {m}")

def check_skin():
    print("\n" + "="*66)
    print("  SKIN — HAM10000 + PAD-UFES-20")
    print("="*66)
    ham_meta = DATASET_DIR / "HAM10000" / "HAM10000_metadata.csv"
    if ham_meta.exists():
        import pandas as pd
        try:
            df = pd.read_csv(ham_meta)
            n = len(df)
            if n >= 10000: ok(f"HAM10000 metadata found: {n} rows (expected 10015)")
            elif n >= 9000: warn(f"HAM10000 metadata: {n} rows (slightly low, expected 10015)")
            else: fail(f"HAM10000 metadata truncated: {n} rows (expected 10015) — re-download")
            # class distribution
            if 'dx' in df.columns:
                print(f"     dx dist: {dict(Counter(df['dx']))}")
            # image dirs
            for folder in ['HAM10000_images_part_1','HAM10000_images_part_2','HAM10000_images','images']:
                p = DATASET_DIR / "HAM10000" / folder
                if p.exists():
                    cnt = len(list(p.glob("*.jpg")))
                    if cnt>0: ok(f"HAM images {folder}: {cnt} files")
        except Exception as e:
            fail(f"Could not read HAM metadata: {e}")
    else:
        fail(f"HAM10000 not found: {ham_meta}")
        print("     Download: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"     Extract to: {DATASET_DIR / 'HAM10000'}")

    pad_meta = DATASET_DIR / "Skin_Cancer" / "metadata.csv"
    pad_alt = DATASET_DIR / "PAD-UFES-20" / "metadata.csv"
    pad_found = None
    for p in [pad_meta, pad_alt] + list(DATASET_DIR.glob("PAD*/metadata.csv")) + list(DATASET_DIR.glob("Skin_Cancer*/metadata.csv")):
        if p.exists():
            pad_found = p; break
    if pad_found:
        import pandas as pd
        try:
            df = pd.read_csv(pad_found)
            n = len(df)
            ok(f"PAD-UFES-20 found: {pad_found} ({n} rows)")
            if 'diagnostic' in df.columns:
                print(f"     diagnostic dist: {dict(Counter(df['diagnostic']))}")
        except Exception as e:
            warn(f"PAD metadata found but unreadable: {e}")
    else:
        warn("PAD-UFES-20 NOT found — model will be dermoscopy-only (phone photos will fail)")
        print("     Download: https://www.kaggle.com/datasets/richardgoulter/pad-ufes-20")
        print(f"     Extract to: {DATASET_DIR / 'Skin_Cancer'}  (must contain metadata.csv)")
        print("     Without PAD, skin model gives 65% melanoma recall on phone uploads and leaks as 'pneumonia' on xray validator.")

    # model file
    cfg = SCRIPT_DIR / "skin_cancer_config.json"
    if cfg.exists():
        with open(cfg) as f: c=json.load(f)
        acc=c.get("accuracy",0)
        if acc>0.75: ok(f"skin_cancer_config.json accuracy {acc:.3f}")
        else: warn(f"skin_cancer accuracy low {acc:.3f} (expected >0.75 after retrain)")
    if not (SCRIPT_DIR / "skin_cancer_model.h5").exists():
        warn("skin_cancer_model.h5 MISSING — run: python ml_model/image_classification.py 1")

def check_chest_xray():
    print("\n" + "="*66)
    print("  CHEST X-RAY — pneumonia")
    print("="*66)
    xray_dir = DATASET_DIR / "chest_xray"
    if not xray_dir.exists():
        fail(f"chest_xray not found: {xray_dir}")
        print("     Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        return
    train_n = len(list((xray_dir/"train"/"NORMAL").glob("*"))) if (xray_dir/"train"/"NORMAL").exists() else 0
    train_p = len(list((xray_dir/"train"/"PNEUMONIA").glob("*"))) if (xray_dir/"train"/"PNEUMONIA").exists() else 0
    test_n = len(list((xray_dir/"test"/"NORMAL").glob("*"))) if (xray_dir/"test"/"NORMAL").exists() else 0
    test_p = len(list((xray_dir/"test"/"PNEUMONIA").glob("*"))) if (xray_dir/"test"/"PNEUMONIA").exists() else 0
    total = train_n+train_p+test_n+test_p
    if total==0:
        fail("chest_xray folder exists but no images found (check NORMAL/PNEUMONIA subfolders)")
    else:
        ok(f"chest_xray found: train NORMAL {train_n} / PNEUMONIA {train_p}, test NORMAL {test_n} / PNEUMONIA {test_p} (total {total})")
        if train_p / max(1, train_n+train_p) > 0.70:
            warn(f"Imbalanced {train_p/(train_n+train_p):.0%} pneumonia — model will bias to pneumonia (TN low). Retrain handles it, but consider focal loss.")
    cfg = SCRIPT_DIR / "pneumonia_config.json"
    if cfg.exists():
        with open(cfg) as f: c=json.load(f)
        acc=c.get("accuracy",0)
        cm=c.get("confusion_matrix",{})
        if isinstance(cm, dict):
            tn, fp, fn, tp = cm.get("TN",0), cm.get("FP",0), cm.get("FN",0), cm.get("TP",0)
            spec = tn / max(1, tn+fp)
            sens = tp / max(1, tp+fn)
            print(f"     config accuracy {acc:.3f}  sens {sens:.3f}  spec {spec:.3f}")
            if spec<0.70: warn(f"Specificity {spec:.1%} is low → any grayscale looks like pneumonia (your bug). Needs retrain without output_bias.")
            else: ok(f"Specificity OK {spec:.1%}")

def check_breast():
    print("\n" + "="*66)
    print("  BREAST — ultrasound BUSI")
    print("="*66)
    found=None
    for name in ['breast_ultrasound','Breast_Ultrasound','Dataset_BUSI_with_GT','BUSI','busi']:
        p=DATASET_DIR/name
        if p.exists() and any((p/d).exists() for d in p.iterdir() if d.is_dir()):
            found=p; break
    if not found:
        # search any subdir with benign/malignant
        for p in DATASET_DIR.iterdir():
            if p.is_dir():
                subs=[d.name.lower() for d in p.iterdir() if d.is_dir()]
                if any('benign' in s for s in subs) or any('malignant' in s for s in subs):
                    found=p; break
    if found:
        import re
        subs={d.name: len(list(d.glob("*.png")))+len(list(d.glob("*.jpg"))) for d in found.iterdir() if d.is_dir()}
        # exclude masks
        for k in list(subs.keys()):
            if 'mask' in k.lower(): subs.pop(k)
        ok(f"BUSI found at {found}: {subs} (masks excluded)")
        total=sum(subs.values())
        if total<500: fail(f"Only {total} images — too small for MobileNetV2 (needs 2000+). Keep BUSI but plan CBIS-DDSM.")
        elif total<1000: warn(f"{total} images — small, will overfit. Augmentation helps but expand to BUS-BRA/CBIS-DDSM for prod.")
        else: ok(f"{total} images — sufficient")
    else:
        fail("Breast ultrasound not found")
        print("     Download: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print(f"     Extract to: {DATASET_DIR/'breast_ultrasound'} (must have benign/malignant/normal subfolders)")

    cfg=SCRIPT_DIR/"breast_cancer_config.json"
    if cfg.exists():
        with open(cfg) as f: c=json.load(f)
        acc=c.get("accuracy",0)
        cm=c.get("confusion_matrix",[])
        # detect empty classes
        if isinstance(cm, list) and len(cm)>=6:
            zero_rows=[i for i,row in enumerate(cm) if sum(row)==0]
            if zero_rows: warn(f"Config 6-class but rows {zero_rows} have 0 samples → fake 6-class (actually 3-class). Frontend BI-RADS 3/4/5 never predicted.")
        print(f"     config accuracy {acc:.3f}")

def check_ecg():
    print("\n" + "="*66)
    print("  HEART ECG — PTB-XL")
    print("="*66)
    ptb_dir=DATASET_DIR/"ptb-xl"
    meta=ptb_dir/"ptbxl_database.csv"
    if not meta.exists():
        fail(f"PTB-XL metadata not found: {meta}")
        print("     Download: https://physionet.org/content/ptb-xl/1.0.3/")
        print(f"     Extract to: {ptb_dir} (must contain ptbxl_database.csv + records100/)")
        return
    import pandas as pd
    try:
        df=pd.read_csv(meta, index_col='ecg_id')
        n=len(df)
        if n>=20000: ok(f"PTB-XL full metadata: {n} records (expected 21799)")
        elif n>=15000: warn(f"PTB-XL metadata {n} rows — seems partial, expected 21799")
        elif n<=500: fail(f"PTB-XL metadata TRUNCATED: {n} rows (expected 21799) — THIS IS YOUR BUG (train on 120 images → 20% acc)")
        else: warn(f"PTB-XL metadata {n} rows (expected 21799)")
        # check fold column
        if 'strat_fold' in df.columns:
            print(f"     strat_fold dist: {dict(Counter(df['strat_fold']))}")
        # records
        rec100=ptb_dir/"records100"
        if rec100.exists():
            cnt=len(list(rec100.rglob("*.dat")))
            ok(f"records100: {cnt} .dat files")
            if cnt<15000: warn(f"Only {cnt} .dat — expected ~21837, check extraction")
        else:
            warn(f"records100 not found: {rec100} — need records100/ for 100Hz training")
    except Exception as e:
        fail(f"Could not read PTB-XL metadata: {e}")
    cfg=SCRIPT_DIR/"heart_image_config.json"
    if cfg.exists():
        with open(cfg) as f: c=json.load(f)
        acc=c.get("accuracy",0)
        if acc<0.50: fail(f"heart_image_config accuracy {acc:.3f} → truncated dataset bug (20% = always Arrhythmia)")
        elif acc<0.75: ok(f"heart_image accuracy {acc:.3f} (good, but can push to 0.80+)")
        else: ok(f"heart_image accuracy {acc:.3f}")

def check_tabular():
    print("\n" + "="*66)
    print("  TABULAR — disease / cancer / heart")
    print("="*66)
    for name, path in [("disease dataset.csv", DATASET_DIR/"dataset.csv"),
                       ("cancer cancer.csv", DATASET_DIR/"cancer.csv"),
                       ("heart heart.csv", DATASET_DIR/"heart.csv")]:
        if path.exists():
            try:
                import pandas as pd
                df=pd.read_csv(path)
                ok(f"{name}: {len(df)} rows, {len(df.columns)} cols")
            except Exception as e:
                warn(f"{name} exists but unreadable: {e}")
        else:
            warn(f"{name} not found at {path} — will fallback to synthetic/sklearn (ok for demo, but real CSV better)")

    for name in ["disease_model.joblib","cancer_model.joblib","heart_disease_model.joblib","cancer_scaler.joblib","heart_scaler.joblib","label_encoder.joblib"]:
        if (SCRIPT_DIR/name).exists():
            ok(f"{name} present")
        else:
            warn(f"{name} MISSING — run training for that modality")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--check", choices=["skin","xray","breast","ecg","tabular","all"], default="all")
    args=ap.parse_args()
    print("="*66)
    print("  MediDiagnose Dataset Audit")
    print(f"  Dataset root: {DATASET_DIR.resolve()}")
    print(f"  Mode: {args.check}")
    print("="*66)
    if not DATASET_DIR.exists():
        fail(f"Dataset dir not found: {DATASET_DIR} — create it and download datasets")
        sys.exit(1)
    checks={"skin": check_skin, "xray": check_chest_xray, "breast": check_breast, "ecg": check_ecg, "tabular": check_tabular}
    if args.check=="all":
        for fn in [check_skin, check_chest_xray, check_breast, check_ecg, check_tabular]:
            fn()
    else:
        checks[args.check]()
    print("\n" + "="*66)
    print("  Audit complete. Fix any ❌ before retraining.")
    print("="*66 + "\n")
