import os
import re
import json
import time
import numpy as np
import pandas as pd
import joblib
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
CSV_PATH = os.path.join(DATASET_DIR, 'disease_symptoms_2023.csv')

SYMPTOM_LIST_PATH = os.path.join(SCRIPT_DIR, 'symptom_list.json')
LABEL_ENCODER_PATH = os.path.join(SCRIPT_DIR, 'label_encoder.joblib')
DISEASE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'disease_model.joblib')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'model_config.json')

MIN_SAMPLES = 10          # drop ultra-rare diseases with fewer than this many rows
TEST_SIZE = 0.2
SEED = 42

SMALL_WORDS = {'of', 'the', 'and', 'in', 'or', 'a', 'an', 'to', 'with', 'on', 'for'}


def to_symptom_id(name):
    return re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')


def title_case(name):
    words = str(name).split()
    out = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return ' '.join(out)


def main():
    t0 = time.time()
    header = pd.read_csv(CSV_PATH, nrows=0)
    dtype = {c: 'int8' for c in header.columns[1:]}
    dtype[header.columns[0]] = 'object'
    df = pd.read_csv(CSV_PATH, dtype=dtype)
    print(f"Loaded {df.shape} in {time.time()-t0:.1f}s")

    label_col = df.columns[0]
    symptom_cols = list(df.columns[1:])

    # drop ultra-rare diseases
    vc = df[label_col].value_counts()
    keep = vc[vc >= MIN_SAMPLES].index
    df = df[df[label_col].isin(keep)].copy()
    print(f"Kept {len(keep)} diseases with >= {MIN_SAMPLES} samples "
          f"(dropped {len(vc) - len(keep)} rare)")

    X = df[symptom_cols].values.astype(np.float32)
    y_raw = df[label_col].values

    # canonical symptom ids (CSV order) -- this becomes symptom_list.json
    symptom_ids = [to_symptom_id(c) for c in symptom_cols]
    assert len(set(symptom_ids)) == len(symptom_ids), "symptom id collision!"

    # disease names (title-cased) in label order
    classes = sorted(set(y_raw))
    disease_names = [title_case(c) for c in classes]
    y = np.array([classes.index(v) for v in y_raw], dtype=np.int64)

    # stratified split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"Train {X_tr.shape}  Test {X_te.shape}")

    # train
    from sklearn.linear_model import SGDClassifier
    t1 = time.time()
    model = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=1e-4, max_iter=20,
        tol=1e-3, random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    print(f"Trained in {time.time()-t1:.1f}s")

    # evaluate top-k accuracy
    proba = model.predict_proba(X_te)          # OVR, not normalized
    proba_n = proba / proba.sum(axis=1, keepdims=True)   # softmax-normalize
    order = np.argsort(proba_n, axis=1)[:, ::-1]
    for k in (1, 3, 5, 10):
        hit = np.mean([y_te[i] in order[i, :k] for i in range(len(y_te))])
        print(f"  top-{k} accuracy: {hit:.4f}")

    # save
    with open(SYMPTOM_LIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(symptom_ids, f, indent=2)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(classes, dtype=object)
    joblib.dump(le, LABEL_ENCODER_PATH)

    joblib.dump(model, DISEASE_MODEL_PATH)

    cfg = {
        'model_type': 'SGDClassifier(log_loss) multinomial-via-SGD',
        'n_features_in': len(symptom_ids),
        'n_symptoms': len(symptom_ids),
        'n_diseases': len(classes),
        'diseases': disease_names,
        'min_samples_per_disease': MIN_SAMPLES,
        'test_size': TEST_SIZE,
        'train_samples': int(len(X_tr)),
        'test_samples': int(len(X_te)),
        'top1_accuracy': round(float(np.mean(np.argmax(proba_n, axis=1) == y_te)), 4),
        'top5_accuracy': round(float(np.mean([y_te[i] in order[i, :5] for i in range(len(y_te))])), 4),
        'source': 'Disease and symptoms dataset 2023 (Mendeley, doi:10.17632/2cxccsxydc.1)',
        'compatible_with': 'server.py /predict-disease + /symptoms/match',
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved: {os.path.basename(SYMPTOM_LIST_PATH)} ({len(symptom_ids)} symptoms), "
          f"{os.path.basename(LABEL_ENCODER_PATH)} ({len(classes)} diseases), "
          f"{os.path.basename(DISEASE_MODEL_PATH)}, {os.path.basename(CONFIG_PATH)}")
    print(f"Total time {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
