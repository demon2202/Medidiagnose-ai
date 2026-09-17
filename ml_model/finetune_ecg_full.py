"""One-shot: continue ECG training from the v1 checkpoint with the ENTIRE
backbone unfrozen (BN frozen) and a cosine LR schedule — the strongest
fine-tuning recipe without re-running phase 1 from scratch."""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, ML_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from medidiagnose import train_utils as TU
from train_heart_image_model import (load_ptbxl_dataset, CLASS_NAMES,
                                     HEART_IMAGE_MODEL_PATH, HEART_CONFIG_PATH,
                                     IMG_SIZE)

import json
import tensorflow as tf
from tensorflow import keras


def main():
    tf.random.set_seed(TU.SEED)
    np.random.seed(TU.SEED)

    print('Loading data (renders once, then cached)...')
    X_train, y_train, X_val, y_val, X_test, y_test, class_weight = \
        load_ptbxl_dataset(sampling_rate=100, norm_train_cap=3000, img_size=IMG_SIZE)

    model = TU.load_model_for_finetune(HEART_IMAGE_MODEL_PATH)
    print('  Model loaded; all non-BatchNorm layers unfrozen.')
    model.summary()

    train_ds = TU.make_dataset(X_train, y_train, 32, training=True)
    val_ds = TU.make_dataset(X_val, y_val, 32)
    steps = int(np.ceil(len(X_train) / 32)) * 25
    opt = keras.optimizers.Adam(
        keras.optimizers.schedules.CosineDecay(3e-5, steps, alpha=1/30))
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, class_weight=class_weight,
              epochs=25, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(
                             monitor='val_accuracy', mode='max', patience=8,
                             restore_best_weights=True, verbose=1),
                         keras.callbacks.ModelCheckpoint(
                             HEART_IMAGE_MODEL_PATH, monitor='val_accuracy',
                             mode='max', save_best_only=True, verbose=0)])

    metrics = TU.evaluate_model(model, X_test, y_test, CLASS_NAMES, tag='ecg-ft')
    model.save(HEART_IMAGE_MODEL_PATH)

    cfg_path = HEART_CONFIG_PATH
    config = {
        'model_path': HEART_IMAGE_MODEL_PATH,
        'model_type': 'image',
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'preprocessing': 'ECG rendered dark-background via '
                         'medidiagnose.inference_utils.signal_to_ecg_image; '
                         'photo uploads auto-inverted at serve time',
        'note': 'Model internally replicates 1ch to 3ch for MobileNetV2',
        'num_classes': 5,
        'classes': {str(k): v for k, v in {0: {'code': 'normal', 'name': 'Normal', 'severity': 'healthy'},
                    1: {'code': 'mi', 'name': 'Myocardial Infarction', 'severity': 'critical'},
                    2: {'code': 'arrhythmia', 'name': 'Arrhythmia', 'severity': 'moderate'},
                    3: {'code': 'hf', 'name': 'Heart Failure Signs', 'severity': 'high'},
                    4: {'code': 'hypertrophy', 'name': 'Ventricular Hypertrophy', 'severity': 'moderate'}}.items()},
        'class_names': CLASS_NAMES,
        'architecture': 'MobileNetV2_transfer_learning_v3_full_finetune',
        'training_notes': ('Full PTB-XL (records100), patient-disjoint strat_fold '
                           'split 1-8/9/10, full-backbone fine-tune, cosine LR'),
        'accuracy': metrics['accuracy'],
        'using_real_data': True,
        'confusion_matrix': metrics['confusion_matrix']
    }
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)
    print('[OK] Model + config saved. Restart server.py to load.')


if __name__ == '__main__':
    main()
