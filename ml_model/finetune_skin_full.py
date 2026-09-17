"""One-shot: continue skin-cancer training from the v1 checkpoint with the
ENTIRE backbone unfrozen (BN frozen) and a cosine LR schedule."""

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
from image_classification import (load_ham10000_data, CLASS_NAMES,
                                  SKIN_MODEL_PATH, SKIN_CONFIG_PATH,
                                  HAM10000_CLASSES, IMG_SIZE)

import json
import tensorflow as tf
from tensorflow import keras


def main():
    tf.random.set_seed(TU.SEED)
    np.random.seed(TU.SEED)

    print('Loading HAM10000...')
    X_train, X_val, y_train, y_val, X_test, y_test, class_weight = \
        load_ham10000_data(IMG_SIZE)

    model = TU.load_model_for_finetune(SKIN_MODEL_PATH)
    print('  Model loaded; all non-BatchNorm layers unfrozen.')
    model.summary()

    train_ds = TU.make_dataset(X_train, y_train, 32, training=True)
    val_ds = TU.make_dataset(X_val, y_val, 32)
    steps = int(np.ceil(len(X_train) / 32)) * 20
    opt = keras.optimizers.Adam(
        keras.optimizers.schedules.CosineDecay(3e-5, steps, alpha=1/30))
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, class_weight=class_weight,
              epochs=20, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(
                             monitor='val_accuracy', mode='max', patience=6,
                             restore_best_weights=True, verbose=1),
                         keras.callbacks.ModelCheckpoint(
                             SKIN_MODEL_PATH, monitor='val_accuracy',
                             mode='max', save_best_only=True, verbose=0)])

    metrics = TU.evaluate_model(model, X_test, y_test, CLASS_NAMES, tag='skin-ft')
    model.save(SKIN_MODEL_PATH)

    config = {
        'model_path': SKIN_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'preprocessing': 'RGB, normalize to [0,1]',
        'num_classes': 7,
        'class_names': CLASS_NAMES,
        'class_mapping': HAM10000_CLASSES,
        'architecture': 'MobileNetV2_transfer_learning_v3_full_finetune',
        'training_notes': ('Full HAM10000, in-model Random* augmentation, '
                           'full-backbone fine-tune with cosine LR'),
        'accuracy': metrics['accuracy'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    with open(SKIN_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print('[OK] Model + config saved. Restart server.py to load.')


if __name__ == '__main__':
    main()
