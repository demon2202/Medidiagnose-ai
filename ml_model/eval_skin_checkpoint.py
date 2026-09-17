"""One-shot: evaluate the skin v4 checkpoint (best val weights saved by
ModelCheckpoint after the training process was interrupted) on the held-out
test split and write skin_cancer_config.json."""

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
from medidiagnose import inference_utils as MDI
from image_classification import (load_skin_combined_data, CLASS_NAMES,
                                  SKIN_MODEL_PATH, SKIN_CONFIG_PATH,
                                  HAM10000_CLASSES, IMG_SIZE)

import json


def main():
    data = load_skin_combined_data(IMG_SIZE)
    if data is None:
        print('dataset missing'); return 1
    X_train, X_val, y_train, y_val, X_test, y_test, _cw = data

    model = MDI.load_keras_model(SKIN_MODEL_PATH)
    TU.compile_model(model, 1e-5)   # evaluate() requires a compiled model
    print('checkpoint loaded; evaluating on test...')
    metrics = TU.evaluate_model(model, X_test, y_test, CLASS_NAMES, tag='skin-v4')

    config = {
        'model_path': SKIN_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'preprocessing': 'RGB, normalize to [0,1]',
        'num_classes': 7,
        'class_names': CLASS_NAMES,
        'class_mapping': HAM10000_CLASSES,
        'architecture': 'MobileNetV2_transfer_learning_v4',
        'training_notes': ('HAM10000 + PAD-UFES-20 (group-disjoint split), '
                           'in-model Random* augmentation, train-only '
                           'rebalancing, full-backbone fine-tune, cosine LR'),
        'accuracy': metrics['accuracy'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    with open(SKIN_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print('[OK] Config saved.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
