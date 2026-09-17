"""Shared transfer-learning training utilities for MediDiagnose.

Every image-model training script in ml_model/ imports the helpers below so
that all four CNNs share one proven recipe:

  * MobileNetV2 ImageNet backbone, image size 224
  * Inputs are float32 in [0, 1] (exactly what backend/server.py feeds the
    loaded models at inference time — see inference_utils.py)
  * Keras built-in Random* augmentation layers INSIDE the model: they are
    active only during .fit() and become no-ops when server.py calls
    model(image, training=False), so train-time and serve-time behaviour
    can never drift apart
  * `Rescaling` (a real, serializable layer) instead of Lambda layers —
    models save to .h5 and load in Keras 3 without safe_mode issues
  * Two phases: (1) frozen backbone, head-only, LR 1e-3
                (2) unfreeze the last `unfreeze` layers, BN kept frozen,
                    LR 1e-5
  * EarlyStopping on val_accuracy with restore_best_weights

Datasets are kept as uint8 arrays in RAM (4x smaller than float32) and
converted per-batch in a tf.data pipeline.
"""

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    TF_AVAILABLE = False

SEED = 42
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(num_classes, channels=1, size=224, dropout=0.3,
                augment=None, name='model', backbone='mobilenetv2'):
    """Transfer-learning model.

    Parameters
    ----------
    num_classes : int   softmax output size
    channels    : int   1 = grayscale input, 3 = RGB input
    size        : int   image is size x size
    augment     : str   None | 'skin' | 'xray' | 'us' | 'ecg'
    backbone    : str   'mobilenetv2' (expects [-1,1]) or
                        'efficientnetb0' (expects [0,255], rescales itself)
    """
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow is required')

    inputs = layers.Input((size, size, channels), name=f'{name}_input')
    x = inputs

    # Augmentation layers are identity at inference (training=False call),
    # so the preprocessing server.py performs stays the single source of truth.
    if augment == 'skin':
        x = layers.RandomFlip('horizontal_and_vertical')(x)
        x = layers.RandomRotation(0.20, fill_mode='reflect')(x)
        x = layers.RandomZoom(0.15, fill_mode='reflect')(x)
        x = layers.RandomTranslation(0.10, 0.10, fill_mode='reflect')(x)
        x = layers.RandomContrast(0.10)(x)
    elif augment == 'xray':
        x = layers.RandomFlip('horizontal')(x)
        x = layers.RandomTranslation(0.05, 0.05, fill_mode='constant')(x)
        x = layers.RandomZoom(0.08, fill_mode='constant')(x)
    elif augment == 'us':
        x = layers.RandomFlip('horizontal')(x)
        x = layers.RandomRotation(0.05, fill_mode='constant')(x)
        x = layers.RandomZoom(0.10, fill_mode='constant')(x)
    elif augment == 'ecg':
        # Small time-shifts (horizontal) are legitimate invariances for ECG;
        # vertical shifts + zoom simulate gain/paper-speed variation.
        x = layers.RandomTranslation(0.05, 0.05, fill_mode='constant')(x)
        x = layers.RandomZoom(0.05, fill_mode='constant')(x)
    elif augment is not None:
        raise ValueError(f'Unknown augment mode: {augment}')

    if channels == 1:
        x = layers.Concatenate(name='gray_to_rgb')([x, x, x])

    if backbone == 'mobilenetv2':
        # [0, 1] -> [-1, 1], the range MobileNetV2 was pretrained on.
        x = layers.Rescaling(2.0, offset=-1.0, name='rescale_to_backbone')(x)
        base_model = keras.applications.MobileNetV2(
            input_shape=(size, size, 3), include_top=False, weights='imagenet')
    elif backbone == 'efficientnetb0':
        # Keras EfficientNet includes its own [0,255] preprocessing, so the
        # [0,1] server inputs are rescaled up before the backbone.
        x = layers.Rescaling(255.0, name='rescale_to_backbone')(x)
        base_model = keras.applications.EfficientNetB0(
            input_shape=(size, size, 3), include_top=False, weights='imagenet')
    else:
        raise ValueError(f'Unknown backbone: {backbone}')
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name=f'{name}_output')(x)

    model = keras.models.Model(inputs, outputs, name=name)
    return model, base_model


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def make_dataset(X_uint8, y_int, batch_size=32, training=False):
    """uint8 images (N,H,W,C) + int labels -> batched float32 [0,1] dataset."""
    ds = tf.data.Dataset.from_tensor_slices((X_uint8, y_int))
    if training:
        ds = ds.shuffle(min(len(X_uint8), 10000), seed=SEED,
                        reshuffle_each_iteration=True)
    ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compile_model(model, learning_rate):
    # NOTE: only 'accuracy' — Keras 3's Precision/Recall metrics misread
    # 2-class softmax probabilities as multi-label binary outputs and crash
    # with a shape error. Precision/recall are reported per-class by the
    # sklearn classification report in evaluate_model() instead.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])


def load_model_for_finetune(path):
    """Load a saved .h5 model and unfreeze everything EXCEPT BatchNorm
    (recursively, including the nested MobileNetV2 backbone)."""
    model = keras.models.load_model(path, compile=False, safe_mode=False)
    model.trainable = True

    def _unfreeze(layer):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        elif hasattr(layer, 'layers'):
            for sub in layer.layers:
                _unfreeze(sub)

    for layer in model.layers:
        _unfreeze(layer)
    return model


# ---------------------------------------------------------------------------
# Two-phase training
# ---------------------------------------------------------------------------

def train_two_phase(model, base_model, X_train, y_train, X_val, y_val, *,
                    class_weight=None, batch_size=32, model_path=None,
                    phase1_epochs=20, phase2_epochs=15, unfreeze=100,
                    phase1_lr=1e-3, phase2_lr=1e-5,
                    phase2_schedule='plateau', tag=''):
    """Phase 1: frozen backbone, train head. Phase 2: unfreeze top layers.

    BatchNorm layers stay frozen in phase 2 (standard practice on small
    medical datasets — recomputing BN statistics on ~10k images would
    destroy the pretrained calibration).

    phase2_schedule: 'plateau' = ReduceLROnPlateau (default), or 'cosine'
    = cosine decay from phase2_lr to phase2_lr/30 over the whole phase
    (works better for full-backbone fine-tuning).
    """
    train_ds = make_dataset(X_train, y_train, batch_size, training=True)
    val_ds = make_dataset(X_val, y_val, batch_size)

    fit_kwargs = dict(validation_data=val_ds, class_weight=class_weight,
                      verbose=2)

    print(f'\n[{tag}] Phase 1 — head training (backbone frozen), '
          f'LR={phase1_lr:g}, up to {phase1_epochs} epochs')
    compile_model(model, phase1_lr)
    cbs = [keras.callbacks.EarlyStopping(
               monitor='val_accuracy', mode='max', patience=6,
               restore_best_weights=True, verbose=1),
           keras.callbacks.ReduceLROnPlateau(
               monitor='val_loss', factor=0.5, patience=3,
               min_lr=1e-6, verbose=1)]
    if model_path:
        cbs.append(keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_accuracy', mode='max',
            save_best_only=True, verbose=0))
    model.fit(train_ds, epochs=phase1_epochs, callbacks=cbs, **fit_kwargs)

    if unfreeze == 'all':
        print(f'\n[{tag}] Phase 2 — fine-tuning the ENTIRE backbone '
              f'(BN frozen), LR={phase2_lr:g} ({phase2_schedule}), '
              f'up to {phase2_epochs} epochs')
        base_model.trainable = True
    else:
        print(f'\n[{tag}] Phase 2 — fine-tuning last {unfreeze} layers '
              f'(BN frozen), LR={phase2_lr:g}, up to {phase2_epochs} epochs')
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze]:
            layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    if phase2_schedule == 'cosine':
        steps = max(1, int(np.ceil(len(X_train) / batch_size))) * phase2_epochs
        lr_sched = keras.optimizers.schedules.CosineDecay(phase2_lr, steps,
                                                          alpha=1.0 / 30.0)
        opt = keras.optimizers.Adam(lr_sched)
    else:
        opt = keras.optimizers.Adam(phase2_lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    cbs = [keras.callbacks.EarlyStopping(
               monitor='val_accuracy', mode='max', patience=6,
               restore_best_weights=True, verbose=1)]
    if phase2_schedule != 'cosine':
        cbs.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7,
            verbose=1))
    if model_path:
        cbs.append(keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_accuracy', mode='max',
            save_best_only=True, verbose=0))
    model.fit(train_ds, epochs=phase2_epochs, callbacks=cbs, **fit_kwargs)

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test_int, class_names, tag=''):
    """Print accuracy / per-class accuracy / report / confusion matrix.

    Returns a dict with scalar metrics for the config JSON.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    X_float = X_test.astype(np.float32) / 255.0
    results = model.evaluate(X_float, y_test_int, verbose=0)
    loss, acc = results[0], results[1]

    probs = model.predict(X_float, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print(f'\n[{tag}] Evaluation on held-out test set '
          f'({len(y_test_int)} samples)')
    print(f'  Loss:     {loss:.4f}')
    print(f'  Accuracy: {acc:.4f}  ({acc * 100:.2f}%)')

    print('\n  Per-class accuracy:')
    for i, cls_name in enumerate(class_names):
        mask = y_test_int == i
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == i).mean()
            print(f'    {cls_name:<20s}: {cls_acc:.4f}  ({mask.sum()} samples)')

    present = sorted(set(y_test_int) | set(y_pred))
    print('\n  Classification Report:')
    print(classification_report(
        y_test_int, y_pred, labels=present, zero_division=0,
        target_names=[class_names[i] for i in present]))

    cm = confusion_matrix(y_test_int, y_pred, labels=range(len(class_names)))
    print('  Confusion Matrix (rows=true, cols=pred):')
    print(cm)

    conf = np.max(probs, axis=1)
    print(f'\n  Confidence: mean={conf.mean():.3f} '
          f'median={np.median(conf):.3f} min={conf.min():.3f} '
          f'max={conf.max():.3f}')

    return {'accuracy': float(acc), 'loss': float(loss),
            'confusion_matrix': cm.tolist()}
