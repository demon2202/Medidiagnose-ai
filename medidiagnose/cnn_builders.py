"""Shared CNN architectures + two-phase transfer-learning trainer.

Every image model in MediDiagnose is MobileNetV2 transfer learning:
  Phase A (cheap):  frozen backbone, features precomputed once, head trained.
  Phase B (short):  top of the backbone fine-tuned with light augmentation.

Inputs follow medidiagnose.inference_utils preprocessing:
  * skin    : (224, 224, 3) in [0, 1]
  * xray    : (224, 224, 1) in [0, 1]  (channel-expanded to 3 internally)
  * breast  : (224, 224, 1) in [0, 1]
  * ecg     : (224, 224, 1) in [0, 1]
"""

import numpy as np

SEED = 42


def build_model(num_classes, input_channels=3, dropout=0.3, dense_units=256,
                base_trainable=False, augmentation=False):
    """Build the transfer-learning model. Returns the full (keras) model."""
    from tensorflow import keras
    from tensorflow.keras import layers

    shape = (224, 224, input_channels)
    inputs = keras.Input(shape=shape)

    x = inputs
    if augmentation:
        x = layers.RandomFlip('horizontal_and_vertical')(x)
        x = layers.RandomRotation(0.10)(x)
        x = layers.RandomZoom(0.10)(x)
        x = layers.RandomTranslation(0.08, 0.08)(x)
        if input_channels == 3:
            x = layers.RandomBrightness(0.10)(x)
            x = layers.RandomContrast(0.10)(x)

    # Map everything into the [-1, 1] range MobileNetV2 was pretrained on.
    x = layers.Rescaling(2.0, offset=-1.0, name='preproc_rescale')(x)
    if input_channels == 1:
        # 1x1 conv gray -> RGB so the ImageNet backbone can be reused
        x = layers.Conv2D(3, (1, 1), padding='same', name='gray_to_rgb')(x)

    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = base_trainable
    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D(name='feat_gap')(x)
    x = layers.Dropout(dropout, name='head_dropout_1')(x)
    x = layers.Dense(dense_units, activation='relu', name='head_dense',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout * 0.5, name='head_dropout_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='head_out')(x)

    model = keras.Model(inputs, outputs)
    model._mdi_backbone = base       # kept for fine-tuning utilities
    return model


def feature_extractor(model):
    """Sub-model mapping the model's raw input to the pooled 1280-d feature."""
    from tensorflow import keras
    return keras.Model(model.input, model.get_layer('feat_gap').output)


def build_head(input_dim, num_classes, dropout=0.3, dense_units=256):
    """Standalone head matching build_model's head layers (by name)."""
    from tensorflow import keras
    head = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dropout(dropout, name='head_dropout_1'),
        keras.layers.Dense(dense_units, activation='relu', name='head_dense',
                           kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dropout(dropout * 0.5, name='head_dropout_2'),
        keras.layers.Dense(num_classes, activation='softmax', name='head_out'),
    ])
    return head


def copy_head_weights(full_model, head):
    """Copy trained head weights into the full model (matched by name)."""
    by_name = {l.name: l for l in full_model.layers}
    by_name['head_dense'].set_weights(head.get_layer('head_dense').get_weights())
    by_name['head_out'].set_weights(head.get_layer('head_out').get_weights())


def set_finetune_layers(model, n_unfreeze=40):
    """Unfreeze the last `n_unfreeze` layers of the backbone (keep BN frozen)."""
    backbone = model._mdi_backbone
    backbone.trainable = True
    for layer in backbone.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in backbone.layers[-n_unfreeze:]:
        if layer.__class__.__name__.startswith('BatchNormalization'):
            layer.trainable = False


def extract_features(model, X):
    """Run the frozen backbone over a stacked array; returns (N, 1280)."""
    fx = feature_extractor(model)
    feats = fx.predict(X.astype(np.float32) / 255.0, batch_size=32, verbose=0)
    return np.asarray(feats, dtype=np.float32)


def default_callbacks(patience=6, min_delta=1e-3):
    from tensorflow import keras
    return [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                      restore_best_weights=True,
                                      min_delta=min_delta, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=max(2, patience // 2),
                                          min_lr=1e-6, verbose=1),
    ]
