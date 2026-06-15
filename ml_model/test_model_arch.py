"""
Quick architecture sanity test.
Tests whether MobileNetV2 gets meaningful features when:
  A) Raw [0,1] images (current broken state)
  B) preprocess_input applied via ImageDataGenerator preprocessing_function
  C) Lambda layer scaling [0,1] -> [-1,1] baked inside the graph

Verdict is based on output activation variance from the backbone.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
np.random.seed(42)

# Simulate a realistic RGB image batch (5 samples, [0,1] normalized)
dummy_imgs = np.random.uniform(0, 1, (5, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)

print("=" * 60)
print("MobileNetV2 input sensitivity test")
print("=" * 60)

# ── Test A: Raw [0,1] input ──────────────────────────────────
print("\n[A] Raw [0,1] input (CURRENT state — wrong for MobileNetV2):")
base_a = keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),
                                         include_top=False, weights='imagenet')
base_a.trainable = False
out_a = base_a(dummy_imgs, training=False)
pool_a = tf.reduce_mean(out_a, axis=[1,2]).numpy()
print(f"    Feature mean:  {pool_a.mean():.4f}")
print(f"    Feature std:   {pool_a.std():.4f}")
print(f"    Nonzero ratio: {(pool_a > 0).mean():.4f}")

# ── Test B: preprocess_input applied externally ──────────────
print("\n[B] preprocess_input([0,1]*255 input) = correct [-1,1] range:")
imgs_255 = (dummy_imgs * 255.0).astype(np.float32)
imgs_preprocessed = preprocess_input(imgs_255.copy())
base_b = keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),
                                          include_top=False, weights='imagenet')
base_b.trainable = False
out_b = base_b(imgs_preprocessed, training=False)
pool_b = tf.reduce_mean(out_b, axis=[1,2]).numpy()
print(f"    Feature mean:  {pool_b.mean():.4f}")
print(f"    Feature std:   {pool_b.std():.4f}")
print(f"    Nonzero ratio: {(pool_b > 0).mean():.4f}")

# ── Test C: Lambda x*2-1 inside graph ───────────────────────
print("\n[C] Lambda(x*2-1) baked into model graph (planned fix):")
inputs_c = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
x_c = layers.Lambda(lambda v: v * 2.0 - 1.0)(inputs_c)
base_c = keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),
                                          include_top=False, weights='imagenet')
base_c.trainable = False
feat_c = base_c(x_c, training=False)
pool_layer_c = layers.GlobalAveragePooling2D()(feat_c)
model_c = models.Model(inputs_c, pool_layer_c)
pool_c = model_c.predict(dummy_imgs, verbose=0)
print(f"    Feature mean:  {pool_c.mean():.4f}")
print(f"    Feature std:   {pool_c.std():.4f}")
print(f"    Nonzero ratio: {(pool_c > 0).mean():.4f}")

print("\n" + "=" * 60)
print("VERDICT:")
print(f"  A (raw [0,1]): std={pool_a.std():.4f}")
print(f"  B (preprocess_input): std={pool_b.std():.4f}")
print(f"  C (Lambda x*2-1): std={pool_c.std():.4f}")
print("  Higher std = richer, more discriminative features")
print("=" * 60)
