import os
import numpy as np
from PIL import Image
import glob
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
xray_dir = os.path.join(DATASET_DIR, 'chest_xray')
train_dir = os.path.join(xray_dir, 'train')

print("Checking directories:")
print("xray_dir exists:", os.path.exists(xray_dir))
print("train_dir exists:", os.path.exists(train_dir))

normal_train = glob.glob(os.path.join(train_dir, 'NORMAL', '*'))
pneumonia_train = glob.glob(os.path.join(train_dir, 'PNEUMONIA', '*'))
print(f"Found {len(normal_train)} normal images and {len(pneumonia_train)} pneumonia images in train.")

# Load one image and check stats
if len(normal_train) > 0:
    p = normal_train[0]
    img = Image.open(p).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    print(f"Sample Normal Image stats - Shape: {arr.shape}, Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}, Std: {arr.std()}")

if len(pneumonia_train) > 0:
    p = pneumonia_train[0]
    img = Image.open(p).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    print(f"Sample Pneumonia Image stats - Shape: {arr.shape}, Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}, Std: {arr.std()}")
