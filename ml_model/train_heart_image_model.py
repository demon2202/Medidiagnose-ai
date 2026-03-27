import os
import numpy as np
import json
import warnings
import ast
warnings.filterwarnings('ignore')

# ── wfdb ────────────────────────────────────────────────────────────────────
WFDB_AVAILABLE = False
try:
    import wfdb
    WFDB_AVAILABLE = True
    print("✓ wfdb library available")
except ImportError:
    print("✗ wfdb not installed. Run: pip install wfdb")

# ── TensorFlow ──────────────────────────────────────────────────────────────
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} available")
except ImportError:
    print("✗ TensorFlow not available")

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
PTBXL_DIR = os.path.join(DATASET_DIR, 'ptb-xl')

HEART_IMAGE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'heart_image_model.h5')
HEART_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'heart_image_config.json')

IMG_SIZE = 224


# ══════════════════════════════════════════════════════════════════════════════
#                         CLASS DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

HEART_CLASSES = {
    0: {'code': 'normal',      'name': 'Normal',                  'severity': 'healthy'},
    1: {'code': 'mi',          'name': 'Myocardial Infarction',   'severity': 'critical'},
    2: {'code': 'arrhythmia',  'name': 'Arrhythmia',              'severity': 'moderate'},
    3: {'code': 'hf',          'name': 'Heart Failure Signs',     'severity': 'high'},
    4: {'code': 'hypertrophy', 'name': 'Ventricular Hypertrophy', 'severity': 'moderate'}
}

CLASS_NAMES = ['Normal', 'MI', 'Arrhythmia', 'Heart Failure', 'Hypertrophy']

SCP_TO_CLASS = {
    'NORM': 0, 'SR': 0,
    'IMI': 1, 'AMI': 1, 'LMI': 1, 'PMI': 1, 'ASMI': 1,
    'ILMI': 1, 'IPLMI': 1, 'IPMI': 1, 'MI': 1,
    'INJAL': 1, 'INJIL': 1, 'INJLA': 1, 'INJIN': 1, 'INJAS': 1,
    'ISC_': 1, 'ISCA': 1, 'ISCI': 1, 'STD_': 1, 'STE_': 1,
    'AFIB': 2, 'AFLT': 2, 'SVTAC': 2, 'PSVT': 2,
    'STACH': 2, 'SBRAD': 2, 'SARRH': 2,
    'BIGU': 2, 'TRIGU': 2, 'PAC': 2, 'PVC': 2,
    'VPRE': 2, 'WPW': 2, 'STTC': 2, 'NST_': 2,
    'LAFB': 3, 'LPFB': 3, 'IRBBB': 3, 'CRBBB': 3,
    'CLBBB': 3, 'ILBBB': 3, '1AVB': 3, '2AVB': 3, '3AVB': 3,
    'CD': 3, 'IVCD': 3,
    'LVH': 4, 'RVH': 4, 'LAO': 4, 'LAE': 4,
    'RAO': 4, 'RAE': 4, 'SEHYP': 4, 'HYP': 4,
}

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ══════════════════════════════════════════════════════════════════════════════
#                  SIGNAL → GRAYSCALE IMAGE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def signal_to_grayscale_image(signal, img_size=(224, 224)):
    """
    Convert 12-lead ECG signal to a high-quality grayscale image.

    Layout: 4×3 grid (standard 12-lead ECG format).
    Renders at high resolution then downscales with LANCZOS for
    clean anti-aliased waveform lines.

    Args:
        signal: numpy array (num_samples, num_leads)
        img_size: tuple (height, width) for output

    Returns:
        numpy array (height, width, 1) float32 [0, 1]
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO

        num_leads = min(signal.shape[1], 12)

        fig, axes = plt.subplots(4, 3, figsize=(12, 8), dpi=100)
        fig.patch.set_facecolor('white')
        axes = axes.flatten()

        # Global y-limits for consistent scaling across all leads
        all_values = signal[:, :num_leads].flatten()
        g_min = np.percentile(all_values, 1)
        g_max = np.percentile(all_values, 99)
        y_range = max(g_max - g_min, 1.0)
        y_margin = y_range * 0.15

        for i in range(num_leads):
            ax = axes[i]
            ax.set_facecolor('#FAFAFA')

            # ECG paper-like grid
            ax.grid(True, which='major', color='#DDDDDD',
                    linewidth=0.8, alpha=0.8)
            ax.minorticks_on()
            ax.grid(True, which='minor', color='#EEEEEE',
                    linewidth=0.4, alpha=0.5)

            # Plot waveform
            ax.plot(signal[:, i], 'k-', linewidth=1.0, antialiased=True)
            ax.set_ylim(g_min - y_margin, g_max + y_margin)
            ax.set_xlim(0, len(signal))

            # Lead label
            name = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f'L{i+1}'
            ax.text(0.02, 0.95, name, transform=ax.transAxes, fontsize=9,
                    fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.8))

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Hide unused subplots
        for i in range(num_leads, 12):
            axes[i].axis('off')

        plt.tight_layout(pad=0.5)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='white', edgecolor='none', dpi=100)
        plt.close(fig)
        buf.seek(0)

        # Load, resize, convert to grayscale
        img = Image.open(buf)
        img = img.resize(img_size, Image.LANCZOS)
        img = img.convert('L')

        img_array = np.array(img, dtype=np.float32) / 255.0

        # Contrast enhancement
        p5, p95 = np.percentile(img_array, (5, 95))
        if p95 - p5 > 0.1:
            img_array = np.clip((img_array - p5) / (p95 - p5), 0, 1)

        img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
        return img_array

    except Exception as e:
        print(f"  Error converting signal to image: {e}")
        return np.ones((img_size[0], img_size[1], 1), dtype=np.float32) * 0.95


# ══════════════════════════════════════════════════════════════════════════════
#                         MODEL CREATION
# ══════════════════════════════════════════════════════════════════════════════

def create_heart_model(input_shape=(224, 224, 1), num_classes=5):
    """
    Heart ECG model — MobileNetV2 transfer learning.

    Input:  grayscale (224, 224, 1)
    Internally replicates to 3 channels for MobileNetV2.
    Output: softmax over 5 classes

    Returns: (model, base_model_reference)
    """
    gray_input = layers.Input(shape=input_shape, name='ecg_input')

    # Replicate grayscale → 3 channels for pretrained backbone
    x = layers.Concatenate()([gray_input, gray_input, gray_input])

    base_model = keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # frozen for phase 1

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(gray_input, outputs, name='heart_ecg_mobilenetv2')
    return model, base_model


# ══════════════════════════════════════════════════════════════════════════════
#                      SYNTHETIC DATA (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def generate_ecg_signal(class_idx, duration_samples=1000, num_leads=12):
    """
    Generate synthetic 12-lead ECG signal with class-specific morphology.

    Class 0: Normal sinus rhythm
    Class 1: MI — ST elevation, pathological Q waves
    Class 2: Arrhythmia — irregular RR, PVCs, absent P waves
    Class 3: Heart failure — wide QRS, low voltage, T inversion
    Class 4: Hypertrophy — tall R waves, strain pattern
    """
    t = np.linspace(0, 10, duration_samples)
    signal = np.zeros((duration_samples, num_leads))
    heart_rate = np.random.uniform(0.8, 1.2)

    for lead in range(num_leads):
        lead_gain = 0.5 + (lead % 6) * 0.1 + np.random.uniform(-0.05, 0.05)
        beat_period = int(duration_samples / (10 * heart_rate))
        ecg = np.zeros(duration_samples)

        for beat_start in range(0, duration_samples - beat_period, beat_period):
            bt = np.arange(beat_period) / beat_period

            # Normal PQRST
            p_wave = 0.15 * np.exp(-0.5 * ((bt - 0.15) / 0.05) ** 2)
            q_wave = -0.1 * np.exp(-0.5 * ((bt - 0.32) / 0.008) ** 2)
            r_wave = 1.0 * np.exp(-0.5 * ((bt - 0.35) / 0.012) ** 2)
            s_wave = -0.2 * np.exp(-0.5 * ((bt - 0.38) / 0.01) ** 2)
            t_wave = 0.3 * np.exp(-0.5 * ((bt - 0.6) / 0.08) ** 2)
            beat = p_wave + q_wave + r_wave + s_wave + t_wave

            # ── Class-specific modifications ────────────────────────────
            if class_idx == 1:  # MI
                st_region = (bt > 0.38) & (bt < 0.55)
                if lead in [1, 5, 7, 8, 9]:
                    beat[st_region] += np.random.uniform(0.15, 0.4)
                else:
                    beat[st_region] -= np.random.uniform(0.05, 0.15)
                if lead in [1, 7, 8]:
                    beat += -0.3 * np.exp(-0.5 * ((bt - 0.3) / 0.02) ** 2)

            elif class_idx == 2:  # Arrhythmia
                beat = np.roll(beat, int(np.random.uniform(
                    -beat_period * 0.1, beat_period * 0.1)))
                if np.random.random() > 0.6:
                    pos = int(beat_period * np.random.uniform(0.3, 0.7))
                    if pos + 20 < beat_period:
                        pvc = 1.5 * np.exp(-0.5 * ((bt - bt[pos]) / 0.015) ** 2)
                        beat += pvc * np.random.uniform(0.5, 1.0)
                if np.random.random() > 0.5:
                    p_mask = (bt > 0.05) & (bt < 0.25)
                    beat[p_mask] *= 0.3
                    beat += 0.05 * np.sin(
                        2 * np.pi * np.random.uniform(5, 8) * bt)

            elif class_idx == 3:  # Heart failure
                r_w = 0.7 * np.exp(-0.5 * ((bt - 0.35) / 0.025) ** 2)
                s_w = -0.3 * np.exp(-0.5 * ((bt - 0.40) / 0.02) ** 2)
                qrs_mask = (bt > 0.28) & (bt < 0.48)
                beat[qrs_mask] = 0
                beat += r_w + s_w
                beat *= 0.6
                if lead in [0, 4, 7, 8, 9, 10]:
                    t_inv = (bt > 0.5) & (bt < 0.75)
                    beat[t_inv] *= -0.5

            elif class_idx == 4:  # Hypertrophy
                beat *= 1.5
                if lead in [7, 8, 9, 10, 11]:
                    strain = (bt > 0.4) & (bt < 0.75)
                    beat[strain] -= 0.2
                    t_m = (bt > 0.55) & (bt < 0.7)
                    beat[t_m] *= -0.8
                if lead in [6, 7]:
                    beat += -0.5 * np.exp(-0.5 * ((bt - 0.42) / 0.015) ** 2)

            end = min(beat_start + beat_period, duration_samples)
            length = end - beat_start
            ecg[beat_start:end] += beat[:length] * lead_gain

        # Baseline wander + noise
        ecg += 0.05 * np.sin(2 * np.pi * 0.15 * t)
        ecg += np.random.uniform(0.01, 0.04) * np.random.randn(duration_samples)
        if np.random.random() > 0.7:
            ecg += 0.02 * np.sin(2 * np.pi * 50 * t)

        signal[:, lead] = ecg

    return signal


def create_synthetic_data(n_samples=800, img_size=224):
    """Create synthetic ECG images as fallback when PTB-XL is unavailable."""
    print("⚠️  Creating synthetic ECG images (fallback)...")
    print("   For best accuracy, use PTB-XL dataset!")

    np.random.seed(42)
    X, y = [], []
    samples_per_class = n_samples // 5

    for class_idx in range(5):
        print(f"  Generating {CLASS_NAMES[class_idx]}...")
        for _ in range(samples_per_class):
            signal = generate_ecg_signal(class_idx, 1000, 12)

            # Normalize per-lead
            for lead in range(12):
                ld = signal[:, lead]
                std = np.std(ld)
                if std > 0:
                    signal[:, lead] = (ld - np.mean(ld)) / std

            img = signal_to_grayscale_image(signal, (img_size, img_size))
            X.append(img)
            y.append(class_idx)
        print(f"    ✓ {samples_per_class} samples")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    print(f"\n  ✓ Created {len(X)} images, shape {X.shape}")
    print(f"  Distribution: {Counter(y)}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#                       PTB-XL DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def find_ptbxl_dataset():
    """Find PTB-XL dataset in common directory names."""
    candidates = ['ptb-xl', 'ptbxl', 'PTB-XL', 'PTBXL', 'ptb_xl',
                   'physionet.org/files/ptb-xl/1.0.3']
    for name in candidates:
        check_dir = os.path.join(DATASET_DIR, name)
        if os.path.exists(check_dir):
            for root, dirs, files in os.walk(check_dir):
                if 'ptbxl_database.csv' in files:
                    meta_path = os.path.join(root, 'ptbxl_database.csv')
                    print(f"  Found PTB-XL at: {root}")
                    return root, meta_path
    return None, None


def load_ptbxl_dataset(sampling_rate=100, max_samples=3000, img_size=224):
    """
    Load PTB-XL dataset, convert signals to grayscale images.

    Returns:
        (X_images, y_labels, class_weight_dict) or None if not found
    """
    if not WFDB_AVAILABLE:
        print("❌ wfdb required. Run: pip install wfdb")
        return None

    ptbxl_dir, metadata_path = find_ptbxl_dataset()
    if ptbxl_dir is None:
        print("❌ PTB-XL dataset not found")
        print(f"📥 Download: https://physionet.org/content/ptb-xl/1.0.3/")
        print(f"📁 Extract to: {PTBXL_DIR}")
        return None

    print("📂 Loading PTB-XL dataset...")
    print(f"  Sampling rate: {sampling_rate} Hz")

    try:
        metadata = pd.read_csv(metadata_path, index_col='ecg_id')
        print(f"  Total records: {len(metadata)}")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None

    # Parse SCP codes
    try:
        metadata['scp_codes'] = metadata['scp_codes'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
        )
    except Exception:
        metadata['scp_codes'] = metadata['scp_codes'].apply(lambda x: {})

    # Limit samples
    if max_samples and len(metadata) > max_samples:
        metadata = metadata.sample(n=max_samples, random_state=42)
        print(f"  Limited to {max_samples} samples")

    records_folder = 'records100' if sampling_rate == 100 else 'records500'
    expected_length = 1000 if sampling_rate == 100 else 5000

    records_path = os.path.join(ptbxl_dir, records_folder)
    if not os.path.exists(records_path):
        print(f"❌ Records folder not found: {records_path}")
        return None

    print(f"  Records folder: {records_folder}")
    print(f"  Converting {len(metadata)} signals to images...")

    X_images, y_labels = [], []
    loaded, errors = 0, 0

    for idx, (ecg_id, row) in enumerate(metadata.iterrows()):
        if idx % 500 == 0 and idx > 0:
            print(f"    Processed {idx}/{len(metadata)} "
                  f"({loaded} loaded, {errors} errors)")
        try:
            filename = row['filename_lr'] if sampling_rate == 100 \
                else row['filename_hr']
            file_path = os.path.join(ptbxl_dir, filename)
            if file_path.endswith('.dat') or file_path.endswith('.hea'):
                file_path = file_path.rsplit('.', 1)[0]
            if not os.path.exists(file_path + '.dat'):
                errors += 1; continue

            signal, _ = wfdb.rdsamp(file_path)

            # Pad or truncate to expected_length
            if len(signal) >= expected_length:
                signal = signal[:expected_length]
            else:
                pad = np.zeros((expected_length - len(signal), signal.shape[1]))
                signal = np.vstack([signal, pad])

            # Normalize per-lead
            for lead in range(signal.shape[1]):
                ld = signal[:, lead]
                std = np.std(ld)
                if std > 0.01:
                    signal[:, lead] = (ld - np.mean(ld)) / std
                else:
                    signal[:, lead] = ld - np.mean(ld)

            # Determine class from SCP codes
            scp_codes = row.get('scp_codes', {})
            primary_class = 0
            max_likelihood = 0
            if isinstance(scp_codes, dict):
                for code, likelihood in scp_codes.items():
                    cu = str(code).upper()
                    if cu in SCP_TO_CLASS and likelihood > max_likelihood:
                        primary_class = SCP_TO_CLASS[cu]
                        max_likelihood = likelihood

            img = signal_to_grayscale_image(signal, (img_size, img_size))
            X_images.append(img)
            y_labels.append(primary_class)
            loaded += 1

        except Exception as e:
            errors += 1
            if errors < 3:
                print(f"    Error: {e}")
            continue

    if loaded == 0:
        print("❌ No records loaded!")
        return None

    X_images = np.array(X_images, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int32)

    print(f"\n  ✓ Loaded {loaded} images ({errors} errors)")
    print(f"  Shape: {X_images.shape}")
    print(f"  Distribution: {Counter(y_labels)}")

    # Class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_labels),
                               y=y_labels)
    cw_dict = {i: 1.0 for i in range(5)}
    for cls, w in zip(np.unique(y_labels), cw):
        cw_dict[cls] = float(w)
    print(f"  Class weights: { {CLASS_NAMES[k]: round(v, 2) for k, v in cw_dict.items()} }")

    return X_images, y_labels, cw_dict


# ══════════════════════════════════════════════════════════════════════════════
#                          TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_heart_image_model():
    """
    Train heart ECG image classifier.

    Method: MobileNetV2 transfer learning (single best approach).
      Phase 1: Frozen backbone, lr=0.001, up to 30 epochs
      Phase 2: Fine-tune last 30 layers, lr=1e-5, up to 15 epochs

    Loss: categorical_crossentropy (standard — no custom loss
          serialization issues when loading in server.py).

    Class imbalance handled via class_weight parameter.
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow required"); return None

    print("\n" + "=" * 70)
    print("  HEART ECG MODEL — MobileNetV2 Transfer Learning")
    print("  (Grayscale input → internal 3-channel conversion)")
    print("=" * 70)

    num_classes = 5

    # ── Load data ───────────────────────────────────────────────────────
    data = load_ptbxl_dataset(sampling_rate=100, max_samples=3000,
                               img_size=IMG_SIZE)

    if data is not None:
        X, y, cw_dict = data
        using_real_data = True
    else:
        print("\n⚠️  PTB-XL not found — using synthetic data...")
        X, y = create_synthetic_data(800, IMG_SIZE)
        cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
        cw_dict = dict(enumerate(cw))
        using_real_data = False

    # One-hot encode
    y_onehot = to_categorical(y, num_classes=num_classes)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Shape: {X_train.shape}")
    print(f"  Real data: {using_real_data}")

    # ── Create model ────────────────────────────────────────────────────
    print("\n🔧 Creating MobileNetV2 model (5-class ECG)...")
    print("   Input: (224, 224, 1) grayscale → internally replicated to 3ch")
    model, base_model = create_heart_model((IMG_SIZE, IMG_SIZE, 1), num_classes)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    model.summary()

    # ── Augmentation (very light — ECG images shouldn't be distorted) ──
    datagen = ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        brightness_range=[0.9, 1.1],
        horizontal_flip=False,   # NEVER flip ECGs
        fill_mode='constant',
        cval=1.0                 # white background fill
    )

    # ── Phase 1: Frozen backbone ────────────────────────────────────────
    epochs_p1 = 30 if using_real_data else 20
    batch_size = 32

    callbacks_p1 = [
        EarlyStopping(monitor='val_auc', mode='max', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(HEART_IMAGE_MODEL_PATH, monitor='val_auc',
                        mode='max', save_best_only=True, verbose=1)
    ]

    print("\n" + "-" * 50)
    print("🚀 Phase 1 — Training classification head (backbone frozen)...")
    print("-" * 50)

    model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs_p1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_p1,
        class_weight=cw_dict,
        verbose=1
    )

    # ── Phase 2: Fine-tune last 30 backbone layers ─────────────────────
    print("\n" + "-" * 50)
    print("🚀 Phase 2 — Fine-tuning last 30 backbone layers...")
    print("-" * 50)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    callbacks_p2 = [
        EarlyStopping(monitor='val_auc', mode='max', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                          min_lr=1e-8, verbose=1),
        ModelCheckpoint(HEART_IMAGE_MODEL_PATH, monitor='val_auc',
                        mode='max', save_best_only=True, verbose=1)
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=15,
        validation_data=(X_test, y_test),
        callbacks=callbacks_p2,
        class_weight=cw_dict,
        verbose=1
    )

    # ── Evaluate ────────────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("📊 Evaluation...")
    print("-" * 50)

    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Loss:      {results[0]:.4f}")
    print(f"  Accuracy:  {results[1]:.4f}  ({results[1] * 100:.2f}%)")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall:    {results[3]:.4f}")
    print(f"  AUC:       {results[4]:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Per-class accuracy
    print("\n  Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true_classes == i
        if mask.sum() > 0:
            acc = (y_pred_classes[mask] == i).mean()
            print(f"    {name:20s}: {acc:.4f}  ({mask.sum()} samples)")

    # Classification report
    present = sorted(set(y_true_classes) | set(y_pred_classes))
    present_names = [CLASS_NAMES[i] for i in present if i < len(CLASS_NAMES)]
    print("\n  Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes,
                                labels=present, target_names=present_names,
                                zero_division=0))

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("  Confusion Matrix:")
    # Header
    header = "              " + "  ".join(f"{CLASS_NAMES[i]:>8s}" for i in present)
    print(header)
    for i, row_idx in enumerate(present):
        row_name = CLASS_NAMES[row_idx] if row_idx < len(CLASS_NAMES) else str(row_idx)
        row_vals = "  ".join(f"{cm[i, j]:>8d}" for j in range(len(present)))
        print(f"  {row_name:>12s}  {row_vals}")

    # Confidence statistics
    max_conf = np.max(y_pred, axis=1)
    print(f"\n  Confidence stats:")
    print(f"    Mean:   {max_conf.mean():.4f}")
    print(f"    Median: {np.median(max_conf):.4f}")
    print(f"    Min:    {max_conf.min():.4f}")
    print(f"    Max:    {max_conf.max():.4f}")

    # Sample predictions
    print("\n  Sample predictions (first 10 test images):")
    for i in range(min(10, len(X_test))):
        true_name = CLASS_NAMES[y_true_classes[i]]
        pred_name = CLASS_NAMES[y_pred_classes[i]]
        conf = max_conf[i]
        status = '✓' if true_name == pred_name else '✗'
        print(f"    {status} True: {true_name:20s}  Pred: {pred_name:20s}  "
              f"Conf: {conf:.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    model.save(HEART_IMAGE_MODEL_PATH)
    print(f"\n✓ Model saved: {HEART_IMAGE_MODEL_PATH}")

    config = {
        'model_path': HEART_IMAGE_MODEL_PATH,
        'model_type': 'image',
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'preprocessing': 'Grayscale, normalize to [0,1]',
        'note': 'Model internally replicates 1ch to 3ch for MobileNetV2',
        'num_classes': num_classes,
        'classes': {str(k): v for k, v in HEART_CLASSES.items()},
        'class_names': CLASS_NAMES,
        'architecture': 'MobileNetV2_transfer_learning',
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4]),
        'mean_confidence': float(max_conf.mean()),
        'using_real_data': using_real_data,
        'confusion_matrix': cm.tolist()
    }
    with open(HEART_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {HEART_CONFIG_PATH}")

    print("\n" + "=" * 70)
    print("  ⚠️  REMINDERS:")
    print(f"  • Grayscale {IMG_SIZE}×{IMG_SIZE} input")
    print(f"  • 5 classes: {CLASS_NAMES}")
    if not using_real_data:
        print("  • ⚠️  SYNTHETIC DATA — retrain with PTB-XL for production!")
    print("  • Restart server.py to load the new model!")
    print("=" * 70)

    return model


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if not TF_AVAILABLE:
        print("❌ TensorFlow required. Install: pip install tensorflow")
    else:
        print("\n" + "=" * 70)
        print("  MediDiagnose-AI: Heart ECG Model Training")
        print("  Method: MobileNetV2 Transfer Learning (best accuracy)")
        print("=" * 70)

        train_heart_image_model()

        print("\n✅ Done!")