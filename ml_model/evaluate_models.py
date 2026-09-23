
import os
import sys
import csv
import glob
import json
import random
import argparse
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ML_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ML_DIR)
for _p in (REPO_ROOT, ML_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from PIL import Image

import eval_utils as EU

DATASET_DIR = os.path.join(ML_DIR, "Dataset")
REPORTS_DIR = os.path.join(ML_DIR, "reports")
SEED = 42
CALIB_FRACTION = 0.30
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_keras(path):
    from medidiagnose import inference_utils as MDI
    return MDI.load_keras_model(path)


def _list_images(folder):
    out = []
    for e in EXTS:
        out += glob.glob(os.path.join(folder, "**", "*" + e), recursive=True)
    return sorted(out)


# ---------------------------------------------------------------------------
# Chest X-ray
# ---------------------------------------------------------------------------
def eval_pneumonia(model_path=None, tta=True):
    model_path = model_path or os.path.join(ML_DIR, "pneumonia_model.h5")
    test_dir = os.path.join(DATASET_DIR, "chest_xray", "test")
    items = []  # (path, true_idx, root)
    for cls, idx in (("NORMAL", 0), ("PNEUMONIA", 1)):
        for p in _list_images(os.path.join(test_dir, cls)):
            items.append((p, idx, "test"))
    items.sort(key=lambda r: r[0])          # deterministic order
    random.Random(SEED).shuffle(items)
    names = ["NORMAL", "PNEUMONIA"]

    model = load_keras(model_path)
    print(f"[pneumonia] model={os.path.basename(model_path)} samples={len(items)}")

    # batch inference
    X = np.stack([EU.load_gray(p, 224, letterbox=True) for p, _, _ in items])
    probs = EU.predict_probs(model, X, tta=tta)
    y_true = np.array([t for _, t, _ in items])

    # calibration subset = first CALIB_FRACTION (deterministic, from the shuffled order)
    n_cal = int(len(items) * CALIB_FRACTION)
    cal_idx = np.arange(n_cal)
    held_idx = np.arange(n_cal, len(items))

    T, curve = EU.fit_temperature(probs[cal_idx], y_true[cal_idx])
    probs_cal = EU.apply_temperature(probs, T)

    reports = {}
    for tag, idx in (("full", np.arange(len(items))), ("heldout", held_idx)):
        pred_raw = probs[idx].argmax(1)
        rep_raw = EU.classification_report(y_true[idx], pred_raw, probs[idx], names)
        pred_cal = probs_cal[idx].argmax(1)
        rep_cal = EU.classification_report(y_true[idx], pred_cal, probs_cal[idx], names)
        reports[tag] = {"uncalibrated": rep_raw, "calibrated": rep_cal}

    samples = [{"path": items[i][0], "root": items[i][2], "true": names[y_true[i]],
                "predicted": names[int(probs_cal.argmax(1)[i])],
                "confidence": float(probs_cal[i].max()),
                "correct": bool(probs_cal.argmax(1)[i] == y_true[i])}
               for i in range(len(items))]

    extra = {
        "model_file": os.path.basename(model_path),
        "preprocessing": "letterbox (aspect-preserving pad) grayscale 224x224, /255, 5x flip-TTA",
        "temperature": T,
        "calibration_subset_n": n_cal,
        "calibration_fraction": CALIB_FRACTION,
        "auc_calibrated_heldout": reports["heldout"]["calibrated"].get("auc"),
        "tta": tta,
        "nll_curve": {"T": curve["grid"], "nll": curve["nll"]},
        "argmax_distribution_full": dict(Counter(
            names[i] for i in probs_cal.argmax(1))),
    }
    main = reports["heldout"]["calibrated"]
    main["n_samples"] = len(held_idx)
    out = EU.write_report(REPORTS_DIR, "pneumonia_eval", main, extra, samples)
    EU.write_report(REPORTS_DIR, "pneumonia_eval_fulltest",
                    reports["full"]["calibrated"],
                    {"note": "full official test set, calibrated", **extra})
    _png_confusion(main["confusion_matrix"], names, "pneumonia_confusion_matrix.png",
                   "Chest X-ray - pneumonia (calibrated, held-out)")
    print(f"[pneumonia] heldout acc={main['accuracy']:.4f} auc={main.get('auc')} T={T}")
    print(f"[pneumonia] full-test acc={reports['full']['calibrated']['accuracy']:.4f} "
          f"argmax={extra['argmax_distribution_full']}")
    return {"temperature": T, "full": reports["full"]["calibrated"],
            "heldout": main, "reports": [str(p) for p in out]}


# ---------------------------------------------------------------------------
# Skin
# ---------------------------------------------------------------------------
def _skin_split():
    """Reproduce image_classification.load_skin_combined_data's test fold."""
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from image_classification import HAM10000_CLASSES, PAD20_CLASSES

    records = []
    ham_dir = os.path.join(DATASET_DIR, "HAM10000")
    meta = os.path.join(ham_dir, "HAM10000_metadata.csv")
    if os.path.exists(meta):
        paths = {}
        for folder in ("HAM10000_images_part_1", "HAM10000_images_part_2",
                       "HAM10000_images", "images"):
            d = os.path.join(ham_dir, folder)
            if os.path.isdir(d):
                for p in glob.glob(os.path.join(d, "*.jpg")):
                    paths[os.path.splitext(os.path.basename(p))[0]] = p
        for _, row in pd.read_csv(meta).iterrows():
            p = paths.get(row["image_id"])
            if p is not None and row["dx"] in HAM10000_CLASSES:
                records.append((p, row["dx"], "ham/" + str(row["lesion_id"]), "HAM10000"))

    pad_dir = os.path.join(DATASET_DIR, "Skin_Cancer")
    pmeta = os.path.join(pad_dir, "metadata.csv")
    if os.path.exists(pmeta):
        paths = {}
        for root, _d, files in os.walk(pad_dir):
            for f in files:
                if f.lower().endswith(EXTS):
                    paths[os.path.splitext(f)[0]] = os.path.join(root, f)
        for _, row in pd.read_csv(pmeta).iterrows():
            code = PAD20_CLASSES.get(str(row["diagnostic"]).upper())
            p = paths.get(str(row["img_id"]).rsplit(".", 1)[0])
            if code and p is not None:
                records.append((p, code, "pad/" + str(row["patient_id"]), "Skin_Cancer"))

    paths = np.array([r[0] for r in records])
    codes = np.array([r[1] for r in records])
    groups = np.array([r[2] for r in records])
    roots = np.array([r[3] for r in records])
    labels = np.array([HAM10000_CLASSES[c] for c in codes])

    sgkf = StratifiedGroupKFold(n_splits=12, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(paths, labels, groups))
    test_idx = folds[0][1]
    return paths[test_idx], labels[test_idx], roots[test_idx], list(HAM10000_CLASSES.keys())


def eval_skin(model_path=None, tta=True):
    model_path = model_path or os.path.join(ML_DIR, "skin_cancer_model.h5")
    paths, labels, roots, names = _skin_split()
    order = np.argsort(paths)
    paths, labels, roots = paths[order], labels[order], roots[order]
    print(f"[skin] model={os.path.basename(model_path)} test samples={len(paths)} "
          f"dist={dict(sorted(Counter(labels).items()))}")

    model = load_keras(model_path)
    X = np.stack([EU.load_rgb(p, 224) for p in paths])
    probs = EU.predict_probs(model, X, tta=tta)

    main = EU.classification_report(labels, probs.argmax(1), probs, names)
    samples = [{"path": paths[i], "root": roots[i], "true": names[labels[i]],
                "predicted": names[int(probs.argmax(1)[i])],
                "confidence": float(probs[i].max()),
                "correct": bool(probs.argmax(1)[i] == labels[i])}
               for i in range(len(paths))]
    extra = {"model_file": os.path.basename(model_path),
             "preprocessing": "RGB 224x224 /255, 5x flip-TTA",
             "split": "group-disjoint StratifiedGroupKFold test fold "
                      "(HAM lesion_id + PAD patient_id)",
             "test_distribution": {names[k]: int(v) for k, v in
                                   sorted(Counter(labels).items())},
             "per_source": _per_source_accuracy(samples),
             "tta": tta}
    out = EU.write_report(REPORTS_DIR, "skin_eval", main, extra, samples)
    _png_confusion(main["confusion_matrix"], names, "skin_confusion_matrix.png",
                   "Skin lesion - 7-class (HAM10000 + PAD-UFES-20)")
    print(f"[skin] acc={main['accuracy']:.4f} macroF1={main['macro_f1']:.4f} "
          f"auc_ovr={main.get('auc_ovr_macro')}")
    return {"metrics": main, "reports": [str(p) for p in out]}


def _per_source_accuracy(samples):
    agg = {}
    for s in samples:
        a = agg.setdefault(s["root"], [0, 0])
        a[1] += 1
        a[0] += int(s["correct"])
    return {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4)}
            for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Breast
# ---------------------------------------------------------------------------
def eval_breast(model_path=None, tta=True):
    model_path = model_path or os.path.join(ML_DIR, "breast_cancer_model.h5")
    root = os.path.join(DATASET_DIR, "breast_ultrasound")
    cls_dirs = [d for d in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, d))]
    names = [d for d in ("normal", "benign", "malignant") if d in cls_dirs] or cls_dirs
    items = []
    for cls in names:
        for p in _list_images(os.path.join(root, cls)):
            items.append((p, names.index(cls)))
    items.sort(key=lambda r: r[0])
    random.Random(SEED).shuffle(items)
    hold = items[int(len(items) * 0.8):]     # deterministic 20% holdout
    print(f"[breast] model={os.path.basename(model_path)} holdout={len(hold)}/{len(items)} "
          f"classes={names}")

    model = load_keras(model_path)
    X = np.stack([EU.load_gray(p, 224, letterbox=True) for p, _ in hold])
    probs = EU.predict_probs(model, X, tta=tta)
    y_true = np.array([t for _, t in hold])
    main = EU.classification_report(y_true, probs.argmax(1), probs, names)
    samples = [{"path": hold[i][0], "root": names[y_true[i]],
                "true": names[y_true[i]], "predicted": names[int(probs.argmax(1)[i])],
                "confidence": float(probs[i].max()),
                "correct": bool(probs.argmax(1)[i] == y_true[i])}
               for i in range(len(hold))]
    extra = {"model_file": os.path.basename(model_path),
             "preprocessing": "letterbox grayscale 224x224 /255, 5x flip-TTA",
             "split": "deterministic 20% holdout of breast_ultrasound",
             "tta": tta}
    out = EU.write_report(REPORTS_DIR, "breast_eval", main, extra, samples)
    _png_confusion(main["confusion_matrix"], names, "breast_confusion_matrix.png",
                   "Breast ultrasound - 3-class")
    print(f"[breast] acc={main['accuracy']:.4f} macroF1={main['macro_f1']:.4f} "
          f"auc_ovr={main.get('auc_ovr_macro')}")
    return {"metrics": main, "reports": [str(p) for p in out]}


# ---------------------------------------------------------------------------
# Confusion-matrix figure (Fathom / clinical-scientific-clarity palette)
# ---------------------------------------------------------------------------
def _png_confusion(cm, names, filename, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as exc:  # pragma: no cover
        print(f"  (matplotlib unavailable, skipping figure: {exc})")
        return None
    cm = np.array(cm, dtype=float)
    palette = LinearSegmentedColormap.from_list(
        "fathom", ["#f6f7f9", "#d7dee8", "#9fb2c8", "#5f7695", "#2f3f59", "#16233a"])
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 3.0, 0.9 * len(names) + 2.6),
                           dpi=160)
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    im = ax.imshow(pct, cmap=palette, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10, color="#2f3f59")
    ax.set_ylabel("True", fontsize=10, color="#2f3f59")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{int(cm[i, j])}\n{pct[i, j] * 100:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="#ffffff" if pct[i, j] > 0.55 else "#16233a")
    ax.set_title(title + "\nrows normalised to 100% (count + row%)",
                 fontsize=11, color="#16233a", pad=12, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure: {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["all", "pneumonia", "pneumonial", "xray", "skin", "breast"])
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()
    tta = not args.no_tta
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary = {}
    which = {"pneumonial": "pneumonia", "xray": "pneumonia", "all": "all",
             "pneumonia": "pneumonia", "skin": "skin", "breast": "breast"}[args.model]
    if which in ("pneumonia", "all"):
        summary["pneumonia"] = eval_pneumonia(tta=tta)
    if which in ("skin", "all"):
        summary["skin"] = eval_skin(tta=tta)
    if which in ("breast", "all"):
        summary["breast"] = eval_breast(tta=tta)
    with open(os.path.join(REPORTS_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("\nDone. Reports in", REPORTS_DIR)


if __name__ == "__main__":
    main()
