import os
import json
import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


# ---------------------------------------------------------------------------
# Image loading / preprocessing
# ---------------------------------------------------------------------------
def letterbox_gray(image, size=224, fill=0):
    """Resize preserving aspect ratio, pad with ``fill`` to a size x size square."""
    img = image.convert("L")
    w, h = img.size
    scale = float(size) / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def load_gray(image_path, size=224, letterbox=True):
    img = Image.open(image_path)
    arr = letterbox_gray(img, size) if letterbox else (
        np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0
    )
    return arr[..., np.newaxis]


def load_rgb(image_path, size=224):
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_probs(model, batch, tta=True):
    """Return the mean softmax over the batch (with optional 4-way flip TTA)."""
    x = np.asarray(batch, dtype=np.float32)
    if x.ndim == 3:
        x = x[np.newaxis, ...]
    p = model(x, training=False).numpy()
    if tta:
        for ax in (1, 2):
            p = p + model(np.flip(x, axis=ax), training=False).numpy()
        for ax in (1, 2):
            p = p + model(np.flip(np.flip(x, axis=1), axis=2), training=False).numpy()
        p = p / 5.0
    return p


def softmax(logits, axis=-1):
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def apply_temperature(probs, T):
    """Temperature-scale a (N, C) probability matrix. T=1 -> unchanged."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0)
    return softmax(np.log(p) / float(T), axis=-1)


def fit_temperature(probs, y_true, grid=None):
    """Fit scalar T minimising negative log-likelihood on a CALIBRATION split.

    Returns (T_best, {"grid": [...], "nll": [...]}).
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    grid = grid if grid is not None else np.round(np.arange(0.5, 4.01, 0.05), 3)
    best_T, best_nll = 1.0, np.inf
    curve = []
    n = len(y_true)
    for T in grid:
        q = apply_temperature(probs, T)
        nll = -np.mean(np.log(np.clip(q[np.arange(n), y_true], 1e-12, 1.0)))
        curve.append(float(nll))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T, {"grid": [float(g) for g in grid], "nll": curve,
                    "nll_at_T1": float(curve[int(np.argmin(np.abs(np.array(grid) - 1.0)))]),
                    "nll_best": float(best_nll)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_report(y_true, y_pred, probs=None, class_names=None):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = sorted(set(list(y_true) + list(y_pred)))
    names = class_names or [str(i) for i in labels]

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))

    per_class = []
    for i, lab in enumerate(labels):
        per_class.append({
            "class": names[lab] if lab < len(names) else str(lab),
            "index": int(lab),
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        })

    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                            average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                               average="weighted", zero_division=0)

    out = {
        "accuracy": acc,
        "n_samples": int(len(y_true)),
        "classes": names,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "missing_labels": int(sum(ti not in list(y_true) for ti in range(len(names)))),
    }

    # AUC: binary -> single value; multiclass -> one-vs-rest (macro)
    if probs is not None:
        probs = np.asarray(probs, dtype=float)
        try:
            if len(names) == 2 or probs.shape[1] == 2:
                out["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
                out["auc_type"] = "binary"
            elif probs.shape[1] == len(names):
                out["auc_ovr_macro"] = float(
                    roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
                out["auc_type"] = "one-vs-rest-macro"
        except Exception as exc:  # pragma: no cover
            out["auc_error"] = str(exc)

    # A tiny bit of derived clinical signal for the binary pneumonia case
    if len(names) == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        out["sensitivity_recall_pos"] = float(tp / (tp + fn)) if (tp + fn) else None
        out["specificity_neg"] = float(tn / (tn + fp)) if (tn + fp) else None
        out["confusion"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    return out


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------
def write_report(out_dir, name, metrics, extra=None, samples=None, config=None):
    os.makedirs(out_dir, exist_ok=True)
    payload = {"report": name, "metrics": metrics}
    if extra:
        payload.update(extra)
    if config:
        payload["config"] = config
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if samples:
        csv_path = os.path.join(out_dir, f"{name}_predictions.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write("image_path,root,true_label,predicted_label,confidence,correct\n")
            for s in samples:
                fh.write("%s,%s,%s,%s,%.4f,%s\n" % (
                    s["path"], s.get("root", ""), s["true"], s["predicted"],
                    s["confidence"], s["correct"]))
    else:
        csv_path = None

    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_markdown_report(name, metrics, extra, csv_path))
    return json_path, csv_path, md_path


def _markdown_report(name, m, extra, csv_path):
    L = []
    L.append(f"# Evaluation report -- {name}\n")
    L.append("_Generated by `ml_model/evaluate_models.py` from the real model file. "
             "Informational only - not a medical device._\n")
    L.append(f"- Samples evaluated: **{m.get('n_samples')}**")
    L.append(f"- Accuracy: **{m.get('accuracy'):.4f}**")
    if "macro_f1" in m:
        L.append(f"- Macro F1: **{m['macro_f1']:.4f}**  |  Weighted F1: **{m['weighted_f1']:.4f}**")
    if "auc" in m:
        L.append(f"- AUC (binary): **{m['auc']:.4f}**")
    if "auc_ovr_macro" in m:
        L.append(f"- AUC (one-vs-rest macro): **{m['auc_ovr_macro']:.4f}**")
    if "confusion" in m:
        c = m["confusion"]
        L.append(f"- Confusion: TN={c['TN']} FP={c['FP']} FN={c['FN']} TP={c['TP']}")
        if m.get("sensitivity_recall_pos") is not None:
            L.append(f"- Sensitivity: **{m['sensitivity_recall_pos']:.4f}**  |  "
                     f"Specificity: **{m['specificity_neg']:.4f}**")
    if extra:
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool)):
                L.append(f"- {k}: {v}")
    L.append("\n## Per-class metrics\n")
    L.append("| Class | Precision | Recall | F1 | Support |")
    L.append("|---|---|---|---|---|")
    for c in m.get("per_class", []):
        L.append("| %s | %.4f | %.4f | %.4f | %d |" % (
            c["class"], c["precision"], c["recall"], c["f1"], c["support"]))
    L.append("\n## Confusion matrix\n")
    L.append("Rows = true, columns = predicted.\n")
    names = m.get("classes", [])
    if m.get("confusion_matrix"):
        L.append("| true\\pred | " + " | ".join(names) + " |")
        L.append("|" + "---|" * (len(names) + 1))
        for i, row in enumerate(m["confusion_matrix"]):
            L.append("| **%s** | " % names[i] + " | ".join(str(v) for v in row) + " |")
    if csv_path:
        L.append(f"\nPer-sample predictions (path / true / predicted / confidence / correct): "
                 f"`{os.path.basename(csv_path)}`\n")
    return "\n".join(L) + "\n"
