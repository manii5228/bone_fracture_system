"""
generate_confusion_matrix.py
============================
Run YOLOv8 inference on the validation split and produce:
  1. A binary confusion matrix (fracture vs. no-fracture at image level)
  2. Detection metrics: precision, recall, F1, mAP@0.5, mAP@0.5:0.95
  3. A PNG confusion matrix figure saved to results/confusion_matrix.png
  4. A text summary saved to results/metrics_summary.txt

Usage:
    python generate_confusion_matrix.py

Requirements: ultralytics, matplotlib, seaborn, numpy
"""

import os
import sys
import json
from pathlib import Path

# ── Dependency check ─────────────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}\n"
             f"Install with: pip install ultralytics matplotlib seaborn numpy")

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_YAML   = "bone_fracture_dataset/dataset.yaml"
RESULTS_DIR    = Path("results")
CM_IMAGE_PATH  = RESULTS_DIR / "confusion_matrix.png"
METRICS_PATH   = RESULTS_DIR / "metrics_summary.txt"
CONF_THRESHOLD = 0.25      # IoU/confidence threshold for detection
IOU_THRESHOLD  = 0.50      # IoU threshold for TP decision

# Fine-tuned model search paths (same priority as app.py)
MODEL_PATHS = [
    "E:/bone_fracture_system/quick_train/fracture_model/weights/best.pt",
    "quick_train/fracture_model/weights/best.pt",
    "models/best.pt",
    "yolov8n.pt",           # base weights – fallback only
]

# ── Load model ───────────────────────────────────────────────────────────────
def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            print(f"✅ Loading model: {path}")
            return YOLO(path), path
    sys.exit("[ERROR] No YOLO model weights found. Run quick_train.py first.")

# ── Validate on dataset ───────────────────────────────────────────────────────
def run_validation(model):
    """
    Run the official Ultralytics validation loop.
    Returns the metrics object produced by model.val().
    """
    print(f"\n📊 Running validation on: {DATASET_YAML}")
    if not os.path.exists(DATASET_YAML):
        sys.exit(f"[ERROR] Dataset YAML not found: {DATASET_YAML}")

    metrics = model.val(
        data    = DATASET_YAML,
        imgsz   = 320,
        conf    = CONF_THRESHOLD,
        iou     = IOU_THRESHOLD,
        device  = "cpu",
        verbose = True,
    )
    return metrics

# ── Extract metrics ───────────────────────────────────────────────────────────
def extract_metrics(metrics):
    """Pull scalar values from the Ultralytics metrics object."""
    box = metrics.box

    precision    = float(box.mp)          # mean precision
    recall       = float(box.mr)          # mean recall
    map50        = float(box.map50)       # mAP@0.5
    map5095      = float(box.map)         # mAP@0.5:0.95
    f1           = (2 * precision * recall / (precision + recall + 1e-9))

    # Confusion matrix values from Ultralytics (class 0 = fracture)
    raw_cm = getattr(box, 'confusion_matrix', None)

    return {
        "precision":  round(precision,  4),
        "recall":     round(recall,     4),
        "f1":         round(f1,         4),
        "map50":      round(map50,      4),
        "map5095":    round(map5095,    4),
        "raw_cm":     raw_cm,
    }

# ── Build image-level binary confusion matrix ─────────────────────────────────
def build_binary_cm_from_ultralytics(metrics_obj):
    """
    Extract TP / FP / FN / TN from the Ultralytics confusion matrix.
    Ultralytics produces an (nc+1) × (nc+1) matrix where:
       rows = ground-truth class index (nc = background)
       cols = predicted class index   (nc = background / no-detection)

    For a single-class detector (nc=1, class 0 = fracture):
       cm[0][0] = TP  (fracture detected, fracture present)
       cm[0][1] = FN  (no detection, fracture present)
       cm[1][0] = FP  (fracture detected, no fracture in GT)
       cm[1][1] = TN  (no detection, no fracture in GT)
    """
    try:
        cm_obj = metrics_obj.confusion_matrix
        cm_arr = cm_obj.matrix          # numpy array, shape (nc+1, nc+1)
        TP = int(cm_arr[0, 0])
        FN = int(cm_arr[0, 1])
        FP = int(cm_arr[1, 0])
        TN = int(cm_arr[1, 1])
        return np.array([[TP, FN], [FP, TN]]), True
    except Exception as e:
        print(f"⚠️  Could not extract Ultralytics confusion matrix: {e}")
        return None, False

# ── Plot confusion matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(cm_array, scalar_metrics, model_path, output_path):
    """Render a styled 2×2 confusion matrix and save as PNG."""
    RESULTS_DIR.mkdir(exist_ok=True)

    TP = cm_array[0, 0]
    FN = cm_array[0, 1]
    FP = cm_array[1, 0]
    TN = cm_array[1, 1]

    total      = TP + FN + FP + TN
    accuracy   = (TP + TN) / total if total > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0   # recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("#f8fafc")

    # ── Left: heatmap ─────────────────────────────────────────────────────────
    ax = axes[0]
    labels = [["TP", "FN"], ["FP", "TN"]]
    annot  = [[f"{TP}\n(TP)", f"{FN}\n(FN)"],
               [f"{FP}\n(FP)", f"{TN}\n(TN)"]]

    cmap = sns.diverging_palette(145, 10, as_cmap=True)
    sns.heatmap(
        cm_array,
        annot=annot, fmt="", cmap=cmap,
        linewidths=2, linecolor="white",
        xticklabels=["Fracture\n(Predicted)", "No Fracture\n(Predicted)"],
        yticklabels=["Fracture\n(Actual)",    "No Fracture\n(Actual)"],
        ax=ax, annot_kws={"size": 16, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Confusion Matrix — Binary Fracture Detection\n"
                 f"(conf≥{CONF_THRESHOLD}, IoU≥{IOU_THRESHOLD})",
                 fontsize=14, fontweight="bold", pad=18, color="#1e293b")
    ax.set_ylabel("Ground Truth", fontsize=12, labelpad=10)
    ax.set_xlabel("Model Prediction", fontsize=12, labelpad=10)
    ax.tick_params(axis="both", labelsize=11)

    # ── Right: metrics panel ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#f8fafc")
    ax2.axis("off")

    metrics_text = [
        ("Detection Metrics", "", "#2563eb", 20, "bold"),
        ("", "", "#000", 12, "normal"),
        ("mAP @ 0.5",          f"{scalar_metrics['map50']*100:.1f}%",  "#1e293b", 14, "normal"),
        ("mAP @ 0.5:0.95",     f"{scalar_metrics['map5095']*100:.1f}%", "#1e293b", 14, "normal"),
        ("Precision",          f"{scalar_metrics['precision']*100:.1f}%", "#1e293b", 14, "normal"),
        ("Recall (Sensitivity)", f"{scalar_metrics['recall']*100:.1f}%", "#1e293b", 14, "normal"),
        ("F1-Score",           f"{scalar_metrics['f1']*100:.1f}%",     "#1e293b", 14, "normal"),
        ("", "", "#000", 12, "normal"),
        ("Image-Level Metrics", "", "#059669", 18, "bold"),
        ("", "", "#000", 12, "normal"),
        ("Accuracy",           f"{accuracy*100:.1f}%",      "#065f46", 14, "normal"),
        ("Sensitivity",        f"{sensitivity*100:.1f}%",   "#065f46", 14, "normal"),
        ("Specificity",        f"{specificity*100:.1f}%",   "#065f46", 14, "normal"),
        ("", "", "#000", 12, "normal"),
        ("Sample Counts", "", "#dc2626", 16, "bold"),
        ("", "", "#000", 12, "normal"),
        ("True Positive (TP)",  str(TP),  "#991b1b", 13, "normal"),
        ("False Negative (FN)", str(FN),  "#991b1b", 13, "normal"),
        ("False Positive (FP)", str(FP),  "#991b1b", 13, "normal"),
        ("True Negative (TN)",  str(TN),  "#991b1b", 13, "normal"),
        ("Total Images",        str(total), "#1e293b", 13, "bold"),
        ("", "", "#000", 10, "normal"),
        ("Model", os.path.basename(model_path), "#64748b", 10, "normal"),
    ]

    y = 0.97
    for label, value, color, size, weight in metrics_text:
        if label == "" and value == "":
            y -= 0.025
            continue
        if value:
            text = f"{label:<28}{value}"
        else:
            text = label
        ax2.text(0.05, y, text, transform=ax2.transAxes,
                 fontsize=size, color=color, fontweight=weight,
                 fontfamily="monospace" if value else "sans-serif",
                 verticalalignment="top")
        y -= 0.047 if size >= 18 else 0.038

    ax2.text(0.05, 0.04,
             "⚠ AI decision-support prototype.\nPhysician review required for clinical use.",
             transform=ax2.transAxes, fontsize=9.5,
             color="#92400e",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7",
                       edgecolor="#d97706", linewidth=1.5))

    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✅ Confusion matrix saved → {output_path}")

# ── Write text summary ────────────────────────────────────────────────────────
def write_summary(scalar_metrics, cm_array, model_path):
    RESULTS_DIR.mkdir(exist_ok=True)

    TP = int(cm_array[0, 0]) if cm_array is not None else "?"
    FN = int(cm_array[0, 1]) if cm_array is not None else "?"
    FP = int(cm_array[1, 0]) if cm_array is not None else "?"
    TN = int(cm_array[1, 1]) if cm_array is not None else "?"

    if cm_array is not None:
        total = TP + FN + FP + TN
        accuracy    = (TP + TN) / total if total else 0
        sensitivity = TP / (TP + FN) if (TP + FN) else 0
        specificity = TN / (TN + FP) if (TN + FP) else 0
    else:
        accuracy = sensitivity = specificity = "?"

    summary = f"""
==============================================================
  MedScan AI — YOLOv8 Validation Metrics Summary
==============================================================
Model file    : {model_path}
Dataset YAML  : {DATASET_YAML}
Conf threshold: {CONF_THRESHOLD}
IoU threshold : {IOU_THRESHOLD}

── Detection Metrics (YOLO box-level) ──────────────────────
  Precision            : {scalar_metrics['precision']*100:.2f}%
  Recall               : {scalar_metrics['recall']*100:.2f}%
  F1-Score             : {scalar_metrics['f1']*100:.2f}%
  mAP @ 0.5            : {scalar_metrics['map50']*100:.2f}%
  mAP @ 0.5:0.95       : {scalar_metrics['map5095']*100:.2f}%

── Image-Level Binary Confusion Matrix ─────────────────────
            | Predicted: Fracture | Predicted: No Fracture |
  ----------|---------------------|------------------------|
  Actual: Fracture   |  TP = {TP:<6}         |  FN = {FN:<6}               |
  Actual: No Fracture|  FP = {FP:<6}         |  TN = {TN:<6}               |

── Derived Image-Level Metrics ─────────────────────────────
  Accuracy             : {accuracy*100:.2f}% (if numeric)
  Sensitivity (Recall) : {sensitivity*100:.2f}%
  Specificity          : {specificity*100:.2f}%

── Notes ────────────────────────────────────────────────────
  • TP/FP/FN/TN are derived from Ultralytics confusion_matrix
    at the bounding-box detection level.
  • "No fracture" images have no ground-truth boxes; TN is
    counted when the model correctly returns zero detections.
  • Results are for the validation split only.
  • This system is an AI decision-support prototype;
    all outputs require physician review.
==============================================================
"""
    with open(METRICS_PATH, "w") as f:
        f.write(summary)

    print(summary)
    print(f"✅ Metrics summary saved → {METRICS_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, model_path = load_model()
    metrics_obj = run_validation(model)
    scalar      = extract_metrics(metrics_obj)

    cm_array, ok = build_binary_cm_from_ultralytics(metrics_obj)

    if ok and cm_array is not None:
        plot_confusion_matrix(cm_array, scalar, model_path, CM_IMAGE_PATH)
        write_summary(scalar, cm_array, model_path)
    else:
        print("⚠️  Confusion matrix data unavailable — printing scalar metrics only.")
        write_summary(scalar, None, model_path)

    print("\n🎯 All done. Files written to:", RESULTS_DIR.resolve())

if __name__ == "__main__":
    main()
