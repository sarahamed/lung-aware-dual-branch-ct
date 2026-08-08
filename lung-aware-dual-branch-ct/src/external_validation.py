"""
External-dataset validation utilities: locating class folders in an
independently downloaded chest-CT dataset, mapping its subtype labels onto
this pipeline's 3-class Normal/Malignant framing, and scoring predictions.
Extracted from the verified pipeline (revision_full_pipeline.ipynb Task 3).
"""
import os

import cv2
import imageio
import numpy as np

from config import CLASS_NAMES, IMG_SIZE
from src.preprocessing import crop_to_lung_np

EXTERNAL_SUBTYPES = ["adenocarcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma", "normal"]

SUBTYPE_TO_TRUE_BUCKET = {
    "normal": "Normal",
    "adenocarcinoma": "Malignant",
    "large.cell.carcinoma": "Malignant",
    "squamous.cell.carcinoma": "Malignant",
}


def download_external_dataset():
    """Downloads the external chest-CT validation dataset via kagglehub. Requires
    Kaggle API credentials configured (see data/README.md)."""
    import kagglehub
    return kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")


def find_leaf_class_dirs(root, subtype_keywords=EXTERNAL_SUBTYPES):
    """Recursively find directories whose name matches one of the subtype keywords
    (case-insensitive substring match), regardless of train/test/valid nesting.
    Returns {subtype_keyword: [image_path, ...]}."""
    hits = {}
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath).lower()
        for kw in subtype_keywords:
            if kw.split(".")[0] in base:  # match on primary keyword, e.g. "adenocarcinoma"
                img_files = [f for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if img_files:
                    hits.setdefault(kw, []).extend(os.path.join(dirpath, f) for f in img_files)
    return hits


def preprocess_external_image(path):
    """Loads an external-dataset image and applies the EXACT training preprocessing
    pipeline (global = full resized image, local = crop_to_lung_np at default/
    baseline settings) -- no reimplementation, for a fair comparison."""
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img_global = img
    img_local = crop_to_lung_np(img_global)
    return img_global, img_local


def score_external_dataset(model, files_by_subtype, class_names=CLASS_NAMES):
    """Runs `model` over every image in files_by_subtype (as returned by
    find_leaf_class_dirs) and returns a list of row dicts: filepath,
    external_subtype, true_bucket, pred_class (native 3-class, not collapsed),
    confidence."""
    rows = []
    for subtype, files in files_by_subtype.items():
        true_bucket = SUBTYPE_TO_TRUE_BUCKET[subtype]
        for fp in files:
            try:
                xg, xl = preprocess_external_image(fp)
            except Exception:
                continue
            preds = model.predict([xg[None], xl[None]], verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_name = class_names[pred_idx]
            rows.append({
                "filepath": fp, "external_subtype": subtype, "true_bucket": true_bucket,
                "pred_class": pred_name, "confidence": float(preds[pred_idx]),
            })
    return rows


def is_correct(row):
    """Scoring rule for the Normal-vs-Malignant external validation framing: a
    Benign prediction is never counted correct either way (no external image is
    ever truly Benign under this dataset's class mapping)."""
    if row["pred_class"] == "Benign":
        return False
    if row["true_bucket"] == "Normal":
        return row["pred_class"] == "Normal"
    if row["true_bucket"] == "Malignant":
        return row["pred_class"] == "Malignant"
    return False


def sensitivity_specificity(ext_df):
    """Sensitivity/specificity for malignant detection (Normal-vs-Malignant framing).
    ext_df must have columns 'true_bucket' and 'pred_class'."""
    mal_true = ext_df["true_bucket"] == "Malignant"
    mal_pred = ext_df["pred_class"] == "Malignant"
    tp = int((mal_true & mal_pred).sum())
    fn = int((mal_true & ~mal_pred).sum())
    tn = int((~mal_true & ~mal_pred).sum())
    fp = int((~mal_true & mal_pred).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return sensitivity, specificity
