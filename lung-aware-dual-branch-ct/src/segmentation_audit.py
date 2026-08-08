"""
Dataset-wide lung-segmentation audit: per-image lung-mask coverage, empty-mask
and outlier flagging, and sub-binning of flagged cases (borderline vs. genuine
outlier). Extracted from the verified pipeline (revision_full_pipeline.ipynb
Task 1, R1#7).
"""
import cv2
import imageio
import numpy as np
import pandas as pd

from config import CLASS_NAMES, IMG_SIZE
from src.preprocessing import lung_mask_np

# Flagging thresholds: an image's lung mask is flagged if it covers less than 2%
# or more than 85% of the frame (near-empty or near-whole-frame segmentation).
LOW_FLAG_THRESHOLD = 0.02
HIGH_FLAG_THRESHOLD = 0.85
# Sub-binning of high-lung-fraction flagged cases.
BORDERLINE_UPPER = 0.90


def run_segmentation_audit(files, labels, class_names=CLASS_NAMES):
    """Runs lung_mask_np over every (file, label) pair and returns a DataFrame with
    one row per image: filepath, class, lung_frac, empty_mask, flagged, n_components."""
    rows = []
    for fp, lbl in zip(files, labels):
        img = imageio.imread(fp)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

        mask, diag = lung_mask_np(img, return_diagnostics=True)
        lung_frac = float(mask.mean())
        empty_mask = bool(mask.sum() == 0)
        flagged = bool((lung_frac < LOW_FLAG_THRESHOLD) or (lung_frac > HIGH_FLAG_THRESHOLD))

        rows.append({
            "filepath": fp,
            "class": class_names[int(lbl)],
            "lung_frac": lung_frac,
            "empty_mask": empty_mask,
            "flagged": flagged,
            "n_components": diag["n_components"],
        })
    return pd.DataFrame(rows)


def find_col(df, candidates):
    """Column auto-detection, robust to naming drift across audit CSV versions."""
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def subbin(lf, borderline_lower=HIGH_FLAG_THRESHOLD, borderline_upper=BORDERLINE_UPPER):
    """Sub-bins a high-lung-fraction flagged case as 'borderline' (just above the
    high-flag threshold) vs. 'genuine_outlier' (well above it)."""
    if borderline_lower <= lf < borderline_upper:
        return "borderline"
    if lf >= borderline_upper:
        return "genuine_outlier"
    return "other"  # flagged via the low-end criterion instead


def subbin_high_flagged(audit_df, col_lung_frac="lung_frac", col_flagged="flagged", col_class="class"):
    """Returns the high-lung-fraction (>=HIGH_FLAG_THRESHOLD) subset of flagged cases,
    with a 'subbin' column added."""
    flagged_df = audit_df[audit_df[col_flagged].astype(bool)].copy()
    high_flagged = flagged_df[flagged_df[col_lung_frac] >= HIGH_FLAG_THRESHOLD].copy()
    high_flagged["subbin"] = high_flagged[col_lung_frac].apply(subbin)
    return flagged_df, high_flagged


def class_subbin_breakdown(high_flagged, col_class="class"):
    """Counts and row-normalized percentages of sub-bin membership by class."""
    counts = high_flagged.groupby([col_class, "subbin"]).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return counts, pct
