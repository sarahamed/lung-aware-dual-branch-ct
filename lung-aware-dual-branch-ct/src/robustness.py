"""
Robustness / perturbation analysis: image-level perturbations (noise, blur,
JPEG compression, brightness/contrast, incomplete lung field), segmentation
degradation and crop-window perturbation, and breaking-point analysis.
Extracted from the verified pipeline (revision_full_pipeline.ipynb Tasks 2 & 4).
"""
import cv2
import numpy as np

from config import IMG_SIZE
from src.preprocessing import crop_to_lung_np, crop_to_lung_np_window_perturbed, load_raw_resized, macro_metrics

# ---------------------------------------------------------------------------
# Segmentation-parameter perturbation (Task 2, mask threshold/kernel/erosion)
# ---------------------------------------------------------------------------


def evaluate_degraded(model, files, labels, threshold_shift=0.0, kernel_delta=0, erode_dilate_px=0.0):
    """Evaluates `model` with the local branch's lung-crop mask perturbed via
    threshold_shift / kernel_delta / erode_dilate_px (global branch unaffected)."""
    yt, yp = [], []
    for fp, lbl in zip(files, labels):
        raw = load_raw_resized(fp)
        img_global = raw
        img_local = crop_to_lung_np(raw, threshold_shift=threshold_shift,
                                     kernel_size_delta=kernel_delta,
                                     erode_dilate_px=erode_dilate_px)
        preds = model.predict([img_global[None], img_local[None]], verbose=0)[0]
        yp.append(int(np.argmax(preds)))
        yt.append(int(lbl))
    return macro_metrics(np.array(yt), np.array(yp))


# ---------------------------------------------------------------------------
# Crop-window perturbation (Task 2, direct position/scale perturbation of the
# local branch's crop window -- what actually stress-tests the local branch)
# ---------------------------------------------------------------------------


def evaluate_window_degraded(model, files, labels, window_shift_px=0, window_scale=1.0):
    """Evaluates `model` with the local branch's crop WINDOW directly shifted/scaled
    (global branch unaffected)."""
    yt, yp = [], []
    for fp, lbl in zip(files, labels):
        raw = load_raw_resized(fp)
        img_global = raw
        img_local = crop_to_lung_np_window_perturbed(
            raw, window_shift_px=window_shift_px, window_scale=window_scale)
        preds = model.predict([img_global[None], img_local[None]], verbose=0)[0]
        yp.append(int(np.argmax(preds)))
        yt.append(int(lbl))
    return macro_metrics(np.array(yt), np.array(yp))


def find_breaking_point(df, family, threshold_pp=2.0):
    """Finds the mildest-severity perturbation level (within `family`) whose
    accuracy_drop_pp first exceeds threshold_pp, worst-case tie-broken. `df` must
    have columns: family, severity, accuracy_drop_pp, level."""
    sub = df[df["family"] == family]
    exceeding = sub[sub["accuracy_drop_pp"] > threshold_pp]
    if len(exceeding) == 0:
        return None
    min_severity = exceeding["severity"].min()
    at_min_severity = exceeding[exceeding["severity"] == min_severity]
    # Multiple levels can share the mildest severity (e.g. shift-10px/shift+10px both
    # have severity=10). pandas sort_values is NOT guaranteed stable on ties, so
    # picking .iloc[0] after a plain sort is non-deterministic. Report the WORST
    # outcome among tied-severity levels instead -- deterministic, and doesn't
    # flatter the robustness claim.
    return at_min_severity.loc[at_min_severity["accuracy_drop_pp"].idxmax()]


def describe_breaking_point(bp, family_label, baseline_acc_pct):
    if bp is None:
        return f"no tested {family_label} magnitude dropped accuracy by more than 2pp from baseline"
    return (f"{family_label} breaking point is '{bp['level']}' "
            f"({bp['accuracy_drop_pp']:.2f}pp drop from baseline {baseline_acc_pct:.2f}%)")


def directional_asymmetry_note(df, family, neg_mask, pos_mask, neg_label, pos_label):
    """Compares worst-case drop between the two directions of a perturbation family
    (e.g. negative vs positive shift, or shrink vs grow scale) and flags it explicitly
    if one direction is substantially more damaging -- rather than letting a single
    "mildest breaking point" number hide that asymmetry."""
    neg_df = df[(df["family"] == family) & neg_mask]
    pos_df = df[(df["family"] == family) & pos_mask]
    if not (len(neg_df) and len(pos_df)):
        return "", None, None
    neg_worst = neg_df["accuracy_drop_pp"].max()
    pos_worst = pos_df["accuracy_drop_pp"].max()
    note = ""
    if abs(neg_worst - pos_worst) > 2.0:
        worse_label = neg_label if neg_worst > pos_worst else pos_label
        note = (
            f" {family.capitalize()} direction is NOT symmetric: {worse_label} is substantially "
            f"more damaging (worst-case drop {max(neg_worst, pos_worst):.2f}pp vs "
            f"{min(neg_worst, pos_worst):.2f}pp for the opposite direction)."
        )
    return note, neg_worst, pos_worst


# ---------------------------------------------------------------------------
# Image-corruption perturbations (Task 4: clinical-realism robustness)
# ---------------------------------------------------------------------------


def perturb_gaussian_noise(img, sigma):
    noisy = img + np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def perturb_motion_blur(img, ksize):
    if ksize <= 1:
        return img
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def perturb_jpeg(img, quality):
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", (img * 255).astype(np.uint8), enc_param)
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return dec.astype(np.float32) / 255.0


def perturb_brightness(img, delta):
    return np.clip(img + delta, 0.0, 1.0)


def perturb_contrast(img, factor):
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0.0, 1.0)


def perturb_incomplete_lung_field(raw_img, crop_frac_top):
    """Crops out the top crop_frac_top of the RAW image BEFORE any segmentation runs,
    then resizes back to IMG_SIZE so the crop pipeline itself has to cope with the loss."""
    h = raw_img.shape[0]
    cut = int(h * crop_frac_top)
    cropped = raw_img[cut:, :, :]
    return cv2.resize(cropped, IMG_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32)


PERTURBATIONS = {
    "gaussian_noise": {"fn": perturb_gaussian_noise, "severities": [0.02, 0.05, 0.10], "stage": "post"},
    "motion_blur": {"fn": perturb_motion_blur, "severities": [3, 7, 15], "stage": "post"},
    "jpeg_compression": {"fn": perturb_jpeg, "severities": [50, 20, 5], "stage": "post"},
    "brightness": {"fn": perturb_brightness, "severities": [-0.10, 0.10], "stage": "post"},
    "contrast": {"fn": perturb_contrast, "severities": [0.7, 1.3], "stage": "post"},
    "incomplete_lung_field": {"fn": perturb_incomplete_lung_field, "severities": [0.10, 0.25, 0.40], "stage": "pre"},
}


def evaluate_perturbation(model, files, labels, pert_name, severity):
    """Evaluates `model` with PERTURBATIONS[pert_name] applied at the given severity.
    'pre'-stage perturbations are applied to the raw image before segmentation/crop
    (so the crop pipeline has to cope with the corrupted input); 'post'-stage
    perturbations are applied identically to both branches after crop/resize."""
    spec = PERTURBATIONS[pert_name]
    yt, yp = [], []
    for fp, lbl in zip(files, labels):
        raw = load_raw_resized(fp)
        if spec["stage"] == "pre":
            img_global = spec["fn"](raw, severity)
            img_local = crop_to_lung_np(img_global)
        else:
            perturbed = spec["fn"](raw, severity)
            img_global = perturbed
            img_local = cv2.resize(perturbed, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        preds = model.predict([img_global[None], img_local[None]], verbose=0)[0]
        yp.append(int(np.argmax(preds)))
        yt.append(int(lbl))
    return macro_metrics(np.array(yt), np.array(yp))
