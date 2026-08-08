"""
Lung-ROI segmentation, cropping, and dataset-building utilities.

Extracted from the manuscript's verified pipeline (revision_full_pipeline.ipynb,
Sections B and C2/C3). Behavior is unchanged from the notebook versions; only
the config constants (IMG_SIZE, BATCH_SIZE, NUM_CLASSES, SEED, CLASS_FOLDERS)
are imported from config.py instead of relying on notebook globals.
"""
import glob
import hashlib
import inspect
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import BATCH_SIZE, CLASS_FOLDERS, CLASS_NAMES, IMG_SIZE, NUM_CLASSES, SEED

AUTOTUNE = tf.data.AUTOTUNE


def lung_roi_mask(image_01):
    """image_01: float32 [0,1], shape [H,W,3] -> mask [H,W,1] in {0,1}"""
    gray = tf.image.rgb_to_grayscale(image_01)
    mean = tf.reduce_mean(gray)
    std = tf.math.reduce_std(gray)
    thresh = mean - 0.2 * std

    raw = tf.cast(gray < thresh, tf.float32)
    raw = tf.nn.avg_pool2d(raw[None, ...], ksize=5, strides=1, padding="SAME")[0]
    raw = tf.cast(raw > 0.3, tf.float32)

    raw2 = tf.nn.avg_pool2d(raw[None, ...], ksize=9, strides=1, padding="SAME")[0]
    mask = tf.cast(raw2 > 0.2, tf.float32)
    return mask


def clahe_lung_np(img, mask):
    """img: float32 [H,W,3] in [0,1]; mask: float32 [H,W,1] in {0,1}"""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mask_u8 = (mask[..., 0] > 0).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray).astype(np.float32) / 255.0

    out = gray.astype(np.float32) / 255.0
    out[mask_u8 == 1] = eq[mask_u8 == 1]
    return np.stack([out, out, out], axis=-1).astype(np.float32)


def lung_mask_np(img_rgb_01, threshold=0.45, kernel_size=9, keep_top_k=2,
                  smooth_kernel=21, smooth_threshold=0.2,
                  return_diagnostics=False):
    """
    img_rgb_01: float32 [H,W,3] in [0,1]
    threshold : intensity threshold (percentile-normalized) below which a pixel is
                considered candidate lung tissue. Lower threshold => smaller mask.
    kernel_size: morphological close/open kernel size (odd int).
    keep_top_k : number of largest connected components retained as "lung" regions
                 (design choice: top-2, since each CT slice typically shows two lung fields).
    smooth_kernel   : Gaussian blur kernel size (odd int) applied to the selected mask
                       to soften its boundary, matching manuscript Section 3.2.
    smooth_threshold: re-binarization threshold applied after the Gaussian blur.
    returns mask [H,W] float32 in {0,1} (and, if requested, a diagnostics dict with the
    raw connected-component count before top-k filtering).
    """
    gray = img_rgb_01.mean(axis=-1)

    lo, hi = np.percentile(gray, [5, 95])
    g = (gray - lo) / (hi - lo + 1e-8)
    g = np.clip(g, 0, 1)

    raw = (g < threshold).astype(np.uint8)

    raw = cv2.medianBlur(raw * 255, 7)
    raw = (raw > 0).astype(np.uint8)

    k = max(3, kernel_size | 1)  # force odd, >=3
    kernel = np.ones((k, k), np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    n_components = max(0, num - 1)  # exclude background label 0

    if num <= 1:
        mask = raw.astype(np.float32)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.argsort(areas)[-keep_top_k:]
        mask = np.zeros_like(raw, dtype=np.uint8)
        for k_idx in keep:
            mask[labels == (k_idx + 1)] = 1
        mask = mask.astype(np.float32)

    # Boundary-softening step (manuscript Section 3.2): Gaussian-blur the selected
    # mask, then re-threshold, instead of leaving the hard component-selection edges.
    sk = max(3, smooth_kernel | 1)  # force odd, >=3
    mask_blur = cv2.GaussianBlur(mask, (sk, sk), 0)
    mask = (mask_blur > smooth_threshold).astype(np.float32)

    if return_diagnostics:
        return mask, {"n_components": n_components}
    return mask


def crop_to_lung_np(img_rgb_01, threshold_shift=0.0, kernel_size_delta=0,
                     erode_dilate_px=0, base_threshold=0.45, base_kernel=9,
                     max_frac=0.80):
    """
    Verbatim reproduction of the original crop_to_lung_np (manuscript Section 3.2).
    Crop geometry:
      - centered on the BOUNDING-BOX MIDPOINT ((y0+y1)//2, (x0+x1)//2) -- NOT the
        pixel-weighted mask centroid;
      - a FIXED window of max_frac (80%) of image height/width, via int() truncation
        -- NOT adaptively capped to the mask's own bbox size, so even a small lung
        mask still gets an 80%-sized crop around its bbox midpoint;
      - near image edges the window is clipped (and can shrink) via max(0,...)/
        min(h,...) -- it is NOT re-centered/shifted to preserve size.
    Extended with controllable perturbation knobs (all default to 0/base values,
    which reproduces the original exactly): threshold_shift, kernel_size_delta,
    erode_dilate_px.
    """
    h, w = img_rgb_01.shape[:2]
    thresh = float(np.clip(base_threshold + threshold_shift, 0.05, 0.95))
    ksize = max(3, (base_kernel + kernel_size_delta) | 1)

    mask = lung_mask_np(img_rgb_01, threshold=thresh, kernel_size=ksize)

    if erode_dilate_px != 0:
        px = int(abs(erode_dilate_px))
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        mask_u8 = (mask > 0).astype(np.uint8)
        if erode_dilate_px > 0:
            mask_u8 = cv2.dilate(mask_u8, struct, iterations=1)
        else:
            mask_u8 = cv2.erode(mask_u8, struct, iterations=1)
        mask = mask_u8.astype(np.float32)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        # degenerate mask: fall back to full-frame crop
        crop = img_rgb_01
    else:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        max_h = int(max_frac * h)
        max_w = int(max_frac * w)

        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2

        y0c = max(0, cy - max_h // 2)
        y1c = min(h, cy + max_h // 2)
        x0c = max(0, cx - max_w // 2)
        x1c = min(w, cx + max_w // 2)

        crop = img_rgb_01[y0c:y1c, x0c:x1c, :]

    crop = cv2.resize(crop.astype(np.float32), IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return crop.astype(np.float32)


def crop_to_lung_np_window_perturbed(img_rgb_01, window_shift_px=0, window_scale=1.0,
                                      base_threshold=0.45, base_kernel=9, max_frac=0.80):
    """
    Direct crop-WINDOW perturbation -- a deliberately SEPARATE function from
    crop_to_lung_np, used for robustness testing of the local branch's crop window:
    symmetric mask perturbations (threshold shift, kernel change, erosion, dilation)
    barely move crop_to_lung_np's bbox-midpoint-centered window, so that crop is
    invariant to them by construction. This function instead computes the SAME
    unperturbed mask/bbox-midpoint as crop_to_lung_np, then perturbs the FINAL
    WINDOW directly:
      window_shift_px : shifts the window center by this many px (same offset on both
                         axes) -- simulates the crop landing in the wrong place, as a
                         real localization/segmentation failure would.
      window_scale     : scales the window size by this factor around its (shifted)
                         center -- simulates the crop being too tight (over-cropped,
                         <1.0) or too loose (under-cropped, >1.0).

    At window_shift_px=0, window_scale=1.0 this delegates straight to
    crop_to_lung_np (rather than re-deriving the same crop with this function's own
    shift-to-fit edge-clamping, whose edge behavior differs from crop_to_lung_np's
    plain-clip for a handful of edge-of-image bbox cases).
    """
    if window_shift_px == 0 and window_scale == 1.0:
        return crop_to_lung_np(img_rgb_01, base_threshold=base_threshold,
                                base_kernel=base_kernel, max_frac=max_frac)

    h, w = img_rgb_01.shape[:2]
    mask = lung_mask_np(img_rgb_01, threshold=base_threshold, kernel_size=base_kernel)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        crop = img_rgb_01
    else:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        cy = (y0 + y1) // 2 + window_shift_px
        cx = (x0 + x1) // 2 + window_shift_px

        win_h = max(1, int(round(max_frac * h * window_scale)))
        win_w = max(1, int(round(max_frac * w * window_scale)))

        y0c = cy - win_h // 2
        y1c = y0c + win_h
        x0c = cx - win_w // 2
        x1c = x0c + win_w

        # Shift (not shrink) back inside bounds so the requested window size is
        # preserved as closely as possible. Independent of crop_to_lung_np, so this
        # does not affect that verified-original behavior.
        if y0c < 0: y1c -= y0c; y0c = 0
        if x0c < 0: x1c -= x0c; x0c = 0
        if y1c > h: y0c -= (y1c - h); y1c = h
        if x1c > w: x0c -= (x1c - w); x1c = w
        y0c = max(0, y0c); x0c = max(0, x0c)
        y1c = min(h, y1c); x1c = min(w, x1c)

        crop = img_rgb_01[y0c:y1c, x0c:x1c, :]

    crop = cv2.resize(crop.astype(np.float32), IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return crop.astype(np.float32)


def crop_to_lung_tf(img, threshold_shift=0.0, kernel_size_delta=0, erode_dilate_px=0):
    """tf.numpy_function wrapper around crop_to_lung_np for use inside tf.data pipelines."""
    def _fn(x):
        return crop_to_lung_np(
            x.numpy(),
            threshold_shift=float(threshold_shift),
            kernel_size_delta=int(kernel_size_delta),
            erode_dilate_px=float(erode_dilate_px),
        )
    out = tf.py_function(_fn, [img], tf.float32)
    out.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return out


def preprocessing_signature():
    """Hash of the preprocessing functions + key config, used as a checkpoint-
    reproduction diagnostic (detects accidental preprocessing drift)."""
    src = "".join([
        inspect.getsource(lung_mask_np),
        inspect.getsource(crop_to_lung_np),
        inspect.getsource(lung_roi_mask),
        str(IMG_SIZE), str(BATCH_SIZE), str(CLASS_NAMES),
    ])
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def load_raw_resized(path):
    """Decode + resize a file path to IMG_SIZE, float32 in [0,1], as a plain NumPy
    array. Used for eager (non-tf.data) per-image evaluation."""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return (tf.cast(img, tf.float32) / 255.0).numpy()


def window_ct(img, p_low=1, p_high=99):
    if img.ndim == 3:
        img = img.mean(axis=-1)
    lo, hi = np.percentile(img, [p_low, p_high])
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo + 1e-8)


def macro_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """Returns dict: accuracy, macro_precision, macro_recall, macro_f1, per-class recall."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0,
                                     labels=list(range(len(class_names))))
    return {
        "accuracy": acc, "macro_precision": prec, "macro_recall": rec, "macro_f1": f1,
        **{f"recall_{cn}": r for cn, r in zip(class_names, per_class_recall)}
    }


def decode_and_preprocess_train(path, label, training=False,
                                 local_threshold_shift=0.0, local_kernel_delta=0,
                                 local_erode_dilate_px=0.0,
                                 perturb_global=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    img_global = img if perturb_global is None else perturb_global(img)
    img_local = crop_to_lung_tf(img_global, threshold_shift=local_threshold_shift,
                                 kernel_size_delta=local_kernel_delta,
                                 erode_dilate_px=local_erode_dilate_px)

    if training:
        img_global = tf.image.random_flip_left_right(img_global)
        img_local = tf.image.random_flip_left_right(img_local)
        img_global = tf.image.random_brightness(img_global, 0.05)
        img_local = tf.image.random_brightness(img_local, 0.05)
        img_global = tf.image.random_contrast(img_global, 0.95, 1.05)
        img_local = tf.image.random_contrast(img_local, 0.95, 1.05)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img_global = tf.image.rot90(img_global, k)
        img_local = tf.image.rot90(img_local, k)

    y = tf.one_hot(label, NUM_CLASSES)
    return (img_global, img_local), y


def make_dataset(files, labels, training=False, batch_size=BATCH_SIZE,
                  local_threshold_shift=0.0, local_kernel_delta=0, local_erode_dilate_px=0.0):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(files), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: decode_and_preprocess_train(
        p, y, training=training,
        local_threshold_shift=local_threshold_shift,
        local_kernel_delta=local_kernel_delta,
        local_erode_dilate_px=local_erode_dilate_px,
    ), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def list_images_and_labels(root_dir, class_folders=CLASS_FOLDERS):
    files, labels = [], []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    for i, cname in enumerate(class_folders):
        cdir = os.path.join(root_dir, cname)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"Missing folder: {cdir}")
        cfiles = sorted(sum((glob.glob(os.path.join(cdir, ext)) for ext in exts), []))
        files.extend(cfiles)
        labels.extend([i] * len(cfiles))
    return np.array(files), np.array(labels)


def download_dataset():
    """Downloads/locates the IQ-OTHNCCD dataset via kagglehub. Requires Kaggle API
    credentials configured (see data/README.md)."""
    import kagglehub
    return kagglehub.dataset_download("hamdallak/the-iqothnccd-lung-cancer-dataset")


def find_data_root(base_path, class_names=CLASS_FOLDERS):
    """Locates the directory (possibly nested under base_path) that directly
    contains all of class_names as subfolders."""
    if all(os.path.isdir(os.path.join(base_path, c)) for c in class_names):
        return base_path
    for item in os.listdir(base_path):
        candidate = os.path.join(base_path, item)
        if os.path.isdir(candidate) and all(os.path.isdir(os.path.join(candidate, c)) for c in class_names):
            return candidate
    for item in os.listdir(base_path):
        candidate = os.path.join(base_path, item)
        if not os.path.isdir(candidate):
            continue
        for item2 in os.listdir(candidate):
            candidate2 = os.path.join(candidate, item2)
            if os.path.isdir(candidate2) and all(os.path.isdir(os.path.join(candidate2, c)) for c in class_names):
                return candidate2
    raise FileNotFoundError("Could not find a DATA_ROOT that contains the class folders.")


def remap_to_data_root(stored_path, data_root, class_folders=CLASS_FOLDERS):
    """Re-anchors a path stored in splits.json onto the current DATA_ROOT, matching
    on '<class folder>/<filename>' rather than trusting the stored absolute path's
    directory prefix (which will differ across machines/sessions)."""
    norm = stored_path.replace("\\", "/")
    parts = norm.split("/")
    filename = parts[-1]
    class_folder = parts[-2] if len(parts) >= 2 else None
    if class_folder not in class_folders:
        raise ValueError(f"Could not identify a known class folder in stored path: {stored_path}")
    return os.path.join(data_root, class_folder, filename)


def remap_file_list(paths, data_root, class_folders=CLASS_FOLDERS):
    remapped = [remap_to_data_root(p, data_root, class_folders) for p in paths]
    missing = [p for p in remapped if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(remapped)} remapped files were not found under DATA_ROOT "
            f"(first missing: {missing[0]}). The downloaded dataset may not match the one used "
            f"to produce this checkpoint's split -- do not silently proceed."
        )
    return np.array(remapped)


def infer_label(path, class_folders=CLASS_FOLDERS):
    for i, cname in enumerate(class_folders):
        if path.replace("\\", "/").split("/")[-2] == cname:
            return i
    raise ValueError(f"Could not infer class label for {path}")
