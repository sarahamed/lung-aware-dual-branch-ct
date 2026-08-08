"""
Explainability (XAI) methods: Grad-CAM, Grad-CAM++, SmoothGrad, Integrated
Gradients, Occlusion Sensitivity, plus the Lung Focus Score and cross-method
agreement metrics used to statistically compare them. Extracted from the
verified pipeline (revision_full_pipeline.ipynb Sections B6-B8).
"""
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from skimage.transform import resize as sk_resize

from config import IMG_SIZE, SEED
from src.preprocessing import window_ct

# Occlusion patch/stride settings used throughout the manuscript's XAI analysis --
# kept fixed rather than exposed as free parameters (Task 5 spec requirement).
OCCLUSION_PATCH = 24
OCCLUSION_STRIDE = 16


def lung_focus_score(heatmap, mask, eps=1e-8):
    """Fraction of (non-negative) heatmap energy that falls inside the lung mask."""
    heatmap = np.maximum(heatmap.astype(np.float32), 0.0)
    mask = mask.astype(np.float32)
    denom = heatmap.sum() + eps
    return float((heatmap * mask).sum() / denom)


def resize_heatmap_to_input(hm, size=IMG_SIZE):
    hm = np.asarray(hm, dtype=np.float32)
    if hm.shape[:2] == size:
        return hm
    return sk_resize(hm, size, order=1, mode="edge", anti_aliasing=True).astype(np.float32)


def show_overlay(ct_img_rgb, heatmap, title="", alpha=0.4):
    import matplotlib.pyplot as plt
    ct = window_ct(ct_img_rgb)
    plt.figure(figsize=(5, 5))
    plt.imshow(ct, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.title(title)
    plt.show()


def _grad_model(model, layer_name):
    return tf.keras.Model(inputs=model.inputs,
                           outputs=[model.get_layer(layer_name).output, model.output])


def _call_grad_model(gm, inputs, training=False):
    """Calls gm(inputs) and returns (conv_out, preds) as plain Tensors.

    tf.nest.flatten guards against Keras (esp. Keras 3) sometimes wrapping a
    multi-output functional model's call result in extra list/tuple nesting rather
    than a flat 2-element list -- direct `a, b = gm(inputs)` unpacking then silently
    assigns a nested list to `b` instead of a Tensor, which fails downstream with a
    confusing "list indices must be integers or slices, not tuple" on `preds[:, c]`.
    """
    outputs = gm(inputs, training=training)
    flat = tf.nest.flatten(outputs)
    return flat[0], flat[1]


def gradcam(model, inputs, class_idx, layer_name):
    gm = _grad_model(model, layer_name)
    with tf.GradientTape() as tape:
        conv_out, preds = _call_grad_model(gm, inputs, training=False)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam.numpy()[0]


def gradcam_pp(model, inputs, class_idx, layer_name):
    """Grad-CAM++: second/third-order-weighted variant, better for multiple/overlapping
    salient regions than vanilla Grad-CAM."""
    gm = _grad_model(model, layer_name)
    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_out, preds = _call_grad_model(gm, inputs, training=False)
                loss = preds[:, class_idx]
            grads = tape1.gradient(loss, conv_out)
        grads2 = tape2.gradient(grads, conv_out)
    grads3 = tape3.gradient(grads2, conv_out)

    grads = grads[0].numpy()
    grads2 = grads2[0].numpy() if grads2 is not None else np.zeros_like(grads)
    grads3 = grads3[0].numpy() if grads3 is not None else np.zeros_like(grads)
    conv_out = conv_out[0].numpy()

    num = grads2
    denom = 2.0 * grads2 + np.sum(conv_out, axis=(0, 1), keepdims=True) * grads3
    denom = np.where(np.abs(denom) > 1e-8, denom, 1e-8)
    alpha = num / denom
    weights = np.sum(alpha * np.maximum(grads, 0.0), axis=(0, 1))

    cam = np.sum(conv_out * weights[None, None, :], axis=-1)
    cam = np.maximum(cam, 0.0)
    cam = cam / (cam.max() + 1e-8)
    return cam


def smoothgrad(model, inputs, class_idx, branch_index=0, n_samples=25, noise_sigma=0.15, seed=SEED):
    """SmoothGrad: average input-space gradient over Gaussian-noised copies of the
    chosen branch input (0=global, 1=local). Returns a [H,W] saliency map."""
    rng = np.random.RandomState(seed)
    base = inputs[branch_index][0].numpy().astype(np.float32)
    accum = np.zeros(base.shape, dtype=np.float32)

    for _ in range(n_samples):
        noise = rng.normal(0.0, noise_sigma, size=base.shape).astype(np.float32)
        noisy = np.clip(base + noise, 0.0, 1.0)
        noisy_t = tf.convert_to_tensor(noisy[None, ...])
        call_inputs = list(inputs)
        call_inputs[branch_index] = noisy_t
        with tf.GradientTape() as tape:
            tape.watch(noisy_t)
            call_inputs[branch_index] = noisy_t
            preds = model(call_inputs, training=False)
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, noisy_t)[0].numpy()
        accum += np.abs(grads)

    sal = accum / n_samples
    sal = np.mean(sal, axis=-1)
    sal = np.maximum(sal, 0.0)
    sal = sal / (sal.max() + 1e-8)
    return sal


def integrated_gradients(model, xg, xm, class_idx, m_steps=32, baseline=None):
    xg = xg.astype(np.float32)
    xm = xm.astype(np.float32)
    if baseline is None:
        baseline = np.zeros_like(xg, dtype=np.float32)
    alphas = np.linspace(0.0, 1.0, m_steps).astype(np.float32)
    grads_accum = np.zeros_like(xg, dtype=np.float32)
    for a in alphas:
        xg_a = baseline + a * (xg - baseline)
        xm_a = baseline + a * (xm - baseline)
        xg_t = tf.convert_to_tensor(xg_a[None, ...])
        xm_t = tf.convert_to_tensor(xm_a[None, ...])
        with tf.GradientTape() as tape:
            tape.watch(xg_t)
            preds = model([xg_t, xm_t], training=False)
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, xg_t)[0].numpy()
        grads_accum += grads
    avg_grads = grads_accum / float(m_steps)
    ig = (xg - baseline) * avg_grads
    heatmap = np.mean(np.abs(ig), axis=-1)
    heatmap = np.maximum(heatmap, 0.0)
    return heatmap / (heatmap.max() + 1e-8)


def occlusion_sensitivity(model, xg, xm, class_idx, patch=OCCLUSION_PATCH, stride=OCCLUSION_STRIDE):
    """Occlusion-sensitivity heatmap."""
    h, w = xg.shape[:2]
    base_pred = model.predict([xg[None], xm[None]], verbose=0)[0, class_idx]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            xg_occ = xg.copy()
            xg_occ[y:y + patch, x:x + patch, :] = 0.0
            xm_occ = xm.copy()
            xm_occ[y:y + patch, x:x + patch, :] = 0.0
            p = model.predict([xg_occ[None], xm_occ[None]], verbose=0)[0, class_idx]
            drop = base_pred - p
            heatmap[y:y + patch, x:x + patch] = max(heatmap[y:y + patch, x:x + patch].max(), drop)
    heatmap = np.maximum(heatmap, 0.0)
    return heatmap / (heatmap.max() + 1e-8)


def agreement_metrics(map_a, map_b, lung_mask, top_frac=0.10):
    """Pearson, Spearman, and IoU@top-k% between two attribution maps, restricted to
    pixels inside the lung mask."""
    m = lung_mask.astype(bool)
    a = map_a[m].astype(np.float64)
    b = map_b[m].astype(np.float64)

    if a.std() < 1e-8 or b.std() < 1e-8 or len(a) < 3:
        pearson_r, spearman_r = np.nan, np.nan
    else:
        pearson_r, _ = pearsonr(a, b)
        spearman_r, _ = spearmanr(a, b)

    k = max(1, int(round(top_frac * len(a))))
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    iou = inter / union if union > 0 else np.nan

    return {"pearson": pearson_r, "spearman": spearman_r, f"iou_top{int(top_frac * 100)}pct": iou}
