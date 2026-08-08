"""
Checkpoint evaluation and metrics-reporting utilities. Extracted from the
verified pipeline (revision_full_pipeline.ipynb Section C, first_part_ofcode.ipynb
training-curve cells). Re-exports macro_metrics from preprocessing.py so callers
only need `from src import evaluate`.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from src.preprocessing import macro_metrics  # noqa: F401 (re-exported)

__all__ = ["macro_metrics", "predict_on_dataset", "evaluate_checkpoint_on",
           "get_curve", "summarize_stage", "plot_confusion_matrix"]


def predict_on_dataset(model, ds):
    """Runs a dual-input (global, local) model over a batched tf.data Dataset of
    ((img_global, img_local), one_hot_label) and returns (y_true, y_pred) as
    integer class-index arrays."""
    y_true, y_pred = [], []
    for (xg, xl), y in ds:
        preds = model.predict([xg, xl], verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(np.argmax(y.numpy(), axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


def evaluate_checkpoint_on(model, ds):
    """Returns percentage accuracy of `model` on a batched (global, local) dataset."""
    y_true, y_pred = predict_on_dataset(model, ds)
    return 100.0 * accuracy_score(y_true, y_pred)


def get_curve(hist, key):
    return np.asarray(hist.get(key, []), dtype=float)


def summarize_stage(name, hist, time_log):
    return {
        "Stage": name,
        "Epochs": len(hist["accuracy"]),
        "Final Train Acc": hist["accuracy"][-1],
        "Final Val Acc": hist["val_accuracy"][-1],
        "Final Train Loss": hist["loss"][-1],
        "Final Val Loss": hist["val_loss"][-1],
        "Total Time (min)": time_log["total_sec"] / 60,
        "Avg Time/Epoch (sec)": sum(time_log["epoch_times_sec"]) / len(time_log["epoch_times_sec"]),
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix",
                           out_path=None, normalize="true"):
    """Heatmap with count + percentage annotations (percentage normalized over
    `normalize` axis: 'true' = row-normalized/per-true-class, 'pred' = column-
    normalized/per-predicted-class, None = counts only)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    if normalize == "true":
        pct = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0) * 100
    elif normalize == "pred":
        pct = cm_df.div(cm_df.sum(axis=0), axis=1).fillna(0) * 100
    else:
        pct = None

    if pct is not None:
        annot = cm_df.astype(str) + "\n(" + pct.round(1).astype(str) + "%)"
        annot_values = annot.values
    else:
        annot_values = cm_df.values
        annot_values = None  # let seaborn use raw counts directly

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm_df, annot=annot_values if annot_values is not None else True,
                fmt="" if annot_values is not None else "d",
                cmap="Blues", cbar=True, ax=ax, annot_kws={"fontsize": 9})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    return fig, ax
