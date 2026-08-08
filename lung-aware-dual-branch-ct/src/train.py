"""
Training entry points: the primary 3-stage progressive-unfreeze protocol used
to produce the manuscript's reported checkpoint, and the multi-seed
stability/ablation training routine used in Task 6. Extracted from the
verified pipeline (first_part_ofcode.ipynb Cells 16-18, revision_full_pipeline.ipynb
Task 6).
"""
import gc
import json
import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import NUM_CLASSES
from src.evaluate import macro_metrics
from src.model import TimeHistory, get_backbone_layer, make_callbacks, make_loss, make_optimizer
from src.preprocessing import make_dataset

# 3-stage progressive-unfreeze schedule used for the manuscript's primary run:
# Stage 1 -- backbone fully frozen; Stage 2A -- top 10% unfrozen; Stage 2B -- top 40%
# unfrozen.
STAGE_LRS = [3e-4, 1e-4, 3e-5]
STAGE_EPOCHS = [25, 15, 20]
STAGE_UNFREEZE_FRACS = [0.0, 0.10, 0.40]  # fraction of backbone layers unfrozen from the top


def train_primary_model(model, train_ds, val_ds, ckpt_dir, log_dir, out_dir,
                         class_weight, stage_lrs=STAGE_LRS, stage_epochs=STAGE_EPOCHS,
                         stage_unfreeze_fracs=STAGE_UNFREEZE_FRACS):
    """Runs the manuscript's primary 3-stage training protocol on `model` (built via
    src.model.build_model_A), saving per-stage history/timing JSON under out_dir and
    checkpointing the best val_accuracy weights under ckpt_dir. Returns a list of
    Keras History objects, one per stage."""
    os.makedirs(out_dir, exist_ok=True)
    loss_fn = make_loss()
    backbone = get_backbone_layer(model)
    histories = []

    for stage_idx, (lr, epochs, unfreeze_frac) in enumerate(
            zip(stage_lrs, stage_epochs, stage_unfreeze_fracs), start=1):
        backbone.trainable = unfreeze_frac > 0.0
        if unfreeze_frac > 0.0:
            n = len(backbone.layers)
            unfreeze_from = int(n * (1.0 - unfreeze_frac))
            for i, layer in enumerate(backbone.layers):
                layer.trainable = (i >= unfreeze_from)

        model.compile(optimizer=make_optimizer(lr), loss=loss_fn, metrics=["accuracy"])

        time_cb = TimeHistory()
        callbacks = make_callbacks(time_cb, ckpt_dir, log_dir)

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                             callbacks=callbacks, class_weight=class_weight)

        stage_name = f"stage{stage_idx}" if stage_idx == 1 else f"stage{stage_idx}{'A' if unfreeze_frac < 0.4 else 'B'}"
        with open(os.path.join(out_dir, f"history_{stage_name}.json"), "w") as f:
            json.dump(history.history, f, indent=2)
        with open(os.path.join(out_dir, f"time_{stage_name}.json"), "w") as f:
            json.dump({"epoch_times_sec": time_cb.epoch_times, "total_sec": time_cb.total_time}, f, indent=2)
        print(f"{stage_name} total time (min): {time_cb.total_time / 60:.2f}")

        histories.append(history)

    return histories


def train_one_run(model_builder, model_tag, seed, multiseed_dir, train_files, train_labels,
                   val_files, val_labels, test_files, test_labels, multi_input=True,
                   batch_size=8, stage_lrs=STAGE_LRS, stage_epochs=STAGE_EPOCHS):
    """Trains one seeded run of `model_builder()` (e.g. build_model_A or
    build_single_branch_ablation) for multi-seed stability analysis. Uses a smaller
    batch_size than the primary training protocol to control peak memory, and
    explicitly clears the Keras backend session + forces garbage collection between
    stages and after returning -- TF/Keras does not reliably release memory when a
    model just goes out of scope in a loop, which otherwise crashes long multi-seed
    sweeps with an out-of-memory error partway through."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = os.path.join(multiseed_dir, f"{model_tag}_seed{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_model.keras")

    model = model_builder()
    loss_fn = make_loss()

    train_ds_r = make_dataset(train_files, train_labels, training=True, batch_size=batch_size)
    val_ds_r = make_dataset(val_files, val_labels, training=False, batch_size=batch_size)
    test_ds_r = make_dataset(test_files, test_labels, training=False, batch_size=batch_size)

    if not multi_input:
        train_ds_r = train_ds_r.map(lambda xs, y: (xs[0], y))
        val_ds_r = val_ds_r.map(lambda xs, y: (xs[0], y))
        test_ds_r = test_ds_r.map(lambda xs, y: (xs[0], y))

    cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    early_stop_log = []
    for stage_idx, (lr, epochs) in enumerate(zip(stage_lrs, stage_epochs), start=1):
        if stage_idx >= 2:
            # unfreeze backbone progressively (best-effort layer index)
            model.get_layer(index=1 if not multi_input else 2).trainable = True
        model.compile(optimizer=make_optimizer(lr), loss=loss_fn, metrics=["accuracy"])
        callbacks = [
            ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=0),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0),
        ]
        hist = model.fit(train_ds_r, validation_data=val_ds_r, epochs=epochs,
                          class_weight=class_weight, callbacks=callbacks, verbose=0)
        triggered = len(hist.epoch) < epochs
        early_stop_log.append({"stage": stage_idx, "epochs_ran": len(hist.epoch),
                                "epochs_budgeted": epochs, "early_stopped": triggered})
        with open(os.path.join(run_dir, f"history_stage{stage_idx}.json"), "w") as f:
            json.dump(hist.history, f)
        del hist, callbacks
        gc.collect()

    model.load_weights(ckpt_path)

    yt, yp, probs = [], [], []
    for batch in test_ds_r:
        xb, yb = batch
        pr = model.predict(xb, verbose=0)
        yp.extend(np.argmax(pr, axis=1).tolist())
        yt.extend(np.argmax(yb.numpy(), axis=1).tolist())
        probs.extend(pr.tolist())
    yt, yp, probs = np.array(yt), np.array(yp), np.array(probs)

    m = macro_metrics(yt, yp)
    try:
        auc = roc_auc_score(tf.one_hot(yt, NUM_CLASSES).numpy(), probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = np.nan
    m["mean_auc"] = auc

    with open(os.path.join(run_dir, "early_stopping_log.json"), "w") as f:
        json.dump(early_stop_log, f, indent=2)

    del model, train_ds_r, val_ds_r, test_ds_r
    tf.keras.backend.clear_session()
    gc.collect()

    return m, early_stop_log
