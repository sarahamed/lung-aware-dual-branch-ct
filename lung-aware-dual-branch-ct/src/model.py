"""
Model architecture, callbacks, and optimizer for the lung-aware dual-branch
CT classifier. Extracted from the verified training pipeline
(first_part_ofcode.ipynb / revision_full_pipeline.ipynb Section C1).
"""
import os
import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from config import IMG_SIZE, NUM_CLASSES


def build_model_A(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    """Dual-branch (global + lung-crop local) EfficientNetV2-S backbone with
    bidirectional cross-attention fusion. This is the manuscript's primary model."""
    inp_global = layers.Input(shape=(*img_size, 3), name="global_ct")
    inp_masked = layers.Input(shape=(*img_size, 3), name="masked_ct")

    backbone = tf.keras.applications.EfficientNetV2S(
        include_top=False, weights="imagenet", input_shape=(*img_size, 3)
    )
    backbone.trainable = False

    f_global = backbone(inp_global)
    f_global = layers.Lambda(lambda x: x, name="feat_global")(f_global)
    f_local = backbone(inp_masked)
    f_local = layers.Lambda(lambda x: x, name="feat_local")(f_local)

    def to_tokens(x, proj_dim=256, name_prefix="tok"):
        x = layers.Conv2D(proj_dim, 1, padding="same", name=f"{name_prefix}_proj")(x)
        x = layers.Reshape((-1, proj_dim), name=f"{name_prefix}_reshape")(x)
        return x

    t_global = to_tokens(f_global, 256, name_prefix="global")
    t_local = to_tokens(f_local, 256, name_prefix="local")

    attn1 = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1, name="xattn_gq_lkv")(
        query=t_global, value=t_local, key=t_local)
    t_fused_g = layers.LayerNormalization(name="fused_g_ln")(layers.Add(name="fused_g_add")([t_global, attn1]))

    attn2 = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1, name="xattn_lq_gkv")(
        query=t_local, value=t_global, key=t_global)
    t_fused_l = layers.LayerNormalization(name="fused_l_ln")(layers.Add(name="fused_l_add")([t_local, attn2]))

    g_pool = layers.GlobalAveragePooling1D(name="g_pool")(t_fused_g)
    l_pool = layers.GlobalAveragePooling1D(name="l_pool")(t_fused_l)

    x = layers.Concatenate(name="fusion_concat")([g_pool, l_pool])
    x = layers.Dense(256, activation="relu", name="fusion_dense1")(x)
    x = layers.Dropout(0.3, name="fusion_dropout1")(x)
    x = layers.Dense(64, activation="relu", name="fusion_dense2")(x)
    x = layers.Dropout(0.2, name="fusion_dropout2")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=[inp_global, inp_masked], outputs=out, name="ModelA_LungROI_CrossAttn")


def get_backbone_layer(model):
    """Returns the shared EfficientNetV2S sub-model inside a model built by
    build_model_A / build_single_branch_ablation, so callers can toggle its
    .trainable / per-layer trainability for progressive unfreezing without
    needing build_model_A to separately return it."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone sub-model found in this model.")


def build_single_branch_ablation(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    """Ablation baseline: global image only, no lung-crop branch, no cross-attention.
    Same backbone (EfficientNetV2-S) for a fair comparison against build_model_A."""
    inp_global = layers.Input(shape=(*img_size, 3), name="global_ct")
    backbone = tf.keras.applications.EfficientNetV2S(
        include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    backbone.trainable = False
    x = backbone(inp_global)
    x = layers.GlobalAveragePooling2D(name="g_pool")(x)
    x = layers.Dense(256, activation="relu", name="fusion_dense1")(x)
    x = layers.Dropout(0.3, name="fusion_dropout1")(x)
    x = layers.Dense(64, activation="relu", name="fusion_dense2")(x)
    x = layers.Dropout(0.2, name="fusion_dropout2")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs=inp_global, outputs=out, name="SingleBranch_Ablation")


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_start)

    def on_train_end(self, logs=None):
        self.total_time = time.time() - self.train_start


def make_callbacks(time_cb, ckpt_dir, log_dir):
    ckpt_path = os.path.join(ckpt_dir, "best_model.keras")
    return [
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        TensorBoard(log_dir=log_dir),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        time_cb,
    ]


def make_loss():
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)


def make_optimizer(lr):
    return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
