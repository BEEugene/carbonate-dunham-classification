"""Grad-CAM heatmaps for the discovered Keras model.

Adapted from the original `AutoDL core carbonate.py` reference. Walks the
exported model in reverse to find the last 4-D `Conv2D` layer, computes
the gradient of the predicted class w.r.t. that layer's output, and
overlays the resulting heatmap on the input image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No 2D Conv2D layer found in the model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]).numpy())
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    base = img if img.max() > 1.0 else img * 255
    return np.clip(jet * alpha + base, 0, 255).astype("uint8")


def render_gradcam_for_samples(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
    num_samples: int = 3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    last_conv_layer = find_last_conv_layer(model)

    images: list[np.ndarray] = []
    labels: list[int] = []
    for img_batch, label_batch in test_ds.take(1):
        images.extend(img_batch.numpy())
        labels.extend(label_batch.numpy())
    if not images:
        return
    n = min(num_samples, len(images))

    plt.figure(figsize=(14, 4 * n))
    for i in range(n):
        img = images[i].astype("uint8")
        true_label = class_names[int(labels[i])]
        batch = np.expand_dims(images[i], axis=0)
        preds = model.predict(batch, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        pred_label = class_names[pred_idx]
        confidence = float(preds[0][pred_idx])

        heatmap = make_gradcam_heatmap(batch, model, last_conv_layer, pred_index=pred_idx)
        overlay = overlay_heatmap(img, heatmap)

        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}")
        plt.axis("off")

        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Grad-CAM activation")
        plt.axis("off")

        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(overlay)
        plt.title(f"Pred: {pred_label} ({confidence:.2%})")
        plt.axis("off")

        fig_path = output_dir / f"sample_{i}_{pred_label}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
    plt.close("all")
