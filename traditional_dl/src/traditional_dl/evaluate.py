"""Evaluation: per-split classification reports and confusion matrices."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def _collect_predictions(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    images: list[np.ndarray] = []
    for img_batch, label_batch in ds:
        y_true.append(label_batch.numpy())
        images.append(img_batch.numpy())
    X = np.concatenate(images, axis=0) if images else np.empty((0,))
    y = np.concatenate(y_true, axis=0) if y_true else np.empty((0,), dtype=np.int64)
    y_pred = model.predict(X, verbose=0).argmax(axis=1)
    return y, y_pred


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.9), max(5, len(class_names) * 0.8)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_split(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    class_names: list[str],
    split_name: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = _collect_predictions(model, ds)
    if len(y_true) == 0:
        return {"split": split_name, "support": 0, "report": "", "confusion_matrix": []}
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )
    (output_dir / f"classification_report_{split_name}.txt").write_text(
        report_text + "\n", encoding="utf-8"
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    _plot_confusion_matrix(
        cm, class_names,
        title=f"Confusion matrix — {split_name}",
        out_path=output_dir / f"confusion_matrix_{split_name}.png",
    )
    return {
        "split": split_name,
        "support": int(len(y_true)),
        "report": report_text,
        "confusion_matrix": cm.tolist(),
    }
