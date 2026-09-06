"""Evaluation: per-split classification reports + confusion matrices."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from autogeo.model import build_classifier


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    images: list[np.ndarray] = []
    labels: list[int] = []
    probs: list[np.ndarray] = []
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        images.append(x.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        probs.append(p)
    return (
        np.concatenate(images),
        np.concatenate(labels),
        np.concatenate(probs),
    )


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
    model: torch.nn.Module,
    loader: DataLoader,
    classes: list[str],
    split_name: str,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run the model over a DataLoader, write report + confusion matrix, return metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _, y_true, probs = _collect_predictions(model, loader, device)
    y_pred = probs.argmax(axis=1)

    report_text = classification_report(
        y_true, y_pred, target_names=classes, digits=3, zero_division=0
    )
    (output_dir / f"classification_report_{split_name}.txt").write_text(
        report_text + "\n", encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    _plot_confusion_matrix(
        cm, classes, title=f"Confusion matrix — {split_name}", out_path=output_dir / f"confusion_matrix_{split_name}.png"
    )
    return {
        "split": split_name,
        "support": int(len(y_true)),
        "report": report_text,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_all_splits(
    *,
    state_dict: dict,
    model_name: str,
    num_classes: int,
    splits: dict[str, DataLoader],
    classes: list[str],
    output_dir: str | Path,
    device: str | torch.device = "cpu",
) -> dict:
    """Reload `state_dict` into a fresh model and report on train/val/test."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model = build_classifier(model_name=model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state_dict)

    results: dict = {"classes": classes, "splits": {}}
    for name, loader in splits.items():
        results["splits"][name] = evaluate_split(
            model, loader, classes, name, output_dir, device
        )
    return results
