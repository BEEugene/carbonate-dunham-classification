"""Top-level training entry point for the AutoKeras NAS pipeline."""
from __future__ import annotations

import json
import random
from pathlib import Path

import autokeras as ak
import numpy as np
import tensorflow as tf

from traditional_dl.data import make_datasets
from traditional_dl.evaluate import evaluate_split
from traditional_dl.gradcam import render_gradcam_for_samples
from traditional_dl.nas import run_nas


def _plot_training_history(tuner: ak.AutoModel, output_dir: Path) -> None:
    """Save a basic loss/accuracy plot from the best trial's history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        best = tuner.tuner.get_best_models(num_models=1)[0]
        history = best.history.history
    except Exception:
        return
    if not history or "loss" not in history:
        return

    epochs = range(1, len(history["loss"]) + 1)
    acc_key = "accuracy" if "accuracy" in history else next(
        (k for k in history if k.endswith("accuracy") and not k.startswith("val_")), None
    )
    val_acc_key = f"val_{acc_key}" if acc_key else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="x", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    if acc_key and val_acc_key and val_acc_key in history:
        axes[1].plot(epochs, history[acc_key], marker="o", label="train")
        axes[1].plot(epochs, history[val_acc_key], marker="x", label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)


def run(
    *,
    data_dir: Path,
    output_dir: Path,
    image_size: int = 128,
    max_trials: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    grad_cam: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_ds, val_ds, test_ds, class_names = make_datasets(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    print(f"== AutoKeras NAS ==")
    print(f"data dir    : {data_dir}")
    print(f"classes     : {class_names}")
    print(f"image size  : {image_size}x{image_size}")
    print(f"max trials  : {max_trials}")
    print(f"epochs/trial: {epochs}")
    print(f"output dir  : {output_dir}")

    tuner = run_nas(
        train_ds, val_ds,
        max_trials=max_trials,
        epochs=epochs,
        seed=seed,
        directory=str(output_dir / "ak_runs"),
    )

    best_model = tuner.export_model()
    _plot_training_history(tuner, output_dir)

    model_path = output_dir / "best_nas_model"
    try:
        best_model.save(str(model_path), save_format="tf")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not save the model: {exc}")

    results = {
        "image_size": image_size,
        "max_trials": max_trials,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "classes": class_names,
        "splits": {},
    }
    for split_name, ds in (("val", val_ds), ("test", test_ds)):
        results["splits"][split_name] = evaluate_split(
            best_model, ds, class_names, split_name, output_dir
        )
        print(f"\n=== {split_name} split ===")
        print(results["splits"][split_name]["report"])

    if grad_cam:
        try:
            render_gradcam_for_samples(
                best_model, test_ds, class_names, output_dir=output_dir / "gradcam",
                num_samples=3,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Warning: Grad-CAM failed: {exc}")

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results
