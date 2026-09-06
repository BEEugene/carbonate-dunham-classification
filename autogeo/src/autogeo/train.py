"""Training loop for the autogeo EfficientNet-B1 pipeline."""
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from autogeo.data import DataSplits
from autogeo.model import build_classifier


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    train_f1_weighted: float
    train_precision_weighted: float
    train_recall_weighted: float
    val_loss: float
    val_acc: float
    val_f1_weighted: float
    val_precision_weighted: float
    val_recall_weighted: float
    epoch_seconds: float


def _move_batch(batch, device: torch.device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    train: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass over `loader`. Returns (mean_loss, y_true, y_pred)."""
    if train:
        model.train()
    else:
        model.eval()

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x, y = _move_batch(batch, device)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            losses.append(loss.item())
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    return float(np.mean(losses)), y_true_arr, y_pred_arr


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def fit(
    splits: DataSplits,
    *,
    epochs: int,
    learning_rate_min: float,
    learning_rate_max: float,
    device: str | torch.device = "cpu",
    output_dir: str | Path | None = None,
    log_every: int = 1,
) -> dict:
    """Run the full training loop and (optionally) persist artifacts."""
    device = torch.device(device)
    model = build_classifier(
        model_name=getattr(splits, "model_name", "efficientnet_b1"),
        num_classes=len(splits.classes),
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=splits.class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=learning_rate_min)
    # Match the original notebook: one full cycle every ~20 mini-steps
    # (step_size_down is in *iterations*, not epochs).
    step_size_down = max(1, 10 * max(1, len(splits.train)))
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate_min,
        max_lr=learning_rate_max,
        cycle_momentum=False,
        step_size_down=step_size_down,
    )

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()
        train_loss, y_t, y_p = _epoch_pass(
            model, splits.train, criterion, device,
            optimizer=optimizer, scheduler=scheduler, train=True,
        )
        train_scores = _score(y_t, y_p)

        val_loss, y_v, y_vp = _epoch_pass(
            model, splits.val, criterion, device, train=False,
        )
        val_scores = _score(y_v, y_vp)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if output_dir is not None:
                torch.save(best_state, output_dir / "best.pt")

        elapsed = time.monotonic() - t0
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_scores.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_scores.items()},
            "epoch_seconds": elapsed,
        }
        history.append(rec)

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:02d}/{epochs} "
                f"| t={elapsed:.1f}s "
                f"| train loss={train_loss:.3f} acc={train_scores['accuracy']:.3f} "
                f"| val loss={val_loss:.3f} acc={val_scores['accuracy']:.3f}",
                flush=True,
            )

    if output_dir is not None:
        torch.save(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            output_dir / "last.pt",
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_state": best_state,
        "num_classes": len(splits.classes),
    }
