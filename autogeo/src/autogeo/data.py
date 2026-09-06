"""Dataset loaders.

Reads the `train/<class>/*.jpg` and `test/<class>/*.jpg` directory layout and
returns PyTorch Datasets. The validation split is carved from the train split
inside the train script (a stratified 80/20 split by default) — the raw zip
does not provide a `val/` split.
"""
from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ImageNet stats — every backbone in timm trained on these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    raw_class: str
    mapped_class: str
    label: int  # mapped_class index in the resolved class list


# -----------------------------------------------------------------------------
# Custom Cutout
# -----------------------------------------------------------------------------


class CustomCutout(albu.DualTransform):
    """Square Cutout that drops a randomly placed patch to a fill value.

    Adapted from the original autogeo notebook (see `references/` for the
    lineage). Kept here as an Albumentations `DualTransform` so it composes
    cleanly with the rest of the augmentation pipeline.
    """

    def __init__(
        self,
        fill_value: int = 0,
        min_cutout_size: int = 30,
        max_cutout_size: int = 50,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.fill_value = fill_value
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size

    def apply(self, image: np.ndarray, **params) -> np.ndarray:  # type: ignore[override]
        image = image.copy()
        h, w = image.shape[:2]
        size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
        size = min(size, h - 1, w - 1)
        if size <= 0:
            return image
        y = np.random.randint(0, h - size + 1)
        x = np.random.randint(0, w - size + 1)
        image[y : y + size, x : x + size, :] = self.fill_value
        return image


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------


def discover_samples(
    root: str | Path,
    class_map: dict[str, str],
    classes: Sequence[str],
) -> list[SampleRecord]:
    """Walk `root/<raw_class>/*.jpg` and produce resolved `SampleRecord`s."""
    root = Path(root)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    records: list[SampleRecord] = []
    for raw_dir in sorted(root.iterdir()):
        if not raw_dir.is_dir():
            continue
        raw_cls = raw_dir.name
        if raw_cls not in class_map:
            continue
        mapped = class_map[raw_cls]
        label = class_to_idx[mapped]
        for p in sorted(raw_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            records.append(SampleRecord(path=p, raw_class=raw_cls, mapped_class=mapped, label=label))
    if not records:
        raise FileNotFoundError(
            f"No image files found under {root}. "
            f"Expected one sub-folder per class from {list(class_map)}."
        )
    return records


def to_dataframe(records: Sequence[SampleRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "path": str(r.path),
        "raw_class": r.raw_class,
        "class": r.mapped_class,
        "label": r.label,
    } for r in records])


# -----------------------------------------------------------------------------
# Augmentations
# -----------------------------------------------------------------------------


def train_transforms(height: int, width: int) -> albu.Compose:
    return albu.Compose([
        CustomCutout(p=1.0),
        albu.Resize(height=height, width=width),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Blur(blur_limit=3, p=0.3),
        albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms(height: int, width: int) -> albu.Compose:
    return albu.Compose([
        albu.Resize(height=height, width=width),
        albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


class CoreDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: albu.Compose,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = cv2.imread(row["path"])
        if image is None:
            raise RuntimeError(f"cv2.imread returned None for {row['path']!r}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out = self.transform(image=image)
        return out["image"], int(row["label"])


def _collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [b for b in batch if b is not None]
    if not batch:
        raise RuntimeError("Empty batch after filtering None samples.")
    return torch.utils.data.default_collate(batch)


# -----------------------------------------------------------------------------
# Top-level helpers
# -----------------------------------------------------------------------------


@dataclass
class DataSplits:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    classes: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_weights: torch.Tensor


def make_dataloaders(
    data_dir: str | Path,
    class_map: dict[str, str],
    classes: Sequence[str],
    *,
    image_height: int = 200,
    image_width: int = 400,
    train_batch: int = 32,
    val_batch: int = 8,
    test_batch: int = 8,
    num_workers: int = 2,
    seed: int = 42,
    pre_shuffle: bool = True,
    val_size: float = 0.2,
) -> DataSplits:
    """Build train/val/test DataLoaders from the standard directory layout.

    The validation split is carved from the train split (stratified by label).
    When `pre_shuffle=False` the split is deterministic; when `pre_shuffle=True`
    the data is shuffled beforehand with a fresh seed (the paper's
    "pre-shuffling" approach).
    """
    data_dir = Path(data_dir)
    train_records = discover_samples(data_dir / "train", class_map, classes)
    test_records = discover_samples(data_dir / "test", class_map, classes)
    train_df = to_dataframe(train_records)
    test_df = to_dataframe(test_records)

    if pre_shuffle:
        # Paper's "pre-shuffling" experiment: re-shuffle the training pool with
        # a different seed each run, then carve a fixed-size val split.
        split_seed = seed + 1
    else:
        # Paper's "non-shuffled" baseline: fixed split, no extra shuffling.
        split_seed = seed

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=split_seed,
        shuffle=True,
        stratify=train_df["label"].values,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_ds = CoreDataset(train_df, train_transforms(image_height, image_width))
    val_ds = CoreDataset(val_df, eval_transforms(image_height, image_width))
    test_ds = CoreDataset(test_df, eval_transforms(image_height, image_width))

    common = dict(collate_fn=_collate, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False, **common)

    class_weights = _compute_class_weights(train_df["label"].values, len(classes))

    return DataSplits(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        classes=list(classes),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_weights=class_weights,
    )


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div-by-zero on unseen classes
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)
