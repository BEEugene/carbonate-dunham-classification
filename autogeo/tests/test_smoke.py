"""Smoke test: build a tiny fake dataset and verify the data pipeline works."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from autogeo.config import load_config
from autogeo.data import (
    CoreDataset,
    CustomCutout,
    discover_samples,
    eval_transforms,
    make_dataloaders,
    train_transforms,
)


def _make_fake_dataset(root: Path, classes: list[str], per_class: int = 2) -> None:
    for split in ("train", "test"):
        for c in classes:
            d = root / split / c
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                img = (np.random.rand(60, 80, 3) * 255).astype("uint8")
                cv2.imwrite(str(d / f"{c}_{i}.jpg"), img)


def test_custom_cutout_runs():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    out = CustomCutout(p=1.0, min_cutout_size=5, max_cutout_size=10)(image=img)["image"]
    assert out.shape == img.shape
    # At least one patch was zeroed
    assert (out == 0).any()


def test_transforms_produce_tensors():
    img = (np.random.rand(60, 80, 3) * 255).astype("uint8")
    t = train_transforms(32, 32)(image=img)["image"]
    assert t.shape == (3, 32, 32)
    t = eval_transforms(32, 32)(image=img)["image"]
    assert t.shape == (3, 32, 32)


def test_discover_and_dataloaders(tmp_path: Path):
    classes = ["Boundstone", "Dolomite", "Sandstone"]
    _make_fake_dataset(tmp_path, classes, per_class=4)
    recs = discover_samples(tmp_path / "train", {c: c for c in classes}, classes)
    assert len(recs) == 3 * 4
    splits = make_dataloaders(
        data_dir=tmp_path,
        class_map={c: c for c in classes},
        classes=classes,
        image_height=32,
        image_width=32,
        train_batch=2,
        val_batch=2,
        test_batch=2,
        num_workers=0,
        seed=0,
        pre_shuffle=False,
    )
    assert len(splits.train_df) >= 6  # 4 per class * 3 classes, 80/20 split
    assert len(splits.test_df) == 3 * 4
    assert splits.classes == classes


def test_load_config_identity():
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "7cls.yaml")
    assert cfg.name == "7cls"
    assert "Boundstone" in cfg.classes
    assert cfg.num_classes == 7


def test_load_config_3cls_collapses_carbonates():
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "3cls.yaml")
    assert cfg.num_classes == 3
    carbonate_super = next(c for c in cfg.classes if "Boundstone" in c)
    assert "Packstone-Grainstone" in carbonate_super
    assert "Dolomite" in cfg.classes
    assert "Sandstone" in cfg.classes
