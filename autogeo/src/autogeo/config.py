"""Configuration loading & default paths.

Configs are plain YAML files. They describe which raw classes exist, how to
merge them for the 6- and 3-class experiments, and the training hyper-params.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Raw class names exactly as they appear in the dataset zip.
RAW_CLASSES: tuple[str, ...] = (
    "Boundstone",
    "Calcareous_Shale",
    "Dolomite",
    "Mudstone-Wackestone",
    "Packstone-Grainstone",
    "Sandstone",
    "Wackestone-Packstone",
)


@dataclass
class TrainConfig:
    """Resolved configuration for one training run."""

    name: str
    class_map: dict[str, str]
    classes: tuple[str, ...]
    image_height: int = 200
    image_width: int = 400
    epochs: int = 30
    train_batch: int = 32
    val_batch: int = 8
    test_batch: int = 8
    learning_rate_min: float = 3e-5
    learning_rate_max: float = 6e-3
    num_workers: int = 2
    model_name: str = "efficientnet_b1"
    seed: int = 42
    # When True, re-shuffles train data each run with a different seed (paper's
    # "pre-shuffling" approach). When False, uses a fixed-seed split (paper's
    # "non-shuffled" baseline).
    pre_shuffle: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def load_config(path: str | Path) -> TrainConfig:
    """Load a YAML config and resolve the merged class list."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    class_map = dict(raw.get("class_map") or {})
    # Anything missing from the map stays as-is.
    for raw_cls in RAW_CLASSES:
        class_map.setdefault(raw_cls, raw_cls)

    classes = tuple(sorted({class_map[c] for c in RAW_CLASSES}))

    train = raw.get("train", {})
    return TrainConfig(
        name=raw.get("name", path.stem),
        class_map=class_map,
        classes=classes,
        image_height=int(train.get("image_height", 200)),
        image_width=int(train.get("image_width", 400)),
        epochs=int(train.get("epochs", 30)),
        train_batch=int(train.get("train_batch", 32)),
        val_batch=int(train.get("val_batch", 8)),
        test_batch=int(train.get("test_batch", 8)),
        learning_rate_min=float(train.get("learning_rate_min", 3e-5)),
        learning_rate_max=float(train.get("learning_rate_max", 6e-3)),
        num_workers=int(train.get("num_workers", 2)),
        model_name=str(train.get("model_name", "efficientnet_b1")),
        seed=int(train.get("seed", 42)),
        pre_shuffle=bool(train.get("pre_shuffle", True)),
        extra={k: v for k, v in train.items() if k not in {
            "image_height", "image_width", "epochs", "train_batch", "val_batch",
            "test_batch", "learning_rate_min", "learning_rate_max",
            "num_workers", "model_name", "seed", "pre_shuffle",
        }},
    )
