"""Data helpers for the AutoKeras pipeline.

The original `AutoDL core carbonate.py` script expected a `train/` and `val/`
layout. The shipped dataset zip only provides `train/` and `test/`, so we
read both, and expose a `make_datasets()` function that returns three
`tf.data.Dataset` objects.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def make_datasets(
    data_dir: str | Path,
    *,
    image_size: int = 128,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build train/val/test `tf.data.Dataset`s from the standard layout.

    The validation split is carved from the train split (stratification is
    not directly available via `image_dataset_from_directory` so we just
    use a `tf.data.experimental.cardinality` shuffle + take/skip pattern).
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train"
    test_path = data_dir / "test"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected both {train_path} and {test_path} — make sure you "
            f"unzipped Carbonate_lithofacies.zip into {data_dir}."
        )

    # Label inference & class order come from the train split.
    full_train = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )
    class_names = list(full_train.class_names)

    # Carve out a val split deterministically.
    val_batches = max(1, int(round(val_split * tf.data.experimental.cardinality(full_train).numpy())))
    val_ds = full_train.take(val_batches)
    train_ds = full_train.skip(val_batches)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names
