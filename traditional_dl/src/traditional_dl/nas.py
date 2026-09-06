"""AutoKeras Neural Architecture Search wrapper."""
from __future__ import annotations

import autokeras as ak
import tensorflow as tf


def run_nas(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    max_trials: int = 5,
    epochs: int = 50,
    seed: int = 42,
    project_name: str = "ak_carbonate_nas",
    directory: str = "ak_runs",
) -> ak.AutoModel:
    """Run the AutoKeras NAS and return the trained `AutoModel` tuner.

    The search space is "vanilla" CNN (per the original `AutoDL core
    carbonate.py` reference) with normalization and augmentation in scope.
    """
    input_node = ak.ImageInput()
    output_node = ak.ImageBlock(
        block_type="vanilla",
        normalize=True,
        augment=True,
    )(input_node)
    output_node = ak.ClassificationHead()(output_node)

    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        max_trials=max_trials,
        seed=seed,
        project_name=project_name,
        directory=directory,
    )
    clf.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return clf
