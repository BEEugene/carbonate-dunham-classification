"""autogeo: PyTorch + timm EfficientNet-B1 baseline for Dunham texture classification."""

from autogeo.config import RAW_CLASSES, TrainConfig, load_config

__all__ = [
    "RAW_CLASSES",
    "TrainConfig",
    "load_config",
    "data",
    "model",
    "train",
    "evaluate",
]

__version__ = "0.1.0"
