"""Model factory.

Wraps `timm` so the rest of the package never imports from it directly. The
backbone is replaced with a fresh linear classifier matching the number of
classes in the resolved config.
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_classifier(model_name: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Create a timm backbone with a fresh classification head.

    EfficientNet-style models expose a `classifier` (a single Linear) at the
    end; we replace it with one whose out_features equals the resolved
    number of classes. For backbones that use a different head attribute
    (e.g. `fc`, `head`) timm normally still exposes `classifier` for
    compatibility, but we fall back just in case.
    """
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = _classifier_in_features(model)
    new_head = nn.Linear(in_features, num_classes)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        model.classifier = new_head
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        model.fc = new_head
    elif hasattr(model, "head") and isinstance(model.head, nn.Module):
        model.head = new_head
    else:
        raise RuntimeError(
            f"Could not find a classification head on {model_name!r} to replace."
        )
    return model


def _classifier_in_features(model: nn.Module) -> int:
    head = (
        getattr(model, "classifier", None)
        or getattr(model, "fc", None)
        or getattr(model, "head", None)
    )
    if head is None:
        raise RuntimeError("Model has no classifier/fc/head attribute.")
    # Walk down nn.Sequential until we find a Linear.
    if isinstance(head, nn.Linear):
        return head.in_features
    if isinstance(head, nn.Sequential):
        for layer in reversed(list(head)):
            if isinstance(layer, nn.Linear):
                return layer.in_features
    raise RuntimeError(f"Could not resolve in_features from head of type {type(head).__name__}.")
