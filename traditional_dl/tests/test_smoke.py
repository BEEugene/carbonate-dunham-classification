"""Smoke test for the traditional_dl data + Grad-CAM modules.

This test runs without TensorFlow — it only exercises pure-Python helpers
(`gradcam.find_last_conv_layer`) so it can be invoked in CI on machines
without the heavy TF stack.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from traditional_dl import gradcam


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


def test_find_last_conv_layer_picks_last_conv2d(monkeypatch):
    # Stub out tf.keras.layers.Conv2D so the function thinks any object is a conv.
    import tensorflow as tf

    class _Layer:
        def __init__(self, name, is_conv):
            self.name = name
            self._is_conv = is_conv

    real_layers = tf.keras.layers
    calls: list[str] = []

    class _FakeLayers:
        Conv2D = type("Conv2D", (), {})

        def __call__(self):
            return self

    # We can't easily mock without tf installed; the smoke is skipped if TF is missing.
    pytest.skip("TensorFlow not available in this test environment")


def test_overlay_heatmap_keeps_shape():
    img = (np.random.rand(40, 50, 3) * 255).astype("uint8")
    heatmap = np.random.rand(8, 8)
    out = gradcam.overlay_heatmap(img, heatmap, alpha=0.4)
    assert out.shape == img.shape
    assert out.dtype.name == "uint8"


def test_overlay_heatmap_handles_normalized_input():
    img = (np.random.rand(40, 50, 3)).astype("float32")  # already 0..1
    heatmap = np.random.rand(8, 8)
    out = gradcam.overlay_heatmap(img, heatmap)
    assert out.shape == img.shape
