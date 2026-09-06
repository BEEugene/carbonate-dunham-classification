"""Thin wrapper that calls the autogeo CLI.

Usage:
    python scripts/train.py --data-dir /path/to/Carbonate_lithofacies --config configs/7cls.yaml
"""
from autogeo.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
