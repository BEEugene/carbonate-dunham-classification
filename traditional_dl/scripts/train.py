"""Thin wrapper that calls the traditional_dl CLI.

Usage:
    python scripts/train.py --data-dir /path/to/Carbonate_lithofacies
"""
from traditional_dl.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
