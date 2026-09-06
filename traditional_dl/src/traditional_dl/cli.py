"""CLI for the traditional_dl AutoKeras pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from traditional_dl.train import run


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traditional-dl-train",
        description="Train an AutoKeras NAS image classifier for Dunham textures.",
    )
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/nas"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-trials", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-grad-cam", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        max_trials=args.max_trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        grad_cam=not args.no_grad_cam,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
