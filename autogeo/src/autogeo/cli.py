"""Command-line entry point: `autogeo-train` / `python scripts/train.py`."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from autogeo import evaluate, load_config
from autogeo.data import make_dataloaders
from autogeo.train import fit


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autogeo-train",
        description="Train EfficientNet-B1 (timm) for Dunham texture classification.",
    )
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Path to the unzipped dataset (contains train/ and test/ sub-folders).")
    p.add_argument("--config", required=True, type=Path,
                   help="YAML config (see configs/).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write checkpoints and reports. Defaults to outputs/<config-stem>.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override train_batch from the config.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None,
                   help="torch device (e.g. 'cuda', 'cuda:0', 'cpu'). Default: cuda if available else cpu.")
    p.add_argument("--no-shuffle-seed", action="store_true",
                   help="Disable the pre-shuffling data split (paper's 'non-shuffled' baseline).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    if args.no_shuffle_seed:
        cfg.pre_shuffle = False
    _seed_everything(seed)

    output_dir = args.output_dir or Path("outputs") / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size is not None:
        cfg.train_batch = args.batch_size
    epochs = args.epochs if args.epochs is not None else cfg.epochs

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Persist the resolved config so the run is self-describing.
    (output_dir / "config.json").write_text(
        json.dumps({
            "name": cfg.name,
            "classes": list(cfg.classes),
            "class_map": cfg.class_map,
            "epochs": epochs,
            "train_batch": cfg.train_batch,
            "val_batch": cfg.val_batch,
            "test_batch": cfg.test_batch,
            "image_height": cfg.image_height,
            "image_width": cfg.image_width,
            "learning_rate_min": cfg.learning_rate_min,
            "learning_rate_max": cfg.learning_rate_max,
            "model_name": cfg.model_name,
            "seed": seed,
            "pre_shuffle": cfg.pre_shuffle,
        }, indent=2),
        encoding="utf-8",
    )

    splits = make_dataloaders(
        data_dir=args.data_dir,
        class_map=cfg.class_map,
        classes=cfg.classes,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        seed=seed,
        pre_shuffle=cfg.pre_shuffle,
    )

    print(f"== autogeo training ==")
    print(f"config      : {args.config}")
    print(f"data dir    : {args.data_dir}")
    print(f"classes     : {splits.classes}")
    print(f"train/val/test sizes: {len(splits.train_df)}/{len(splits.val_df)}/{len(splits.test_df)}")
    print(f"epochs      : {epochs}")
    print(f"device      : {device}")
    print(f"output dir  : {output_dir}")

    fit_result = fit(
        splits,
        epochs=epochs,
        learning_rate_min=cfg.learning_rate_min,
        learning_rate_max=cfg.learning_rate_max,
        device=device,
        output_dir=output_dir,
    )

    if fit_result["best_state"] is not None:
        eval_results = evaluate.evaluate_all_splits(
            state_dict=fit_result["best_state"],
            model_name=cfg.model_name,
            num_classes=len(splits.classes),
            splits={"train": splits.train, "val": splits.val, "test": splits.test},
            classes=splits.classes,
            output_dir=output_dir,
            device=device,
        )
        (output_dir / "evaluation.json").write_text(
            json.dumps(eval_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("\n=== Test split classification report ===")
        print(eval_results["splits"]["test"]["report"])

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
