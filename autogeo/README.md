# `autogeo` — PyTorch + timm EfficientNet-B1

A clean, reproducible training script for the **AutoGEO** approach described
in the paper. EfficientNet-B1 (timm) is fine-tuned on the same dataset that
the AutoKeras baseline (`../traditional_dl/`) trains on, so the two numbers
are directly comparable.

## What it does

1. Reads `data/Carbonate_lithofacies/{train,test}/<class>/*.jpg` (or any
   directory with the same shape).
2. Optionally remaps the seven raw classes to the 6- or 3-class setting
   described in the paper.
3. Trains EfficientNet-B1 (ImageNet pre-trained) with:
   - 200 × 400 input resolution
   - Albumentations: `CustomCutout`, `HorizontalFlip`, `VerticalFlip`, `Blur`,
     `Normalize` (ImageNet stats)
   - Adam, `CyclicLR` between 3e-5 and 6e-3
   - Weighted `CrossEntropyLoss`
4. Saves the best checkpoint by validation loss.
5. Reloads the best checkpoint and writes classification reports + confusion
   matrices for the train, validation, and test splits into `outputs/`.

## Install

```bash
poetry install
```

The lockfile pins `torch`, `torchvision`, `timm`, `albumentations`, etc.

## Train

```bash
poetry run python scripts/train.py \
    --data-dir /path/to/Carbonate_lithofacies \
    --config configs/7cls.yaml \
    --output-dir outputs/7cls
```

Available configs: `configs/7cls.yaml`, `configs/6cls.yaml`, `configs/3cls.yaml`.
They differ only in the class-merge map and the number of output classes.

Useful flags:

| flag                  | default                       | meaning                                   |
| --------------------- | ----------------------------- | ----------------------------------------- |
| `--data-dir`          | required                      | path to the unzipped dataset              |
| `--config`            | `configs/7cls.yaml`           | YAML config with class map + hyper-params |
| `--output-dir`        | `outputs/<config-stem>`       | where checkpoints and figures go          |
| `--epochs`            | from config (30)              | override the epoch count                  |
| `--batch-size`        | from config (32)              | train batch size                          |
| `--seed`              | 42                            | random seed                               |
| `--device`            | `cuda` if available else `cpu`| torch device                              |
| `--no-shuffle-seed`   | off                           | disable the pre-shuffling data split (paper's "non-shuffled" baseline) |

## Output layout

```
<output-dir>/
├── best.pt                          # best model weights by val loss
├── last.pt                          # final-epoch weights
├── history.json                     # per-epoch metrics
├── classification_report_train.txt
├── classification_report_val.txt
├── classification_report_test.txt
├── confusion_matrix_train.png
├── confusion_matrix_val.png
└── confusion_matrix_test.png
```

## Reproducing paper numbers

The paper reports weighted F1 ≈ 0.86 on a held-out well when training with
7 classes. Run all three configs back-to-back:

```bash
for cfg in 7cls 6cls 3cls; do
    poetry run python scripts/train.py \
        --data-dir /path/to/Carbonate_lithofacies \
        --config configs/${cfg}.yaml \
        --output-dir outputs/${cfg}
done
```

Then compare `outputs/*/classification_report_test.txt` to the paper's
Tables 2, 6 and 7.

## License

MIT.
