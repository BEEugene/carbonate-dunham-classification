# `traditional_dl` — AutoKeras Neural Architecture Search

A clean, reproducible re-implementation of the **AutoDL baseline** described
in the paper. The original `AutoDL core carbonate.py` notebook shipped with
the dataset is kept as a reference; this package is the production version.

AutoKeras performs a Bayesian-optimised Neural Architecture Search (NAS)
across a small search space of CNN topologies, picking both the architecture
and its pre-processing (normalization + augmentation) blocks.

## What it does

1. Reads `data/Carbonate_lithofacies/{train,test}/<class>/*.jpg`.
2. Carves an internal validation split off the train split.
3. Runs `ak.ImageClassifier` with `max_trials` NAS trials × `epochs` epochs.
4. Evaluates the best discovered model on the test split and writes
   classification reports and confusion matrices.

## Install

```bash
poetry install
```

> AutoKeras 1.x pulls in `tensorflow` 2.10–2.15. If you are on a GPU
> machine, install the matching `tensorflow-gpu` build before this
> command.

## Train

```bash
poetry run python scripts/train.py \
    --data-dir /path/to/Carbonate_lithofacies \
    --output-dir outputs/nas
```

Useful flags:

| flag               | default            | meaning                                                |
| ------------------ | ------------------ | ------------------------------------------------------ |
| `--data-dir`       | required           | path to the unzipped dataset                           |
| `--output-dir`     | `outputs/nas`      | where reports and the best model go                    |
| `--image-size`     | 128                | square input resolution (NAS-friendly)                 |
| `--max-trials`     | 5                  | number of architectures AutoKeras explores             |
| `--epochs`         | 50                 | epochs per trial                                       |
| `--batch-size`     | 32                 | training batch size                                    |
| `--seed`           | 42                 | random seed                                            |
| `--no-grad-cam`    | off                | skip the Grad-CAM interpretability step                |

## Output layout

```
<output-dir>/
├── best_nas_model/                  # exported Keras model
├── classification_report_test.txt
├── confusion_matrix_test.png
├── training_history.png             # loss / accuracy curves
└── gradcam/                         # heatmaps per sample
    ├── sample_0_*.png
    ├── ...
```

## Reproducing paper numbers

Run the same command three times — once per class configuration — by
passing the appropriate `--class-map` flag (or editing the script):

```bash
poetry run python scripts/train.py --data-dir /path/to/Carbonate_lithofacies --output-dir outputs/nas_7cls
# 6-class and 3-class configs are handled by adjusting the train folder
# layout — see the `class_map` discussion in autogeo/README.md.
```

The paper's Table 2 (right) reports the AutoDL numbers; this script aims
to reproduce them.

## License

MIT.
