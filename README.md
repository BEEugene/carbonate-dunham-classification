# Carbonate Dunham Texture Classification

Reproducible code for the paper:

> Baraboshkin, E., Koeshidayatullah, A., Demidov, A., Panchenko, E., Orlov, D., Koroteev, D.
> *Self-recognition of Dunham textures in subsurface carbonate cores with deep convolutional neural networks.*
> Computers & Geosciences (under review).

The repository contains two independently runnable pipelines that train a classifier on
subsurface carbonate core images (Dunham textures + non-carbonate facies):

| package             | approach                                      | framework                       |
| ------------------- | --------------------------------------------- | ------------------------------- |
| `autogeo/`          | hand-designed CNN (EfficientNet-B1, timm)     | PyTorch + Albumentations        |
| `traditional_dl/`   | Neural Architecture Search (NAS, AutoKeras)   | TensorFlow + AutoKeras          |

Both pipelines consume the **same** dataset (see `data/`), support the same three
class-configurations studied in the paper (7 / 6 / 3 classes), and report precision,
recall, F1, accuracy and a confusion matrix.

## Repository layout

```
carbonate-dunham-classification/
├── README.md                  # this file
├── LICENSE                    # MIT
├── CITATION.cff
├── .gitignore
├── .gitattributes
├── data/
│   └── README.md             # how to fetch the dataset release asset
├── autogeo/                   # PyTorch + timm EfficientNet-B1
│   ├── pyproject.toml         # poetry env
│   ├── README.md
│   ├── configs/               # YAML configs for 7/6/3 classes
│   ├── scripts/train.py
│   └── src/autogeo/
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       └── evaluate.py
└── traditional_dl/            # AutoKeras NAS
    ├── pyproject.toml
    ├── README.md
    ├── scripts/train.py
    └── src/traditional_dl/
        ├── data.py
        ├── nas.py
        ├── train.py
        └── evaluate.py
```

## Dataset

The labelled image dataset is distributed as a **GitHub Release asset** for
this repo (≈130 MB, see `data/README.md`). Pull it with:

```bash
gh release download --repo BEEugene/carbonate-dunham-classification \
    --pattern 'Carbonate_lithofacies.zip' --dir .
unzip Carbonate_lithofacies.zip -d /path/to/data
```

The expected layout after unzipping is:

```
<unzipped>/train/<class_name>/*.jpg
<unzipped>/test/<class_name>/*.jpg
```

The seven Dunham + non-carbonate classes are:

| folder name          | description                                         |
| -------------------- | --------------------------------------------------- |
| `Boundstone`         | Dunham boundstone (autochthonous framework)         |
| `Calcareous_Shale`   | calcareous / argillaceous shale                     |
| `Dolomite`           | dolostone                                           |
| `Mudstone-Wackestone`| mud-dominated carbonate                             |
| `Packstone-Grainstone`| grain-dominated carbonate                          |
| `Sandstone`          | siliciclastic sandstone                             |
| `Wackestone-Packstone`| transitional carbonate                             |

For the 6-class experiment `Wackestone-Packstone` is folded into
`Packstone-Grainstone`; for the 3-class experiment all carbonates are merged
into one super-class. The mappings live in the configs of each pipeline.

## Quick start

```bash
# 1. one-time environment per pipeline
cd autogeo
poetry install
cd ../traditional_dl
poetry install
cd ..

# 2. materialise dataset (downloads the release asset)
gh release download --repo BEEugene/carbonate-dunham-classification \
    --pattern 'Carbonate_lithofacies.zip' --dir .
unzip Carbonate_lithofacies.zip -d /path/to/data

# 3. train one of the pipelines
poetry --directory autogeo     run python scripts/train.py --data-dir /path/to/data --config configs/7cls.yaml
poetry --directory traditional_dl run python scripts/train.py --data-dir /path/to/data
```

See each sub-package's `README.md` for the full set of flags and outputs.

## Citation

If you use this code in academic work, please cite the paper using the metadata
in `CITATION.cff`.

## License

MIT — see `LICENSE`.
