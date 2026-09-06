# Dataset

The dataset is distributed as a **GitHub Release asset** for this repository
(download link in the repository's "Releases" section). It is **not**
tracked in git because the file is ≈130 MB and would exceed the Git LFS
free-tier quota.

## Download

Get the latest release here:

```
https://github.com/BEEugene/carbonate-dunham-classification/releases/latest
```

The asset is named `Carbonate_lithofacies.zip` (≈130 MB). You can also
fetch it with `gh`:

```bash
gh release download --repo BEEugene/carbonate-dunham-classification \
    --pattern 'Carbonate_lithofacies.zip' --dir .
unzip Carbonate_lithofacies.zip -d /path/to/data
```

## Layout

After unzipping, point the training scripts at the resulting directory. The
expected layout is:

```
Carbonate_lithofacies/
├── train/
│   ├── Boundstone/*.jpg
│   ├── Calcareous_Shale/*.jpg
│   ├── Dolomite/*.jpg
│   ├── Mudstone-Wackestone/*.jpg
│   ├── Packstone-Grainstone/*.jpg
│   ├── Sandstone/*.jpg
│   └── Wackestone-Packstone/*.jpg
└── test/
    └── <same 7 class folders>
```

## Provenance

Images are 10×10 cm crops of full-bore core box photographs collected from
the Finnmark Platform (Barents Sea, Permian/Carboniferous) and the Khoreyver
depression (Pechora syneclise, Silurian–Permian). Labels were assigned by a
geologist according to the Dunham classification; the siliciclastic
(`Sandstone`) and shaly (`Calcareous_Shale`) classes are non-carbonate but
share the same photographic scale and were retained for the multi-class
experiments reported in the paper.

For full attribution see the paper's *Materials and methods → Dataset*
section.
