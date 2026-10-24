# COS711-A2 data preparation

This repository contains a reproducible preparation pipeline for the
MAGFiLO solar-filament segmentation assignment. The pipeline does not require
Kaggle access for tests.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

KaggleHub uses its normal credential configuration. Credentials must never be
committed to this project.

On Windows, authenticate before downloading using one of KaggleHub's supported
methods:

```powershell
# Option 1: set the token for the current PowerShell session
$env:KAGGLE_API_TOKEN = "<your-kaggle-api-token>"

# Option 2: use Kaggle's credentials file
# Save the downloaded Kaggle API credentials as:
# $env:USERPROFILE\.kaggle\kaggle.json
```

Alternatively, authenticate interactively from Python with
`kagglehub.login()`. Do not paste tokens into source files, `.env` files that
are committed, or command history shared with others. The competition must
also be accessible to the Kaggle account being used.

## Workflow

Download and discover the competition files:

```powershell
python scripts/download_data.py
```

By default, KaggleHub downloads the original competition files directly into
`data/raw/`. Use `--staging-dir` to select another raw-data directory. The
pipeline never places generated masks or preprocessed images in `data/raw/`;
those belong in `data/processed/`.

Audit an extracted dataset:

```powershell
python scripts/audit_data.py --root data/raw/MAGFiLO_1.0_Kaggle_2026 `
  --train-images data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images `
  --annotations data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
```

Validate and split by normalised physical filename:

```powershell
python scripts/prepare_data.py --stage all --train-images <train-images> --annotations <annotations>
python scripts/verify_prepared_data.py
```

The reusable modules are in `src/data`: discovery, inventory, validation,
annotation inspection, duplicate reporting, grouped splitting, polygon
rasterisation, and deterministic image preprocessing. Raw files are never
edited. Polygon overlaps are reported and use last-annotation-wins raster
order as an explicit provisional policy; confirm the official Kaggle rules
before using these masks for final PQ scoring.

Run tests with:

```powershell
python -m pytest -q
```

## Dataset uncertainty

The documented dataset is expected to contain grayscale 2048x2048 images,
but the implementation inspects actual files and does not require those
dimensions. Exact PQ matching, ignored-region treatment, category semantics,
and submission encoding remain subject to the official competition rules.
