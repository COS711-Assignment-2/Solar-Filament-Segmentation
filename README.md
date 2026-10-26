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

Run these commands from the project root:

```powershell
cd "C:\Users\morwa\Desktop\COS711 Assignment 2"
```

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

The audit writes machine-readable reports to `outputs/data_audit/`, including
the inventory, validation, annotation, and duplicate reports.

Create splits, masks, and processed images:

```powershell
python scripts/prepare_data.py --stage all `
  --train-images data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images `
  --annotations data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json `
  --split-dir data/splits `
  --processed-dir data/processed
```

Verify all generated outputs:

```powershell
python scripts/verify_prepared_data.py `
  --split-dir data/splits `
  --processed-dir data/processed
```

The `prepare_data.py --stage all` command performs the following stages:

1. Validates the raw images and COCO annotation JSON.
2. Creates leakage-safe grouped `train.csv` and `val.csv` manifests.
3. Rasterises polygons into instance and category masks.
4. Creates black-and-white binary masks.
5. Writes deterministic grayscale images.
6. Saves annotation provenance metadata.

Progress bars show the number of records completed, processing rate, and
estimated time remaining during long-running stages.

The generated structure is:

```text
data/
├── raw/
│   └── MAGFiLO_1.0_Kaggle_2026/
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── split_summary.json
└── processed/
    ├── train/
    │   ├── images/
    │   ├── masks/
    │   │   ├── binary/       # 0 = black background, 255 = white filament
    │   │   ├── categories/   # category ID masks
    │   │   └── instances/    # instance ID masks
    │   └── metadata/
    ├── val/
    │   └── same structure as train/
    └── preparation_summary.json
```

Binary masks are intended for easy visual inspection. They use `0` for
background and `255` for any filament pixel. Instance and category masks are
retained for model training and instance-level evaluation. Raw files are
never edited. Polygon overlaps are reported and currently use
last-annotation-wins raster order; confirm the official Kaggle rules before
using these masks for final PQ scoring.

Individual stages can be rerun when needed:

```powershell
# Validate only
python scripts/prepare_data.py --stage validate `
  --train-images data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images `
  --annotations data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json

# Recreate only the train/validation split
python scripts/prepare_data.py --stage split `
  --train-images data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images `
  --annotations data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json

# Regenerate processed images and masks
python scripts/prepare_data.py --stage process `
  --train-images data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images `
  --annotations data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json `
  --split-dir data/splits `
  --processed-dir data/processed
```

## Model training and experiments

The segmentation trainer reads the processed train/validation manifests and
instance masks. Before training, regenerate prepared data with the command
above if it was created by an earlier version of this repository; image
preprocessing previously could overwrite generated images with mask pixels.
Training output reports the training loss and validation panoptic quality
(PQ), along with the selected postprocessing threshold. PQ is the validation
selection metric for this instance-segmentation task; the output does not
report pixel-level Dice or IoU.

```powershell
python3 scripts/train.py --experiment unet
```

Keep one clearly named configuration under `experiments/<name>/config.yaml`.
The repository includes a U-Net baseline and a separate boundary-aware U-Net
example. Copy an experiment folder for each new run, give it a unique
`experiment.name`, then train it by folder name. For example:

```powershell
python3 scripts/train.py --experiment boundary_aware_unet
```

Override maximum epochs and early-stopping patience for a run:

```bash
python3 scripts/train.py --experiment unet --max-epochs 100 --patience 15
```

Use `--patience 0` to disable early stopping and train for the full maximum
epoch count.

Configuration options:

- `model.name`: a registered architecture (`unet` or `boundary_aware_unet`).
  Each architecture has its own module under `src/models/`; register future
  architectures with `register_model` in `src/models/__init__.py`. The trainer
  depends on the model registry, not on model-specific imports or conditionals.
- `experiment.name`: a unique name for the run. Checkpoints and metrics are
  stored separately under `runs/<model.name>/<experiment.name>/<run-tag>/`, so
  comparing architectures, seeds, or experiment settings does not overwrite
  earlier results.
- `data.resolution`: resize images and masks to `512` or `1024`, or use
  `tiled` with `data.tile_size` for fixed-size non-overlapping crops.
- `loss.bce_weight` and `loss.dice_weight`: independently weight BCE and soft
  Dice loss; `loss.boundary_weight` controls the auxiliary boundary loss.
- `evaluation.postprocessing`: `connected_components` or `watershed`.
- `evaluation.thresholds`: candidate foreground thresholds. Each epoch scores
  every candidate using instance-level validation PQ and records the best one
  in the run's checkpoint and metrics files.
- `training.device`: `auto` prefers an available TPU, then CUDA, then CPU;
  `cpu`, `cuda`, and `tpu` can also be selected explicitly. TPU training uses
  PyTorch/XLA for device placement, optimizer steps, and checkpoint saving.
  Install a `torch_xla` release compatible with the PyTorch version supplied by
  the TPU runtime; avoid replacing the runtime's PyTorch/XLA packages with
  unrelated versions from `requirements.txt`.
- TPU launch can set the PJRT backend before Python starts:
  `PJRT_DEVICE=TPU GLS_DEVICE=xla python3 scripts/train.py --experiment unet
  --device xla`. The trainer also sets `PJRT_DEVICE=TPU` before importing
  PyTorch/XLA when no PJRT backend was explicitly configured.
- `training.max_epochs` sets the maximum number of epochs. `training.patience`
  enables early stopping when validation PQ stops improving. Progress bars
  show epoch, training-batch, and validation-image progress; the best-PQ
  checkpoint is kept.

See [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) for the complete layout,
single-experiment commands, TPU usage, and multi-seed runner.

PQ uses the project's provisional class-agnostic instance matching rule
(IoU > 0.5). Confirm the competition's exact PQ, category, and ignored-region
rules before interpreting these scores as official results. Thresholds and
post-processing must be selected on validation data, never on the unlabelled
test set.

Run tests with:

```powershell
python -m pytest -q
```

## Dataset uncertainty

The documented dataset is expected to contain grayscale 2048x2048 images,
but the implementation inspects actual files and does not require those
dimensions. Exact PQ matching, ignored-region treatment, category semantics,
and submission encoding remain subject to the official competition rules.
